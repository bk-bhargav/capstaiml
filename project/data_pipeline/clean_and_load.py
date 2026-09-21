"""
clean_and_load.py
==================
Reads data_pipeline/data/raw_books.csv (produced by scrape.py), cleans and
types every field, converts GBP -> INR at the project's fixed baseline rate,
and loads the result into a normalized two-table SQLite schema.

Cleaning rules & how messy rows are handled
--------------------------------------------
1. price_gbp (float):
   Strip the leading "£" and parse as float. If a price string is malformed
   (e.g. "£N/A"), we DO NOT drop the row. Price is a continuous numeric
   field describing books we already know the title/category/rating for, so
   throwing the whole book away because of one bad numeric token loses
   information for no good reason. Instead we flag it and, at the end,
   MEDIAN-IMPUTE it from the successfully-parsed prices *within the same
   category* (falling back to the global median if a whole category failed).
   Median (not mean) is used because book prices are not symmetric and a
   single bad value could otherwise skew a mean-based fill.

2. rating (int, 1-5):
   The site encodes rating as one of exactly five words: One..Five. This is
   a categorical/ordinal field with no natural "typical" value to impute --
   inventing a star rating for a book risks materially misrepresenting it in
   a dataset for consumer-book listings, and there are only two rows (if
   any) with bad values, so being conservative costs us almost nothing.
   Decision: rows whose star_rating text doesn't map to one of the five
   known words are DROPPED, and each drop is logged to stdout with a reason.

3. in_stock (bool):
   Parsed from the availability text: "in stock" (case-insensitive,
   substring match) -> True, "out of stock" -> False. Like rating, stock
   status is categorical (there is no numeric "average" stock status to
   impute), so any availability text that matches neither pattern causes
   that row to be DROPPED and logged, for the same reason as (2).

4. price_inr (float):
   price_gbp * GBP_TO_INR, where GBP_TO_INR = 105.50 is a fixed,
   project-defined constant (NOT a live/historical FX rate -- see README).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
RAW_CSV = REPO_ROOT / "data" / "raw_books.csv"
CLEAN_CSV = REPO_ROOT / "data" / "clean_books.csv"
DB_PATH = REPO_ROOT / "db" / "books.db"

# Fixed, project-defined baseline conversion rate (see README "Currency
# conversion" section). This is a constant, not a market lookup.
GBP_TO_INR = 105.50

RATING_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(raw: str) -> float | None:
    """Strip currency symbol and parse to float. Returns None on failure."""
    try:
        cleaned = str(raw).replace("£", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_rating(raw: str) -> int | None:
    return RATING_WORD_TO_INT.get(str(raw).strip().title())


def parse_in_stock(raw: str) -> bool | None:
    text = str(raw).strip().lower()
    if "out of stock" in text:
        return False
    if "in stock" in text:
        return True
    return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_gbp_parsed"] = df["price"].apply(parse_price)
    df["rating_parsed"] = df["star_rating"].apply(parse_rating)
    df["in_stock_parsed"] = df["availability"].apply(parse_in_stock)

    # --- rule 2 & 3: drop rows with unparseable categorical fields --------
    bad_rating_mask = df["rating_parsed"].isna()
    bad_stock_mask = df["in_stock_parsed"].isna()
    for _, row in df[bad_rating_mask].iterrows():
        print(f"[drop] '{row['title']}': unparseable star_rating={row['star_rating']!r}")
    for _, row in df[bad_stock_mask].iterrows():
        print(f"[drop] '{row['title']}': unparseable availability={row['availability']!r}")
    df = df[~bad_rating_mask & ~bad_stock_mask].copy()

    # --- rule 1: median-impute unparseable numeric price -------------------
    bad_price_mask = df["price_gbp_parsed"].isna()
    if bad_price_mask.any():
        # per-category median first, global median as a fallback
        category_medians = df.groupby("category")["price_gbp_parsed"].transform("median")
        global_median = df["price_gbp_parsed"].median()
        for _, row in df[bad_price_mask].iterrows():
            print(f"[impute] '{row['title']}': unparseable price={row['price']!r}, "
                  f"filling with category median")
        df.loc[bad_price_mask, "price_gbp_parsed"] = category_medians[bad_price_mask].fillna(global_median)

    df = df.rename(columns={
        "price_gbp_parsed": "price_gbp",
        "rating_parsed": "rating",
        "in_stock_parsed": "in_stock",
    })
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_gbp"] = df["price_gbp"].astype(float).round(2)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL REFERENCES categories(category_id)
        );
        """
    )


def load(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        build_schema(conn)

        categories = sorted(df["category"].unique())
        cat_df = pd.DataFrame({"category_name": categories})
        cat_df.to_sql("categories", conn, if_exists="append", index=False)

        cat_id_lookup = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
        merged = df.merge(cat_id_lookup, left_on="category", right_on="category_name")
        books_df = merged[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].copy()
        books_df["in_stock"] = books_df["in_stock"].astype(int)
        books_df.to_sql("books", conn, if_exists="append", index=False)

        conn.commit()
    finally:
        conn.close()


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(raw)} raw rows from {RAW_CSV}")

    cleaned = clean(raw)
    print(f"\n{len(raw) - len(cleaned)} row(s) dropped; {len(cleaned)} rows remain after cleaning.")

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)
    print(f"Wrote cleaned data -> {CLEAN_CSV}")

    load(cleaned, DB_PATH)
    print(f"Loaded cleaned data into SQLite DB -> {DB_PATH}")


if __name__ == "__main__":
    main()
