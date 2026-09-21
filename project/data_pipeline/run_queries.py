"""
run_queries.py
===============
Runs >=5 SQL queries against db/books.db that collectively cover:
  - SELECT / WHERE
  - ORDER BY
  - LIMIT
  - DISTINCT
  - IN or BETWEEN
  - a JOIN between books and categories

Prints each query and its result, and writes the same to
data_pipeline/sql/query_results.md so the executed output is saved
alongside the query text (per the assignment's acceptance criteria).

Also demonstrates, for the JOIN query:
  - reading it back into a DataFrame with pd.read_sql(...)
  - reproducing the same result with pd.merge(...) on in-memory DataFrames
    (no SQL), and shows the two are equivalent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "db" / "books.db"
RESULTS_MD = REPO_ROOT / "sql" / "query_results.md"

QUERIES: list[tuple[str, str]] = [
    (
        "Q1 - SELECT / WHERE: in-stock books priced under £15",
        """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp < 15.0;
        """,
    ),
    (
        "Q2 - ORDER BY + LIMIT: 10 most expensive books overall (GBP)",
        """
        SELECT title, price_gbp, category_id
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
        """,
    ),
    (
        "Q3 - DISTINCT: distinct rating values present in the dataset",
        """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
        """,
    ),
    (
        "Q4 - BETWEEN: books priced between £20 and £30 (inclusive)",
        """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20.0 AND 30.0
        ORDER BY price_gbp;
        """,
    ),
    (
        "Q5 - IN: books in a chosen subset of categories",
        """
        SELECT b.title, c.category_name, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Fiction', 'Young Adult')
        ORDER BY c.category_name, b.title;
        """,
    ),
    (
        "Q6 - JOIN: top-3 highest-rated (then cheapest) books per category",
        """
        SELECT c.category_name,
               b.title,
               b.rating,
               b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE (
            SELECT COUNT(*)
            FROM books b2
            WHERE b2.category_id = b.category_id
              AND (b2.rating > b.rating
                   OR (b2.rating = b.rating AND b2.price_gbp < b.price_gbp))
        ) < 3
        ORDER BY c.category_name, b.rating DESC, b.price_gbp ASC;
        """,
    ),
]


def run_all() -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(DB_PATH)
    results: dict[str, pd.DataFrame] = {}
    try:
        lines = ["# SQL query results\n"]
        for title, sql in QUERIES:
            df = pd.read_sql(sql, conn)
            results[title] = df
            print(f"\n=== {title} ===")
            print(sql.strip())
            print(df.to_string(index=False))

            lines.append(f"## {title}\n")
            lines.append("```sql")
            lines.append(sql.strip())
            lines.append("```\n")
            lines.append(df.to_markdown(index=False))
            lines.append("\n")

        RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSaved all query text + output -> {RESULTS_MD}")
    finally:
        conn.close()
    return results


def demonstrate_merge_equivalence() -> None:
    """
    Read books & categories back into DataFrames with pd.read_sql, then
    reproduce the Q6 join-query result via pd.merge on those in-memory
    DataFrames (no SQL), and show the two outputs match.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        books_df = pd.read_sql("SELECT * FROM books", conn)
        categories_df = pd.read_sql("SELECT * FROM categories", conn)
        sql_join_result = pd.read_sql(
            """
            SELECT c.category_name, b.title, b.rating, b.price_gbp
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY c.category_name, b.title;
            """,
            conn,
        )
    finally:
        conn.close()

    # Reproduce the same join purely in pandas (no SQL)
    pandas_join_result = (
        books_df.merge(categories_df, on="category_id", how="inner")
        [["category_name", "title", "rating", "price_gbp"]]
        .sort_values(["category_name", "title"])
        .reset_index(drop=True)
    )
    sql_join_result_sorted = sql_join_result.sort_values(["category_name", "title"]).reset_index(drop=True)

    are_equal = pandas_join_result.equals(sql_join_result_sorted)

    print("\n=== pd.read_sql JOIN result (head) ===")
    print(sql_join_result_sorted.head(5).to_string(index=False))
    print("\n=== pd.merge JOIN result (head) ===")
    print(pandas_join_result.head(5).to_string(index=False))
    print(f"\npd.read_sql result and pd.merge result are equivalent: {are_equal}")

    assert are_equal, "pd.read_sql and pd.merge results diverged!"


if __name__ == "__main__":
    run_all()
    demonstrate_merge_equivalence()
