# data_pipeline

Scrapes books.toscrape.com, cleans/types the data, converts GBP → INR at a
fixed baseline rate, loads it into a normalized SQLite database, and runs a
set of SQL queries — with a pandas/SQL equivalence check for the join query.

## Contents

```
data_pipeline/
├── scrape.py              # requests + BeautifulSoup scraper -> data/raw_books.csv
├── clean_and_load.py       # cleans/types raw data -> data/clean_books.csv, loads db/books.db
├── run_queries.py          # runs the 5+ required SQL queries, saves sql/query_results.md
├── raw/
│   ├── bookstoscrape_snapshot.csv            # offline fallback snapshot (see note below)
│   └── bookstoscrape_snapshot_SOURCE_NOTE.md # provenance of that snapshot
├── data/
│   ├── raw_books.csv        # scraper output (raw, uncleaned strings)
│   └── clean_books.csv      # cleaned, typed output
├── db/
│   └── books.db              # SQLite database (regenerable via clean_and_load.py)
└── sql/
    └── query_results.md      # the 5+ SQL queries with their executed output
```

## Install / run

```bash
pip install requests beautifulsoup4 pandas tabulate
cd data_pipeline
python scrape.py           # -> data/raw_books.csv
python clean_and_load.py   # -> data/clean_books.csv, db/books.db
python run_queries.py      # -> prints + saves sql/query_results.md
```

Each script can be re-run independently and will overwrite its own output;
running all three in order fully regenerates the module's artifacts from
scratch.

## What's scraped

`scrape.py` pulls every book from **4 categories** on books.toscrape.com —
Fiction, Nonfiction, Sequential Art, and Young Adult — which together yield
**140 books** (well over the 60-book / 3-category minimum). For each book it
captures exactly what's on the category listing page: `title`, `price`
(raw, e.g. `"£51.77"`), `star_rating` (raw text, e.g. `"Three"`),
`availability` (raw text, e.g. `"In stock (16 available)"`), and `category`.

Live scraping uses `requests.Session()` + `BeautifulSoup`, follows each
category's "next page" pagination link until exhausted, and parses:
- title from `<h3><a title="...">` (the full, untruncated title)
- price from `<p class="price_color">`
- star rating from the second CSS class on `<p class="star-rating X">`
- availability from `<p class="instock availability">`

### A note on network access in this submission's execution environment

`books.toscrape.com` is a real, live, public site, and `scrape.py` is
written to hit it directly — on a machine with normal internet access it
fetches fresh data with no configuration needed. The sandbox used to *build*
this submission has outbound network access blocked at the proxy level
(live requests return `403 Forbidden`). To keep the pipeline runnable
end-to-end there anyway, `scrape.py` catches request failures and falls back
to a bundled **offline snapshot** of the real books.toscrape.com catalogue
(`raw/bookstoscrape_snapshot.csv`) — this is not fabricated data; it's a
public CC-BY 4.0 dataset ("Books to scrape dataset", Hernandez & Cebey,
2021, https://doi.org/10.5281/zenodo.5594938) that was itself produced by
scraping the same site, and its columns map 1:1 onto the live page's fields.
The fallback re-derives the exact raw text formats (e.g. `cantidad_stock=16`
→ `"In stock (16 available)"`, `rating=3` → `"Three"`) before handing rows to
the *same* parsing/cleaning code path used for live HTML, so the rest of the
pipeline is exercised identically either way. See
`raw/bookstoscrape_snapshot_SOURCE_NOTE.md` for full details. Run the script
yourself on a normal internet connection to see it scrape live.

## Cleaning decisions

To prove the pipeline doesn't just crash on messy input, a handful of
deliberately malformed rows were injected into `data/raw_books.csv` before
cleaning (a book with `price = "£N/A"`, one with an invalid
`star_rating = "Zero"`, one with unparseable `availability = "Ask in store"`).
`clean_and_load.py` handles each field differently, based on what makes
sense for that field's type:

- **`price_gbp` (float)** — strip the `£` symbol and parse as float. If a
  price fails to parse, the row is **kept** and the price is
  **median-imputed** (using the median of successfully-parsed prices within
  the same category, falling back to the global median). Price is a
  continuous numeric field on a book we already have a title/category/rating
  for, so discarding the whole row over one bad numeric token throws away
  information for no good reason, and imputing a *typical* price for that
  category is a defensible placeholder. Median (not mean) is used because
  book prices aren't symmetric and a single bad value could skew a mean.

- **`rating` (int, 1–5)** — mapped from the star-rating word (`One`…`Five`).
  If the text doesn't match one of those five words, the row is **dropped**.
  Unlike price, rating is a categorical/ordinal field with no natural
  "typical" value — inventing a star rating risks materially misrepresenting
  a specific book, and only a couple of rows are ever affected, so dropping
  costs us almost nothing.

- **`in_stock` (bool)** — parsed by substring match on the availability text
  (`"in stock"` → `True`, `"out of stock"` → `False`). Any other text is
  **dropped**, for the same reason as rating: there's no sensible "average"
  stock status to impute for a boolean/categorical field.

Both drop and impute decisions are logged to stdout by field and title when
`clean_and_load.py` runs, so nothing happens silently.

## Currency conversion

`price_inr = price_gbp * 105.50`

**1 GBP = 105.50 INR** is a fixed, project-defined constant used for this
assignment only — it is *not* a live or historical market exchange rate and
carries no date reference. It requires no API call and no network access;
it's simply a hard-coded multiplier (`GBP_TO_INR = 105.50` in
`clean_and_load.py`). No optional live-FX lookup was implemented, since the
assignment states the fixed-rate path alone is what's graded.

## Database schema

Two tables, normalized, with a primary/foreign-key relationship:

```sql
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
```

`db/books.db` is committed, but is also fully regenerable from scratch by
running `scrape.py` then `clean_and_load.py`.

## SQL queries

`run_queries.py` runs 6 queries against `db/books.db`, covering every
required clause (full text + output saved in `sql/query_results.md`):

| # | Demonstrates |
|---|---|
| Q1 | `SELECT` / `WHERE` — in-stock books under £15 |
| Q2 | `ORDER BY` + `LIMIT` — 10 priciest books overall |
| Q3 | `DISTINCT` — distinct rating values present |
| Q4 | `BETWEEN` — books priced £20–£30 |
| Q5 | `IN` (+ `JOIN`) — books in a chosen subset of categories |
| Q6 | `JOIN` — top-3 highest-rated (then cheapest) books per category |

## pandas ↔ SQL equivalence

`run_queries.py` also reads `books` and `categories` back into DataFrames
with `pd.read_sql(...)`, reproduces the Q6-style join purely with
`pd.merge(...)` on those in-memory DataFrames (no SQL), sorts both the same
way, and asserts the two results are identical — this passes
(`pd.read_sql result and pd.merge result are equivalent: True`) and is shown
in the script's printed output.

## Reproducing from scratch

```bash
python scrape.py && python clean_and_load.py && python run_queries.py
```
