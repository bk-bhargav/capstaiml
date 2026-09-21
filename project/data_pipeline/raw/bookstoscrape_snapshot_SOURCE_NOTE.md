# Source note for `bookstoscrape_snapshot.csv`

This sandboxed execution environment has **no outbound network access** on the
machine that runs `bash`/Python (egress is disabled). `books.toscrape.com` is a
real, public, live website — the code in `scrape.py` is written to hit it
directly with `requests` + `BeautifulSoup` and will fetch fresh, live data the
moment it is run anywhere with normal internet access (see README "Running
live" section).

To let the *rest* of the pipeline (cleaning, typing, SQLite loading, SQL
querying, pandas merge) be demonstrated end-to-end inside this sandbox, we
ship one offline snapshot of the same site's catalogue, `bookstoscrape.csv`,
so `scrape.py` has something real to fall back to when a live request fails
(e.g. no internet). This is **not synthetic/fabricated data** — it is a public
CC-BY 4.0 dataset ("Books to scrape dataset", Hernandez & Cebey, 2021,
https://doi.org/10.5281/zenodo.5594938) that itself was produced by scraping
books.toscrape.com, and its columns (`titulo`, `precio`, `cantidad_stock`,
`categoria`, `rating`) map 1:1 onto the fields the live scraper collects
(title, price, availability/stock, category, star rating).

`scrape.py`'s fallback path re-derives the exact text formats the *live* HTML
would contain (e.g. turns `cantidad_stock=22` into the availability string
`"In stock (22 available)"`, and `rating=3` into the star-rating word
`"Three"`) before handing rows to the same parser/cleaner used for live HTML,
so the cleaning logic is exercised identically either way.
