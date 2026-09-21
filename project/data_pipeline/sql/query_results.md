# SQL query results

## Q1 - SELECT / WHERE: in-stock books priced under £15

```sql
SELECT title, price_gbp, price_inr, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp < 15.0;
```

| title                                                                             |   price_gbp |   price_inr |   rating |
|:----------------------------------------------------------------------------------|------------:|------------:|---------:|
| I Am Pilgrim (Pilgrim #1)                                                         |       10.6  |     1118.3  |        4 |
| Eight Hundred Grapes                                                              |       14.39 |     1518.14 |        4 |
| Dear Mr. Knightley                                                                |       11.21 |     1182.66 |        5 |
| Mothering Sunday                                                                  |       13.34 |     1407.37 |        2 |
| So You've Been Publicly Shamed                                                    |       12.23 |     1290.27 |        2 |
| Agnostic: A Spirited Manifesto                                                    |       12.51 |     1319.8  |        5 |
| The Sleep Revolution: Transforming Your Life, One Night at a Time                 |       11.68 |     1232.24 |        4 |
| Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Princess Jellyfish 2-in-1 Omnibus #1) |       13.61 |     1435.86 |        5 |
| Patience                                                                          |       10.16 |     1071.88 |        3 |
| Superman Vol. 1: Before Truth (Superman by Gene Luen Yang #1)                     |       11.89 |     1254.4  |        5 |
| Adulthood Is a Myth: A "Sarah's Scribbles" Collection                             |       10.9  |     1149.95 |        2 |
| Obsidian (Lux #1)                                                                 |       14.86 |     1567.73 |        2 |
| Wild Swans                                                                        |       14.36 |     1514.98 |        2 |
| The Epidemic (The Program 0.6)                                                    |       14.44 |     1523.42 |        5 |


## Q2 - ORDER BY + LIMIT: 10 most expensive books overall (GBP)

```sql
SELECT title, price_gbp, category_id
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
```

| title                                                                                                                  |   price_gbp |   category_id |
|:-----------------------------------------------------------------------------------------------------------------------|------------:|--------------:|
| The Diary of a Young Girl                                                                                              |       59.9  |             2 |
| The Improbability of Love                                                                                              |       59.45 |             1 |
| Hamilton: The Revolution                                                                                               |       58.79 |             2 |
| Aristotle and Dante Discover the Secrets of the Universe (Aristotle and Dante Discover the Secrets of the Universe #1) |       58.14 |             4 |
| El Deafo                                                                                                               |       57.62 |             3 |
| The Dinner Party                                                                                                       |       56.54 |             1 |
| Abstract City                                                                                                          |       56.37 |             2 |
| The Electric Pencil: Drawings from Inside State Hospital No. 3                                                         |       56.06 |             2 |
| Shtum                                                                                                                  |       55.84 |             1 |
| Don't Get Caught                                                                                                       |       55.35 |             4 |


## Q3 - DISTINCT: distinct rating values present in the dataset

```sql
SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
```

|   rating |
|---------:|
|        1 |
|        2 |
|        3 |
|        4 |
|        5 |


## Q4 - BETWEEN: books priced between £20 and £30 (inclusive)

```sql
SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20.0 AND 30.0
        ORDER BY price_gbp;
```

| title                                                                                                                         |   price_gbp |
|:------------------------------------------------------------------------------------------------------------------------------|------------:|
| Tuesday Nights in 1980                                                                                                        |       21.04 |
| Snatched: How A Drug Queen Went Undercover for the DEA and Was Kidnapped By Colombian Guerillas                               |       21.21 |
| No Dream Is Too High: Life Lessons From a Man Who Walked on the Moon                                                          |       21.95 |
| In the Country We Love: My Family Divided                                                                                     |       22    |
| Giant Days, Vol. 2 (Giant Days #5-8)                                                                                          |       22.11 |
| The Requiem Red                                                                                                               |       22.65 |
| #HigherSelfie: Wake Up Your Life. Free Your Soul. Find Your Tribe.                                                            |       23.11 |
| My Mrs. Brown                                                                                                                 |       24.48 |
| 10% Happier: How I Tamed the Voice in My Head, Reduced Stress Without Losing My Edge, and Found Self-Help That Actually Works |       24.57 |
| Cometh the Hour (The Clifton Chronicles #6)                                                                                   |       25.01 |
| Through the Woods                                                                                                             |       25.38 |
| Red Hood/Arsenal, Vol. 1: Open for Business (Red Hood/Arsenal #1)                                                             |       25.48 |
| The First Hostage (J.B. Collins #2)                                                                                           |       25.85 |
| Still Life with Bread Crumbs                                                                                                  |       26.41 |
| Reasons to Stay Alive                                                                                                         |       26.41 |
| Let It Out: A Journey Through Journaling                                                                                      |       26.79 |
| 13 Hours: The Inside Account of What Really Happened In Benghazi                                                              |       27.06 |
| Eligible (The Austen Project #4)                                                                                              |       27.09 |
| This Is Where It Ends                                                                                                         |       27.12 |
| Becoming Wise: An Inquiry into the Mystery and Art of Living                                                                  |       27.43 |
| The Time Keeper                                                                                                               |       27.88 |
| A Fierce and Subtle Poison                                                                                                    |       28.13 |
| Burning                                                                                                                       |       28.81 |
| Mr. Mercedes (Bill Hodges Trilogy #1)                                                                                         |       28.9  |
| Call the Nurse: True Stories of a Country Nurse on a Scottish Isle                                                            |       29.14 |
| Looking for Lovely: Collecting the Moments that Matter                                                                        |       29.14 |
| I Hate Fairyland, Vol. 1: Madly Ever After (I Hate Fairyland (Compilations) #1-5)                                             |       29.17 |
| Frostbite (Vampire Academy #2)                                                                                                |       29.99 |


## Q5 - IN: books in a chosen subset of categories

```sql
SELECT b.title, c.category_name, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Fiction', 'Young Adult')
        ORDER BY c.category_name, b.title;
```

| title                                                                                                                  | category_name   |   rating |
|:-----------------------------------------------------------------------------------------------------------------------|:----------------|---------:|
| 11/22/63                                                                                                               | Fiction         |        3 |
| A Glitchy Listing                                                                                                      | Fiction         |        3 |
| A Man Called Ove                                                                                                       | Fiction         |        1 |
| Balloon Animals                                                                                                        | Fiction         |        3 |
| Cometh the Hour (The Clifton Chronicles #6)                                                                            | Fiction         |        3 |
| Daredevils                                                                                                             | Fiction         |        3 |
| Dear Mr. Knightley                                                                                                     | Fiction         |        5 |
| Eight Hundred Grapes                                                                                                   | Fiction         |        4 |
| Eligible (The Austen Project #4)                                                                                       | Fiction         |        3 |
| Finders Keepers (Bill Hodges Trilogy #2)                                                                               | Fiction         |        5 |
| I Am Pilgrim (Pilgrim #1)                                                                                              | Fiction         |        4 |
| Lies and Other Acts of Love                                                                                            | Fiction         |        1 |
| Mothering Sunday                                                                                                       | Fiction         |        2 |
| Mr. Mercedes (Bill Hodges Trilogy #1)                                                                                  | Fiction         |        1 |
| My Mrs. Brown                                                                                                          | Fiction         |        3 |
| My Name Is Lucy Barton                                                                                                 | Fiction         |        1 |
| Private Paris (Private #10)                                                                                            | Fiction         |        5 |
| Shtum                                                                                                                  | Fiction         |        4 |
| Soumission                                                                                                             | Fiction         |        1 |
| Still Life with Bread Crumbs                                                                                           | Fiction         |        3 |
| Take Me with You                                                                                                       | Fiction         |        3 |
| The Dinner Party                                                                                                       | Fiction         |        2 |
| The First Hostage (J.B. Collins #2)                                                                                    | Fiction         |        3 |
| The Improbability of Love                                                                                              | Fiction         |        1 |
| The Murder That Never Was (Forensic Instincts #5)                                                                      | Fiction         |        3 |
| The Regional Office Is Under Attack!                                                                                   | Fiction         |        5 |
| The Silent Sister (Riley MacPherson #1)                                                                                | Fiction         |        5 |
| The Testament of Mary                                                                                                  | Fiction         |        4 |
| The Time Keeper                                                                                                        | Fiction         |        5 |
| The Vacationers                                                                                                        | Fiction         |        4 |
| Thirst                                                                                                                 | Fiction         |        5 |
| Tuesday Nights in 1980                                                                                                 | Fiction         |        2 |
| We Love You, Charlie Freeman                                                                                           | Fiction         |        5 |
| A Fierce and Subtle Poison                                                                                             | Young Adult     |        4 |
| Aristotle and Dante Discover the Secrets of the Universe (Aristotle and Dante Discover the Secrets of the Universe #1) | Young Adult     |        4 |
| Burning                                                                                                                | Young Adult     |        3 |
| Catching Jordan (Hundred Oaks)                                                                                         | Young Adult     |        3 |
| Don't Get Caught                                                                                                       | Young Adult     |        1 |
| Exit, Pursued by a Bear                                                                                                | Young Adult     |        4 |
| Frostbite (Vampire Academy #2)                                                                                         | Young Adult     |        5 |
| Library of Souls (Miss Peregrine's Peculiar Children #3)                                                               | Young Adult     |        5 |
| My Kind of Crazy                                                                                                       | Young Adult     |        1 |
| Nightingale, Sing                                                                                                      | Young Adult     |        1 |
| No Love Allowed (Dodge Cove #1)                                                                                        | Young Adult     |        4 |
| Obsidian (Lux #1)                                                                                                      | Young Adult     |        2 |
| Scarlett Epstein Hates It Here                                                                                         | Young Adult     |        5 |
| Set Me Free                                                                                                            | Young Adult     |        5 |
| Stars Above (The Lunar Chronicles #4.5)                                                                                | Young Adult     |        2 |
| The Darkest Lie                                                                                                        | Young Adult     |        5 |
| The Epidemic (The Program 0.6)                                                                                         | Young Adult     |        5 |
| The Natural History of Us (The Fine Art of Pretending #2)                                                              | Young Adult     |        3 |
| The Requiem Red                                                                                                        | Young Adult     |        1 |
| This Is Where It Ends                                                                                                  | Young Adult     |        2 |
| Until Friday Night (The Field Party #1)                                                                                | Young Adult     |        2 |
| Wild Swans                                                                                                             | Young Adult     |        2 |


## Q6 - JOIN: top-3 highest-rated (then cheapest) books per category

```sql
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
```

| category_name   | title                                                                             |   rating |   price_gbp |
|:----------------|:----------------------------------------------------------------------------------|---------:|------------:|
| Fiction         | Dear Mr. Knightley                                                                |        5 |       11.21 |
| Fiction         | Thirst                                                                            |        5 |       17.27 |
| Fiction         | The Time Keeper                                                                   |        5 |       27.88 |
| Nonfiction      | Agnostic: A Spirited Manifesto                                                    |        5 |       12.51 |
| Nonfiction      | Mother, Can You Not?                                                              |        5 |       16.89 |
| Nonfiction      | #HigherSelfie: Wake Up Your Life. Free Your Soul. Find Your Tribe.                |        5 |       23.11 |
| Sequential Art  | Superman Vol. 1: Before Truth (Superman by Gene Luen Yang #1)                     |        5 |       11.89 |
| Sequential Art  | Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Princess Jellyfish 2-in-1 Omnibus #1) |        5 |       13.61 |
| Sequential Art  | Batman: The Dark Knight Returns (Batman)                                          |        5 |       15.38 |
| Young Adult     | The Epidemic (The Program 0.6)                                                    |        5 |       14.44 |
| Young Adult     | Set Me Free                                                                       |        5 |       17.46 |
| Young Adult     | Frostbite (Vampire Academy #2)                                                    |        5 |       29.99 |

