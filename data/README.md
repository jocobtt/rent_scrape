# Data

Listing data is scraped from [Suumo](https://suumo.jp/) and is **not** redistributed
in this repository. Check Suumo's terms of service and `robots.txt` before scraping,
and keep request rates low.

## Regenerating the dataset

```bash
cd api
# scrape only: all 23 wards x 10 pages -> api/dataset_versions/<hash>/data.csv
uv run python scripts/scrape_dataset.py
WARDS=13104,13113 PAGES_PER_WARD=5 uv run python scripts/scrape_dataset.py

# scrape + train + register in MLflow
# one listing URL, pages 1-50
SUUMO_URL="<suumo search results url>" PAGES=1,50 uv run python scripts/scrape_and_train.py

# or specific / all Tokyo wards
WARDS=13104,13113 PAGES_PER_WARD=10 uv run python scripts/scrape_and_train.py
WARDS=all uv run python scripts/scrape_and_train.py
```

Each run writes a versioned snapshot (CSV plus manifest with a content hash) to
`api/dataset_versions/<hash>/` and trains the models against it. See
[`docs/`](../docs) for the full pipeline.

## Schema (model input)

| column | meaning | unit |
| --- | --- | --- |
| `rent_price` | target: monthly rent | 万円 |
| `sqr_m` | floor area | m² |
| `rei_price` | key money (礼金) | 万円 |
| `shikikin` | security deposit (敷金) | 万円 |
| `maintenence_price` | monthly maintenance fee | yen in raw data |
| `year_built` | building age | years |
| `floor` | floor number | |
| `eki_walk` | walk to nearest station | minutes |
| `building_id` | stable hash of building name + address; **grouping key for splits** | |
| `scraped_at` | UTC date of the scrape run; enables time-based splits across snapshots | date |

`building_id` and `scraped_at` (and `address`) are identifiers: they are kept in the data
but excluded from the model features. Files scraped before these columns existed fall back
to grouping by `address`.
