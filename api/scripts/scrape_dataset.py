"""Scrape Suumo listings, clean them and save a versioned dataset (no training).

Usage:
    cd api
    uv run python scripts/scrape_dataset.py

Environment variables:
    WARDS             Comma-separated Suumo sc codes (e.g. "13104,13113"), or "all"
                      for every Tokyo special ward. Default: all.
    PAGES_PER_WARD    Result pages per ward (default: 10). One page is one request
                      and yields roughly 100+ unit listings.
    WAIT_MIN, WAIT_MAX  Seconds to sleep between listings/pages (default: 2, 5).

The cleaned data is written to api/dataset_versions/<hash>/data.csv with a
manifest (content hash, row count, source). Train on it with the model scripts or
scripts/scrape_and_train.py. Check Suumo's terms and robots.txt before running,
and keep the delays polite.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_lineage_service import DataLineageService  # noqa: E402
from utils.scrape_data import TOKYO_WARD_CODES, ScrapeData, build_suumo_url  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    wards_env = os.environ.get("WARDS", "all").strip()
    ward_codes = TOKYO_WARD_CODES if wards_env.lower() == "all" else [w.strip() for w in wards_env.split(",")]
    pages = int(os.environ.get("PAGES_PER_WARD", "10"))
    wait_min = int(os.environ.get("WAIT_MIN", "2"))
    wait_max = int(os.environ.get("WAIT_MAX", "5"))

    raw = ScrapeData.scrape_wards(
        ward_codes=ward_codes, pages_per_ward=pages, wait_time_min=wait_min, wait_time_max=wait_max
    )
    if raw.empty:
        logger.error("Nothing scraped")
        return 1

    clean = ScrapeData(build_suumo_url()).clean_data(raw)
    manifest = DataLineageService().save_versioned_dataset(
        clean, source_description=f"Suumo scrape wards={wards_env} pages_per_ward={pages}", kind="scrape"
    )
    logger.info("Saved %d rows (%d buildings) to %s",
                len(clean), clean["building_id"].nunique(), manifest["data_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
