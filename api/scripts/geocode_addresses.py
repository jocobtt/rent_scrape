"""Geocode the scraped street-block addresses with the GSI (国土地理院) address-search API.

Results are cached in geo_cache/address_coords.json and only missing addresses are fetched, so
reruns are cheap. Training also calls this automatically for addresses new to the cache.

Usage (from api/):
    uv run python scripts/geocode_addresses.py [--delay 0.25]
"""
import argparse
import logging
import os
import sys

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)

from services.data_cleaning import load_training_frame  # noqa: E402
from services.geocoding import geocode_missing, load_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between requests")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    addresses = sorted(load_training_frame(None)["address"].unique())
    before = load_cache()
    print(f"{len(addresses)} addresses, {sum(a not in before for a in addresses)} to geocode", flush=True)
    cache = geocode_missing(addresses, before, delay=args.delay)
    missing = [a for a in addresses if not cache.get(a)]
    print(f"done: {len(addresses) - len(missing)} geocoded, {len(missing)} missing", flush=True)


if __name__ == "__main__":
    main()
