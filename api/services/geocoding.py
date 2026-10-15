"""Address -> (lat, lon) via the GSI (国土地理院) address-search API, with a JSON cache.

Listing addresses are only precise to a block (e.g. 東京都杉並区西荻南４), so every unique address
becomes one point. Used by `scripts/geocode_addresses.py` and by training (to fill in addresses
new to the cache) and, optionally, by the API for an address it has never seen.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

API_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = API_DIR / "geo_cache" / "address_coords.json"
URL = "https://msearch.gsi.go.jp/address-search/AddressSearch"
LON_RANGE, LAT_RANGE = (138.9, 140.0), (35.4, 35.95)  # Tokyo's 23 wards sit well inside this

Coords = Dict[str, Optional[List[float]]]


_FULLWIDTH_DIGITS = str.maketrans("0123456789", "０１２３４５６７８９")


def normalize_address(address: str) -> str:
    """Listing form: prefecture prefix, full-width digits, no surrounding space
    ('渋谷区猿楽町1' -> '東京都渋谷区猿楽町１')."""
    address = str(address).strip().translate(_FULLWIDTH_DIGITS)
    return address if address.startswith("東京都") else f"東京都{address}"


def load_cache(path: Path = CACHE_PATH) -> Coords:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Coords, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)


def geocode(address: str, session: Optional[requests.Session] = None, timeout: float = 15.0) -> Optional[List[float]]:
    """[lat, lon] of the first hit that lands inside Tokyo, else None. Raises on network errors."""
    getter = session or requests
    resp = getter.get(URL, params={"q": normalize_address(address)}, timeout=timeout,
                      headers=None if session else {"User-Agent": "rent-scrape-geocoder (research project)"})
    resp.raise_for_status()
    for hit in resp.json():
        lon, lat = hit["geometry"]["coordinates"]
        if LON_RANGE[0] <= lon <= LON_RANGE[1] and LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
            return [lat, lon]
    return None


def geocode_missing(addresses: Iterable[str], cache: Optional[Coords] = None, delay: float = 0.25,
                    save_every: int = 100, path: Path = CACHE_PATH) -> Coords:
    """Geocode the addresses not yet in the cache (network failures are skipped and retried next run)."""
    cache = load_cache(path) if cache is None else cache
    todo = [a for a in dict.fromkeys(addresses) if a not in cache]
    if todo:
        logger.info("Geocoding %d new addresses", len(todo))
    session = requests.Session()
    session.headers["User-Agent"] = "rent-scrape-geocoder (research project)"
    for i, addr in enumerate(todo, 1):
        try:
            cache[addr] = geocode(addr, session)
        except requests.RequestException as e:
            logger.warning("Geocoding %s failed: %s", addr, e)
            continue
        if i % save_every == 0:
            save_cache(cache, path)
        time.sleep(delay)
    if todo:
        save_cache(cache, path)
    return cache
