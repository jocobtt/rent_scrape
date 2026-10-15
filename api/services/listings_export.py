"""The newest scraped listings, shaped for the dashboard.

`scripts/scrape_dataset.py` writes each scrape to `dataset_versions/<hash>/` (manifest kind
"scrape"). The dashboard reads one CSV; this picks the latest scrape and converts it to the
columns and conventions the frontend's `parseTokyoRentData` expects, so the charts follow each
re-scrape without anything being committed or copied.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from services.data_cleaning import valid_layout_mask
from services.data_lineage_service import DATASET_STORE_DIR

logger = logging.getLogger(__name__)

# Dashboard column order (frontend/src/lib/dataUtils.js)
DASHBOARD_COLUMNS = [
    "rent_price", "address", "sqr_m", "rei_price", "shikikin", "maintenence_price",
    "apartment_type", "year_built", "ku_name", "floor", "house_type", "eki_walk",
]


def latest_scrape(store_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Manifest of the most recent raw scrape, or None. Processed training snapshots are skipped."""
    best = None
    for manifest_path in Path(store_dir or DATASET_STORE_DIR).glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
        data_path = manifest_path.parent / "data.csv"
        if manifest.get("kind") != "scrape" or not data_path.exists():
            continue
        if best is None or manifest.get("created_at", "") > best["created_at"]:
            best = {**manifest, "data_path": str(data_path)}
    return best


def to_dashboard_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw scrape -> dashboard columns. Wards lose their 区 suffix, the walk time is `eki_walk`,
    and listings with an impossible layout are dropped."""
    # Mis-parsed layouts ("16K") would otherwise show up as filter options
    df = raw[valid_layout_mask(raw["apartment_type"])].rename(columns={"nearest_eki_walk": "eki_walk"})
    # The map joins on the ward name without the suffix ("中野"); the scrape stores "中野区"
    df["ku_name"] = df["ku_name"].astype(str).str.removesuffix("区")
    return df[DASHBOARD_COLUMNS]


@lru_cache(maxsize=2)
def _csv_for(data_path: str, mtime_ns: int) -> bytes:
    # mtime is part of the key so a re-written file is re-read
    raw = pd.read_csv(data_path, usecols=lambda c: c in set(DASHBOARD_COLUMNS) | {"nearest_eki_walk"})
    return to_dashboard_frame(raw).to_csv(index=False).encode("utf-8")


def dashboard_csv(manifest: Dict[str, Any]) -> Tuple[bytes, str]:
    """(csv bytes, etag) for a scrape manifest from `latest_scrape`."""
    path = manifest["data_path"]
    body = _csv_for(path, Path(path).stat().st_mtime_ns)
    return body, f'"{manifest["hash_short"]}"'
