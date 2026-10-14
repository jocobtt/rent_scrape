"""Per-ward / per-layout / per-station medians from the newest scrape.

The rent form uses them as starting values (so a renter only changes what they care about)
and to say how an estimate compares with what is typically listed in that area.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from services.data_cleaning import valid_layout_mask

logger = logging.getLogger(__name__)

# Below this many listings a cell's medians are too noisy to be a default
MIN_CELL = 5

_COLUMNS = [
    "rent_price", "sqr_m", "maintenence_price", "year_built", "floor", "nearest_eki_walk",
    "apartment_type", "ku_name", "nearest_station", "nearest_line",
]
_STATS = {
    "rent": "rent_price", "sqr_m": "sqr_m", "maintenance": "maintenence_price",
    "year_built": "year_built", "floor": "floor", "eki_walk": "nearest_eki_walk",
}


def _cell(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"count": int(len(df))}
    for key, col in _STATS.items():
        out[key] = round(float(df[col].median()), 2)
    # Median of per-m² rent, not rent / median size: comparisons are made on per-m²
    out["rent_per_sqm"] = round(float((df["rent_price"] / df["sqr_m"]).median()), 3)
    return out


def _by_layout(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {
        str(layout): _cell(g)
        for layout, g in df.groupby("apartment_type")
        if len(g) >= MIN_CELL
    }


def summarize(raw: pd.DataFrame) -> Dict[str, Any]:
    """Raw scrape -> {"tokyo": {...}, "wards": {...}, "stations": [...]} (rents in 万円)."""
    df = raw[valid_layout_mask(raw["apartment_type"])].dropna(subset=["rent_price", "sqr_m", "apartment_type"])
    df = df[df["sqr_m"] > 0]
    df = df.assign(
        maintenence_price=df["maintenence_price"].fillna(0),
        nearest_eki_walk=df["nearest_eki_walk"].fillna(df["nearest_eki_walk"].median()),
    )

    wards = {
        str(ward): {**_cell(g), "layouts": _by_layout(g)}
        for ward, g in df.groupby("ku_name")
        if len(g) >= MIN_CELL
    }

    stations = []
    for (station, line), g in df.dropna(subset=["nearest_station", "nearest_line"]).groupby(
        ["nearest_station", "nearest_line"]
    ):
        if len(g) < MIN_CELL:
            continue
        stations.append({
            "name": str(station), "line": str(line),
            "ward": str(g["ku_name"].mode().iat[0]),
            "count": int(len(g)),
            "eki_walk": round(float(g["nearest_eki_walk"].median()), 1),
        })
    stations.sort(key=lambda s: (-s["count"], s["name"]))

    return {"tokyo": {**_cell(df), "layouts": _by_layout(df)}, "wards": wards, "stations": stations}


@lru_cache(maxsize=2)
def _summary_for(data_path: str, mtime_ns: int) -> Dict[str, Any]:
    raw = pd.read_csv(data_path, usecols=lambda c: c in set(_COLUMNS))
    return summarize(raw)


def market_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Summary for a scrape manifest from `listings_export.latest_scrape`."""
    path = manifest["data_path"]
    body = _summary_for(path, Path(path).stat().st_mtime_ns)
    return {
        "dataset": {"hash": manifest["hash_short"], "created_at": manifest["created_at"]},
        **body,
    }
