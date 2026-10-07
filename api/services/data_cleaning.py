"""Dataset resolution and training-time cleaning.

Every training script loads its data through `load_training_frame()`, so the
dedupe/outlier/unit rules are applied identically for all model families.

Dataset selection, in order:
  1. an explicit path argument,
  2. the DATASET_PATH environment variable,
  3. the newest folder in `api/dataset_versions/` (written by
     scripts/scrape_dataset.py or scripts/scrape_and_train.py).
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

API_DIR = Path(__file__).resolve().parent.parent
DATASET_VERSIONS_DIR = API_DIR / "dataset_versions"

# Plausible rent per square metre, in 万円/㎡/month (1,500 to 15,000 yen/㎡).
# Outside this range the row is almost certainly a parse error or a data-entry slip.
MIN_RENT_PER_SQM = 0.15
MAX_RENT_PER_SQM = 1.5

# Maintenance fees >= this are yen, not 万円 (1M yen/month is not a maintenance fee).
YEN_MAINTENANCE_FLOOR = 100.0

_IDENTITY_EXCLUDE = ["scraped_at"]  # duplicates are identical apart from the scrape date


def resolve_dataset_path(path: Optional[str] = None) -> Path:
    """Return the dataset CSV to train on (see module docstring for the order)."""
    candidate = path or os.environ.get("DATASET_PATH")
    if candidate:
        p = Path(candidate).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p

    latest, latest_ts = None, ""
    if DATASET_VERSIONS_DIR.is_dir():
        for manifest_path in DATASET_VERSIONS_DIR.glob("*/manifest.json"):
            data_path = manifest_path.parent / "data.csv"
            if not data_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                continue
            # Only real scrapes: lineage snapshots of encoded training frames live
            # in the same store, and older manifests predate the `kind` field.
            if manifest.get("kind") != "scrape":
                continue
            ts = manifest.get("created_at", "")
            if ts > latest_ts:
                latest, latest_ts = data_path, ts
    if latest is None:
        raise FileNotFoundError(
            "No training dataset found. Run `uv run python scripts/scrape_dataset.py` "
            "(see data/README.md) or set DATASET_PATH to a CSV."
        )
    logger.info("Using latest dataset: %s", latest)
    return latest


def clean_listings(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Dedupe, fix units and drop implausible rows. Returns (cleaned, counts)."""
    report = {"rows_in": len(df)}
    df = df.copy()

    # Units: legacy files store maintenance in yen; the model/API use 万円.
    if "maintenence_price" in df.columns:
        yen = df["maintenence_price"] >= YEN_MAINTENANCE_FLOOR
        report["maintenance_converted_from_yen"] = int(yen.sum())
        df.loc[yen, "maintenence_price"] = df.loc[yen, "maintenence_price"] / 10000.0

    # Exact duplicates (identical units listed repeatedly in one building)
    subset = [c for c in df.columns if c not in _IDENTITY_EXCLUDE]
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    report["duplicates_dropped"] = before - len(df)

    # Unparsed / non-positive core fields
    if {"rent_price", "sqr_m"} <= set(df.columns):
        valid = (df["rent_price"] > 0) & (df["sqr_m"] > 0)
        report["nonpositive_dropped"] = int((~valid).sum())
        df = df[valid]

        per_sqm = df["rent_price"] / df["sqr_m"]
        plausible = per_sqm.between(MIN_RENT_PER_SQM, MAX_RENT_PER_SQM)
        report["implausible_rent_per_sqm_dropped"] = int((~plausible).sum())
        df = df[plausible]

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def load_training_frame(path: Optional[str] = None) -> pd.DataFrame:
    """Resolve, read and clean the training dataset."""
    data_path = resolve_dataset_path(path)
    df = pd.read_csv(data_path)
    df, report = clean_listings(df)
    logger.info("Loaded %s: %s", data_path, report)
    return df
