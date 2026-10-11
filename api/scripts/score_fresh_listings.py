"""Time-forward check of the served models on listings from a newer scrape.

    uv run python scripts/score_fresh_listings.py                      # newest scrape vs the dataset the models trained on
    uv run python scripts/score_fresh_listings.py --new path/to/data.csv --trained-on path/to/older/data.csv
    ... --log                                                          # also feed the prediction log (performance monitoring)

The dataset the models were trained on is read from the bundle stamp (`dataset` in `<variant>-features.json`,
written by training); pass --trained-on for models that predate it. Scoring listings the models were
trained on would flatter them, so the script refuses when the "new" dataset is the training dataset.
Writes evaluation_reports/fresh_listings_<timestamp>.json.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)
os.environ.setdefault("GEO_LIVE_GEOCODE", "0")  # addresses of scraped listings are in the geocode cache

from services.data_cleaning import DATASET_VERSIONS_DIR, load_training_frame, resolve_dataset_path  # noqa: E402
from services.fresh_listings import new_units, score, summarise  # noqa: E402

MODELS_DIR = Path(API_DIR) / "models"
VARIANTS = ("challenger", "passed", "challenger-noterms", "passed-noterms")


def served_training_datasets() -> set:
    ids = set()
    for v in VARIANTS:
        f = MODELS_DIR / f"{v}-features.json"
        if f.exists():
            d = (json.loads(f.read_text()).get("bundle") or {}).get("dataset")
            if d:
                ids.add(d)
    return ids


def frame(path) -> pd.DataFrame:
    df = load_training_frame(str(path))
    return df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", help="newer scrape CSV (default: the newest scrape)")
    ap.add_argument("--trained-on", help="CSV the served models were trained on (default: from the bundle stamp)")
    ap.add_argument("--max-rows", type=int, default=3000, help="score at most this many new listings (random sample)")
    ap.add_argument("--log", action="store_true", help="also write the rows, with actual rents, to the prediction log")
    args = ap.parse_args()

    new_path = resolve_dataset_path(args.new)
    if args.trained_on:
        trained_path = Path(args.trained_on)
    else:
        ids = served_training_datasets()
        if len(ids) != 1:
            sys.exit(f"Cannot tell which dataset the served models were trained on (stamps say {sorted(ids) or 'nothing'}); "
                     "pass --trained-on")
        trained_path = DATASET_VERSIONS_DIR / ids.pop() / "data.csv"
    if new_path.resolve() == trained_path.resolve():
        sys.exit(f"{new_path} is the dataset the models were trained on; scoring it would only measure the training data. "
                 "Re-scrape and try again, or pass --new.")

    fresh = new_units(frame(new_path), frame(trained_path))
    print(f"{len(fresh)} listings in {new_path.parent.name} whose unit is not in {trained_path.parent.name}")
    if fresh.empty:
        sys.exit("Nothing new to score.")
    fresh = fresh.sample(min(len(fresh), args.max_rows), random_state=0)

    from services import PredictionService
    load = lambda n: joblib.load(MODELS_DIR / f"{n}-model.joblib")  # noqa: E731
    service = PredictionService(load("passed"), load("challenger"),
                                passed_noterms_model=load("passed-noterms"),
                                challenger_noterms_model=load("challenger-noterms"))
    scored = score(service, fresh)
    summary = summarise(scored)
    print(summary.round(3).to_string(index=False))

    out_dir = Path(API_DIR) / "evaluation_reports"
    out_dir.mkdir(exist_ok=True)
    report = out_dir / f"fresh_listings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report.write_text(json.dumps({"new": new_path.parent.name, "trained_on": trained_path.parent.name,
                                  "n_listings": len(fresh), "summary": summary.to_dict("records")}, indent=2))
    print(f"wrote {report}")

    if args.log:
        from services.data_capture_service import DataCaptureService
        capture = DataCaptureService()
        for r in scored.itertuples():
            capture.log_prediction(r.model_type, r.input, r.prediction, actual_value=r.actual_value,
                                   variant=r.variant, source="rescrape")
        capture.flush()
        print(f"logged {len(scored)} predictions with actual rents (source=rescrape)")


if __name__ == "__main__":
    main()
