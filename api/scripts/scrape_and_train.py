"""Standalone scrape-and-train script for CI and local use.

Usage:
    cd api
    uv run python scripts/scrape_and_train.py

Environment variables:
    SUUMO_URL          - Single Suumo listing URL to scrape.  Ignored when
                         WARDS is set.
    WARDS              - Comma-separated Suumo sc codes for multi-ward mode
                         (e.g. "13104,13113,13112").  Set to "all" to scrape
                         every Tokyo special ward.  When set, SUUMO_URL and
                         PAGES are ignored.
    PAGES_PER_WARD     - Pages per ward in multi-ward mode (default: 10).
    MODELS_TO_RETRAIN  - Comma-separated list: challenger,passed (default: both)
    PAGES              - Page range as start,end for single-URL mode (default: 1,51)
    MLFLOW_TRACKING_URI - MLflow URI (default: http://localhost:5001; use sqlite:///mlflow.db in CI)
"""
import json
import logging
import os
import sys

# Resolve to api/ directory regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.training_service import TrainingService
from utils.scrape_data import TOKYO_WARD_CODES, build_suumo_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = os.environ.get("MODELS_TO_RETRAIN", "challenger,passed").split(",")
service = TrainingService()

_wards_env = os.environ.get("WARDS", "").strip()

if _wards_env:
    # ── Multi-ward mode ──────────────────────────────────────────────────────
    ward_codes = TOKYO_WARD_CODES if _wards_env.lower() == "all" else _wards_env.split(",")
    pages_per_ward = int(os.environ.get("PAGES_PER_WARD", "10"))

    logger.info("Starting multi-ward scrape + retrain")
    logger.info("  Wards          : %s (%d)", _wards_env[:60], len(ward_codes))
    logger.info("  Pages per ward : %d (~%d max listings)", pages_per_ward, pages_per_ward * 30 * len(ward_codes))
    logger.info("  Models         : %s", MODELS)
    logger.info("  MLflow         : %s", os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    result = service.retrain_with_wards(
        ward_codes=ward_codes,
        pages_per_ward=pages_per_ward,
        wait_time_min=2,
        wait_time_max=5,
        models_to_retrain=MODELS,
    )
else:
    # ── Single-URL mode (backward-compatible) ────────────────────────────────
    SUUMO_URL = os.environ.get("SUUMO_URL") or build_suumo_url()
    _pages_raw = os.environ.get("PAGES", "1,51").split(",")
    PAGES = (int(_pages_raw[0]), int(_pages_raw[1]))

    logger.info("Starting single-URL scrape + retrain")
    logger.info("  URL   : %s...", SUUMO_URL[:80])
    logger.info("  Pages : %s", PAGES)
    logger.info("  Models: %s", MODELS)
    logger.info("  MLflow: %s", os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    result = service.retrain_models(
        url=SUUMO_URL,
        wait_time_min=2,
        wait_time_max=5,
        pages=PAGES,
        models_to_retrain=MODELS,
    )

print(json.dumps(result, indent=2, default=str))

if result.get("status") == "error":
    logger.error("Retrain failed: %s", result.get("message", "unknown error"))
    sys.exit(1)

logger.info("Done. Dataset: %s", result.get("dataset_version_path"))
