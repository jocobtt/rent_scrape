"""Stamp every variant in models/ with its model / geo-state hashes (see services/artifact_bundle.py).

Training does this automatically; run it by hand to adopt files produced before stamping existed.
    uv run python scripts/stamp_bundles.py
"""
import os
import sys

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)

from services.artifact_bundle import stamp  # noqa: E402

MODELS_DIR = os.path.join(API_DIR, "models")

if __name__ == "__main__":
    for variant in ("challenger", "passed", "challenger-noterms", "passed-noterms"):
        if os.path.exists(os.path.join(MODELS_DIR, f"{variant}-features.json")):
            print(variant, stamp(MODELS_DIR, variant))
