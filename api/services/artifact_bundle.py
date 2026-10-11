"""Keep a model file, its feature list and its geo state from drifting apart.

A no-terms variant is three files: `<variant>-model.joblib`, `<variant>-features.json` and (when it was
trained with geo features) `<variant>-geo.json`. They are written by one training run and only make
sense together: a model paired with another run's geo state keeps predicting, just wrongly. The geo
state is derived from scraped listings and is not committed, so the features file (which is) records
what the other two must look like:

    "bundle": {"model_sha256": ..., "geo_sha256": ... | null}

`stamp` writes it at the end of training; `verify` runs when the API loads the variant. A model or geo
file that does not match raises `BundleMismatch`; a geo file that is absent only warns (the features
become NaN, which the model handles but with less accuracy).
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)


class BundleMismatch(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _paths(models_dir: Union[str, Path], variant: str):
    d = Path(models_dir)
    return d / f"{variant}-model.joblib", d / f"{variant}-features.json", d / f"{variant}-geo.json"


def stamp(models_dir: Union[str, Path], variant: str) -> dict:
    """Record the current model and geo-state hashes in `<variant>-features.json`."""
    model, features, geo = _paths(models_dir, variant)
    with open(features, encoding="utf-8") as f:
        state = json.load(f)
    state["bundle"] = {"model_sha256": _sha256(model), "geo_sha256": _sha256(geo) if geo.exists() else None}
    with open(features, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return state["bundle"]


def problems(models_dir: Union[str, Path], variant: str) -> List[str]:
    """Why the three files do not belong together; empty when they do (or were never stamped)."""
    model, features, geo = _paths(models_dir, variant)
    if not features.exists():
        return []
    with open(features, encoding="utf-8") as f:
        recorded = json.load(f).get("bundle")
    if not recorded:
        return []
    out = []
    if model.exists() and _sha256(model) != recorded["model_sha256"]:
        out.append(f"{model.name} is not the model this feature file was trained with")
    expected_geo = recorded.get("geo_sha256")
    if expected_geo and geo.exists() and _sha256(geo) != expected_geo:
        out.append(f"{geo.name} is not the geo state this model was trained with")
    return out


def verify(models_dir: Union[str, Path], variant: str) -> None:
    """Raise `BundleMismatch` for mismatched files; warn when a recorded geo state is missing.
    Set BUNDLE_CHECK=0 to skip."""
    if os.getenv("BUNDLE_CHECK", "1") == "0":
        return
    bad = problems(models_dir, variant)
    if bad:
        raise BundleMismatch(
            f"{variant}: " + "; ".join(bad) + ". Retrain the variant (training re-stamps it) or restore the "
            "matching files; BUNDLE_CHECK=0 skips this check.")
    model, features, geo = _paths(models_dir, variant)
    if features.exists():
        recorded = (json.load(open(features, encoding="utf-8")).get("bundle") or {})
        if recorded.get("geo_sha256") and not geo.exists():
            logger.warning("%s: %s is missing, so address / station features will be NaN for this variant",
                           variant, geo.name)
