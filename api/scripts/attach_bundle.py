"""Attach the model files currently in models/ to the registry run they came from.

Runs trained before bundles existed have no `bundle/` artifact, so services/model_sync.py cannot serve
them from the registry. If the local files really are that Production version, this logs them to its
run and stamps the run id into the features file. "Really are" is checked, not assumed: the local
model and the registered artifact must give identical predictions on a random matrix.

    MLFLOW_TRACKING_URI=sqlite:///mlflow_train.db uv run python scripts/attach_bundle.py challenger-noterms
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)

from services.artifact_bundle import stamp  # noqa: E402
from services.model_sync import VARIANT_MODELS, local_run_id  # noqa: E402

MODELS_DIR = os.path.join(API_DIR, "models")


def same_predictions(local, registered, n: int = 200, seed: int = 0) -> bool:
    """Identical predictions on a random matrix (False for any model that cannot even take the same input)."""
    names = getattr(local, "feature_names_in_", None)
    if names is None:
        names = getattr(registered, "feature_names_in_", None)
    if names is None:
        return False
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(1, 60, size=(n, len(names))), columns=list(names))
    try:
        a, b = local.predict(X), registered.predict(X)
        return bool(np.allclose(a, b, rtol=1e-6, atol=1e-6))
    except Exception:
        return False


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANT_MODELS:
        sys.exit(f"usage: attach_bundle.py {{{'|'.join(VARIANT_MODELS)}}}")
    variant = sys.argv[1]
    name = VARIANT_MODELS[variant]

    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    production = client.get_latest_versions(name, stages=["Production"])
    prod_version = production[0].version if production else None

    from scripts.backfill_multiseed import load_registered_estimator
    local = joblib.load(os.path.join(MODELS_DIR, f"{variant}-model.joblib"))
    # Which registered version are the files on disk? (Not necessarily Production: training used to
    # overwrite models/ whether or not the candidate was promoted.)
    match = None
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version), reverse=True):
        try:
            candidate = load_registered_estimator(f"models:/{name}/{mv.version}")
        except Exception:
            continue
        if same_predictions(local, candidate):
            match = mv
            break
    if match is None:
        sys.exit(f"models/{variant}-model.joblib matches no registered version of {name}; retrain to get a bundle")
    if local_run_id(MODELS_DIR, variant) == match.run_id:
        sys.exit(f"{variant} already belongs to run {match.run_id}")
    if any(a.path == "bundle" for a in client.list_artifacts(match.run_id)):
        sys.exit(f"run {match.run_id} already has a bundle; let sync install it")

    stamp(MODELS_DIR, variant, match.run_id)
    for suffix in ("model.joblib", "features.json", "geo.json"):
        f = os.path.join(MODELS_DIR, f"{variant}-{suffix}")
        if os.path.exists(f):
            client.log_artifact(match.run_id, f, "bundle")
    note = "Production" if match.version == prod_version else f"NOT Production (Production is v{prod_version})"
    print(f"{variant}: the files on disk are {name} v{match.version} ({note}); attached to run {match.run_id}")


if __name__ == "__main__":
    main()
