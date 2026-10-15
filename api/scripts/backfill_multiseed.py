"""Give an already-registered model version the multi-seed metrics the promotion gate compares on.

The gate (services/multiseed_gate.py) only uses multi-seed evidence when both the candidate and
Production have it; Production runs trained before it existed do not. This replays a registered
version's estimator (cloned and refitted, never its stored weights) on the same grouped splits as a
fresh training run and logs the metrics to that run. It uses the CURRENT feature pipeline and data, so
run it on the version you intend to compare against, and only that.

    MLFLOW_TRACKING_URI=sqlite:///mlflow_train.db uv run python scripts/backfill_multiseed.py \
        tokyo_rent_lgbm            # Production version
    ... tokyo_rent_lgbm_noterms --version 6
"""
import argparse
import os
import sys

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)


def load_registered_estimator(uri: str):
    import mlflow.lightgbm
    import mlflow.sklearn
    try:
        return mlflow.sklearn.load_model(uri)       # linear models and the per-m² wrapper
    except Exception:
        return mlflow.lightgbm.load_model(uri)      # older runs logged the bare booster wrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name")
    ap.add_argument("--version", help="version number; default: the Production version")
    args = ap.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.base import clone

    client = MlflowClient()
    if args.version:
        mv = client.get_model_version(args.model_name, args.version)
    else:
        versions = client.get_latest_versions(args.model_name, stages=["Production"])
        if not versions:
            sys.exit(f"No Production version of {args.model_name}")
        mv = versions[0]
    run = client.get_run(mv.run_id)
    if "rmse_multiseed" in run.data.metrics:
        sys.exit(f"{args.model_name} v{mv.version} already has multi-seed metrics")

    # The training modules read their variant from env / argv at import time
    if args.model_name.endswith("_noterms"):
        os.environ["NO_TERMS"] = "1"
    if run.data.params.get("geo_features") == "False":
        os.environ["GEO_FEATURES"] = "0"
    family = "challenger_model" if "lgbm" in args.model_name else "reg_model"
    module = __import__(f"models.{family}", fromlist=["load_data"])

    from services import multiseed_gate
    from services.data_split import groups_for

    estimator = load_registered_estimator(f"models:/{args.model_name}/{mv.version}")

    def split(s):
        Xtr, Xte, ytr, yte, df_l = module.load_data(seed=s, save=False)
        return Xtr, Xte, ytr, yte, groups_for(df_l, Xtr)

    result = multiseed_gate.evaluate_seeds(split, lambda: clone(estimator))
    multiseed_gate.log_metrics(result, lambda k, v: client.log_metric(mv.run_id, k, v))
    print(f"{args.model_name} v{mv.version}: RMSE {result['rmse_multiseed']:.2f} ± {result['rmse_multiseed_std']:.2f} "
          f"over {result['multiseed_n']} splits (logged to run {mv.run_id})")


if __name__ == "__main__":
    main()
