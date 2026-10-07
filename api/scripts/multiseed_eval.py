"""Re-score every model family on several grouped train/test splits.

A single split is noisy here: with identical LightGBM parameters, test RMSE ranged from 4.0 to
9.2 across five splits depending on which buildings were held out. This script replays the
hyperparameters already found (it does NOT re-tune) on seeds 0..N-1 and reports mean ± std.

Hyperparameters come from the Production version of each registered model (or the latest
training run for families that have no registry entry). Fast families (LightGBM, linear) take
minutes; `tabnet`, `tabpfn` and `torch` take much longer.

Usage (from api/):
    MLFLOW_TRACKING_URI=sqlite:///mlflow_train.db \\
        uv run python scripts/multiseed_eval.py --seeds 5 --families lightgbm,linear
    ... --families lightgbm,linear,tabnet,tabpfn,torch

Writes evaluation_reports/multiseed_<timestamp>.json and prints a markdown table.
"""
import argparse
import ast
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)

from services.data_cleaning import load_training_frame  # noqa: E402
from services.data_split import group_labels, grouped_train_test_split, split_meta  # noqa: E402
from services.feature_preprocessor import FeaturePreprocessor, OPTIONAL_TERMS  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("multiseed")


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float).ravel()
    err = y_true - y_pred
    ss_res, ss_tot = float((err ** 2).sum()), float(((y_true - y_true.mean()) ** 2).sum())
    nz = y_true != 0
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": 1 - ss_res / ss_tot,
        "mape": float(np.mean(np.abs(err[nz] / y_true[nz])) * 100),
    }


# ---------------------------------------------------------------- hyperparameters from MLflow

def _parse(v: str):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def registered_params(name: str) -> dict:
    """Params of the run behind the Production version of a registered model."""
    from mlflow.tracking import MlflowClient
    c = MlflowClient()
    versions = c.get_latest_versions(name, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production version of {name}")
    return {k: _parse(v) for k, v in c.get_run(versions[0].run_id).data.params.items()}


def latest_run_params(run_name: str) -> dict:
    import mlflow
    runs = mlflow.search_runs(search_all_experiments=True, order_by=["start_time DESC"],
                              filter_string=f"tags.mlflow.runName = '{run_name}' and metrics.rmse > 0")
    if runs.empty:
        raise RuntimeError(f"No finished run named {run_name}")
    params = {c[len("params."):]: runs.iloc[0][c] for c in runs.columns if c.startswith("params.")}
    return {k: _parse(v) for k, v in params.items() if isinstance(v, str)}


def lgbm_params(name: str) -> dict:
    p = registered_params(name)
    drop = {"tuning_trials", "terms_mask_frac", "variant", "task", "boosting_type", "metric"}
    return {k: v for k, v in p.items() if k not in drop}


# ---------------------------------------------------------------- data

GROUP_BY = "building_id"  # or "address": a street block, so a much stricter hold-out


def load_frames():
    df = load_training_frame(None)
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    groups = df[GROUP_BY].astype(str) if GROUP_BY != "building_id" else group_labels(df)
    return df, groups


def encoded_frame(df: pd.DataFrame, drop_terms: bool = False):
    d = df.drop(columns=OPTIONAL_TERMS) if drop_terms else df
    enc, _ = FeaturePreprocessor().fit_transform(d)
    return enc.drop(columns="rent_price").astype(float), enc["rent_price"]


def dummies_frame(df: pd.DataFrame):
    """The representation TabNet / TabPFN train on (see models/tabular_pretrained_model.py)."""
    d, groups = split_meta(df)
    for col, prefix in (("ku_name", "ku"), ("apartment_type", "apt"), ("house_type", "house")):
        d[col] = d[col].astype("category")
        d = pd.get_dummies(d, columns=[col], prefix=prefix)
    return d.drop(columns="rent_price").astype(float), d["rent_price"], groups


# ---------------------------------------------------------------- families

def fit_lgbm(params, Xtr, ytr, gtr, seed):
    import lightgbm as lgb
    Xf, Xv, yf, yv, _ = grouped_train_test_split(Xtr, ytr, gtr, test_size=0.1, random_state=seed)
    return lgb.LGBMRegressor(**{**params, "random_state": seed, "verbose": -1}).fit(
        Xf, yf, eval_set=[(Xv, yv)], eval_metric="l1", callbacks=[lgb.early_stopping(50, verbose=False)])


def run_lightgbm(df, groups, seeds, shared_p, dedicated_p, frac):
    from services.feature_preprocessor import mask_optional_terms
    X, y = encoded_frame(df)
    Xd, _ = encoded_frame(df, drop_terms=True)
    out = {"lightgbm_shared_with_terms": [], "lightgbm_shared_no_terms": [], "lightgbm_dedicated_no_terms": []}
    for s in seeds:
        Xtr, Xte, ytr, yte, gtr = grouped_train_test_split(X, y, groups, 0.2, s)
        m = fit_lgbm(shared_p, mask_optional_terms(Xtr, frac, s), ytr, gtr, s)
        out["lightgbm_shared_with_terms"].append(metrics(yte, m.predict(Xte)))
        nt = Xte.copy()
        nt[OPTIONAL_TERMS] = np.nan
        out["lightgbm_shared_no_terms"].append(metrics(yte, m.predict(nt)))
        Xd_tr, Xd_te = Xd.loc[Xtr.index], Xd.loc[Xte.index]
        md = fit_lgbm(dedicated_p, Xd_tr, ytr, gtr, s)
        out["lightgbm_dedicated_no_terms"].append(metrics(yte, md.predict(Xd_te)))
        log.warning("seed %s lightgbm done", s)
    return out


def run_linear(df, groups, seeds):
    from sklearn.linear_model import LinearRegression
    X, y = encoded_frame(df)
    Xd, _ = encoded_frame(df, drop_terms=True)
    out = {"linear_with_terms": [], "linear_median_imputed_no_terms": [], "linear_dedicated_no_terms": []}
    for s in seeds:
        Xtr, Xte, ytr, yte, _ = grouped_train_test_split(X, y, groups, 0.2, s)
        m = LinearRegression().fit(Xtr, ytr)
        out["linear_with_terms"].append(metrics(yte, m.predict(Xte)))
        imp = Xte.copy()
        for c in OPTIONAL_TERMS:
            imp[c] = Xtr[c].median()
        out["linear_median_imputed_no_terms"].append(metrics(yte, m.predict(imp)))
        md = LinearRegression().fit(Xd.loc[Xtr.index], ytr)
        out["linear_dedicated_no_terms"].append(metrics(yte, md.predict(Xd.loc[Xte.index])))
        log.warning("seed %s linear done", s)
    return out


def _regroup(df, groups):
    """Swap in the --group-by key (dummies_frame groups by building)."""
    return df[GROUP_BY].astype(str).set_axis(groups.index) if GROUP_BY != "building_id" else groups


def run_tabnet(df, seeds):
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor
    X, y, groups = dummies_frame(df)
    groups = _regroup(df, groups)
    p = latest_run_params("tabnet_model")
    lr = p["optimizer_params"]["lr"] if isinstance(p["optimizer_params"], dict) else _parse(p["optimizer_params"])["lr"]
    batch = int(p.get("batch_size", 256))
    out = {"tabnet": []}
    for s in seeds:
        Xtr, Xte, ytr, yte, gtr = grouped_train_test_split(X, y, groups, 0.2, s)
        Xf, Xv, yf, yv, _ = grouped_train_test_split(Xtr, ytr, gtr, 0.2, s)
        model = TabNetRegressor(
            n_d=int(p["n_d"]), n_a=int(p["n_a"]), n_steps=int(p["n_steps"]), gamma=float(p["gamma"]),
            n_independent=2, n_shared=2, lambda_sparse=float(p["lambda_sparse"]),
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=lr),
            scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={"step_size": 50, "gamma": 0.9},
            mask_type="entmax", verbose=0, seed=s)
        model.fit(X_train=Xf.values, y_train=yf.values.reshape(-1, 1),
                  eval_set=[(Xv.values, yv.values.reshape(-1, 1))], eval_name=["valid"], eval_metric=["rmse"],
                  max_epochs=200, patience=20, batch_size=batch, virtual_batch_size=min(128, batch),
                  num_workers=0, drop_last=False)
        out["tabnet"].append(metrics(yte, model.predict(Xte.values).ravel()))
        log.warning("seed %s tabnet done", s)
    return out


def run_tabpfn(df, seeds):
    from tabpfn import TabPFNRegressor
    from models.tabular_pretrained_model import sample_context
    X, y, groups = dummies_frame(df)
    groups = _regroup(df, groups)
    n_ctx = int(os.environ.get("TABPFN_MAX_SAMPLES", "8000"))
    out = {"tabpfn": []}
    for s in seeds:
        Xtr, Xte, ytr, yte, _ = grouped_train_test_split(X, y, groups, 0.2, s)
        Xc, yc = sample_context(Xtr, ytr, n_ctx, "stratified", seed=s)
        model = TabPFNRegressor(device="cpu", n_estimators=8, ignore_pretraining_limits=True, random_state=s)
        model.fit(Xc.values, yc.values)
        out["tabpfn"].append(metrics(yte, model.predict(Xte.values)))
        log.warning("seed %s tabpfn done", s)
    return out


def run_torch(seeds, model_types):
    from models.torch_model import TorchRentPredictor
    from services.data_split import grouped_train_val_test_split
    out = {}
    for mt in model_types:
        p = latest_run_params(f"torch_{mt}")
        keep = {"learning_rate", "weight_decay", "batch_size", "dropout_rate", "use_attention", "patience", "hidden_dims"}
        hp = {k: v for k, v in p.items() if k in keep}
        if mt == "tabular" and "hidden_dims" not in hp:
            raise RuntimeError("Latest torch_tabular run predates the hidden_dims fix; retrain it first.")
        out[f"torch_{mt}"] = []
        for s in seeds:
            pred = TorchRentPredictor(model_type=mt, device="cpu", random_seed=s)
            X, y = pred.load_and_prepare_data()
            if GROUP_BY != "building_id":
                pred.groups_ = _regroup(DF, pred.groups_)
            Xtr, Xv, Xte, ytr, yv, yte = grouped_train_val_test_split(X, y, pred.groups_, 0.2, 0.2, s)
            pred.train_model(Xtr, ytr, Xv, yv, hp, num_epochs=int(os.environ.get("TORCH_FINAL_EPOCHS", "100")))
            m = pred.evaluate_model(Xte, yte)
            out[f"torch_{mt}"].append({k: float(m[k]) for k in ("rmse", "r2", "mape")})
            log.warning("seed %s torch_%s done", s, mt)
    return out


# ---------------------------------------------------------------- reporting

def summarise(results: dict) -> list:
    rows = []
    for name, runs in results.items():
        r = {k: np.array([x[k] for x in runs]) for k in ("rmse", "r2", "mape")}
        rows.append({"model": name, "n": len(runs),
                     **{f"{k}_mean": float(v.mean()) for k, v in r.items()},
                     **{f"{k}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in r.items()},
                     "rmse_per_seed": [round(float(x), 2) for x in r["rmse"]]})
    return rows


def markdown(rows: list) -> str:
    lines = ["| Model | RMSE (mean ± sd) | R² (mean ± sd) | MAPE % (mean ± sd) | RMSE per seed |",
             "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['rmse_mean']:.2f} ± {r['rmse_std']:.2f} | "
                     f"{r['r2_mean']:.3f} ± {r['r2_std']:.3f} | {r['mape_mean']:.1f} ± {r['mape_std']:.1f} | "
                     f"{', '.join(map(str, r['rmse_per_seed']))} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--families", default="lightgbm,linear",
                    help="comma list of: lightgbm, linear, tabnet, tabpfn, torch")
    ap.add_argument("--torch-types", default="tabular,ensemble")
    ap.add_argument("--mask-frac", type=float, default=float(os.getenv("TERMS_MASK_FRAC", "0.25")))
    ap.add_argument("--group-by", default="building_id", choices=["building_id", "address"])
    args = ap.parse_args()
    global GROUP_BY, DF
    GROUP_BY = args.group_by

    seeds = list(range(args.seeds))
    fams = {f.strip() for f in args.families.split(",")}
    df, groups = load_frames()
    DF = df
    results, t0 = {}, time.time()

    if "lightgbm" in fams:
        results.update(run_lightgbm(df, groups, seeds, lgbm_params("tokyo_rent_lgbm"),
                                    lgbm_params("tokyo_rent_lgbm_noterms"), args.mask_frac))
    if "linear" in fams:
        results.update(run_linear(df, groups, seeds))
    if "tabnet" in fams:
        results.update(run_tabnet(df, seeds))
    if "tabpfn" in fams:
        results.update(run_tabpfn(df, seeds))
    if "torch" in fams:
        results.update(run_torch(seeds, args.torch_types.split(",")))

    rows = summarise(results)
    out_dir = os.path.join(API_DIR, "evaluation_reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"multiseed_{GROUP_BY}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump({"seeds": seeds, "rows": len(df), "elapsed_s": round(time.time() - t0),
                   "per_seed": results, "summary": rows}, f, indent=2)
    print(f"\n{len(df)} rows, {len(seeds)} 80/20 splits grouped by {GROUP_BY}, {time.time() - t0:.0f}s\n")
    print(markdown(rows))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
