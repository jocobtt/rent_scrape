"""Shared (with-terms) LightGBM: plain rent target vs per-m² wrapper, on seeds 0..N-1 of the building hold-out.

Compares Production's hyperparameters and the Optuna-tuned ones from the latest training log, scored with
deposit/key money present and hidden. Usage (from api/):
    MLFLOW_TRACKING_URI=sqlite:///mlflow_train.db uv run python scripts/per_sqm_shared_check.py logs/perm_challenger.log
"""
import ast
import re
import sys
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.multiseed_eval import encoded_frame, lgbm_params, load_frames, metrics  # noqa: E402
from services.data_split import grouped_train_test_split  # noqa: E402
from services.feature_preprocessor import OPTIONAL_TERMS, mask_optional_terms  # noqa: E402
from services.per_sqm_model import PerSqmRegressor  # noqa: E402


def tuned_from_log(path):
    txt = open(path).read()
    d = ast.literal_eval(re.search(r"params=(\{.*\})", txt).group(1))
    base = lgbm_params("tokyo_rent_lgbm")
    return {**base, **d}


def run(params, X, y, groups, seed, wrap):
    Xtr, Xte, ytr, yte, gtr = grouped_train_test_split(X, y, groups, 0.2, seed)
    Xtr = mask_optional_terms(Xtr, 0.25, seed)
    Xf, Xv, yf, yv, _ = grouped_train_test_split(Xtr, ytr, gtr, test_size=0.1, random_state=seed)
    g = lgb.LGBMRegressor(**{**params, "random_state": seed, "verbose": -1})
    m = PerSqmRegressor(g, 2.0) if wrap else g
    m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric="l1", callbacks=[lgb.early_stopping(50, verbose=False)])
    nt = Xte.copy()
    nt[OPTIONAL_TERMS] = np.nan
    return metrics(yte, m.predict(Xte))["rmse"], metrics(yte, m.predict(nt))["rmse"]


SEEDS = [int(x) for x in os.getenv("SEEDS", "0,1,2,3,4").split(",")]

if __name__ == "__main__":
    df, groups = load_frames()
    X, y = encoded_frame(df)
    sets = {"production params": lgbm_params("tokyo_rent_lgbm"), **({"tuned params": tuned_from_log(sys.argv[1])} if len(sys.argv) > 1 else {})}
    for pname, p in sets.items():
        for wrap in (False, True):
            r = np.array([run(p, X, y, groups, s, wrap) for s in SEEDS])
            print(f"{pname:18s} {'per-m²' if wrap else 'rent  '}  with terms {r[:,0].mean():.2f} (sd {r[:,0].std():.2f})"
                  f"  no terms {r[:,1].mean():.2f}   per-seed with: {np.round(r[:,0],2).tolist()}", flush=True)
