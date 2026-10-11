"""Follow-up geo experiments: residual lag, rent-per-m² target, density ablation, spatial block CV.

Questions (all LightGBM with Production `tokyo_rent_lgbm` hyperparameters, deposit/key money masked in
training and scored with and without them):

  1. Residual lag: smooth the *out-of-fold residual* of a structural model (size, age, floor, walk,
     layout, building type; no location) instead of raw log rent/m², so unit mix does not leak
     into the neighbourhood signal.
  2. Target: predict rent per m² (the lag's own unit) and multiply by size, weighting samples by
     size^w so the loss still tracks rent errors.
  3. Density ablation: drop geo_dist_s_km and geo_kern_density, which mostly encode the scrape cap
     (about 300 buildings per ward), not real building density.
  4. Spatial block CV: hold out whole spatial clusters and drop every training building within
     `--buffer-km` of a held-out one, to measure generalisation to a new area (the building and street-
     block schemes leave the neighbours in training).

Usage (from api/):
    MLFLOW_TRACKING_URI=sqlite:///mlflow_train.db uv run python scripts/geo_residual_experiment.py --seeds 5
Writes evaluation_reports/geo_residual_<timestamp>.json.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)

from scripts.multiseed_eval import encoded_frame, lgbm_params, markdown, metrics, summarise  # noqa: E402
from services.data_cleaning import load_training_frame  # noqa: E402
from services.data_split import grouped_train_test_split  # noqa: E402
from services.feature_preprocessor import OPTIONAL_TERMS, mask_optional_terms  # noqa: E402
from services.geo_features import DENSITY_COLS, GEO_COLS, GeoRentFeatures, load_coords, oof_residuals  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("geo_residual")

NO_DENSITY = [c for c in GEO_COLS if c not in DENSITY_COLS]
# name -> (lag source, columns, target mode, sample-weight power on size)
VARIANTS = {
    "base / rent": (None, [], "rent", 0),
    "base / rent per m²": (None, [], "ppm", 2),
    "raw lag / rent  [current]": ("raw", GEO_COLS, "rent", 0),
    "raw lag, no density / rent": ("raw", NO_DENSITY, "rent", 0),
    "raw lag / rent per m²": ("raw", GEO_COLS, "ppm", 2),
    "residual lag / rent": ("resid", GEO_COLS, "rent", 0),
    "residual lag, no density / rent": ("resid", NO_DENSITY, "rent", 0),
    "residual lag / rent per m² (w=size²)": ("resid", GEO_COLS, "ppm", 2),
    "residual lag / rent per m² (w=size)": ("resid", GEO_COLS, "ppm", 1),
}
CURRENT = "raw lag / rent  [current]"


# ------------------------------------------------------------------ residual target

# ------------------------------------------------------------------ splits

def spatial_split(df, X, y, groups, seed, coords, n_clusters=30, test_frac=0.2, buffer_km=1.0):
    """Hold out whole k-means spatial clusters; drop training rows within buffer_km of any held-out row."""
    pts = np.radians(np.array([coords[a] for a in df["address"]]))
    labels = KMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit_predict(np.degrees(pts))
    rng = np.random.default_rng(seed)
    test_clusters = rng.choice(n_clusters, size=max(1, int(round(n_clusters * test_frac))), replace=False)
    is_test = np.isin(labels, test_clusters)
    dist, _ = BallTree(pts[is_test], metric="haversine").query(pts[~is_test], k=1)
    keep = np.flatnonzero(~is_test)[dist[:, 0] * 6371.0088 >= buffer_km]
    tr_idx, te_idx = df.index[keep], df.index[is_test]
    return X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx], groups.loc[tr_idx]


# ------------------------------------------------------------------ model

def fit_predict(params, A, B, ytr, gtr, sqm_tr, sqm_te, mode, power, frac, seed):
    """Train on A (terms masked), return rent predictions for B with and without deposit/key money."""
    A = mask_optional_terms(A, frac, seed)
    target = ytr / sqm_tr if mode == "ppm" else ytr
    w = (sqm_tr ** power) if power else pd.Series(1.0, index=ytr.index)
    Xf, Xv, yf, yv, _ = grouped_train_test_split(A, target, gtr, test_size=0.1, random_state=seed)
    m = lgb.LGBMRegressor(**{**params, "random_state": seed, "verbose": -1}).fit(
        Xf, yf, sample_weight=w.loc[Xf.index], eval_set=[(Xv, yv)], eval_sample_weight=[w.loc[Xv.index]],
        eval_metric="l1", callbacks=[lgb.early_stopping(50, verbose=False)])
    nt = B.copy()
    nt[OPTIONAL_TERMS] = np.nan
    scale = sqm_te if mode == "ppm" else 1.0
    return m.predict(B) * scale, m.predict(nt) * scale


def run_scheme(df, scheme, seeds, params, frac, coords):
    X, y = encoded_frame(df)
    groups = df["address" if scheme == "address" else "building_id"].astype(str)
    out = {f"{n} | {t}": [] for n in VARIANTS for t in ("with_terms", "no_terms")}
    kept = []
    for s in seeds:
        if scheme == "spatial_block":
            Xtr, Xte, ytr, yte, gtr = spatial_split(df, X, y, groups, s, coords)
        else:
            Xtr, Xte, ytr, yte, gtr = grouped_train_test_split(X, y, groups, 0.2, s)
        kept.append((len(Xtr), len(Xte)))
        dtr, dte = df.loc[Xtr.index], df.loc[Xte.index]
        feats = {}
        for lag in ("raw", "resid"):
            target = oof_residuals(dtr, gtr) if lag == "resid" else None
            geo = GeoRentFeatures(coords).fit(dtr, gtr, target=target)
            feats[lag] = (geo.transform(dtr, gtr, in_train=True), geo.transform(dte, groups.loc[Xte.index], in_train=False))
        for name, (lag, cols, mode, power) in VARIANTS.items():
            A, B = Xtr, Xte
            if lag:
                A, B = pd.concat([Xtr, feats[lag][0][cols]], axis=1), pd.concat([Xte, feats[lag][1][cols]], axis=1)
            p_with, p_without = fit_predict(params, A, B, ytr, gtr, dtr["sqr_m"], dte["sqr_m"], mode, power, frac, s)
            out[f"{name} | with_terms"].append(metrics(yte, p_with))
            out[f"{name} | no_terms"].append(metrics(yte, p_without))
        log.warning("%s seed %s done (train %d, test %d)", scheme, s, *kept[-1])
    return out, kept


def paired(results, ref):
    lines = [f"| Variant | ΔRMSE vs `{ref}` | better on |", "| --- | --- | --- |"]
    for key, runs in results.items():
        name, terms = key.split(" | ")
        if name == ref:
            continue
        d = np.array([r["rmse"] - b["rmse"] for r, b in zip(runs, results[f"{ref} | {terms}"])])
        lines.append(f"| {name} ({terms}) | {d.mean():+.2f} | {(d < 0).sum()}/{len(d)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--schemes", default="building_id,spatial_block")
    args = ap.parse_args()

    df = load_training_frame(None)
    coords = load_coords()
    params = lgbm_params("tokyo_rent_lgbm")
    seeds = list(range(args.seeds))
    report = {"seeds": seeds, "rows": len(df), "schemes": {}}
    for scheme in args.schemes.split(","):
        results, kept = run_scheme(df, scheme, seeds, params, args.mask_frac, coords)
        rows = summarise(results)
        report["schemes"][scheme] = {"per_seed": results, "summary": rows, "train_test_sizes": kept}
        print(f"\n### Held-out {scheme}   (train/test rows per seed: {kept})\n{markdown(rows)}", flush=True)
        print(f"\nvs the no-geo baseline\n{paired(results, 'base / rent')}", flush=True)
        print(f"\nvs the current production setup\n{paired(results, CURRENT)}", flush=True)

    path = os.path.join(API_DIR, "evaluation_reports", f"geo_residual_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(report, f)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
