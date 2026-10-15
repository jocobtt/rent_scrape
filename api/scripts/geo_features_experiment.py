"""Which geo / station feature sets and settings help LightGBM on held-out buildings?

All variants are LightGBM with the hyperparameters of the Production `tokyo_rent_lgbm` run, deposit /
key money randomly masked in training (as in the shared model) and scored with and without them, on
several grouped splits. A training row's geo features never include its own group's target (see
services/geo_features.py).

Variant sets (`--set`):
  features   base | address geo | station only | address geo + station
  tuning     address geo + station, one setting changed at a time (bandwidth, neighbour counts, shrinkage)

Hold-out schemes (`--schemes`): whole buildings (`building_id`, realistic) and whole street blocks
(`address`, a stress test where the block itself is unseen).

Usage (from api/):
    MLFLOW_TRACKING_URI=sqlite:///mlflow_train.db uv run python scripts/geo_features_experiment.py --set features
    ... --set tuning --seeds 5 --schemes building_id,address
Writes evaluation_reports/geo_<set>_<timestamp>.json and prints markdown tables, including how
often each variant beat `base` on the same split.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)

from scripts.multiseed_eval import encoded_frame, fit_lgbm, lgbm_params, markdown, metrics, summarise  # noqa: E402
from services.data_cleaning import load_training_frame  # noqa: E402
from services.data_split import grouped_train_test_split  # noqa: E402
from services.feature_preprocessor import OPTIONAL_TERMS, mask_optional_terms  # noqa: E402
from services.geo_features import ADDRESS_COLS, GEO_COLS, STATION_COLS, GeoRentFeatures, load_coords  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("geo_experiment")

ALL = ADDRESS_COLS + STATION_COLS
# name -> (GeoRentFeatures kwargs, columns added to the base features)
SETS = {
    "features": {
        "base": ({}, []),
        "address geo": ({}, ADDRESS_COLS),
        "station only": ({}, STATION_COLS),
        "address geo + station": ({}, ALL),
    },
    "tuning": {
        "default (bw .4, ks 5/20, shrink 5/5/20)": ({}, ALL),
        "bandwidth 0.2": ({"bandwidth_km": 0.2}, ALL),
        "bandwidth 0.8": ({"bandwidth_km": 0.8}, ALL),
        "ks 3/10": ({"ks": (3, 10)}, ALL),
        "ks 10/40": ({"ks": (10, 40)}, ALL),
        "shrink x0.4 (town 2, station 2, line 8)": ({"town_shrink": 2, "station_shrink": 2, "line_shrink": 8}, ALL),
        "shrink x3 (town 15, station 15, line 60)": ({"town_shrink": 15, "station_shrink": 15, "line_shrink": 60}, ALL),
    },
}


def run_scheme(df, group_col, seeds, params, frac, coords, variants):
    X, y = encoded_frame(df)
    groups = df[group_col].astype(str)
    out = {f"{name} | {terms}": [] for name in variants for terms in ("with_terms", "no_terms")}
    for s in seeds:
        Xtr, Xte, ytr, yte, gtr = grouped_train_test_split(X, y, groups, 0.2, s)
        cache = {}  # identical GeoRentFeatures settings share one fit per split
        for name, (kwargs, cols) in variants.items():
            if cols:
                key = json.dumps(kwargs, sort_keys=True)
                if key not in cache:
                    geo = GeoRentFeatures(coords, **kwargs).fit(df.loc[Xtr.index], groups.loc[Xtr.index])
                    cache[key] = (geo.transform(df.loc[Xtr.index], groups.loc[Xtr.index], in_train=True),
                                  geo.transform(df.loc[Xte.index], groups.loc[Xte.index], in_train=False))
                f_tr, f_te = cache[key]
                A, B = pd.concat([Xtr, f_tr[cols]], axis=1), pd.concat([Xte, f_te[cols]], axis=1)
            else:
                A, B = Xtr, Xte
            model = fit_lgbm(params, mask_optional_terms(A, frac, s), ytr, gtr, s)
            out[f"{name} | with_terms"].append(metrics(yte, model.predict(B)))
            nt = B.copy()
            nt[OPTIONAL_TERMS] = np.nan
            out[f"{name} | no_terms"].append(metrics(yte, model.predict(nt)))
        log.warning("%s seed %s done", group_col, s)
    return out


def paired_vs_base(results, base_name):
    """Per variant: mean RMSE change vs the base variant and on how many splits it was better."""
    lines = ["| Variant | mean ΔRMSE vs base | better on |", "| --- | --- | --- |"]
    for key, runs in results.items():
        name, terms = key.split(" | ")
        if name == base_name:
            continue
        base = results[f"{base_name} | {terms}"]
        d = np.array([r["rmse"] - b["rmse"] for r, b in zip(runs, base)])
        lines.append(f"| {name} ({terms}) | {d.mean():+.2f} | {(d < 0).sum()}/{len(d)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="features", choices=list(SETS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--schemes", default="building_id,address")
    args = ap.parse_args()

    df = load_training_frame(None)
    if "nearest_station" not in df.columns:
        sys.exit("The dataset has no nearest_station / nearest_line columns: rescrape first.")
    coords = load_coords()
    print(f"rows={len(df)}  without coordinates={df['address'].map(lambda a: not coords.get(a)).mean():.1%}", flush=True)
    params = lgbm_params("tokyo_rent_lgbm")
    variants = SETS[args.set]
    base_name = next(iter(variants))
    seeds = list(range(args.seeds))

    report = {"set": args.set, "seeds": seeds, "rows": len(df), "schemes": {}}
    for scheme in args.schemes.split(","):
        results = run_scheme(df, scheme, seeds, params, args.mask_frac, coords, variants)
        rows = summarise(results)
        report["schemes"][scheme] = {"per_seed": results, "summary": rows}
        print(f"\n### Held-out {scheme}\n{markdown(rows)}\n\n{paired_vs_base(results, base_name)}", flush=True)

    path = os.path.join(API_DIR, "evaluation_reports", f"geo_{args.set}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(report, f)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
