"""Score the served models on listings that appeared after they were trained.

There is no feedback loop for a rent predictor: nobody tells the API what a flat really rented for. Each
re-scrape is the substitute. Listings in a newer scrape whose unit was never in the training data are
genuinely unseen, so the error on them is an honest, time-forward estimate of how the models do now
(the train/test split in training says how they do on other buildings from the same moment).

`new_units` finds those listings, `score` runs them through the real inference path
(`PredictionService`, so address / station features, imputation and variant routing are exercised), and
`summarise` turns the result into per-model, per-variant error numbers. scripts/score_fresh_listings.py
is the command line; with `--log` the rows also go into the prediction log with their actual rent, which
is what the auto-retrain performance check reads.
"""
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# What identifies "the same unit" across scrapes: everything but the rent, fees and scrape date.
UNIT_KEY = ["address", "floor", "sqr_m", "apartment_type", "house_type", "year_built"]


def _keys(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    return df[columns].astype(str).agg("|".join, axis=1)


def new_units(fresh: pd.DataFrame, trained_on: pd.DataFrame) -> pd.DataFrame:
    """Rows of `fresh` whose unit never appeared in `trained_on`."""
    columns = [c for c in UNIT_KEY if c in fresh.columns and c in trained_on.columns]
    if len(columns) < 4:
        raise ValueError(f"Not enough identifying columns to match units across datasets: {columns}")
    seen = set(_keys(trained_on, columns))
    return fresh[~_keys(fresh, columns).isin(seen)]


def payloads(df: pd.DataFrame, with_terms: bool) -> List[Dict]:
    """API-shaped request bodies for each row. `with_terms=False` leaves deposit / key money out, which is
    what routes a request to the no-terms variant."""
    out = []
    for r in df.to_dict("records"):
        body = {
            "sqr_m": r["sqr_m"], "maintenence_price": r["maintenence_price"], "year_built": r["year_built"],
            "floor": r["floor"], "eki_walk": r["nearest_eki_walk"],
            "address": r.get("address"), "nearest_station": r.get("nearest_station"), "nearest_line": r.get("nearest_line"),
        }
        if with_terms:
            body.update(rei_price=r["rei_price"], shikikin=r["shikikin"])
        out.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in body.items()})
    return out


def score(prediction_service, df: pd.DataFrame, models: Iterable[str] = ("passed", "challenger")) -> pd.DataFrame:
    """One row per (listing, model, terms given?): the prediction next to the actual rent."""
    rows = []
    for with_terms in (True, False):
        bodies = payloads(df, with_terms)
        for model in models:
            predict = getattr(prediction_service, f"predict_with_{model}_model")
            for body, actual, idx in zip(bodies, df["rent_price"], df.index):
                res = predict(dict(body))
                rows.append({"index": idx, "model_type": model, "variant": res["variant"], "with_terms": with_terms,
                             "prediction": res["prediction"], "actual_value": float(actual), "input": body})
    return pd.DataFrame(rows)


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """RMSE / MAE / MAPE / R² per (model, variant)."""
    out = []
    for (model, variant), g in scored.groupby(["model_type", "variant"]):
        err = g["actual_value"] - g["prediction"]
        ss_tot = ((g["actual_value"] - g["actual_value"].mean()) ** 2).sum()
        out.append({"model_type": model, "variant": variant, "n": len(g),
                    "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(err.abs().mean()),
                    "mape": float((err.abs() / g["actual_value"]).mean() * 100),
                    "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")})
    return pd.DataFrame(out)
