"""Train the demo baseline models from synthetic data so the API works out of the box.

The repo does not redistribute scraped Suumo listings, and the preprocessor files
that the prediction service needs (`models/*-features.json`) are produced at
training time. This script generates a seeded synthetic dataset with the same
schema as `ScrapeData.clean_data()` output, fits the `FeaturePreprocessor`, and
writes the files `ModelLoader`/`PredictionService` expect (plus a `-noterms` pair of each,
trained without deposit / key money):

    models/challenger-model.joblib   (LightGBM)
    models/challenger-features.json
    models/passed-model.joblib       (Ridge)
    models/passed-features.json

These models are for demonstration and smoke tests only. Their predictions are
not market estimates. Train on real data (see data/README.md) for that.

Usage:
    cd api
    uv run python scripts/bootstrap_demo_models.py
"""
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, API_DIR)

from services.feature_preprocessor import FeaturePreprocessor, mask_optional_terms  # noqa: E402

MODELS_DIR = os.path.join(API_DIR, "models")
SEED = 42

# Rough relative rent level per ward (synthetic, illustrative only)
WARD_FACTOR = {
    "港区": 1.45, "渋谷区": 1.35, "千代田区": 1.40, "中央区": 1.30, "新宿区": 1.20,
    "文京区": 1.15, "目黒区": 1.20, "世田谷区": 1.05, "品川区": 1.10, "豊島区": 1.00,
    "中野区": 0.95, "杉並区": 0.95, "台東区": 1.00, "江東区": 0.95, "足立区": 0.75,
}
LAYOUT_SQM = {"1R": 20, "1K": 25, "1DK": 32, "1LDK": 40, "2LDK": 55, "3LDK": 70}
HOUSE_TYPES = ["賃貸マンション", "賃貸アパート"]


def make_synthetic_listings(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ku = rng.choice(list(WARD_FACTOR), n)
    layout = rng.choice(list(LAYOUT_SQM), n, p=[0.15, 0.30, 0.10, 0.15, 0.20, 0.10])
    sqr_m = np.array([LAYOUT_SQM[a] for a in layout]) * rng.uniform(0.8, 1.3, n)
    year_built = rng.integers(0, 45, n)
    floor = rng.integers(1, 20, n)
    walk = rng.integers(1, 21, n)
    house = rng.choice(HOUSE_TYPES, n, p=[0.7, 0.3])
    contract = rng.choice([0, 1], n, p=[0.9, 0.1])

    rent = (
        sqr_m * 0.28 * np.array([WARD_FACTOR[k] for k in ku])
        - year_built * 0.04
        + floor * 0.04
        - walk * 0.06
        + (house == "賃貸マンション") * 0.8
        - contract * 0.3
        + rng.normal(0, 0.6, n)
    ).clip(2.5, None)
    maintenance = (sqr_m * 0.004 + rng.normal(0, 0.1, n)).clip(0, None)
    return pd.DataFrame({
        "rent_price": rent.round(1),
        "sqr_m": sqr_m.round(1),
        "rei_price": rng.choice([0.0, 1.0], n, p=[0.6, 0.4]) * rent.round(1),
        "shikikin": rng.choice([0.0, 1.0, 2.0], n, p=[0.3, 0.4, 0.3]) * rent.round(1),
        "maintenence_price": maintenance.round(2),
        "year_built": year_built,
        "floor": floor,
        "nearest_eki_walk": walk,
        "contract_type": contract,
        "apartment_type": layout,
        "ku_name": ku,
        "house_type": house,
    })


def _fit_and_save(name: str, model, df: pd.DataFrame, mask_terms: bool = False,
                  drop_terms: bool = False) -> None:
    if drop_terms:  # dedicated variant that never sees deposit / key money
        df = df.drop(columns=["rei_price", "shikikin"])
    prep = FeaturePreprocessor()
    encoded, _ = prep.fit_transform(df)
    X = encoded.drop(columns="rent_price").astype(float)
    y = encoded["rent_price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    if mask_terms:  # NaN-aware model: also learn to predict without deposit / key money
        X_tr = mask_optional_terms(X_tr, frac=0.5, random_state=SEED)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"{name:<11} MAE={mean_absolute_error(y_te, pred):.2f}万円  R²={r2_score(y_te, pred):.3f}  (synthetic holdout)")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}-model.joblib"))
    prep.save(os.path.join(MODELS_DIR, f"{name}-features.json"))


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = make_synthetic_listings()
    _fit_and_save("challenger", lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                                  num_leaves=31, random_state=SEED, verbose=-1), df, mask_terms=True)
    _fit_and_save("passed", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), df)
    # Dedicated no-deposit / no-key-money variants, served when a request omits both
    _fit_and_save("challenger-noterms", lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                                          num_leaves=31, random_state=SEED, verbose=-1),
                  df, drop_terms=True)
    _fit_and_save("passed-noterms", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), df, drop_terms=True)
    print(f"Wrote demo models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
