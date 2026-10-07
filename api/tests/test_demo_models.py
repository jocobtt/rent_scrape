"""The committed demo models must stay consistent with their feature files, or a
fresh clone cannot serve /predict (this broke once when tests overwrote the files)."""
from pathlib import Path

import joblib
import pytest

from services.feature_preprocessor import FeaturePreprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SAMPLE = {
    "sqr_m": 30, "rei_price": 2, "shikikin": 2, "maintenence_price": 0.1,
    "year_built": 10, "floor": 3, "eki_walk": 8, "ku_name": "渋谷区",
}


@pytest.mark.parametrize("name", ["challenger", "passed"])
def test_feature_file_matches_model(name):
    prep = FeaturePreprocessor.load(MODELS_DIR / f"{name}-features.json")
    model = joblib.load(MODELS_DIR / f"{name}-model.joblib")
    X = prep.transform_single(SAMPLE).astype(float)
    assert X.shape[1] == model.n_features_in_
    assert 1 < float(model.predict(X)[0]) < 100  # plausible rent in 万円


NO_TERMS = {k: v for k, v in SAMPLE.items() if k not in ("rei_price", "shikikin")}


def test_challenger_predicts_without_terms():
    """LightGBM gets NaN for missing terms; it was trained with masked rows so it copes."""
    prep = FeaturePreprocessor.load(MODELS_DIR / "challenger-features.json")
    model = joblib.load(MODELS_DIR / "challenger-model.joblib")
    X = prep.transform_single(NO_TERMS).astype(float)
    assert X[["rei_price", "shikikin"]].isna().all().all()
    assert 1 < float(model.predict(X)[0]) < 100


def test_passed_model_imputes_missing_terms():
    """The linear model can't take NaN: missing terms become the training median."""
    prep = FeaturePreprocessor.load(MODELS_DIR / "passed-features.json")
    model = joblib.load(MODELS_DIR / "passed-model.joblib")
    X = prep.transform_single(NO_TERMS, impute_missing=True).astype(float)
    assert not X.isna().any().any()
    assert X["shikikin"].iloc[0] == prep.fill_values_["shikikin"]
    assert 1 < float(model.predict(X)[0]) < 100


def test_prediction_service_without_terms():
    from services.prediction_service import PredictionService
    svc = PredictionService(joblib.load(MODELS_DIR / "passed-model.joblib"),
                            joblib.load(MODELS_DIR / "challenger-model.joblib"))
    payload = {**NO_TERMS, "rei_price": None, "shikikin": None}
    assert svc.predict_with_passed_model(payload)["status"] == "success"
    assert svc.predict_with_challenger_model(payload)["status"] == "success"


def test_mask_optional_terms():
    import numpy as np
    import pandas as pd
    from services.feature_preprocessor import mask_optional_terms
    X = pd.DataFrame({"sqr_m": np.arange(1000.0), "rei_price": 1.0, "shikikin": 2.0})
    out = mask_optional_terms(X, frac=0.5, random_state=0)
    assert out["sqr_m"].notna().all()
    assert (out["rei_price"].isna() == out["shikikin"].isna()).all()  # masked together
    assert 0.4 < out["shikikin"].isna().mean() < 0.6
    assert X["shikikin"].notna().all()  # input untouched


def _service():
    from services.prediction_service import PredictionService
    load = lambda n: joblib.load(MODELS_DIR / f"{n}-model.joblib")  # noqa: E731
    return PredictionService(load("passed"), load("challenger"),
                             passed_noterms_model=load("passed-noterms"),
                             challenger_noterms_model=load("challenger-noterms"))


@pytest.mark.parametrize("name", ["challenger-noterms", "passed-noterms"])
def test_noterms_feature_file_matches_model(name):
    prep = FeaturePreprocessor.load(MODELS_DIR / f"{name}-features.json")
    model = joblib.load(MODELS_DIR / f"{name}-model.joblib")
    assert "shikikin" not in prep.feature_columns_ and "rei_price" not in prep.feature_columns_
    X = prep.transform_single(NO_TERMS).astype(float)
    assert X.shape[1] == model.n_features_in_


def test_requests_without_terms_use_dedicated_models():
    svc = _service()
    payload = {**NO_TERMS, "rei_price": None, "shikikin": None}
    assert svc.predict_with_challenger_model(payload)["variant"] == "no_terms"
    assert svc.predict_with_passed_model(payload)["variant"] == "no_terms"


def test_requests_with_terms_use_shared_models():
    svc = _service()
    assert svc.predict_with_challenger_model(SAMPLE)["variant"] == "shared"
    assert svc.predict_with_passed_model(SAMPLE)["variant"] == "shared"


def test_only_one_term_given_uses_shared_model():
    """A single missing term isn't the no-terms case; the shared, NaN-trained model handles it."""
    svc = _service()
    out = svc.predict_with_challenger_model({**SAMPLE, "rei_price": None})
    assert out["variant"] == "shared" and out["status"] == "success"
