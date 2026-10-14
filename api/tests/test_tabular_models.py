"""TabPFN / TabNet bundles: loaded from models/, encoded like the other models, served via /torch."""
import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from services import artifact_bundle
from services.feature_preprocessor import FeaturePreprocessor
from services.tabular_models import load_deep_models
from services.torch_prediction_service import PyTorchPredictionService
from tests.test_ensemble_model import listings


@pytest.fixture
def bundle_dir(tmp_path):
    """A 'tabpfn' bundle whose model is a linear stand-in (the encoding and loading are what is tested)."""
    df = listings(n_buildings=20)
    encoded, _ = FeaturePreprocessor().fit_transform(df.drop(columns=["address", "building_id", "scraped_at"]))
    prep = FeaturePreprocessor()
    prep.fit_transform(df.drop(columns=["address", "building_id", "scraped_at"]))
    X, y = encoded.drop(columns="rent_price").astype(float), encoded["rent_price"]
    joblib.dump(LinearRegression().fit(X, y), tmp_path / "tabpfn-model.joblib")
    prep.save(tmp_path / "tabpfn-features.json")
    artifact_bundle.stamp(tmp_path, "tabpfn", run_id="run-1", dataset="d1")
    return tmp_path, prep, X, y


INPUT = {"sqr_m": 30.0, "rei_price": 5.0, "shikikin": 8.0, "maintenence_price": 0.5, "year_built": 10.0,
         "floor": 3.0, "eki_walk": 5.0, "ku_name": "渋谷区", "apartment_type": "1K", "house_type": "賃貸マンション"}


def test_bundle_loads_under_the_registered_name(bundle_dir):
    path, *_ = bundle_dir
    models = load_deep_models(path)
    assert list(models) == ["tokyo_rent_tabpfn"]
    assert models["tokyo_rent_tabpfn"].describe()["run_id"] == "run-1"


def test_prediction_uses_the_shared_encoding(bundle_dir):
    path, prep, X, y = bundle_dir
    model = load_deep_models(path)["tokyo_rent_tabpfn"]
    row = prep.transform_single(INPUT, impute_missing=True).astype(float)
    expected = float(LinearRegression().fit(X, y).predict(row.values)[0])
    assert model.predict_single(INPUT) == pytest.approx(expected)


def test_served_through_the_torch_service(bundle_dir):
    path, *_ = bundle_dir
    deep = load_deep_models(path)
    svc = PyTorchPredictionService(None, None, deep)
    out = svc.predict_with_torch_model(INPUT, "tokyo_rent_tabpfn")
    assert out["status"] == "success" and np.isfinite(out["prediction"])
    assert svc.get_available_torch_models()["models"]["tokyo_rent_tabpfn"]["ready_for_inference"]


def test_a_model_from_another_run_is_not_served(bundle_dir):
    path, *_ = bundle_dir
    joblib.dump(LinearRegression().fit(np.zeros((3, 2)), [1, 2, 3]), path / "tabpfn-model.joblib")  # swapped file
    assert load_deep_models(path) == {}


def test_no_bundle_means_no_models(tmp_path):
    assert load_deep_models(tmp_path) == {}
