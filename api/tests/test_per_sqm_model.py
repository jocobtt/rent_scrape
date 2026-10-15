import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from services.per_sqm_model import PerSqmRegressor


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"sqr_m": rng.uniform(15, 80, n), "age": rng.uniform(0, 40, n)})
    y = X["sqr_m"] * (3.0 - 0.02 * X["age"]) + rng.normal(0, 0.3, n)  # rent in 万円
    return X, y


def _model():
    import lightgbm as lgb
    return PerSqmRegressor(lgb.LGBMRegressor(n_estimators=80, verbose=-1, random_state=0))


def test_predicts_rent_not_rent_per_sqm(data):
    X, y = data
    m = _model().fit(X, y)
    pred = m.predict(X)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 2.0
    np.testing.assert_allclose(pred, m.predict_per_sqm(X) * X["sqr_m"])


def test_eval_set_is_converted_and_ndarray_predict_works(data):
    import lightgbm as lgb
    X, y = data
    m = PerSqmRegressor(lgb.LGBMRegressor(n_estimators=500, verbose=-1, random_state=0)).fit(
        X.iloc[:300], y.iloc[:300], eval_set=[(X.iloc[300:], y.iloc[300:])],
        eval_metric="l1", callbacks=[lgb.early_stopping(10, verbose=False)])
    assert m.best_iteration_ < 500
    np.testing.assert_allclose(m.predict(X.iloc[300:].values), m.predict(X.iloc[300:]))


def test_clone_and_pickle(data, tmp_path):
    import joblib
    X, y = data
    fitted = clone(_model()).fit(X, y)
    joblib.dump(fitted, tmp_path / "m.joblib")
    np.testing.assert_allclose(joblib.load(tmp_path / "m.joblib").predict(X), fitted.predict(X))


def test_shap_attributions_sum_to_rent(data):
    from services.explainability_service import ExplainabilityService
    X, y = data
    m = _model().fit(X, y)
    row = X.iloc[[0]]
    explainer, _ = ExplainabilityService()._get_shap_explainer_and_values(m, X.iloc[:50])
    sv = explainer(row)
    assert float(sv.base_values.sum() + sv.values.sum()) == pytest.approx(float(m.predict(row)[0]), rel=1e-3)
