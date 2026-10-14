"""Ensembles combine base-model predictions; training data is grouped out-of-fold."""
import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import ensemble_model
from models.ensemble_model import METHODS, EnsembleModel


@pytest.fixture
def preds():
    """'challenger' is accurate on cheap flats, 'passed' on expensive ones."""
    rng = np.random.default_rng(0)
    y = rng.uniform(5, 40, 3000)
    noise = rng.normal(0, 1, (2, len(y)))
    cheap = y < 20
    return (
        {"challenger": y + noise[0] * np.where(cheap, 0.5, 4.0),
         "passed": y + noise[1] * np.where(cheap, 4.0, 0.5)},
        y,
    )


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@pytest.mark.parametrize("method", METHODS)
def test_every_method_beats_a_single_poor_model(preds, method):
    p, y = preds
    ens = EnsembleModel(method).fit(p, y)
    assert rmse(ens.combine(p), y) < min(rmse(p["challenger"], y), rmse(p["passed"], y)) + 0.05


def test_weights_are_a_blend(preds):
    p, y = preds
    w = EnsembleModel("weighted_average").fit(p, y).weights
    assert sum(w.values()) == pytest.approx(1.0) and all(v >= 0 for v in w.values())


def test_voting_is_the_mean(preds):
    p, y = preds
    out = EnsembleModel("voting").fit(p, y).combine(p)
    assert np.allclose(out, (p["challenger"] + p["passed"]) / 2)


def test_dynamic_weights_follow_the_rent_band(preds):
    p, y = preds
    ens = EnsembleModel("dynamic_weights").fit(p, y)
    cheap_band, _, dear_band = ens.band_weights
    # base order is (challenger, passed): trust the first on cheap flats, the second on expensive ones
    assert cheap_band[0] > cheap_band[1] and dear_band[1] > dear_band[0]
    # being band-aware beats the single global blend here
    assert rmse(ens.combine(p), y) < rmse(EnsembleModel("weighted_average").fit(p, y).combine(p), y)


def test_save_and_load_round_trip(preds, tmp_path):
    p, y = preds
    for method in METHODS:
        ens = EnsembleModel(method).fit(p, y)
        ens.meta = {"eval_rmse": 1.5}
        ens.save(tmp_path / f"{method}.joblib")
        again = EnsembleModel.load(tmp_path / f"{method}.joblib")
        assert np.allclose(again.combine(p), ens.combine(p))
        assert again.describe()["eval_rmse"] == 1.5


def test_untrained_or_incomplete_inputs_are_refused(preds):
    p, y = preds
    with pytest.raises(ValueError, match="trained"):
        EnsembleModel("voting").combine(p)
    with pytest.raises(ValueError, match="Missing base predictions"):
        EnsembleModel("voting").fit(p, y).combine({"challenger": p["challenger"]})
    with pytest.raises(ValueError, match="Unknown"):
        EnsembleModel("magic")


def test_service_combines_the_base_predictions(preds):
    from services.ensemble_prediction_service import EnsemblePredictionService
    p, y = preds
    svc = EnsemblePredictionService.__new__(EnsemblePredictionService)
    svc.ensemble_models = {"voting": EnsembleModel("voting").fit(p, y)}
    svc._base_predictions = lambda data: {"challenger": 10.0, "passed": 14.0}
    out = svc.predict_with_ensemble({}, "voting")
    assert out["prediction"] == pytest.approx(12.0) and out["model_type"] == "ensemble_voting"
    with pytest.raises(ValueError, match="not available"):
        svc.predict_with_ensemble({}, "stacking")


def listings(n_buildings=60, per=6, seed=3):
    rng = np.random.default_rng(seed)
    bid = np.repeat([f"b{i}" for i in range(n_buildings)], per)
    n = len(bid)
    sqr = rng.uniform(15, 60, n)
    return pd.DataFrame({
        "rent_price": sqr * rng.uniform(0.3, 0.5, n), "address": [f"a-{b}" for b in bid], "building_id": bid,
        "scraped_at": "2026-01-01", "sqr_m": sqr, "rei_price": rng.uniform(0, 20, n),
        "shikikin": rng.uniform(0, 20, n), "maintenence_price": rng.uniform(0, 2, n),
        "year_built": rng.integers(0, 40, n), "floor": rng.integers(1, 15, n),
        "nearest_eki_walk": rng.integers(1, 20, n), "apartment_type": rng.choice(["1K", "2LDK"], n),
        "ku_name": rng.choice(["渋谷区", "新宿区"], n), "house_type": rng.choice(["賃貸マンション", "賃貸アパート"], n),
    })


def test_out_of_fold_predictions_never_see_the_row_they_predict(tmp_path):
    """Base models that memorise their training rows still can't be exact out-of-fold."""
    df = listings()
    X, y, groups = ensemble_model._encoded_frame(df)
    joblib.dump(LGBMRegressor(n_estimators=300, min_child_samples=1, num_leaves=64, verbose=-1),
                tmp_path / "challenger-model.joblib")
    joblib.dump(make_pipeline(StandardScaler(), Ridge()), tmp_path / "passed-model.joblib")

    oof, y_out, groups_out = ensemble_model.out_of_fold_predictions(df, n_splits=5, models_dir=tmp_path)

    assert set(oof) == {"challenger", "passed"} and len(y_out) == len(df)
    assert all(np.isfinite(v).all() for v in oof.values())
    # a memorising model fitted on the whole set would be near-exact; held-out buildings are not
    in_sample = LGBMRegressor(n_estimators=300, min_child_samples=1, num_leaves=64, verbose=-1).fit(X, y)
    assert rmse(oof["challenger"], y_out) > 3 * rmse(in_sample.predict(X), y_out)
