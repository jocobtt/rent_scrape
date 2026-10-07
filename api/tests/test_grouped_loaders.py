"""Every model family's data loader must hold out whole buildings and keep
identifier columns out of the features."""
import numpy as np
import pandas as pd
import pytest

IDENTIFIERS = {"address", "building_id", "scraped_at"}


@pytest.fixture(autouse=True)
def no_feature_file_writes(monkeypatch):
    """`load_data()` saves models/<name>-features.json as a side effect; tests must
    not overwrite the committed demo models' feature files."""
    from services.feature_preprocessor import FeaturePreprocessor
    monkeypatch.setattr(FeaturePreprocessor, "save", lambda self, path: None)


@pytest.fixture
def raw_listings():
    rng = np.random.default_rng(1)
    n_b, per = 40, 5
    bid = np.repeat([f"b{i}" for i in range(n_b)], per)
    n = len(bid)
    sqr_m = rng.uniform(15, 60, n)
    return pd.DataFrame({
        # plausible rent (0.3-0.5 万円/㎡) so the cleaning step keeps every row
        "rent_price": sqr_m * rng.uniform(0.3, 0.5, n),
        "address": [f"addr-{b}" for b in bid],
        "building_id": bid,
        "scraped_at": "2026-01-01",
        "sqr_m": sqr_m,
        "rei_price": rng.uniform(0, 20, n),
        "shikikin": rng.uniform(0, 20, n),
        "maintenence_price": rng.uniform(0, 2, n),
        "contract_type": 0,
        "year_built": rng.integers(0, 40, n),
        "floor": rng.integers(1, 15, n),
        "nearest_eki_walk": rng.integers(1, 20, n),
        "apartment_type": rng.choice(["1K", "2LDK"], n),
        "ku_name": rng.choice(["渋谷区", "新宿区"], n),
        "house_type": rng.choice(["賃貸マンション", "賃貸アパート"], n),
    }), bid


@pytest.fixture
def dataset_csv(monkeypatch, tmp_path, raw_listings):
    """Point every loader at a synthetic CSV via DATASET_PATH."""
    path = tmp_path / "data.csv"
    raw_listings[0].to_csv(path, index=False)
    monkeypatch.setenv("DATASET_PATH", str(path))
    return path


def _assert_grouped(X_tr, X_te, bid):
    assert not IDENTIFIERS & set(X_tr.columns)
    assert not IDENTIFIERS & set(X_te.columns)
    assert set(bid[X_tr.index]).isdisjoint(set(bid[X_te.index]))
    assert len(X_te) > 0


def test_tabular_pretrained_loader(dataset_csv, raw_listings):
    from models import tabular_pretrained_model as m
    X_tr, X_te, y_tr, y_te, lineage = m.load_data()
    _assert_grouped(X_tr, X_te, raw_listings[1])
    assert "_group" in lineage.columns


def test_challenger_and_regression_loaders(dataset_csv, raw_listings):
    from models.challenger_model import load_data as load_lgbm
    from models.reg_model import load_data as load_reg
    for load in (load_lgbm, load_reg):
        X_tr, X_te, *_ = load()
        _assert_grouped(X_tr, X_te, raw_listings[1])


def test_ensemble_loader(dataset_csv, raw_listings):
    from models.ensemble_model import load_data_for_ensemble
    X_tr, X_te, y_tr, y_te = load_data_for_ensemble()
    _assert_grouped(X_tr, X_te, raw_listings[1])


@pytest.mark.parametrize("module,cls", [
    ("models.torch_model", "TorchRentPredictor"),
    ("models.transformer_model", "TransformerRentPredictor"),
])
def test_neural_predictor_loaders(dataset_csv, raw_listings, module, cls):
    import importlib
    mod = importlib.import_module(module)
    from services.data_split import grouped_train_val_test_split
    predictor = getattr(mod, cls)()
    X, y = predictor.load_and_prepare_data()
    X_tr, X_val, X_te, *_ = grouped_train_val_test_split(X, y, predictor.groups_)
    bid = raw_listings[1]
    _assert_grouped(X_tr, X_te, bid)
    assert set(bid[X_tr.index]).isdisjoint(set(bid[X_val.index]))
