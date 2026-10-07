import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from services.data_split import (
    attach_groups,
    group_labels,
    grouped_train_test_split,
    groups_for,
)
from services.feature_preprocessor import FeaturePreprocessor
from services.model_evaluation_service import ModelEvaluationService
from utils.scrape_data import make_building_id


@pytest.fixture
def listings():
    rng = np.random.default_rng(0)
    n_buildings, per = 40, 6
    bid = np.repeat([f"b{i}" for i in range(n_buildings)], per)
    base = np.repeat(rng.normal(10, 3, n_buildings), per)
    df = pd.DataFrame({
        # building-level attribute: lets a flexible model memorise a building's rent
        "year_built": np.repeat(rng.uniform(0, 40, n_buildings), per),
        "building_id": bid,
        "address": bid,
        "scraped_at": "2026-01-01",
        "sqr_m": rng.uniform(15, 60, len(bid)),
        "rent_price": base + rng.normal(0, 0.3, len(bid)),
    })
    return df


class TestGroupLabels:
    def test_prefers_building_id_over_address(self, listings):
        listings["address"] = "same"
        assert group_labels(listings).nunique() == 40

    def test_falls_back_to_address(self, listings):
        assert group_labels(listings.drop(columns="building_id")).nunique() == 40

    def test_none_without_group_columns(self, listings):
        assert group_labels(listings[["sqr_m", "rent_price"]]) is None


class TestGroupedSplit:
    def test_no_group_in_both_sides(self, listings):
        X, y = listings[["sqr_m"]], listings["rent_price"]
        g = group_labels(listings)
        X_tr, X_te, y_tr, y_te, g_tr = grouped_train_test_split(X, y, g)
        assert set(g.loc[X_tr.index]).isdisjoint(set(g.loc[X_te.index]))
        assert len(X_tr) + len(X_te) == len(X)
        assert g_tr.index.equals(X_tr.index)

    def test_is_deterministic(self, listings):
        X, y, g = listings[["sqr_m"]], listings["rent_price"], group_labels(listings)
        a = grouped_train_test_split(X, y, g, random_state=1)[1].index
        b = grouped_train_test_split(X, y, g, random_state=1)[1].index
        assert a.equals(b)

    def test_falls_back_to_random_with_too_few_groups(self, listings):
        few = listings.assign(building_id=["a", "b"] * (len(listings) // 2))
        X, y = few[["sqr_m"]], few["rent_price"]
        *_, g_tr = grouped_train_test_split(X, y, group_labels(few))
        assert g_tr is None

    def test_attach_and_recover_groups(self, listings):
        X, y, g = listings[["sqr_m"]], listings["rent_price"], group_labels(listings)
        X_tr, *_ = grouped_train_test_split(X, y, g)
        lineage = attach_groups(listings.copy(), g)
        assert groups_for(lineage, X_tr).equals(g.loc[X_tr.index])
        assert groups_for(listings, X_tr) is None


class TestGroupedCV:
    def test_random_cv_leaks_building_siblings_grouped_cv_does_not(self, listings):
        X, y, g = listings[["sqr_m", "year_built"]], listings["rent_price"], group_labels(listings)
        svc = ModelEvaluationService(n_folds=4, random_state=0)
        model = DecisionTreeRegressor(random_state=0)
        random_rmse = svc.cross_validate_model(model, X, y)["metrics"]["rmse_mean"]
        grouped_rmse = svc.cross_validate_model(model, X, y, groups=g)["metrics"]["rmse_mean"]
        assert grouped_rmse > 2 * random_rmse


class TestMetadataNotFeatures:
    def test_preprocessor_drops_identifier_columns(self, listings):
        _, cols = FeaturePreprocessor().fit_transform(listings)
        assert not {"building_id", "address", "scraped_at"} & set(cols)


class TestBuildingId:
    def test_stable_and_normalised(self):
        assert make_building_id("Foo House", "東京都A") == make_building_id(" foo house ", "東京都A ")

    def test_distinguishes_buildings(self):
        assert make_building_id("Foo", "東京都A") != make_building_id("Bar", "東京都A")
