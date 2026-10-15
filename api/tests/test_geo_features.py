"""Spatial-lag rent features: leakage safety, persistence, inference fallbacks."""
import numpy as np
import pandas as pd
import pytest

from services.feature_preprocessor import FeaturePreprocessor
from services.geo_features import ADDRESS_COLS as GEO_COLS, GeoRentFeatures, mask_geo_features, town_of, ward_of
from services.geocoding import normalize_address

TOWNS = {  # address -> (lat, lon); rent per m² rises towards the east
    "東京都渋谷区猿楽町１": (35.650, 139.700),
    "東京都渋谷区猿楽町２": (35.651, 139.701),
    "東京都渋谷区恵比寿１": (35.646, 139.710),
    "東京都新宿区西新宿１": (35.690, 139.690),
    "東京都新宿区西新宿２": (35.691, 139.691),
    "東京都江東区豊洲１": (35.650, 139.790),
    "東京都江東区豊洲２": (35.651, 139.791),
}


@pytest.fixture
def listings():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(70):  # 70 buildings, 3 units each
        addr = list(TOWNS)[i % len(TOWNS)]
        lon = TOWNS[addr][1]
        for _ in range(3):
            sqm = rng.uniform(20, 50)
            rows.append({"building_id": f"b{i}", "address": addr, "sqr_m": sqm,
                         "rent_price": sqm * (0.3 + (lon - 139.69) * 2) * rng.uniform(0.97, 1.03)})
    df = pd.DataFrame(rows)
    return df, df["building_id"]


@pytest.fixture
def coords():
    return {a: list(c) for a, c in TOWNS.items()}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("GEO_LIVE_GEOCODE", "0")


def test_town_and_ward():
    assert town_of("東京都杉並区西荻南４") == "杉並区西荻南"
    assert ward_of("東京都杉並区西荻南４") == "杉並区"
    assert normalize_address("渋谷区猿楽町1") == "東京都渋谷区猿楽町１"


def test_training_features_never_see_own_target(listings, coords):
    df, groups = listings
    geo = GeoRentFeatures(coords).fit(df, groups)
    before = geo.transform(df, groups, in_train=True)
    changed = df.copy()
    own = changed["building_id"] == "b0"
    changed.loc[own, "rent_price"] *= 10
    geo2 = GeoRentFeatures(coords).fit(changed, groups)
    after = geo2.transform(changed, groups, in_train=True)
    # b0's own features are untouched by its own (now 10x) rent...
    pd.testing.assert_frame_equal(before[own], after[own])
    # ...while a neighbour in the same town does see the change
    neighbour = (df["building_id"] == "b7")  # same address as b0
    assert not np.allclose(before.loc[neighbour, "te_town"], after.loc[neighbour, "te_town"])


def test_heldout_rows_use_all_training_groups(listings, coords):
    df, groups = listings
    geo = GeoRentFeatures(coords).fit(df, groups)
    train_feats = geo.transform(df, groups, in_train=True)
    eval_feats = geo.transform(df, groups, in_train=False)  # as if these buildings were new
    # without the exclusion each row's own building contributes -> features differ
    assert not np.allclose(train_feats["geo_knn_s"], eval_feats["geo_knn_s"])
    assert eval_feats[GEO_COLS].notna().all().all()


def test_features_track_location(listings, coords):
    df, groups = listings
    geo = GeoRentFeatures(coords).fit(df, groups)
    east = geo.transform_address("東京都江東区豊洲１")
    west = geo.transform_address("東京都渋谷区猿楽町１")
    assert east["geo_knn_s"] > west["geo_knn_s"]  # log rent/m² is higher in the east


def test_roundtrip(listings, coords, tmp_path):
    df, groups = listings
    geo = GeoRentFeatures(coords).fit(df, groups)
    geo.save(tmp_path / "x-geo.json")
    loaded = GeoRentFeatures.load(tmp_path / "x-geo.json")
    pd.testing.assert_frame_equal(geo.transform(df, groups, in_train=False),
                                  loaded.transform(df, groups, in_train=False))


def test_unknown_address_falls_back_to_town_then_ward_then_nan(listings, coords):
    df, groups = listings
    geo = GeoRentFeatures(coords).fit(df, groups)
    town = geo.transform_address("東京都渋谷区猿楽町９")      # unseen block, known town
    ward = geo.transform_address("東京都渋谷区南平台町１")    # unseen town, known ward
    none = geo.transform_address("東京都千代田区丸の内１")    # unseen ward
    blank = geo.transform_address(None)
    assert not np.isnan(town["geo_knn_s"]) and not np.isnan(ward["geo_knn_s"])
    assert np.isnan(none["geo_knn_s"]) and all(np.isnan(v) for v in blank.values())


def test_preprocessor_computes_geo_from_address(listings, coords, tmp_path):
    df, groups = listings
    geo = GeoRentFeatures(coords).fit(df, groups)
    feats = geo.transform(df, groups, in_train=True)

    prep = FeaturePreprocessor()
    prep.fit_transform(df.assign(ku_name="渋谷区", apartment_type="1K", house_type="賃貸マンション",
                                 year_built=5, floor=3, nearest_eki_walk=5))
    prep.add_geo(geo, feats)
    path = tmp_path / "toy-noterms-features.json"
    prep.save(path)
    assert (tmp_path / "toy-noterms-geo.json").exists()

    loaded = FeaturePreprocessor.load(path)
    base = {"sqr_m": 30, "year_built": 5, "floor": 3, "eki_walk": 5, "ku_name": "渋谷区"}
    with_addr = loaded.transform_single({**base, "address": "東京都江東区豊洲１"})
    no_addr = loaded.transform_single(base)
    assert list(with_addr.columns) == loaded.feature_columns_
    assert with_addr[GEO_COLS].notna().all().all()
    assert no_addr[GEO_COLS].isna().all().all()
    # models that cannot take NaN get the training medians instead
    imputed = loaded.transform_single(base, impute_missing=True)
    assert imputed[GEO_COLS].notna().all().all()
    assert imputed["te_town"].iloc[0] == pytest.approx(float(feats["te_town"].median()))


def test_mask_geo_features():
    X = pd.DataFrame({"sqr_m": np.arange(1000.0), **{c: 1.0 for c in GEO_COLS}})
    out = mask_geo_features(X, frac=0.3, random_state=0)
    assert out["sqr_m"].notna().all()
    assert 0.2 < out["te_town"].isna().mean() < 0.4
    assert (out["te_town"].isna() == out["geo_knn_s"].isna()).all()
    assert X["te_town"].notna().all()


# --------------------------------------------------------------------------- station / line

@pytest.fixture
def with_stations(listings):
    df, groups = listings
    station = {a: f"st{i % 4}" for i, a in enumerate(TOWNS)}
    df = df.assign(nearest_station=df["address"].map(station),
                   nearest_line=df["address"].map(lambda a: "lineA" if TOWNS[a][1] < 139.75 else "lineB"))
    return df, groups


def test_station_columns_only_when_data_has_them(listings, with_stations, coords):
    plain = GeoRentFeatures(coords).fit(*listings)
    assert "te_station" not in plain.transform(listings[0], listings[1], in_train=True).columns
    full = GeoRentFeatures(coords).fit(*with_stations)
    feats = full.transform(with_stations[0], with_stations[1], in_train=True)
    assert {"te_station", "te_line"} <= set(feats.columns) and feats[["te_station", "te_line"]].notna().all().all()


def test_station_encoding_never_sees_own_target(with_stations, coords):
    df, groups = with_stations
    before = GeoRentFeatures(coords).fit(df, groups).transform(df, groups, in_train=True)
    changed = df.copy()
    own = changed["building_id"] == "b0"
    changed.loc[own, "rent_price"] *= 10
    after = GeoRentFeatures(coords).fit(changed, groups).transform(changed, groups, in_train=True)
    pd.testing.assert_frame_equal(before[own], after[own])


def test_station_roundtrip_and_request(with_stations, coords, tmp_path):
    df, groups = with_stations
    geo = GeoRentFeatures(coords, ks=(3, 10), bandwidth_km=0.8).fit(df, groups)
    geo.save(tmp_path / "g.json")
    loaded = GeoRentFeatures.load(tmp_path / "g.json")
    assert loaded.ks == (3, 10) and loaded.bandwidth_km == 0.8
    pd.testing.assert_frame_equal(geo.transform(df, groups), loaded.transform(df, groups))
    both = loaded.transform_address("東京都江東区豊洲１", "st0", "lineB")
    assert not np.isnan(both["te_station"]) and not np.isnan(both["te_line"])
    # station given, no address: only the station columns are available
    only_station = loaded.transform_address(None, "st0", "lineA")
    assert not np.isnan(only_station["te_station"]) and np.isnan(only_station["geo_knn_s"])
    # address given, no station
    no_station = loaded.transform_address("東京都江東区豊洲１")
    assert np.isnan(no_station["te_station"]) and not np.isnan(no_station["geo_knn_s"])


def test_mask_station_and_address_columns_independently():
    from services.geo_features import ADDRESS_COLS, STATION_COLS
    X = pd.DataFrame({"sqr_m": np.arange(2000.0), **{c: 1.0 for c in ADDRESS_COLS + STATION_COLS}})
    out = mask_geo_features(X, frac=0.3, random_state=0)
    assert (out["te_station"].isna() == out["te_line"].isna()).all()
    assert not (out["te_station"].isna() == out["te_town"].isna()).all()


def test_drop_density_and_residual_target_roundtrip(tmp_path):
    from services.geo_features import DENSITY_COLS, GeoRentFeatures, oof_residuals
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "address": [f"東京都渋谷区猿楽町{i % 30}" for i in range(n)],
        "rent_price": rng.uniform(8, 20, n), "sqr_m": rng.uniform(20, 60, n),
        "year_built": rng.uniform(0, 30, n), "floor": rng.integers(1, 10, n),
        "nearest_eki_walk": rng.integers(1, 15, n), "maintenence_price": rng.uniform(0, 1, n),
        "apartment_type": "1K", "house_type": "mansion",
    })
    groups = pd.Series((np.arange(n) % 30).astype(str), index=df.index)
    coords = {a: [35.65 + 0.001 * i, 139.70 + 0.001 * i] for i, a in enumerate(df["address"].unique())}
    resid = oof_residuals(df, groups)
    assert resid.index.equals(df.index) and resid.notna().all()
    geo = GeoRentFeatures(coords, drop_density=True).fit(df, groups, target=resid)
    feats = geo.transform(df, groups, in_train=True)
    assert not set(DENSITY_COLS) & set(feats.columns)
    path = tmp_path / "geo.json"
    geo.save(path)
    again = GeoRentFeatures.load(path)
    assert list(again.columns_) == list(geo.columns_)
    assert set(again.transform_address("東京都渋谷区猿楽町3")) == set(geo.columns_)
