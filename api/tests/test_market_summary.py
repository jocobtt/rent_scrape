"""/data/market: per-ward / layout / station medians for the prediction form."""
import pandas as pd
import pytest

from services import listings_export, market_summary
from tests.test_listings_export import write_dataset  # noqa: F401  (fixtures below reuse it)
from tests.test_listings_export import api, store  # noqa: F401


def frame(n=6, ward="中野区", layout="1K", rent=10.0, sqr=20.0, station="中野", line="JR中央線", walk=4):
    return pd.DataFrame({
        "rent_price": [rent + i for i in range(n)], "sqr_m": [sqr] * n, "maintenence_price": [0.5] * n,
        "year_built": [10] * n, "floor": [3] * n, "nearest_eki_walk": [walk] * n,
        "apartment_type": [layout] * n, "ku_name": [ward] * n,
        "nearest_station": [station] * n, "nearest_line": [line] * n,
    })


def test_medians_per_ward_and_layout():
    out = market_summary.summarize(frame())
    cell = out["wards"]["中野区"]["layouts"]["1K"]
    assert cell["count"] == 6 and cell["rent"] == 12.5 and cell["sqr_m"] == 20.0
    assert cell["rent_per_sqm"] == pytest.approx(0.625)
    assert out["tokyo"]["layouts"]["1K"]["count"] == 6


def test_small_cells_are_left_out():
    df = pd.concat([frame(), frame(n=2, ward="港区", station="新橋", line="JR山手線")])
    out = market_summary.summarize(df)
    assert "港区" not in out["wards"]
    assert [s["name"] for s in out["stations"]] == ["中野"]


def test_mis_parsed_layouts_are_dropped():
    out = market_summary.summarize(pd.concat([frame(), frame(layout="16K")]))
    assert list(out["tokyo"]["layouts"]) == ["1K"]


def test_stations_carry_ward_and_walk_time():
    st = market_summary.summarize(frame())["stations"][0]
    assert st == {"name": "中野", "line": "JR中央線", "ward": "中野区", "count": 6, "eki_walk": 4.0}


def test_route_serves_summary_with_etag(store, api):  # noqa: F811
    write_dataset(store, "abc", "scrape", "2026-10-10T00:00:00+00:00")
    market_summary._summary_for.cache_clear()
    # one row is below MIN_CELL, so the summary is empty but well-formed
    res = api.get("/data/market")
    assert res.status_code == 200 and res.json()["dataset"]["hash"] == "abc"
    assert api.get("/data/market", headers={"If-None-Match": res.headers["etag"]}).status_code == 304


def test_route_404_without_a_scrape(store, api):  # noqa: F811
    assert api.get("/data/market").status_code == 404
