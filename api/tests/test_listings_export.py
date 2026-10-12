"""/data/listings: the newest raw scrape, in the columns the dashboard parses."""
import csv
import io
import json

import pytest

from services import listings_export

RAW_HEADER = ["building_id", "rent_price", "address", "sqr_m", "rei_price", "shikikin", "house_type", "floor",
              "ku_name", "apartment_type", "maintenence_price", "scraped_at", "year_built", "nearest_eki_walk",
              "nearest_station", "nearest_line", "n_stations"]


def write_dataset(store, short, kind, created_at, rent=12.5, ward="中野区"):
    d = store / short
    d.mkdir(parents=True)
    row = dict(zip(RAW_HEADER, ["b1", rent, "東京都中野区中野２", 25.1, 0.0, 13.0, "賃貸マンション", 8, ward, "2K",
                                1.5, "2026-10-09", 3, 5, "中野", "JR中央線", 2]))
    with open(d / "data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_HEADER)
        w.writeheader()
        w.writerow(row)
    (d / "manifest.json").write_text(json.dumps(
        {"hash_short": short, "kind": kind, "created_at": created_at, "n_rows": 1}))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(listings_export, "DATASET_STORE_DIR", tmp_path)
    listings_export._csv_for.cache_clear()
    return tmp_path


@pytest.fixture
def api(client, monkeypatch):
    from deps import limiter
    monkeypatch.setattr(limiter, "enabled", False)
    return client


def test_picks_the_newest_scrape_and_ignores_training_snapshots(store):
    write_dataset(store, "old", "scrape", "2026-10-01T00:00:00+00:00")
    write_dataset(store, "new", "scrape", "2026-10-10T00:00:00+00:00")
    write_dataset(store, "snap", "snapshot", "2026-10-12T00:00:00+00:00")  # newer, but processed
    assert listings_export.latest_scrape()["hash_short"] == "new"


def test_no_scrape_is_none(store):
    assert listings_export.latest_scrape() is None


def test_csv_has_dashboard_columns_and_conventions(store):
    write_dataset(store, "abc", "scrape", "2026-10-10T00:00:00+00:00")
    body, etag = listings_export.dashboard_csv(listings_export.latest_scrape())
    rows = list(csv.DictReader(io.StringIO(body.decode())))
    assert list(rows[0]) == listings_export.DASHBOARD_COLUMNS
    assert rows[0]["ku_name"] == "中野"      # the map joins on the name without 区
    assert rows[0]["eki_walk"] == "5"        # renamed from nearest_eki_walk
    assert rows[0]["rent_price"] == "12.5" and etag == '"abc"'


def test_route_serves_csv_with_etag_and_304(store, api):
    write_dataset(store, "abc", "scrape", "2026-10-10T00:00:00+00:00")
    r = api.get("/data/listings")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.text.splitlines()[0].startswith("rent_price,address")
    again = api.get("/data/listings", headers={"If-None-Match": r.headers["etag"]})
    assert again.status_code == 304 and again.content == b""


def test_route_follows_a_new_scrape(store, api):
    write_dataset(store, "one", "scrape", "2026-10-01T00:00:00+00:00", rent=10.0)
    assert api.get("/data/listings").headers["etag"] == '"one"'
    write_dataset(store, "two", "scrape", "2026-10-02T00:00:00+00:00", rent=20.0)
    r = api.get("/data/listings")
    assert r.headers["etag"] == '"two"' and "\n20.0," in r.text


def test_route_404_before_any_scrape(store, api):
    assert api.get("/data/listings").status_code == 404
