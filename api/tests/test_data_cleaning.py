import json

import pandas as pd
import pytest

from services import data_cleaning as dc
from utils.scrape_data import convert_money_to_man


def _frame(**over):
    base = {
        "rent_price": [10.0, 10.0, 12.0, 20.0, 30.0],
        "sqr_m": [30.0, 30.0, 30.0, 40.0, 1000.0],
        "maintenence_price": [10000.0, 10000.0, 0.8, 5000.0, 0.0],
        "scraped_at": ["2026-10-07", "2026-10-08", "2026-10-07", "2026-10-07", "2026-10-07"],
    }
    base.update(over)
    return pd.DataFrame(base)


class TestCleanListings:
    def test_duplicates_ignore_scrape_date(self):
        out, rep = dc.clean_listings(_frame())
        assert rep["duplicates_dropped"] == 1

    def test_yen_maintenance_converted_and_man_left_alone(self):
        out, _ = dc.clean_listings(_frame())
        assert sorted(out["maintenence_price"]) == pytest.approx([0.5, 0.8, 1.0])

    def test_implausible_rent_per_sqm_dropped(self):
        out, rep = dc.clean_listings(_frame())
        assert rep["implausible_rent_per_sqm_dropped"] == 1  # 30万円 for 1000㎡
        assert (out["sqr_m"] < 1000).all()

    def test_nonpositive_rows_dropped(self):
        out, rep = dc.clean_listings(_frame(sqr_m=[0.0, 30.0, 30.0, 40.0, 40.0], rent_price=[10, 10, 12, 20, 0]))
        assert rep["nonpositive_dropped"] == 2


class TestResolveDataset:
    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        f = tmp_path / "a.csv"; f.write_text("x\n1\n")
        monkeypatch.setenv("DATASET_PATH", str(tmp_path / "other.csv"))
        assert dc.resolve_dataset_path(str(f)) == f

    def test_env_var(self, tmp_path, monkeypatch):
        f = tmp_path / "a.csv"; f.write_text("x\n1\n")
        monkeypatch.setenv("DATASET_PATH", str(f))
        assert dc.resolve_dataset_path() == f

    def test_missing_explicit_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dc.resolve_dataset_path(str(tmp_path / "nope.csv"))

    def test_latest_version_chosen_by_manifest_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DATASET_PATH", raising=False)
        for name, ts in (("old", "2026-01-01T00:00:00"), ("new", "2026-06-01T00:00:00")):
            d = tmp_path / name; d.mkdir()
            (d / "data.csv").write_text("x\n1\n")
            (d / "manifest.json").write_text(json.dumps({"created_at": ts, "kind": "scrape"}))
        monkeypatch.setattr(dc, "DATASET_VERSIONS_DIR", tmp_path)
        assert dc.resolve_dataset_path().parent.name == "new"

    def test_lineage_snapshots_and_legacy_manifests_are_ignored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DATASET_PATH", raising=False)
        for name, manifest in (
            ("scrape", {"created_at": "2026-01-01T00:00:00", "kind": "scrape"}),
            ("snapshot", {"created_at": "2026-09-01T00:00:00", "kind": "snapshot"}),
            ("legacy", {"created_at": "2026-09-02T00:00:00"}),
        ):
            d = tmp_path / name; d.mkdir()
            (d / "data.csv").write_text("x\n1\n")
            (d / "manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setattr(dc, "DATASET_VERSIONS_DIR", tmp_path)
        assert dc.resolve_dataset_path().parent.name == "scrape"

    def test_no_dataset_gives_helpful_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DATASET_PATH", raising=False)
        monkeypatch.setattr(dc, "DATASET_VERSIONS_DIR", tmp_path / "empty")
        with pytest.raises(FileNotFoundError, match="scrape_dataset.py"):
            dc.resolve_dataset_path()


@pytest.mark.parametrize("text,expected", [
    ("28万円", 28.0), ("12.5万円", 12.5), ("10000円", 1.0), ("5,000円", 0.5), ("-", 0.0), ("", 0.0),
])
def test_convert_money_to_man(text, expected):
    assert convert_money_to_man(text) == pytest.approx(expected)
