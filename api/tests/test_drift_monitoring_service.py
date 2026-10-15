"""Drift detection against real Evidently runs on small synthetic frames."""
import numpy as np
import pandas as pd
import pytest

from services.drift_monitoring_service import DriftMonitoringService


def frame(n, seed, shift=0.0, terms=True):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "sqr_m": rng.normal(30, 8, n) + shift * 20,
        "rei_price": rng.choice([0.0, 1.0, 2.0], n), "shikikin": rng.choice([0.0, 1.0, 2.0], n),
        "maintenence_price": rng.normal(0.5, 0.2, n), "year_built": rng.uniform(0, 40, n) + shift * 30,
        "floor": rng.integers(1, 10, n).astype(float), "eki_walk": rng.integers(1, 15, n).astype(float),
    })
    d["prediction"] = d["sqr_m"] * 0.4
    if not terms:
        d[["rei_price", "shikikin"]] = np.nan
    return d


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    return DriftMonitoringService(frame(2000, 0), reports_dir=str(tmp_path_factory.mktemp("reports")))


def test_no_drift_when_the_distribution_is_unchanged(svc):
    _, summary = svc.generate_data_drift_report(frame(500, 1), save_html=False, save_json=False)
    assert summary["drifted_features"] == 0 and not summary["dataset_drift"]
    assert summary["drifted_columns"] == []


def test_shifted_columns_are_flagged_and_unshifted_discrete_ones_are_not(svc):
    _, summary = svc.generate_data_drift_report(frame(500, 2, shift=1.0), save_html=False, save_json=False)
    flagged = set(summary["drifted_columns"])
    assert {"sqr_m", "year_built"} <= flagged                       # the two columns that moved
    assert not flagged & {"floor", "eki_walk", "rei_price", "shikikin"}
    assert summary["drift_share"] == pytest.approx(len(flagged) / 7)  # a borderline continuous column may join


def test_dataset_drift_follows_the_requested_share(svc):
    cur = frame(500, 3, shift=1.0)
    share = svc.generate_data_drift_report(cur, False, False)[1]["drift_share"]
    assert 0.2 < share < 0.5
    assert not svc.generate_data_drift_report(cur, False, False, drift_share_threshold=0.5)[1]["dataset_drift"]
    assert svc.generate_data_drift_report(cur, False, False, drift_share_threshold=0.2)[1]["dataset_drift"]


def test_requests_without_terms_do_not_crash_or_look_like_drift(svc):
    """Logged requests that all omitted deposit / key money give all-NaN columns, which Evidently rejects."""
    _, summary = svc.generate_data_drift_report(frame(500, 4, terms=False), save_html=False, save_json=False)
    assert summary["total_features"] == 5 and summary["drifted_features"] == 0
    _, passed = svc.run_drift_tests(frame(500, 4, terms=False), drift_share_threshold=0.3)
    assert passed  # missing optional terms are not a data-quality failure


def test_drift_tests_fail_on_drift_and_on_missing_required_values(svc):
    assert svc.run_drift_tests(frame(500, 5), drift_share_threshold=0.3)[1]
    assert not svc.run_drift_tests(frame(500, 6, shift=1.0), drift_share_threshold=0.1)[1]
    broken = frame(500, 7)
    broken.loc[:20, "sqr_m"] = np.nan
    assert not svc.run_drift_tests(broken, drift_share_threshold=0.3)[1]


def test_single_column_drift(svc):
    cur = frame(500, 8, shift=1.0)
    assert svc.check_column_drift(cur, "sqr_m")["drift_detected"]
    assert not svc.check_column_drift(cur, "floor")["drift_detected"]
    assert svc.check_column_drift(cur, "sqr_m")["drift_score"] > svc.check_column_drift(cur, "sqr_m")["threshold"]


def test_small_windows_are_reported_but_never_called_drift(svc):
    _, summary = svc.generate_data_drift_report(frame(60, 9, shift=1.0), save_html=False, save_json=False)
    assert summary["current_rows"] == 60 and not summary["reliable"] and not summary["dataset_drift"]
    assert summary["drifted_columns"]  # the per-column numbers are still there for a human
    _, big = svc.generate_data_drift_report(frame(400, 9, shift=1.0), save_html=False, save_json=False, drift_share_threshold=0.2)
    assert big["reliable"] and big["dataset_drift"]
