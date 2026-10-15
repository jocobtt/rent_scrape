import pandas as pd
import pytest

from services.data_capture_service import DataCaptureService


@pytest.fixture
def cap(tmp_path):
    return DataCaptureService(storage_dir=str(tmp_path), buffer_size=1000)


def log(cap, n, **kw):
    return [cap.log_prediction("passed", {"sqr_m": 30.0 + i, "eki_walk": 5.0}, 10.0 + i, **kw) for i in range(n)]


def test_each_prediction_gets_a_unique_id_and_variant(cap):
    ids = log(cap, 5, variant="no_terms")
    assert len(set(ids)) == 5
    df = cap.get_recent_predictions()
    assert set(df["prediction_id"]) == set(ids) and set(df["variant"]) == {"no_terms"}


def test_labelling_keeps_every_other_row(cap):
    ids = log(cap, 30)
    assert cap.update_actual_value(ids[7], 99.0)
    df = cap.get_recent_predictions()
    assert len(df) == 30
    assert df.loc[df["prediction_id"] == ids[7], "actual_value"].item() == 99.0
    assert df["actual_value"].notna().sum() == 1
    for i in ids:  # labelling all of them, one by one, works too
        assert cap.update_actual_value(i, 1.0)
    assert len(cap.get_predictions_with_actuals()) == 30


def test_unknown_id_is_reported_and_changes_nothing(cap):
    log(cap, 3)
    assert not cap.update_actual_value("missing", 1.0)
    assert len(cap.get_recent_predictions()) == 3 and cap.get_recent_predictions()["actual_value"].isna().all()


def test_logs_written_before_ids_existed_can_still_be_labelled_by_timestamp(cap, tmp_path):
    legacy = pd.DataFrame({"timestamp": ["2026-10-01T10:00:00.000001", "2026-10-01T10:00:01.000001"],
                           "model_type": "passed", "sqr_m": 30.0, "prediction": [10.0, 11.0], "actual_value": None})
    legacy.to_csv(tmp_path / "predictions_20261001.csv", index=False)
    assert cap.update_actual_value("2026-10-01T10:00:01.000001", 12.0)
    df = cap.get_recent_predictions()
    assert len(df) == 2 and df["actual_value"].notna().sum() == 1


def test_labelling_only_touches_the_file_that_holds_the_prediction(cap, tmp_path):
    pd.DataFrame({"prediction_id": ["old1", "old2"], "timestamp": ["2026-10-01T10:00:00", "2026-10-01T11:00:00"],
                  "model_type": "passed", "prediction": [1.0, 2.0], "actual_value": None}
                 ).to_csv(tmp_path / "predictions_20261001.csv", index=False)
    new = log(cap, 2)
    assert cap.update_actual_value("old2", 5.0)
    assert len(pd.read_csv(tmp_path / "predictions_20261001.csv")) == 2
    assert len(cap.get_recent_predictions()) == 4 and cap.update_actual_value(new[0], 3.0)
    assert not list(tmp_path.glob("*.tmp"))
