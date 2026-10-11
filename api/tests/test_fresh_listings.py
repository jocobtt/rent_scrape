import numpy as np
import pandas as pd
import pytest

from services.artifact_bundle import stamp
from services.fresh_listings import new_units, payloads, score, summarise


def listings(n, start=0, rent_bump=0.0):
    return pd.DataFrame({
        "address": [f"東京都渋谷区猿楽町{i}" for i in range(start, start + n)],
        "floor": 3.0, "sqr_m": [20.0 + i for i in range(start, start + n)], "apartment_type": "1K", "house_type": "mansion",
        "year_built": 10.0, "rent_price": [10.0 + i + rent_bump for i in range(start, start + n)],
        "rei_price": 1.0, "shikikin": 1.0, "maintenence_price": 0.5, "nearest_eki_walk": 5.0,
        "nearest_station": "渋谷", "nearest_line": "JR山手線",
    })


def test_only_units_absent_from_the_training_data_count_as_new():
    trained = listings(10)
    fresh = pd.concat([listings(10, rent_bump=2.0), listings(5, start=10)], ignore_index=True)
    out = new_units(fresh, trained)
    assert len(out) == 5 and set(out["address"]) == set(listings(5, start=10)["address"])  # re-listed units are not new


def test_matching_needs_identifying_columns():
    with pytest.raises(ValueError):
        new_units(listings(3)[["floor", "sqr_m", "rent_price"]], listings(3)[["floor", "sqr_m", "rent_price"]])


def test_payloads_are_api_shaped_and_terms_are_optional():
    df = listings(2)
    df.loc[0, "nearest_station"] = np.nan
    with_terms, without = payloads(df, True), payloads(df, False)
    assert with_terms[0]["rei_price"] == 1.0 and "rei_price" not in without[0] and "shikikin" not in without[0]
    assert with_terms[0]["eki_walk"] == 5.0 and with_terms[0]["nearest_station"] is None  # NaN -> None
    assert with_terms[1]["address"].endswith("1")


class FakeService:
    """Variants route like the real service: no terms -> no_terms."""

    def __init__(self, bias):
        self.bias = bias

    def _predict(self, body, model):
        variant = "shared" if body.get("rei_price") is not None else "no_terms"
        return {"prediction": body["sqr_m"] * 0.5 + self.bias[model], "variant": variant}

    def predict_with_passed_model(self, body):
        return self._predict(body, "passed")

    def predict_with_challenger_model(self, body):
        return self._predict(body, "challenger")


def test_score_runs_every_model_and_variant_and_summarise_measures_error():
    df = listings(6)
    df["rent_price"] = df["sqr_m"] * 0.5
    scored = score(FakeService({"passed": 1.0, "challenger": 0.0}), df)
    assert len(scored) == 6 * 2 * 2 and set(scored["variant"]) == {"shared", "no_terms"}
    s = summarise(scored).set_index(["model_type", "variant"])
    assert s.loc[("challenger", "shared"), "rmse"] == pytest.approx(0.0)
    assert s.loc[("passed", "no_terms"), "rmse"] == pytest.approx(1.0) and s.loc[("passed", "shared"), "n"] == 6


def test_logging_rescrape_rows_makes_the_performance_check_usable(tmp_path):
    from services.data_capture_service import DataCaptureService
    cap = DataCaptureService(storage_dir=str(tmp_path))
    scored = score(FakeService({"passed": 2.0, "challenger": 2.0}), listings(4))
    for r in scored.itertuples():
        cap.log_prediction(r.model_type, r.input, r.prediction, actual_value=r.actual_value, variant=r.variant, source="rescrape")
    labelled = cap.get_predictions_with_actuals()
    assert len(labelled) == len(scored) and set(labelled["source"]) == {"rescrape"}


def test_bundle_records_the_training_dataset(tmp_path):
    (tmp_path / "v-model.joblib").write_bytes(b"m")
    (tmp_path / "v-features.json").write_text('{"feature_columns": []}')
    assert stamp(tmp_path, "v", "run1", dataset="076812c27246")["dataset"] == "076812c27246"
