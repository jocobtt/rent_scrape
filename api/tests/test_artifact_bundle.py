import json

import pytest

from services.artifact_bundle import BundleMismatch, problems, stamp, verify


@pytest.fixture
def models_dir(tmp_path):
    (tmp_path / "v-model.joblib").write_bytes(b"model-a")
    (tmp_path / "v-features.json").write_text(json.dumps({"feature_columns": ["a"]}))
    (tmp_path / "v-geo.json").write_text('{"groups": []}')
    return tmp_path


def test_stamped_bundle_verifies_and_keeps_other_keys(models_dir):
    bundle = stamp(models_dir, "v")
    assert bundle["geo_sha256"] and bundle["model_sha256"]
    assert json.loads((models_dir / "v-features.json").read_text())["feature_columns"] == ["a"]
    verify(models_dir, "v")


def test_other_runs_model_or_geo_state_is_rejected(models_dir):
    stamp(models_dir, "v")
    (models_dir / "v-geo.json").write_text('{"groups": [["b", 1.0, "x"]]}')
    with pytest.raises(BundleMismatch, match="geo state"):
        verify(models_dir, "v")
    stamp(models_dir, "v")
    (models_dir / "v-model.joblib").write_bytes(b"model-b")
    with pytest.raises(BundleMismatch, match="model"):
        verify(models_dir, "v")


def test_missing_geo_state_only_warns(models_dir, caplog):
    stamp(models_dir, "v")
    (models_dir / "v-geo.json").unlink()
    assert problems(models_dir, "v") == []
    with caplog.at_level("WARNING"):
        verify(models_dir, "v")
    assert "missing" in caplog.text


def test_unstamped_and_disabled_checks_pass(models_dir, monkeypatch):
    verify(models_dir, "v")  # never stamped
    stamp(models_dir, "v")
    (models_dir / "v-model.joblib").write_bytes(b"changed")
    monkeypatch.setenv("BUNDLE_CHECK", "0")
    verify(models_dir, "v")
