"""Registry-driven serving: models/ follows the Production version's bundle; failures keep serving what is there."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import artifact_bundle, model_sync

V = "challenger-noterms"
NAME = "tokyo_rent_lgbm_noterms"


def make_bundle(directory: Path, run_id: str, model: bytes, geo: bytes = b'{"groups": []}') -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{V}-model.joblib").write_bytes(model)
    (directory / f"{V}-features.json").write_text(json.dumps({"feature_columns": ["a"]}))
    if geo is not None:
        (directory / f"{V}-geo.json").write_bytes(geo)
    artifact_bundle.stamp(directory, V, run_id)
    return directory


class FakeClient:
    def __init__(self, runs):
        self.runs = runs            # run_id -> bundle dir (or None = run without bundle)
        self.production = None      # (version, run_id) or None
        self.registry_down = False

    def get_latest_versions(self, name, stages):
        if self.registry_down:
            raise ConnectionError("registry down")
        if name == NAME and self.production:
            return [SimpleNamespace(version=self.production[0], run_id=self.production[1])]
        return []

    def download_artifacts(self, run_id, path, dst):
        src = self.runs.get(run_id)
        if src is None:
            raise FileNotFoundError("no bundle")
        target = Path(dst) / path
        shutil.copytree(src, target)
        return str(target)


@pytest.fixture
def env(tmp_path):
    serving = make_bundle(tmp_path / "models", "run-old", b"old-model", b'{"v": "old"}')
    runs = {"run-old": make_bundle(tmp_path / "r_old", "run-old", b"old-model", b'{"v": "old"}'),
            "run-new": make_bundle(tmp_path / "r_new", "run-new", b"new-model", b'{"v": "new"}'),
            "run-nogeo": make_bundle(tmp_path / "r_nogeo", "run-nogeo", b"nogeo-model", geo=None),
            "run-nobundle": None}
    client = FakeClient(runs)
    return SimpleNamespace(models=serving, client=client, runs=runs, tmp=tmp_path)


def sync(env):
    return model_sync.sync_variant(env.client, env.models, V)


def test_matching_run_is_left_alone(env):
    env.client.production = ("1", "run-old")
    assert sync(env)["action"] == "unchanged"
    assert (env.models / f"{V}-model.joblib").read_bytes() == b"old-model"


def test_new_production_run_replaces_the_whole_bundle(env):
    env.client.production = ("2", "run-new")
    out = sync(env)
    assert out["action"] == "updated" and out["version"] == "2"
    assert (env.models / f"{V}-model.joblib").read_bytes() == b"new-model"
    assert json.loads((env.models / f"{V}-geo.json").read_text()) == {"v": "new"}
    assert model_sync.local_run_id(env.models, V) == "run-new"
    artifact_bundle.verify(env.models, V)             # files belong together
    assert not list(env.models.glob("*.tmp"))
    assert sync(env)["action"] == "unchanged"


def test_rollback_to_an_older_production_run_is_served_too(env):
    env.client.production = ("2", "run-new")
    sync(env)
    env.client.production = ("1", "run-old")
    assert sync(env)["action"] == "updated"
    assert (env.models / f"{V}-model.joblib").read_bytes() == b"old-model"


def test_stale_geo_file_is_removed_when_the_new_bundle_has_none(env):
    env.client.production = ("3", "run-nogeo")
    assert sync(env)["action"] == "updated"
    assert not (env.models / f"{V}-geo.json").exists()
    artifact_bundle.verify(env.models, V)


@pytest.mark.parametrize("setup, fragment", [
    (lambda e: setattr(e.client, "production", None), "no Production"),
    (lambda e: setattr(e.client, "production", ("4", "run-nobundle")), "no downloadable bundle"),
    (lambda e: (setattr(e.client, "production", ("4", "run-new")), setattr(e.client, "registry_down", True)), "registry unavailable"),
])
def test_failures_keep_serving_the_files_on_disk(env, setup, fragment):
    setup(env)
    out = sync(env)
    assert out["action"] == "skipped" and fragment in out["reason"]
    assert (env.models / f"{V}-model.joblib").read_bytes() == b"old-model"
    assert model_sync.local_run_id(env.models, V) == "run-old"


def test_a_bundle_failing_verification_is_not_installed(env):
    (env.runs["run-new"] / f"{V}-model.joblib").write_bytes(b"tampered")
    env.client.production = ("2", "run-new")
    out = sync(env)
    assert out["action"] == "skipped" and "verification" in out["reason"]
    assert (env.models / f"{V}-model.joblib").read_bytes() == b"old-model"


def test_unstamped_bundle_is_not_installed(env):
    state = json.loads((env.runs["run-new"] / f"{V}-features.json").read_text())
    del state["bundle"]
    (env.runs["run-new"] / f"{V}-features.json").write_text(json.dumps(state))
    env.client.production = ("2", "run-new")
    assert sync(env)["action"] == "skipped"


def test_sync_all_variants_and_status(env, monkeypatch):
    monkeypatch.setattr(model_sync, "VARIANT_MODELS", {V: NAME, "passed": "tokyo_passed_rent_model"})
    env.client.production = ("2", "run-new")
    report = model_sync.sync_production_models(env.models, env.client)
    assert report[V]["action"] == "updated" and report["passed"]["action"] == "skipped"
    status = model_sync.serving_status(env.models, env.client)
    assert status[V]["in_sync"] and status[V]["serving_run_id"] == "run-new" and status[V]["bundle_problems"] == []
    assert status["passed"]["production_version"] is None and not status["passed"]["in_sync"]
    env.client.registry_down = True
    assert "registry_error" in model_sync.serving_status(env.models, env.client)[V]


# ------------------------------------------------------------------ API integration

def test_refresh_reloads_services_only_when_something_changed(monkeypatch):
    import model_api
    import model_runtime
    from app_state import S
    monkeypatch.setenv("MODEL_SYNC", "1")
    before = S.prediction_service
    calls = []

    def fake_sync(models_dir):
        calls.append(models_dir)
        return {"challenger": {"action": "unchanged"}}

    monkeypatch.setattr(model_sync, "sync_production_models", fake_sync)
    out = model_runtime.refresh_models_from_registry(force=True)
    assert out["status"] == "ok" and out["updated"] == [] and S.prediction_service is before

    monkeypatch.setattr(model_sync, "sync_production_models", lambda d: {"challenger": {"action": "updated"}})
    reloaded = []
    monkeypatch.setattr(model_runtime, "_reload_models", lambda: reloaded.append(1))
    assert model_runtime.refresh_models_from_registry(force=True)["updated"] == ["challenger"] and reloaded == [1]


def test_refresh_is_throttled_disabled_and_never_raises(monkeypatch):
    import model_api
    import model_runtime
    from app_state import S
    monkeypatch.setenv("MODEL_SYNC", "0")
    assert model_runtime.refresh_models_from_registry(force=True)["status"] == "disabled"
    monkeypatch.setenv("MODEL_SYNC", "1")
    monkeypatch.setattr(model_sync, "sync_production_models", lambda d: 1 / 0)
    assert model_runtime.refresh_models_from_registry(force=True)["status"] == "error"
    monkeypatch.setattr(model_runtime, "_model_sync_last", [model_runtime.time.monotonic()])
    assert model_runtime.refresh_models_from_registry()["status"] == "skipped"


def test_reload_builds_services_from_disk_and_survives_a_bad_reload(client, monkeypatch):
    import model_api
    import model_runtime
    from app_state import S
    client.get("/health")  # lets the middleware load models
    model_runtime._reload_models()
    assert S.prediction_service is not None and S.ensemble_prediction_service is not None
    old = S.prediction_service
    monkeypatch.setattr(S.model_loader, "load_model",
                        lambda name, **k: (_ for _ in ()).throw(FileNotFoundError(name)))
    with pytest.raises(RuntimeError):
        model_runtime._reload_models()
    assert S.prediction_service is old  # the running service was not replaced


def test_promote_endpoint_reports_what_is_now_served(client, admin_headers, monkeypatch):
    import model_api
    import model_runtime
    from app_state import S
    monkeypatch.setattr(S.auto_promotion_service, "promote_specific_version",
                        lambda m, v, f: {"promoted": True, "model_name": m, "version": v})
    monkeypatch.setattr(model_runtime, "refresh_models_from_registry", lambda force=False: {"status": "ok", "updated": ["challenger"]})
    r = client.post("/models/tokyo_rent_lgbm/promote/9", headers=admin_headers)
    assert r.status_code == 200 and r.json()["serving"]["updated"] == ["challenger"]
