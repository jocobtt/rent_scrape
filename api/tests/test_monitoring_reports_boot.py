import services.monitoring_reports as mr


def test_manager_boots_without_a_training_dataset(monkeypatch, tmp_path):
    """Scraped data is not committed; the API must still import on a fresh checkout / in CI."""
    def no_data(path=None):
        raise FileNotFoundError("No training dataset found")

    monkeypatch.setattr(mr, "load_reference_data", no_data)
    manager = mr.MonitoringReportsManager(reports_dir=str(tmp_path / "r"), predictions_dir=str(tmp_path / "p"))
    assert manager.reference_data.empty
