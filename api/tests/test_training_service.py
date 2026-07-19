"""
Tests for the data-quality gate in api/services/training_service.py.

These mock out scraping, dataset persistence, and model training entirely —
the goal is only to verify that a scrape which fails validate_scraped_data's
`blocking` check short-circuits before any training happens.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.training_service import TrainingService


GOOD_DATA = pd.DataFrame({"rent_price": [10.0] * 20, "sqr_m": [45.0] * 20})
BROKEN_DATA = pd.DataFrame({"rent_price": [0.0] * 2, "sqr_m": [0.0] * 2})


@pytest.fixture(autouse=True)
def _no_disk_persistence(monkeypatch):
    """Every test in this file should skip real dataset_versions/ writes."""
    monkeypatch.setattr(
        TrainingService, "_persist_scraped_dataset", lambda self, df, url: "fake/dataset/path.csv"
    )


class TestDataQualityGate:
    @patch("services.training_service.ScrapeData")
    def test_blocking_quality_skips_training(self, mock_scrape_cls):
        # Use the real validate_scraped_data logic via an unbound ScrapeData instance
        from utils.scrape_data import ScrapeData

        real_scraper = ScrapeData.__new__(ScrapeData)
        mock_instance = mock_scrape_cls.return_value
        mock_instance.scrape.return_value = pd.DataFrame({"placeholder": [1]})
        mock_instance.clean_data.return_value = BROKEN_DATA
        mock_instance.validate_scraped_data.side_effect = real_scraper.validate_scraped_data

        service = TrainingService()
        with patch.object(service, "_retrain_challenger_model") as mock_challenger, \
             patch.object(service, "_retrain_passed_model") as mock_passed:
            result = service.retrain_models(
                url="https://example.com", wait_time_min=0, wait_time_max=0,
                pages=(0, 1), models_to_retrain=["challenger", "passed"],
            )

        assert result["status"] == "error"
        assert result["data_quality"]["blocking"] is True
        assert result["retrained_models"] == {}
        mock_challenger.assert_not_called()
        mock_passed.assert_not_called()

    @patch("services.training_service.ScrapeData")
    def test_good_quality_proceeds_to_training(self, mock_scrape_cls):
        from utils.scrape_data import ScrapeData

        real_scraper = ScrapeData.__new__(ScrapeData)
        mock_instance = mock_scrape_cls.return_value
        mock_instance.scrape.return_value = pd.DataFrame({"placeholder": [1]})
        mock_instance.clean_data.return_value = GOOD_DATA
        mock_instance.validate_scraped_data.side_effect = real_scraper.validate_scraped_data

        service = TrainingService()
        with patch.object(service, "_retrain_challenger_model", return_value="model") as mock_challenger, \
             patch.object(service, "_retrain_passed_model", return_value="model") as mock_passed, \
             patch("os.remove"), patch("pandas.DataFrame.to_csv"):
            result = service.retrain_models(
                url="https://example.com", wait_time_min=0, wait_time_max=0,
                pages=(0, 1), models_to_retrain=["challenger", "passed"],
            )

        assert result["status"] == "success"
        assert result["data_quality"]["blocking"] is False
        mock_challenger.assert_called_once()
        mock_passed.assert_called_once()
