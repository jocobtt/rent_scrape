import pytest


class TestHealth:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_startup_without_models_returns_503(self, client, no_models):
        response = client.get("/startup")
        assert response.status_code == 503

    def test_ready_without_models_returns_503(self, client, no_models):
        response = client.get("/ready")
        assert response.status_code == 503


class TestInputValidation:
    def test_predict_missing_required_field(self, client):
        response = client.post("/predict", json={"sqr_m": 30.0})
        assert response.status_code == 422

    def test_predict_empty_body(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_sqr_m_above_max(self, client, valid_input):
        response = client.post("/predict", json={**valid_input, "sqr_m": 600.0})
        assert response.status_code == 422

    def test_predict_sqr_m_below_min(self, client, valid_input):
        response = client.post("/predict", json={**valid_input, "sqr_m": 1.0})
        assert response.status_code == 422

    def test_predict_negative_floor(self, client, valid_input):
        response = client.post("/predict", json={**valid_input, "floor": -1.0})
        assert response.status_code == 422

    def test_predict_eki_walk_above_max(self, client, valid_input):
        response = client.post("/predict", json={**valid_input, "eki_walk": 200.0})
        assert response.status_code == 422

    def test_predict_string_value_rejects(self, client, valid_input):
        response = client.post("/predict", json={**valid_input, "sqr_m": "big"})
        assert response.status_code == 422


class TestServiceUnavailable:
    def test_predict_without_models_returns_503(self, client, valid_input, no_models):
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 503

    def test_challenger_predict_without_models_returns_503(self, client, valid_input, no_models):
        response = client.post("/challenger_predict", json=valid_input)
        assert response.status_code == 503

    def test_compare_without_models_returns_503(self, client, valid_input, no_models):
        response = client.post("/compare_predictions", json=valid_input)
        assert response.status_code == 503

    def test_ensemble_methods_without_models_returns_503(self, client, no_models):
        response = client.get("/ensemble/methods")
        assert response.status_code == 503


class TestPredictions:
    def test_predict_success(self, client, valid_input, mock_prediction_svc):
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_predict_returns_correct_value(self, client, valid_input, mock_prediction_svc):
        response = client.post("/predict", json=valid_input)
        assert response.json()["prediction"] == 12.5

    def test_challenger_predict_success(self, client, valid_input, mock_prediction_svc):
        response = client.post("/challenger_predict", json=valid_input)
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_challenger_predict_returns_correct_value(self, client, valid_input, mock_prediction_svc):
        response = client.post("/challenger_predict", json=valid_input)
        assert response.json()["prediction"] == 13.0

    def test_compare_predictions_success(self, client, valid_input, mock_prediction_svc):
        response = client.post("/compare_predictions", json=valid_input)
        assert response.status_code == 200
        data = response.json()
        assert "passed_prediction" in data
        assert "challenger_prediction" in data

    def test_predict_calls_passed_model_method(self, client, valid_input, mock_prediction_svc):
        client.post("/predict", json=valid_input)
        mock_prediction_svc.predict_with_passed_model.assert_called_once()

    def test_challenger_predict_calls_challenger_method(self, client, valid_input, mock_prediction_svc):
        client.post("/challenger_predict", json=valid_input)
        mock_prediction_svc.predict_with_challenger_model.assert_called_once()

    def test_compare_calls_compare_method(self, client, valid_input, mock_prediction_svc):
        client.post("/compare_predictions", json=valid_input)
        mock_prediction_svc.compare_predictions.assert_called_once()


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        response = client.get("/model_info")
        assert response.status_code == 200

    def test_model_info_shape(self, client):
        data = client.get("/model_info").json()
        assert "models" in data
        assert "services" in data
        assert "api_version" in data

    def test_model_info_reports_models_not_loaded(self, client, no_models):
        data = client.get("/model_info").json()
        assert data["models"]["passed_model"]["loaded"] is False
        assert data["models"]["challenger_model"]["loaded"] is False

    def test_model_info_reports_prediction_service_unavailable(self, client, no_models):
        data = client.get("/model_info").json()
        assert data["services"]["prediction_service_available"] is False


class TestABTesting:
    def test_list_ab_tests_returns_200(self, client):
        response = client.get("/ab-test/list")
        assert response.status_code == 200

    def test_create_ab_test(self, client):
        response = client.post("/ab-test/create", json={
            "experiment_name": "ci_test_experiment",
            "control_model": "passed",
            "treatment_model": "challenger",
            "traffic_split": 0.2,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "experiment" in data

    def test_create_ab_test_invalid_name_with_spaces(self, client):
        response = client.post("/ab-test/create", json={
            "experiment_name": "invalid name spaces",
            "control_model": "passed",
            "treatment_model": "challenger",
        })
        assert response.status_code == 422

    def test_create_ab_test_traffic_split_above_max(self, client):
        response = client.post("/ab-test/create", json={
            "experiment_name": "bad_split",
            "traffic_split": 1.5,
        })
        assert response.status_code == 422

    def test_create_ab_test_missing_name(self, client):
        response = client.post("/ab-test/create", json={
            "control_model": "passed",
            "treatment_model": "challenger",
        })
        assert response.status_code == 422
