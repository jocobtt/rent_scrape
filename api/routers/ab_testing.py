"""A/B experiments: create, assign, predict, analyse, promote the winner."""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app_state import S
from config.logging_config import get_logger
from deps import DEFAULT_RATE_LIMIT, PREDICTION_RATE_LIMIT, limiter, require_admin
from schemas import ABTestCreateRequest, ABTestPredictRequest
from services.ab_mlflow_integration import ABTestMLflowIntegration
from services.ab_testing_service import ABTestConfig

logger = get_logger(__name__)
router = APIRouter()


# A/B Testing Endpoints

def get_ab_mlflow_integration():
    """Lazy initialization of A/B MLflow integration"""
    if S.ab_mlflow_integration is None:
        S.ab_mlflow_integration = ABTestMLflowIntegration(S.ab_testing_service)
    return S.ab_mlflow_integration


@router.post("/ab-test/create", tags=["A/B Testing"], dependencies=[Depends(require_admin)])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def create_ab_test(request: Request, body: ABTestCreateRequest):
    """
    Create a new A/B test experiment

    Request Body:
        experiment_name: Unique name for the experiment
        control_model: Control model name (default: "passed")
        treatment_model: Treatment model name (default: "challenger")
        traffic_split: Percentage to treatment 0.0-1.0 (default: 0.1 = 10%)
        shadow_mode: Both models predict, only control returned (default: False)
        min_samples: Min samples for statistical tests (default: 100)
        confidence_level: Statistical confidence level (default: 0.95)
        min_effect_size: Minimum meaningful improvement (default: 0.05 = 5%)

    Returns:
        dict: Experiment creation status
    """
    try:
        config = ABTestConfig(
            experiment_name=body.experiment_name,
            control_model=body.control_model,
            treatment_model=body.treatment_model,
            traffic_split=body.traffic_split,
            shadow_mode=body.shadow_mode,
            min_samples=body.min_samples,
            confidence_level=body.confidence_level,
            min_effect_size=body.min_effect_size,
        )

        S.ab_testing_service.validate_config(config)  # ValueError -> 400
        success = S.ab_testing_service.create_experiment(config)

        if success:
            return {
                "status": "success",
                "message": f"Created A/B test experiment: {body.experiment_name}",
                "experiment": config.to_dict()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create experiment"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating A/B test: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create A/B test: {str(e)}"
        )


@router.post("/ab-test/{experiment_name}/start", dependencies=[Depends(require_admin)])
async def start_ab_test(experiment_name: str):
    """Start an A/B test experiment"""
    try:
        success = S.ab_testing_service.start_experiment(experiment_name)
        if success:
            return {
                "status": "success",
                "message": f"Started experiment: {experiment_name}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {experiment_name}"
            )
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/ab-test/{experiment_name}/stop", dependencies=[Depends(require_admin)])
async def stop_ab_test(experiment_name: str):
    """Stop an A/B test experiment"""
    try:
        success = S.ab_testing_service.stop_experiment(experiment_name)
        if success:
            return {
                "status": "success",
                "message": f"Stopped experiment: {experiment_name}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {experiment_name}"
            )
    except Exception as e:
        logger.error(f"Error stopping experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/ab-test/list")
async def list_ab_tests():
    """List all A/B test experiments"""
    try:
        experiments = S.ab_testing_service.list_experiments()
        return {
            "status": "success",
            "experiments": [exp.to_dict() for exp in experiments],
            "total": len(experiments)
        }
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/ab-test/{experiment_name}")
async def get_ab_test(experiment_name: str):
    """Get details of a specific A/B test"""
    try:
        config = S.ab_testing_service.get_experiment(experiment_name)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {experiment_name}"
            )
        return {
            "status": "success",
            "experiment": config.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/ab-test/predict", tags=["A/B Testing"])
@limiter.limit(PREDICTION_RATE_LIMIT)
async def ab_test_predict(request: Request, body: ABTestPredictRequest):
    """
    Make prediction with A/B testing

    This endpoint:
    1. Determines which model to use based on traffic split
    2. Makes predictions with both models (for shadow mode)
    3. Returns the appropriate prediction
    4. Logs everything for analysis

    Request Body:
        experiment_name: Name of active experiment
        input_data: Apartment features
        user_id: Optional user ID for consistent assignment
        force_model: Force specific model (for testing)

    Returns:
        dict: Prediction result with experiment metadata
    """
    try:
        if S.prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )

        # Get experiment config
        config = S.ab_testing_service.get_experiment(body.experiment_name)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {body.experiment_name}"
            )

        # Assign model
        import time
        start_time = time.time()

        assigned_model = S.ab_testing_service.assign_model(
            body.experiment_name,
            user_id=body.user_id,
            force_model=body.force_model
        )

        # Make predictions with both models
        input_dict = body.input_data.dict()

        control_result = S.prediction_service.predict_with_passed_model(input_dict)
        treatment_result = S.prediction_service.predict_with_challenger_model(input_dict)

        control_pred = control_result["prediction"]
        treatment_pred = treatment_result["prediction"]

        # Determine which prediction to return
        if config.shadow_mode:
            # Shadow mode: always return control
            returned_pred = control_pred
            returned_model = "control"
        else:
            # Normal mode: return assigned model's prediction
            if assigned_model == "treatment":
                returned_pred = treatment_pred
                returned_model = "treatment"
            else:
                returned_pred = control_pred
                returned_model = "control"

        latency_ms = (time.time() - start_time) * 1000

        # Log to A/B testing service
        S.ab_testing_service.log_prediction(
            experiment_name=body.experiment_name,
            assigned_model=assigned_model,
            input_features=input_dict,
            control_prediction=control_pred,
            treatment_prediction=treatment_pred,
            returned_prediction=returned_pred,
            latency_ms=latency_ms
        )

        # Also log to data capture for monitoring
        S.data_capture_service.log_prediction(
            model_type=returned_model,
            input_features=input_dict,
            prediction=returned_pred
        )

        return {
            "prediction": returned_pred,
            "status": "success",
            "experiment": {
                "name": body.experiment_name,
                "assigned_model": assigned_model,
                "returned_model": returned_model,
                "shadow_mode": config.shadow_mode,
            },
            "debug_info": {
                "control_prediction": control_pred,
                "treatment_prediction": treatment_pred,
                "difference": abs(control_pred - treatment_pred),
                "latency_ms": round(latency_ms, 2)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in A/B test prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/ab-test/{experiment_name}/analyze")
async def analyze_ab_test(experiment_name: str, days_back: int = None):
    """
    Analyze A/B test results with statistical tests

    Query Parameters:
        days_back: Number of days to analyze (optional, defaults to all data)

    Returns:
        dict: Complete statistical analysis with winner recommendation
    """
    try:
        analysis = S.ab_testing_service.analyze_experiment(
            experiment_name,
            days_back=days_back
        )

        if "error" in analysis:
            return analysis

        # Log to MLflow
        try:
            config = S.ab_testing_service.get_experiment(experiment_name)
            if config:
                integration = get_ab_mlflow_integration()
                run_id = integration.log_experiment_to_mlflow(
                    experiment_name,
                    analysis,
                    config
                )
                analysis["mlflow_run_id"] = run_id
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

        return {
            "status": "success",
            "analysis": analysis
        }

    except Exception as e:
        logger.error(f"Error analyzing experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/ab-test/{experiment_name}/promote", dependencies=[Depends(require_admin)])
async def promote_ab_test_winner(
    experiment_name: str,
    model_registry_name: str = "tokyo_rent_lgbm",
    auto_promote: bool = False
):
    """
    Promote winning model to production

    Query Parameters:
        model_registry_name: MLflow model registry name (default: "tokyo_rent_lgbm")
        auto_promote: Auto-promote even with low confidence (default: False)

    Returns:
        dict: Promotion status and results
    """
    try:
        # First analyze to get winner
        analysis = S.ab_testing_service.analyze_experiment(experiment_name)

        if "error" in analysis:
            return {
                "status": "error",
                "message": analysis.get("error", "Unknown error"),
                "analysis": analysis
            }

        # Promote winner
        integration = get_ab_mlflow_integration()
        result = integration.promote_winner_to_production(
            experiment_name,
            analysis,
            model_registry_name,
            auto_promote=auto_promote
        )

        return result

    except Exception as e:
        logger.error(f"Error promoting winner: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/ab-test/{experiment_name}/report")
async def get_ab_test_report(experiment_name: str):
    """
    Get markdown report of A/B test results

    Returns:
        dict: Markdown formatted report
    """
    try:
        analysis = S.ab_testing_service.analyze_experiment(experiment_name)

        if "error" in analysis:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=analysis.get("error", "Unknown error")
            )

        integration = get_ab_mlflow_integration()
        report = integration.create_experiment_report(experiment_name, analysis)

        return {
            "status": "success",
            "experiment_name": experiment_name,
            "report": report
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
