"""SHAP / LIME explanations."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from app_state import S
from config.logging_config import get_logger
from deps import DEFAULT_RATE_LIMIT, limiter
from schemas import ExplainRequest, InputData, _require_terms

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# End PyTorch Endpoints
# ============================================================================

# ============================================================================
# Explainability Endpoints (SHAP + LIME)
# ============================================================================

@router.post("/explain/predict", tags=["Explainability"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def explain_prediction(request: Request, data: ExplainRequest):
    """
    Explain a single prediction using SHAP or LIME.

    Returns per-feature attribution values that show why the model predicted
    a particular rent price for the given apartment.

    - **model_name**: `"challenger"` (LightGBM) or `"passed"` (Linear)
    - **method**: `"shap"` (default, faster) or `"lime"` (model-agnostic)

    Response includes:
    - `prediction`: The predicted rent in 万円
    - `base_value`: SHAP baseline (average prediction over training data)
    - `features`: Per-feature SHAP/LIME values sorted by absolute importance
    - `top_positive`: Top features that *increase* the predicted rent
    - `top_negative`: Top features that *decrease* the predicted rent
    """
    try:
        if S.prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded. Ensure the API has started successfully."
            )

        model_name = data.model_name.lower()
        if model_name == "challenger":
            model = S.app.state.challenger_model
        elif model_name == "passed":
            model = S.app.state.passed_model
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model_name '{data.model_name}'. Use 'challenger' or 'passed'."
            )

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{data.model_name}' is not loaded."
            )

        if data.method not in ("shap", "lime"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown method '{data.method}'. Use 'shap' or 'lime'."
            )

        result = S.explainability_service.explain_prediction(
            model=model,
            input_data=_require_terms(data.input_data),
            background_data=None,  # uses single-row baseline; sufficient for SHAP tree/linear
            method=data.method,
        )

        return {
            "model_name": data.model_name,
            "method": data.method,
            **result,
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/explain/global/{model_name}", tags=["Explainability"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def global_feature_importance(request: Request, model_name: str, n_samples: int = 200):
    """
    Compute global SHAP feature importance for a model over a sample of typical inputs.

    Returns all features ranked by mean absolute SHAP value — showing which features
    the model relies on most across all predictions.

    - **model_name**: `"challenger"` or `"passed"`
    - **n_samples**: Number of synthetic background samples to use (default 200)
    """
    try:
        if S.prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded."
            )

        name = model_name.lower()
        if name == "challenger":
            model = S.app.state.challenger_model
        elif name == "passed":
            model = S.app.state.passed_model
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model_name '{model_name}'. Use 'challenger' or 'passed'."
            )

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{model_name}' is not loaded."
            )

        from services.explainability_service import FEATURE_NAMES

        # Build a synthetic background dataset using sensible Tokyo apartment ranges
        rng = np.random.default_rng(42)
        n = min(n_samples, 500)
        background = pd.DataFrame({
            "sqr_m":              rng.uniform(15, 120, n),
            "rei_price":          rng.uniform(0, 3, n),
            "shikikin":           rng.uniform(0, 3, n),
            "maintenence_price":  rng.uniform(0, 2, n),
            "year_built":         rng.integers(1970, 2024, n).astype(float),
            "floor":              rng.integers(1, 30, n).astype(float),
            "eki_walk":           rng.integers(1, 25, n).astype(float),
        })

        result = S.explainability_service.global_feature_importance(
            model=model,
            background_data=background,
            feature_names=FEATURE_NAMES,
            n_samples=n,
        )

        return {"model_name": model_name, **result}

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Global importance error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/explain/compare", tags=["Explainability"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def compare_model_explanations(request: Request, data: InputData):
    """
    Compare SHAP explanations from both the challenger and passed models side-by-side.

    Useful for understanding where the two models agree or disagree about which
    features drive the rent prediction.
    """
    try:
        if S.prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded."
            )

        results = {}
        for model_name, model in [
            ("challenger", S.app.state.challenger_model),
            ("passed", S.app.state.passed_model),
        ]:
            if model is None:
                results[model_name] = {"error": "model not loaded"}
                continue
            try:
                results[model_name] = S.explainability_service.explain_prediction(
                    model=model,
                    input_data=_require_terms(data),
                    method="shap",
                )
            except Exception as e:
                logger.warning(f"Could not explain {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        # Agreement score: how similar are the top features across both models?
        try:
            ch_top = {f["name"] for f in results.get("challenger", {}).get("features", [])[:3]}
            pa_top = {f["name"] for f in results.get("passed", {}).get("features", [])[:3]}
            overlap = len(ch_top & pa_top)
            agreement = f"{overlap}/3 top features agree"
        except Exception:
            agreement = "unavailable"

        return {
            "input_data": _require_terms(data),
            "explanations": results,
            "top_feature_agreement": agreement,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation compare error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
