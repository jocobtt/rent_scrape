"""Dataset lineage for registered models."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from app_state import S
from config.logging_config import get_logger
from deps import DEFAULT_RATE_LIMIT, limiter

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# End Explainability Endpoints
# ============================================================================

# ============================================================================
# Lineage Endpoints
# ============================================================================

@router.get("/lineage/{model_name}", tags=["Lineage"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_model_lineage(request: Request, model_name: str, version: Optional[str] = None):
    """
    Get dataset lineage for a registered model version.

    Returns the dataset hash, source, row/column counts, and columns used
    to train the specified model version (defaults to Production stage).
    Returns 404 if the model or lineage data is not found.
    """
    try:
        result = S.data_lineage_service.get_lineage_for_model(model_name, version)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No lineage data found for model '{model_name}'. "
                       "The model may not exist or was trained before lineage tracking was enabled."
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching lineage for {model_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/lineage/{model_name}/compare", tags=["Lineage"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def compare_model_lineage(
    request: Request,
    model_name: str,
    version_a: str,
    version_b: str,
):
    """
    Compare the training datasets of two model versions.

    Returns whether both versions were trained on the same dataset, along with
    a diff showing changes in row count, column count, columns added/removed,
    source URL changes, and creation dates.
    """
    if not version_a or not version_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both version_a and version_b query parameters are required."
        )
    try:
        result = S.data_lineage_service.compare_model_datasets(model_name, version_a, version_b)
        return result
    except Exception as e:
        logger.error(f"Error comparing lineage for {model_name} v{version_a} vs v{version_b}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
