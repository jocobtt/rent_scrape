"""Drift detection, monitoring reports, ground truth and prediction statistics."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app_state import S
from config.logging_config import get_logger
from deps import MONITORING_RATE_LIMIT, limiter, require_admin
from schemas import GroundTruthRequest
from services.monitoring_reports import MonitoringReportsManager
from utils import ScrapeData

logger = get_logger(__name__)
router = APIRouter()


# Monitoring Endpoints

def get_monitoring_manager():
    """Lazy initialization of monitoring manager"""
    if S.monitoring_manager is None:
        try:
            S.monitoring_manager = MonitoringReportsManager()
            logger.info("Initialized MonitoringReportsManager")
        except Exception as e:
            logger.error(f"Failed to initialize MonitoringReportsManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Monitoring service initialization failed: {str(e)}"
            )
    return S.monitoring_manager


@router.get("/monitoring/summary")
async def get_monitoring_summary():
    """
    Get a summary of monitoring status including prediction statistics and recent reports.

    Returns:
        dict: Monitoring summary with prediction statistics and report information
    """
    try:
        manager = get_monitoring_manager()
        summary = manager.get_monitoring_summary()
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting monitoring summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring summary: {str(e)}"
        )


@router.get("/scrape/health-check", tags=["Monitoring"])
@limiter.limit(MONITORING_RATE_LIMIT)
async def scrape_health_check(request: Request, url: str):
    """
    Scrape exactly one page from the given Suumo URL and verify the site is still parseable.

    Does NOT save data or trigger retraining — safe to call at any time.

    Returns field completeness (% non-zero) for all expected training features.
    """
    try:
        scraper = ScrapeData(url, wait_time_min=1, wait_time_max=2, pages=(0, 1))
        raw = scraper.scrape()
        if raw.empty:
            return {
                "status": "no_listings_parsed",
                "rows_parsed": 0,
                "field_completeness": {},
                "data_quality": scraper.validate_scraped_data(raw),
            }
        cleaned = scraper.clean_data(raw)
        expected_cols = [
            "sqr_m", "rei_price", "shikikin", "maintenence_price",
            "year_built", "floor", "nearest_eki_walk", "rent_price",
        ]
        completeness = {
            col: float((cleaned[col] != 0).mean()) if col in cleaned.columns else 0.0
            for col in expected_cols
        }
        return {
            "status": "ok" if len(cleaned) > 0 else "no_listings_parsed",
            "rows_parsed": len(cleaned),
            "field_completeness": completeness,
            "data_quality": scraper.validate_scraped_data(cleaned),
        }
    except Exception as e:
        logger.error(f"Scrape health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scrape health check failed: {str(e)}",
        )


@router.get("/monitoring/drift")
async def check_drift(
    days_back: int = 7,
    model_type: Optional[str] = None,
    drift_threshold: float = 0.3
):
    """
    Run drift detection on recent predictions.

    Query Parameters:
        days_back: Number of days of prediction data to analyze (default: 7)
        model_type: Filter by model type - 'passed' or 'challenger' (optional)
        drift_threshold: Maximum allowed share of drifted features (default: 0.3)

    Returns:
        dict: Drift detection results including test pass/fail and drift metrics
    """
    try:
        manager = get_monitoring_manager()
        result = manager.run_drift_detection(
            days_back=days_back,
            model_type=model_type,
            drift_threshold=drift_threshold
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking drift: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}"
        )


@router.post("/monitoring/reports/generate", tags=["Monitoring"], dependencies=[Depends(require_admin)])
@limiter.limit(MONITORING_RATE_LIMIT)
async def generate_monitoring_reports(
    request: Request,
    days_back: int = 7,
    model_type: Optional[str] = None
):
    """
    Generate all monitoring reports (data drift, data quality, model performance).

    Request Body (JSON):
        days_back: Number of days of data to analyze (default: 7)
        model_type: Filter by model type - 'passed' or 'challenger' (optional)

    Returns:
        dict: Paths to generated report files
    """
    try:
        manager = get_monitoring_manager()
        report_paths = manager.generate_all_reports(
            days_back=days_back,
            model_type=model_type
        )
        return {
            "status": "success",
            "reports": report_paths,
            "message": "Reports generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating monitoring reports: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )


@router.post("/monitoring/ground-truth", tags=["Monitoring"], dependencies=[Depends(require_admin)])
@limiter.limit(MONITORING_RATE_LIMIT)
async def submit_ground_truth(request: Request, body: GroundTruthRequest):
    """Attach actual rents to logged predictions; performance monitoring and A/B analysis need them."""
    def apply():
        return [S.data_capture_service.update_actual_value(i.prediction_id, i.actual_value) for i in body.labels]

    outcomes = await run_in_threadpool(apply)
    return {"status": "success", "updated": sum(outcomes),
            "not_found": [i.prediction_id for i, ok in zip(body.labels, outcomes) if not ok]}


@router.get("/monitoring/compare_models")
async def compare_models_drift(days_back: int = 7):
    """
    Compare drift between challenger and passed models.

    Query Parameters:
        days_back: Number of days of data to analyze (default: 7)

    Returns:
        dict: Comparison of drift metrics between models
    """
    try:
        manager = get_monitoring_manager()
        comparison = manager.compare_models_drift(days_back=days_back)
        return {
            "status": "success",
            "comparison": comparison,
            "days_analyzed": days_back
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing model drift: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model drift comparison failed: {str(e)}"
        )


@router.get("/monitoring/predictions/stats")
async def get_prediction_stats():
    """
    Get statistics about captured predictions.

    Returns:
        dict: Statistics including total predictions, predictions with actuals, and model breakdown
    """
    try:
        stats = S.data_capture_service.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting prediction stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction statistics: {str(e)}"
        )
