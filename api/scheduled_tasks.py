"""
Scheduled Tasks for Automated Model Lifecycle

This script runs periodic checks and automation tasks:
1. Check for models to promote (daily)
2. Check for drift and retrain if needed (hourly)
3. Monitor production performance (hourly)

Usage:
    python scheduled_tasks.py

Or as a background daemon:
    nohup python scheduled_tasks.py > scheduled_tasks.log 2>&1 &
"""

import logging
import schedule
import time
from datetime import datetime, timezone

from services.auto_promotion_service import AutoPromotionService
from services.auto_retrain_service import AutoRetrainService
from services.monitoring_reports import MonitoringReportsManager
from services.sla_service import ModelSLAService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduled_tasks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize services
auto_promotion = AutoPromotionService()
auto_retrain = AutoRetrainService()
monitoring_manager = MonitoringReportsManager()


def run_promotion_check():
    """
    Daily check for models ready to be promoted to production

    Runs at 2:00 AM daily
    """
    logger.info("="*60)
    logger.info("Running daily promotion check...")
    logger.info("="*60)

    try:
        result = auto_promotion.check_and_promote_new_models(dry_run=False)

        if result["models_promoted"]:
            logger.info(f"✅ Promoted {len(result['models_promoted'])} models:")
            for model in result["models_promoted"]:
                logger.info(f"   - {model['model_name']} v{model['version']}: {model['reason']}")
        else:
            logger.info("ℹ️  No models qualified for promotion")

        if result["models_skipped"]:
            logger.info(f"Skipped {len(result['models_skipped'])} models:")
            for model in result["models_skipped"]:
                logger.info(f"   - {model['model_name']}: {model['reason']}")

    except Exception as e:
        logger.error(f"❌ Error in promotion check: {e}", exc_info=True)


def run_drift_check():
    """
    Hourly drift check with potential retraining

    Checks for:
    - Data drift
    - Performance degradation

    Triggers retraining if thresholds exceeded
    """
    logger.info("-"*60)
    logger.info("Running hourly drift/performance check...")
    logger.info("-"*60)

    try:
        # Check and retrain if needed
        result = auto_retrain.check_and_retrain(days_back=1, dry_run=False)

        if result["retraining_triggered"]:
            logger.warning("⚠️  RETRAINING TRIGGERED!")
            logger.info(f"   Drift detected: {result['checks_performed']['drift']['drift_detected']}")
            logger.info(f"   Performance degraded: {result['checks_performed']['performance']['performance_degraded']}")

            if result["models_retrained"]:
                logger.info(f"   Retrained models: {', '.join(result['models_retrained'])}")

            if result["models_promoted"]:
                logger.info(f"   Auto-promoted: {len(result['models_promoted'])} models")
        else:
            logger.info("✅ All systems healthy - no retraining needed")

    except Exception as e:
        logger.error(f"❌ Error in drift check: {e}", exc_info=True)


def run_monitoring_reports():
    """
    Daily monitoring report generation

    Generates comprehensive drift and performance reports
    """
    logger.info("-"*60)
    logger.info("Generating daily monitoring reports...")
    logger.info("-"*60)

    try:
        # Generate all reports
        report_paths = monitoring_manager.generate_all_reports(days_back=7)

        logger.info("Generated reports:")
        for report_type, path in report_paths.items():
            if path:
                logger.info(f"   - {report_type}: {path}")

    except Exception as e:
        logger.error(f"❌ Error generating reports: {e}", exc_info=True)


STALENESS_THRESHOLD_DAYS = 14  # warn if production model is older than this


def run_sla_check():
    """
    Daily SLA floor check for all production models.

    Automatically demotes models to Staging if they fall below minimum thresholds.
    Runs at 04:00 daily.
    """
    logger.info("-"*60)
    logger.info("Running SLA check...")
    logger.info("-"*60)

    try:
        sla_service = ModelSLAService()
        result = sla_service.check_all_models()
        logger.info(
            f"SLA check complete: {result['models_passing']}/{result['models_checked']} passing, "
            f"{result['models_demoted']} demoted"
        )
        for model_name, r in result["results"].items():
            if r.get("demoted"):
                logger.warning(f"ALERT: {model_name} demoted to Staging — {r['reason']}")
            elif not r.get("sla_passing", True):
                logger.warning(f"ALERT: {model_name} SLA failing — {r.get('reason')}")
    except Exception as e:
        logger.error(f"❌ Error in SLA check: {e}", exc_info=True)


def run_monthly_scrape_and_retrain():
    """
    Monthly data refresh: scrape fresh Suumo listings and retrain both models.

    The `schedule` library has no native monthly cadence, so this runs daily
    at 01:00 and exits early on every day that isn't the 1st of the month.
    Set SUUMO_URL env var to override the default Tokyo apartment listing URL.
    """
    if datetime.now().day != 1:
        return
    logger.info("="*60)
    logger.info("Running monthly scrape + retrain...")
    logger.info("="*60)
    try:
        import os
        from services.training_service import TrainingService
        url = os.environ.get("SUUMO_URL") or (
            "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=30"
            "&smk=&po1=25&po2=99&tc=0400101&tc=0400201&tc=0400301&tc=0400401"
            "&tc=0400501&tc=0400601&tc=0400701&tc=0400801&tc=0400901&tc=0401001&tc=0401101"
        )
        result = TrainingService().retrain_models(
            url=url,
            wait_time_min=2,
            wait_time_max=5,
            pages=(0, 50),
            models_to_retrain=["challenger", "passed"],
        )
        logger.info(
            f"Monthly retrain complete: status={result.get('status')} "
            f"dataset={result.get('dataset_version_path')}"
        )
        if result.get("status") == "error" and result.get("data_quality"):
            logger.error(
                f"Scrape rejected by data quality gate: {result['data_quality'].get('critical')}"
            )
    except Exception as e:
        logger.error(f"Monthly scrape/retrain failed: {e}", exc_info=True)


def run_cleanup(dry_run: bool = False):
    """
    Weekly version cleanup — deletes Archived versions beyond retention limit.

    Runs Sundays at 05:00.
    """
    logger.info("-"*60)
    logger.info(f"Running version cleanup (dry_run={dry_run})...")
    logger.info("-"*60)

    try:
        result = auto_promotion.cleanup_all_models(max_versions_to_keep=10, dry_run=dry_run)
        logger.info(f"Cleanup complete: {result['total_deleted']} versions deleted")
        for model_name, r in result["results"].items():
            if "error" in r:
                logger.error(f"  {model_name}: cleanup error — {r['error']}")
            else:
                logger.info(
                    f"  {model_name}: deleted {len(r['deleted'])}, "
                    f"kept {len(r['archived_kept'])} archived"
                )
    except Exception as e:
        logger.error(f"❌ Error in version cleanup: {e}", exc_info=True)


def check_model_health():
    """
    Hourly health check for production models.

    Monitors:
    - Prediction latency
    - Error rates
    - Drift status
    - Model staleness
    """
    logger.info("-"*60)
    logger.info("Checking model health...")
    logger.info("-"*60)

    now_ts = datetime.now(timezone.utc).timestamp()

    try:
        for model_name in ["tokyo_rent_lgbm", "tokyo_passed_rent_model"]:
            status = auto_promotion.get_promotion_status(model_name)

            if "error" not in status:
                logger.info(f"{model_name}:")
                logger.info(f"   Total versions: {status['total_versions']}")

                if "Production" in status["latest_by_stage"]:
                    prod = status["latest_by_stage"]["Production"]
                    logger.info(f"   Production: v{prod['version']}")
                    logger.info(f"   Metrics: {prod['metrics']}")

                    # Staleness alert
                    created_ms = prod.get("creation_timestamp")
                    if created_ms:
                        age_days = (now_ts - created_ms / 1000) / 86400
                        if age_days > STALENESS_THRESHOLD_DAYS:
                            logger.warning(
                                f"   STALENESS ALERT: {model_name} production model is "
                                f"{age_days:.0f} days old — consider retraining"
                            )
                else:
                    logger.warning(f"   ⚠️  No production version!")

    except Exception as e:
        logger.error(f"❌ Error in health check: {e}", exc_info=True)


# Schedule tasks
logger.info("Starting scheduled tasks...")
logger.info("Schedules:")
logger.info("  - Monthly scrape:    Daily at 01:00 (runs only on 1st of month)")
logger.info("  - Promotion check:   Daily at 02:00")
logger.info("  - Monitoring reports: Daily at 03:00")
logger.info("  - SLA check:         Daily at 04:00")
logger.info("  - Version cleanup:   Weekly (Sunday) at 05:00")
logger.info("  - Drift check:       Every hour at :00")
logger.info("  - Health check:      Every hour at :30")
logger.info("")

# Daily tasks
schedule.every().day.at("01:00").do(run_monthly_scrape_and_retrain)
schedule.every().day.at("02:00").do(run_promotion_check)
schedule.every().day.at("03:00").do(run_monitoring_reports)
schedule.every().day.at("04:00").do(run_sla_check)

# Weekly tasks
schedule.every().sunday.at("05:00").do(run_cleanup)

# Hourly tasks
schedule.every().hour.at(":00").do(run_drift_check)
schedule.every().hour.at(":30").do(check_model_health)

# Run once immediately on startup
logger.info("Running initial checks...")
check_model_health()


# Main loop
if __name__ == "__main__":
    logger.info("Scheduled tasks service started")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*60)

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        logger.info("\nStopping scheduled tasks...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Scheduled tasks service stopped")
