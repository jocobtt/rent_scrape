"""
MLflow Integration for A/B Testing

This module extends the A/B testing framework with MLflow tracking
to log experiments, metrics, and promote winners to production.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from .ab_testing_service import ABTestingService, ABTestConfig, ExperimentStatus

logger = logging.getLogger(__name__)


class ABTestMLflowIntegration:
    """Integrate A/B testing with MLflow tracking"""

    def __init__(
        self,
        ab_service: ABTestingService,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow integration for A/B testing

        Args:
            ab_service: ABTestingService instance
            tracking_uri: MLflow tracking URI (optional)
        """
        self.ab_service = ab_service

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()
        logger.info("ABTestMLflowIntegration initialized")

    def log_experiment_to_mlflow(
        self,
        experiment_name: str,
        analysis_results: Dict[str, Any],
        config: ABTestConfig,
    ) -> str:
        """
        Log A/B test results to MLflow

        Args:
            experiment_name: Name of the A/B test
            analysis_results: Results from analyze_experiment()
            config: Experiment configuration

        Returns:
            MLflow run ID
        """
        try:
            # Create or get MLflow experiment
            mlflow_experiment_name = f"ab_test_{experiment_name}"

            try:
                experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                else:
                    experiment_id = mlflow.create_experiment(mlflow_experiment_name)
            except Exception:
                experiment_id = mlflow.create_experiment(mlflow_experiment_name)

            # Start MLflow run
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log configuration parameters
                mlflow.log_param("control_model", config.control_model)
                mlflow.log_param("treatment_model", config.treatment_model)
                mlflow.log_param("traffic_split", config.traffic_split)
                mlflow.log_param("shadow_mode", config.shadow_mode)
                mlflow.log_param("min_samples", config.min_samples)
                mlflow.log_param("confidence_level", config.confidence_level)
                mlflow.log_param("min_effect_size", config.min_effect_size)

                # Log sample sizes
                sample_sizes = analysis_results.get("sample_sizes", {})
                for key, value in sample_sizes.items():
                    mlflow.log_metric(f"samples_{key}", value)

                # Log control metrics
                control_metrics = analysis_results.get("control_metrics", {})
                for key, value in control_metrics.items():
                    mlflow.log_metric(f"control_{key}", value)

                # Log treatment metrics
                treatment_metrics = analysis_results.get("treatment_metrics", {})
                for key, value in treatment_metrics.items():
                    mlflow.log_metric(f"treatment_{key}", value)

                # Log statistical test results
                stat_tests = analysis_results.get("statistical_tests", {})
                if "t_test" in stat_tests:
                    mlflow.log_metric("t_statistic", stat_tests["t_test"]["t_statistic"])
                    mlflow.log_metric("p_value", stat_tests["t_test"]["p_value"])
                    mlflow.log_metric("significant", 1 if stat_tests["t_test"]["significant"] else 0)

                if "effect_size" in stat_tests:
                    mlflow.log_metric("cohens_d", stat_tests["effect_size"]["cohens_d"])

                if "mean_errors" in stat_tests:
                    mlflow.log_metric(
                        "percent_improvement",
                        stat_tests["mean_errors"]["percent_improvement"]
                    )

                # Log recommendation
                recommendation = analysis_results.get("recommendation", {})
                if recommendation.get("winner"):
                    mlflow.log_param("winner", recommendation["winner"])
                    mlflow.log_param("confidence", recommendation["confidence"])
                    mlflow.log_param("recommendation_action", recommendation["recommendation_action"])

                # Log latency comparison if available
                latency = analysis_results.get("latency_comparison", {})
                for key, value in latency.items():
                    mlflow.log_metric(key, value)

                # Add tags
                mlflow.set_tag("experiment_type", "ab_test")
                mlflow.set_tag("experiment_name", experiment_name)
                mlflow.set_tag("experiment_status", config.status.value)
                mlflow.set_tag("analysis_date", datetime.now().isoformat())

                logger.info(f"Logged A/B test results to MLflow run: {run.info.run_id}")
                return run.info.run_id

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            raise

    def promote_winner_to_production(
        self,
        experiment_name: str,
        analysis_results: Dict[str, Any],
        model_registry_name: str,
        auto_promote: bool = False,
    ) -> Dict[str, Any]:
        """
        Promote winning model to production in MLflow Model Registry

        Args:
            experiment_name: Name of A/B test
            analysis_results: Results from analyze_experiment()
            model_registry_name: Name of model in MLflow registry
            auto_promote: If True, automatically promote without confirmation

        Returns:
            Dictionary with promotion results
        """
        try:
            recommendation = analysis_results.get("recommendation", {})
            winner = recommendation.get("winner")
            confidence = recommendation.get("confidence")
            action = recommendation.get("recommendation_action")

            if not winner:
                return {
                    "status": "no_winner",
                    "message": "No clear winner determined",
                    "recommendation": recommendation,
                }

            if action != "promote_treatment_to_production":
                return {
                    "status": "promotion_not_recommended",
                    "message": f"Recommended action: {action}",
                    "recommendation": recommendation,
                }

            if confidence != "high" and not auto_promote:
                return {
                    "status": "low_confidence",
                    "message": "Confidence is not high. Set auto_promote=True to override.",
                    "recommendation": recommendation,
                }

            # Promote treatment model to production
            if winner == "treatment":
                # Get the latest version of the model
                versions = self.client.get_latest_versions(
                    model_registry_name,
                    stages=["None", "Staging"]
                )

                if not versions:
                    return {
                        "status": "error",
                        "message": f"No model versions found for {model_registry_name}",
                    }

                latest_version = versions[0]

                # Transition to Production
                self.client.transition_model_version_stage(
                    name=model_registry_name,
                    version=latest_version.version,
                    stage="Production",
                    archive_existing_versions=True,
                )

                # Update experiment status
                config = self.ab_service.get_experiment(experiment_name)
                if config:
                    config.status = ExperimentStatus.WINNER_SELECTED
                    self.ab_service._save_experiment(config)

                logger.info(
                    f"Promoted {model_registry_name} v{latest_version.version} "
                    f"to Production based on A/B test: {experiment_name}"
                )

                return {
                    "status": "success",
                    "message": f"Promoted {model_registry_name} v{latest_version.version} to Production",
                    "model_name": model_registry_name,
                    "version": latest_version.version,
                    "winner": winner,
                    "confidence": confidence,
                    "improvement": recommendation.get("percent_improvement", 0),
                }

            else:
                return {
                    "status": "control_won",
                    "message": "Control model won. No promotion needed.",
                    "recommendation": recommendation,
                }

        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def create_experiment_report(
        self,
        experiment_name: str,
        analysis_results: Dict[str, Any],
    ) -> str:
        """
        Create a markdown report of A/B test results

        Args:
            experiment_name: Name of experiment
            analysis_results: Analysis results

        Returns:
            Markdown formatted report
        """
        recommendation = analysis_results.get("recommendation", {})
        control_metrics = analysis_results.get("control_metrics", {})
        treatment_metrics = analysis_results.get("treatment_metrics", {})
        stat_tests = analysis_results.get("statistical_tests", {})
        sample_sizes = analysis_results.get("sample_sizes", {})

        report = f"""# A/B Test Results: {experiment_name}

**Analysis Date:** {analysis_results.get('analysis_date', 'N/A')}
**Status:** {analysis_results.get('status', 'N/A')}

## Summary

**Winner:** {recommendation.get('winner', 'None')}
**Confidence:** {recommendation.get('confidence', 'N/A')}
**Recommended Action:** {recommendation.get('recommendation_action', 'N/A')}

{recommendation.get('message', '')}

## Sample Sizes

- Total Predictions: {sample_sizes.get('total_predictions', 0):,}
- Predictions with Actuals: {sample_sizes.get('predictions_with_actuals', 0):,}
- Control Group: {sample_sizes.get('control_group', 0):,}
- Treatment Group: {sample_sizes.get('treatment_group', 0):,}

## Performance Metrics

### Control Model

- **MAE:** {control_metrics.get('mae', 0):.2f}
- **RMSE:** {control_metrics.get('rmse', 0):.2f}
- **MAPE:** {control_metrics.get('mape', 0):.2f}%
- **R²:** {control_metrics.get('r2', 0):.4f}

### Treatment Model

- **MAE:** {treatment_metrics.get('mae', 0):.2f}
- **RMSE:** {treatment_metrics.get('rmse', 0):.2f}
- **MAPE:** {treatment_metrics.get('mape', 0):.2f}%
- **R²:** {treatment_metrics.get('r2', 0):.4f}

### Improvement

- **Percent Improvement in MAE:** {recommendation.get('percent_improvement', 0):.2f}%
- **Absolute Difference:** {abs(control_metrics.get('mae', 0) - treatment_metrics.get('mae', 0)):.2f}

## Statistical Analysis
"""

        if "t_test" in stat_tests:
            t_test = stat_tests["t_test"]
            report += f"""
### T-Test

- **t-statistic:** {t_test.get('t_statistic', 0):.4f}
- **p-value:** {t_test.get('p_value', 0):.4f}
- **Statistically Significant:** {'Yes' if t_test.get('significant', False) else 'No'}
"""

        if "effect_size" in stat_tests:
            effect = stat_tests["effect_size"]
            report += f"""
### Effect Size

- **Cohen's d:** {effect.get('cohens_d', 0):.4f}
- **Interpretation:** {effect.get('interpretation', 'N/A')}
"""

        if "confidence_intervals" in stat_tests:
            ci = stat_tests["confidence_intervals"]
            report += f"""
### Confidence Intervals

**Control:**
- Lower: {ci['control']['lower']:.2f}
- Upper: {ci['control']['upper']:.2f}

**Treatment:**
- Lower: {ci['treatment']['lower']:.2f}
- Upper: {ci['treatment']['upper']:.2f}
"""

        if "latency_comparison" in analysis_results:
            latency = analysis_results["latency_comparison"]
            report += f"""
## Latency Comparison

- **Control Avg Latency:** {latency.get('control_avg_ms', 0):.2f} ms
- **Treatment Avg Latency:** {latency.get('treatment_avg_ms', 0):.2f} ms
"""

        report += f"""
## Recommendation

**{recommendation.get('recommendation_action', 'N/A').replace('_', ' ').title()}**

- Statistically Significant: {'Yes' if recommendation.get('is_statistically_significant', False) else 'No'}
- Practically Significant: {'Yes' if recommendation.get('is_practically_significant', False) else 'No'}

{recommendation.get('message', '')}
"""

        return report

    def compare_with_previous_experiments(
        self,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """
        Compare current experiment with previous experiments

        Args:
            experiment_name: Current experiment name

        Returns:
            Dictionary with comparison results
        """
        try:
            # Get all completed experiments
            completed = self.ab_service.list_experiments(
                status=ExperimentStatus.COMPLETED
            )

            if not completed:
                return {
                    "status": "no_previous_experiments",
                    "message": "No previous experiments to compare",
                }

            # Analyze current experiment
            current_results = self.ab_service.analyze_experiment(experiment_name)

            comparisons = []
            for exp in completed:
                if exp.experiment_name == experiment_name:
                    continue

                prev_results = self.ab_service.analyze_experiment(exp.experiment_name)

                comparison = {
                    "experiment_name": exp.experiment_name,
                    "current_winner": current_results.get("recommendation", {}).get("winner"),
                    "previous_winner": prev_results.get("recommendation", {}).get("winner"),
                    "current_improvement": current_results.get("recommendation", {}).get("percent_improvement", 0),
                    "previous_improvement": prev_results.get("recommendation", {}).get("percent_improvement", 0),
                }

                comparisons.append(comparison)

            return {
                "status": "success",
                "current_experiment": experiment_name,
                "comparisons": comparisons,
            }

        except Exception as e:
            logger.error(f"Error comparing experiments: {e}")
            return {"status": "error", "message": str(e)}
