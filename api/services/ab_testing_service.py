"""
A/B Testing Service for Model Comparison

This service implements a comprehensive A/B testing framework for comparing
challenger and passed models in production with proper statistical analysis.

Features:
- Traffic splitting (e.g., 90% passed, 10% challenger)
- Shadow mode (both models predict, only one returned)
- Statistical significance testing
- Experiment tracking in MLflow
- Automated winner selection
- Performance metrics comparison
"""

import logging
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B test experiment"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    WINNER_SELECTED = "winner_selected"


class TrafficSplitStrategy(Enum):
    """Strategy for splitting traffic between models"""
    PERCENTAGE = "percentage"  # Split by percentage (e.g., 90/10)
    RANDOM_HASH = "random_hash"  # Use hash of user/request ID
    MANUAL = "manual"  # Manual assignment via API parameter


class ABTestConfig:
    """Configuration for an A/B test experiment"""

    def __init__(
        self,
        experiment_name: str,
        control_model: str = "passed",
        treatment_model: str = "challenger",
        traffic_split: float = 0.5,
        split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.PERCENTAGE,
        shadow_mode: bool = False,
        min_samples: int = 100,
        confidence_level: float = 0.95,
        min_effect_size: float = 0.05,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """
        Initialize A/B test configuration

        Args:
            experiment_name: Unique name for the experiment
            control_model: Name of control model (default: "passed")
            treatment_model: Name of treatment model (default: "challenger")
            traffic_split: Percentage of traffic to treatment (0.0-1.0)
            split_strategy: How to split traffic
            shadow_mode: If True, both models predict but only control is returned
            min_samples: Minimum samples needed before statistical tests
            confidence_level: Confidence level for statistical tests (e.g., 0.95 = 95%)
            min_effect_size: Minimum meaningful effect size (e.g., 0.05 = 5% improvement)
            start_date: When experiment starts
            end_date: When experiment ends
        """
        self.experiment_name = experiment_name
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.traffic_split = traffic_split
        self.split_strategy = split_strategy
        self.shadow_mode = shadow_mode
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        self.min_effect_size = min_effect_size
        self.start_date = start_date or datetime.now()
        self.end_date = end_date
        self.status = ExperimentStatus.DRAFT
        self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "experiment_name": self.experiment_name,
            "control_model": self.control_model,
            "treatment_model": self.treatment_model,
            "traffic_split": self.traffic_split,
            "split_strategy": self.split_strategy.value,
            "shadow_mode": self.shadow_mode,
            "min_samples": self.min_samples,
            "confidence_level": self.confidence_level,
            "min_effect_size": self.min_effect_size,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ABTestConfig":
        """Create config from dictionary"""
        config = cls(
            experiment_name=data["experiment_name"],
            control_model=data.get("control_model", "passed"),
            treatment_model=data.get("treatment_model", "challenger"),
            traffic_split=data.get("traffic_split", 0.5),
            shadow_mode=data.get("shadow_mode", False),
            min_samples=data.get("min_samples", 100),
            confidence_level=data.get("confidence_level", 0.95),
            min_effect_size=data.get("min_effect_size", 0.05),
        )

        if data.get("split_strategy"):
            config.split_strategy = TrafficSplitStrategy(data["split_strategy"])

        if data.get("status"):
            config.status = ExperimentStatus(data["status"])

        if data.get("start_date"):
            config.start_date = datetime.fromisoformat(data["start_date"])

        if data.get("end_date"):
            config.end_date = datetime.fromisoformat(data["end_date"])

        if data.get("created_at"):
            config.created_at = datetime.fromisoformat(data["created_at"])

        return config


class ABTestingService:
    """Service for managing A/B tests between models"""

    def __init__(
        self,
        experiments_dir: str = "./ab_experiments",
        results_dir: str = "./ab_results",
    ):
        """
        Initialize A/B testing service

        Args:
            experiments_dir: Directory to store experiment configs
            results_dir: Directory to store experiment results
        """
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = Path(results_dir)

        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Cache for active experiments
        self._active_experiments: Dict[str, ABTestConfig] = {}

        # Load existing experiments
        self._load_experiments()

        logger.info("ABTestingService initialized")

    def create_experiment(self, config: ABTestConfig) -> bool:
        """
        Create a new A/B test experiment

        Args:
            config: Experiment configuration

        Returns:
            True if successful
        """
        try:
            # Validate config
            if config.traffic_split < 0 or config.traffic_split > 1:
                raise ValueError("traffic_split must be between 0 and 1")

            if config.min_samples < 10:
                raise ValueError("min_samples must be at least 10")

            # Save config
            config_path = self.experiments_dir / f"{config.experiment_name}.json"
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)

            # Add to cache
            self._active_experiments[config.experiment_name] = config

            logger.info(f"Created experiment: {config.experiment_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            return False

    def start_experiment(self, experiment_name: str) -> bool:
        """Start an experiment"""
        config = self.get_experiment(experiment_name)
        if not config:
            return False

        config.status = ExperimentStatus.RUNNING
        config.start_date = datetime.now()
        self._save_experiment(config)

        logger.info(f"Started experiment: {experiment_name}")
        return True

    def stop_experiment(self, experiment_name: str) -> bool:
        """Stop an experiment"""
        config = self.get_experiment(experiment_name)
        if not config:
            return False

        config.status = ExperimentStatus.PAUSED
        self._save_experiment(config)

        logger.info(f"Stopped experiment: {experiment_name}")
        return True

    def complete_experiment(self, experiment_name: str) -> bool:
        """Mark experiment as completed"""
        config = self.get_experiment(experiment_name)
        if not config:
            return False

        config.status = ExperimentStatus.COMPLETED
        config.end_date = datetime.now()
        self._save_experiment(config)

        logger.info(f"Completed experiment: {experiment_name}")
        return True

    def get_experiment(self, experiment_name: str) -> Optional[ABTestConfig]:
        """Get experiment configuration"""
        return self._active_experiments.get(experiment_name)

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ABTestConfig]:
        """
        List experiments, optionally filtered by status

        Args:
            status: Filter by status (optional)

        Returns:
            List of experiment configs
        """
        experiments = list(self._active_experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        return experiments

    def assign_model(
        self,
        experiment_name: str,
        user_id: Optional[str] = None,
        force_model: Optional[str] = None,
    ) -> str:
        """
        Assign which model to use for this request

        Args:
            experiment_name: Name of the experiment
            user_id: Optional user ID for consistent assignment
            force_model: Force specific model (for testing)

        Returns:
            Model name to use ("control" or "treatment")
        """
        config = self.get_experiment(experiment_name)

        if not config or config.status != ExperimentStatus.RUNNING:
            # Default to control if no active experiment
            return "control"

        # Check if experiment is within date range
        now = datetime.now()
        if config.end_date and now > config.end_date:
            return "control"

        # Force model if specified
        if force_model:
            return force_model

        # Apply traffic split strategy
        if config.split_strategy == TrafficSplitStrategy.PERCENTAGE:
            # Random assignment based on traffic split
            if random.random() < config.traffic_split:
                return "treatment"
            else:
                return "control"

        elif config.split_strategy == TrafficSplitStrategy.RANDOM_HASH:
            # Consistent assignment based on user_id
            if user_id:
                hash_value = hash(user_id) % 100
                if hash_value < (config.traffic_split * 100):
                    return "treatment"
            return "control"

        else:
            # Manual - requires force_model
            return "control"

    def log_prediction(
        self,
        experiment_name: str,
        assigned_model: str,
        input_features: Dict,
        control_prediction: float,
        treatment_prediction: float,
        returned_prediction: float,
        actual_value: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> bool:
        """
        Log prediction result for A/B test

        Args:
            experiment_name: Name of experiment
            assigned_model: Which model was assigned (control/treatment)
            input_features: Input features
            control_prediction: Control model prediction
            treatment_prediction: Treatment model prediction
            returned_prediction: Prediction returned to user
            actual_value: Actual value if available
            latency_ms: Prediction latency in milliseconds

        Returns:
            True if successful
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": experiment_name,
                "assigned_model": assigned_model,
                "control_prediction": control_prediction,
                "treatment_prediction": treatment_prediction,
                "returned_prediction": returned_prediction,
                "actual_value": actual_value,
                "latency_ms": latency_ms,
                "input_features": input_features,
            }

            # Save to daily log file
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.results_dir / f"{experiment_name}_{date_str}.jsonl"

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            return True

        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            return False

    def get_experiment_results(
        self,
        experiment_name: str,
        days_back: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get experiment results as DataFrame

        Args:
            experiment_name: Name of experiment
            days_back: Number of days to look back (optional)

        Returns:
            DataFrame with experiment results
        """
        try:
            # Find all log files for this experiment
            pattern = f"{experiment_name}_*.jsonl"
            log_files = list(self.results_dir.glob(pattern))

            if not log_files:
                return pd.DataFrame()

            # Filter by date if specified
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                cutoff_str = cutoff_date.strftime("%Y%m%d")
                log_files = [
                    f for f in log_files
                    if f.stem.split("_")[-1] >= cutoff_str
                ]

            # Load all data
            data = []
            for log_file in log_files:
                with open(log_file, "r") as f:
                    for line in f:
                        data.append(json.loads(line))

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df

        except Exception as e:
            logger.error(f"Error loading experiment results: {e}")
            return pd.DataFrame()

    def analyze_experiment(
        self,
        experiment_name: str,
        days_back: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze experiment results with statistical tests

        Args:
            experiment_name: Name of experiment
            days_back: Number of days to analyze (optional)

        Returns:
            Dictionary with analysis results
        """
        config = self.get_experiment(experiment_name)
        if not config:
            return {"error": "Experiment not found"}

        # Load results
        df = self.get_experiment_results(experiment_name, days_back)

        if df.empty:
            return {
                "error": "No data available",
                "experiment_name": experiment_name,
            }

        # Filter to only predictions with actual values
        df_with_actuals = df[df["actual_value"].notna()].copy()

        if len(df_with_actuals) < config.min_samples:
            return {
                "status": "insufficient_data",
                "experiment_name": experiment_name,
                "samples_collected": len(df),
                "samples_with_actuals": len(df_with_actuals),
                "min_samples_required": config.min_samples,
                "message": f"Need at least {config.min_samples} samples with actual values for statistical analysis",
            }

        # Calculate errors for each model
        df_with_actuals["control_error"] = abs(
            df_with_actuals["actual_value"] - df_with_actuals["control_prediction"]
        )
        df_with_actuals["treatment_error"] = abs(
            df_with_actuals["actual_value"] - df_with_actuals["treatment_prediction"]
        )

        # Separate by assigned model
        control_group = df_with_actuals[df_with_actuals["assigned_model"] == "control"]
        treatment_group = df_with_actuals[df_with_actuals["assigned_model"] == "treatment"]

        # Calculate metrics
        analysis = {
            "experiment_name": experiment_name,
            "status": config.status.value,
            "analysis_date": datetime.now().isoformat(),
            "sample_sizes": {
                "total_predictions": len(df),
                "predictions_with_actuals": len(df_with_actuals),
                "control_group": len(control_group),
                "treatment_group": len(treatment_group),
            },
            "control_metrics": self._calculate_metrics(
                df_with_actuals["control_prediction"],
                df_with_actuals["actual_value"],
            ),
            "treatment_metrics": self._calculate_metrics(
                df_with_actuals["treatment_prediction"],
                df_with_actuals["actual_value"],
            ),
        }

        # Perform statistical tests
        if len(control_group) >= 10 and len(treatment_group) >= 10:
            analysis["statistical_tests"] = self._perform_statistical_tests(
                control_group["control_error"].values,
                treatment_group["treatment_error"].values,
                config.confidence_level,
            )

            # Add winner recommendation
            analysis["recommendation"] = self._determine_winner(
                analysis["control_metrics"],
                analysis["treatment_metrics"],
                analysis["statistical_tests"],
                config.min_effect_size,
            )
        else:
            analysis["statistical_tests"] = {
                "error": "Not enough samples in both groups for statistical tests",
                "control_samples": len(control_group),
                "treatment_samples": len(treatment_group),
            }
            analysis["recommendation"] = {
                "winner": None,
                "confidence": "insufficient_data",
            }

        # Add latency comparison if available
        if "latency_ms" in df.columns and df["latency_ms"].notna().any():
            analysis["latency_comparison"] = {
                "control_avg_ms": float(df[df["assigned_model"] == "control"]["latency_ms"].mean()),
                "treatment_avg_ms": float(df[df["assigned_model"] == "treatment"]["latency_ms"].mean()),
            }

        return analysis

    def _calculate_metrics(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        errors = actuals - predictions
        abs_errors = abs(errors)

        return {
            "mae": float(abs_errors.mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "mape": float((abs_errors / actuals).mean() * 100),
            "r2": float(1 - ((errors ** 2).sum() / ((actuals - actuals.mean()) ** 2).sum())),
            "mean_prediction": float(predictions.mean()),
            "std_prediction": float(predictions.std()),
        }

    def _perform_statistical_tests(
        self,
        control_errors: np.ndarray,
        treatment_errors: np.ndarray,
        confidence_level: float,
    ) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        # Two-sample t-test
        t_statistic, p_value = stats.ttest_ind(control_errors, treatment_errors)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (control_errors.std() ** 2 + treatment_errors.std() ** 2) / 2
        )
        cohens_d = (control_errors.mean() - treatment_errors.mean()) / pooled_std

        # Confidence intervals
        control_ci = stats.t.interval(
            confidence_level,
            len(control_errors) - 1,
            loc=control_errors.mean(),
            scale=stats.sem(control_errors),
        )

        treatment_ci = stats.t.interval(
            confidence_level,
            len(treatment_errors) - 1,
            loc=treatment_errors.mean(),
            scale=stats.sem(treatment_errors),
        )

        # Mann-Whitney U test (non-parametric alternative)
        u_statistic, u_p_value = stats.mannwhitneyu(
            control_errors, treatment_errors, alternative="two-sided"
        )

        return {
            "t_test": {
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
                "significant": p_value < (1 - confidence_level),
            },
            "mann_whitney_u": {
                "u_statistic": float(u_statistic),
                "p_value": float(u_p_value),
                "significant": u_p_value < (1 - confidence_level),
            },
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": self._interpret_effect_size(abs(cohens_d)),
            },
            "confidence_intervals": {
                "control": {
                    "lower": float(control_ci[0]),
                    "upper": float(control_ci[1]),
                },
                "treatment": {
                    "lower": float(treatment_ci[0]),
                    "upper": float(treatment_ci[1]),
                },
            },
            "mean_errors": {
                "control": float(control_errors.mean()),
                "treatment": float(treatment_errors.mean()),
                "difference": float(control_errors.mean() - treatment_errors.mean()),
                "percent_improvement": float(
                    ((control_errors.mean() - treatment_errors.mean()) / control_errors.mean()) * 100
                ),
            },
        }

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def _determine_winner(
        self,
        control_metrics: Dict,
        treatment_metrics: Dict,
        statistical_tests: Dict,
        min_effect_size: float,
    ) -> Dict[str, Any]:
        """Determine experiment winner based on statistical criteria"""
        # Check if statistically significant
        is_significant = statistical_tests["t_test"]["significant"]

        # Check if effect size is meaningful
        percent_improvement = abs(
            statistical_tests["mean_errors"]["percent_improvement"]
        )
        is_meaningful = percent_improvement >= (min_effect_size * 100)

        # Determine winner
        treatment_better = (
            treatment_metrics["mae"] < control_metrics["mae"]
        )

        if is_significant and is_meaningful:
            if treatment_better:
                winner = "treatment"
                confidence = "high"
                message = (
                    f"Treatment model is statistically significantly better "
                    f"({percent_improvement:.1f}% improvement in MAE)"
                )
            else:
                winner = "control"
                confidence = "high"
                message = (
                    f"Control model is statistically significantly better "
                    f"({percent_improvement:.1f}% improvement in MAE)"
                )
        elif is_significant and not is_meaningful:
            winner = "control"
            confidence = "medium"
            message = (
                f"Difference is statistically significant but not practically meaningful "
                f"({percent_improvement:.1f}% < {min_effect_size * 100}% threshold)"
            )
        elif not is_significant and treatment_better:
            winner = None
            confidence = "low"
            message = (
                "Treatment appears better but difference is not statistically significant. "
                "Collect more data or continue with control."
            )
        else:
            winner = "control"
            confidence = "medium"
            message = "No significant improvement detected. Continue with control model."

        return {
            "winner": winner,
            "confidence": confidence,
            "message": message,
            "is_statistically_significant": is_significant,
            "is_practically_significant": is_meaningful,
            "percent_improvement": percent_improvement,
            "recommendation_action": self._get_recommendation_action(winner, confidence),
        }

    def _get_recommendation_action(
        self, winner: Optional[str], confidence: str
    ) -> str:
        """Get recommended action based on winner"""
        if winner == "treatment" and confidence == "high":
            return "promote_treatment_to_production"
        elif winner == "control" and confidence == "high":
            return "keep_control_discard_treatment"
        elif confidence == "low":
            return "continue_experiment_collect_more_data"
        else:
            return "keep_control"

    def _save_experiment(self, config: ABTestConfig) -> None:
        """Save experiment configuration"""
        config_path = self.experiments_dir / f"{config.experiment_name}.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        self._active_experiments[config.experiment_name] = config

    def _load_experiments(self) -> None:
        """Load existing experiments from disk"""
        for config_file in self.experiments_dir.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                    config = ABTestConfig.from_dict(data)
                    self._active_experiments[config.experiment_name] = config
            except Exception as e:
                logger.error(f"Error loading experiment {config_file}: {e}")
