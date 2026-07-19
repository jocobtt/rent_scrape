"""
MLflow service for model registry and metadata management.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np

logger = logging.getLogger(__name__)


class MLflowService:
    """Service class for MLflow model registry and metadata operations."""

    def __init__(self, tracking_uri: str = None):
        """
        Initialize MLflow service.

        Args:
            tracking_uri: MLflow tracking URI (defaults to env variable or local)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        else:
            # Default to local tracking
            mlflow.set_tracking_uri("file:///tmp/mlruns")

        self.client = MlflowClient()
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    def get_production_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for the production version of a model.

        Args:
            model_name: Name of the registered model

        Returns:
            Dict containing model metadata or None if not found
        """
        try:
            # Get the latest production model version
            versions = self.client.get_latest_versions(model_name, stages=["Production"])

            if not versions:
                # If no Production stage, try None stage (default)
                versions = self.client.get_latest_versions(model_name, stages=["None"])

            if not versions:
                logger.warning(f"No model version found for {model_name}")
                return None

            model_version = versions[0]
            return self._build_model_metadata(model_version, model_name)

        except MlflowException as e:
            logger.error(f"MLflow error getting model metadata: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting model metadata: {str(e)}")
            return None

    def get_model_metadata_by_version(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific version of a model.

        Args:
            model_name: Name of the registered model
            version: Version number

        Returns:
            Dict containing model metadata or None if not found
        """
        try:
            model_version = self.client.get_model_version(model_name, version)
            return self._build_model_metadata(model_version, model_name)

        except MlflowException as e:
            logger.error(f"MLflow error getting model version: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting model version: {str(e)}")
            return None

    def _build_model_metadata(self, model_version, model_name: str) -> Dict[str, Any]:
        """
        Build metadata dictionary from model version.

        Args:
            model_version: MLflow model version object
            model_name: Name of the model

        Returns:
            Dict containing formatted model metadata
        """
        try:
            # Get the run information
            run = self.client.get_run(model_version.run_id)

            # Extract metrics
            metrics = {}
            for key, value in run.data.metrics.items():
                metrics[key] = value

            # Calculate additional metrics if needed
            if 'rmse' in metrics and 'mae' not in metrics:
                # Some models might only log certain metrics
                pass

            # Extract parameters
            params = dict(run.data.params)

            # Convert string params to appropriate types
            typed_params = self._convert_param_types(params)

            # Get feature importance if available
            feature_importance = self._get_feature_importance(model_version.run_id)

            # Map stage to status
            stage_mapping = {
                "Production": "active",
                "Staging": "staging",
                "Archived": "archived",
                "None": "active"  # Default to active if no stage set
            }
            status = stage_mapping.get(model_version.current_stage, "active")

            # Format timestamp
            last_updated = datetime.fromtimestamp(
                model_version.last_updated_timestamp / 1000
            ).isoformat() + "Z"

            # Build response
            metadata = {
                "name": self._get_display_name(model_name),
                "version": str(model_version.version),
                "run_id": model_version.run_id,
                "status": status,
                "last_updated": last_updated,
                "description": model_version.description or f"ML model for Tokyo rent prediction ({model_name})",
                "metrics": metrics,
                "params": typed_params,
                "feature_importance": feature_importance,
                "model_name": model_name,  # Original model name for reference
                "stage": model_version.current_stage,
                "dataset_lineage": self._get_dataset_lineage(model_version.run_id),
            }

            return metadata

        except Exception as e:
            logger.error(f"Error building model metadata: {str(e)}")
            raise

    def _get_dataset_lineage(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset lineage for a run from tags and manifest artifact.

        Returns None if the run predates lineage tracking.
        """
        try:
            run = self.client.get_run(run_id)
            tags = run.data.tags
            dataset_hash = tags.get("dataset.hash")
            if not dataset_hash:
                return None

            lineage = {
                "hash": dataset_hash,
                "name": tags.get("dataset.name"),
                "source": tags.get("dataset.source"),
            }

            # Try to enrich with manifest artifact
            try:
                manifest_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/dataset_manifest.json"
                )
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                lineage.update({
                    "n_rows": manifest.get("n_rows"),
                    "n_cols": manifest.get("n_cols"),
                    "columns": manifest.get("columns"),
                    "created_at": manifest.get("created_at"),
                })
            except Exception:
                pass

            return lineage

        except Exception as e:
            logger.debug(f"Could not load dataset lineage for run {run_id}: {e}")
            return None

    def _get_feature_importance(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get feature importance from MLflow artifacts.

        Args:
            run_id: MLflow run ID

        Returns:
            List of dicts with feature names and importance scores
        """
        try:
            # Try to download feature importance artifact
            artifact_path = f"runs:/{run_id}/feature_importance.json"
            local_path = mlflow.artifacts.download_artifacts(artifact_path)

            with open(local_path, 'r') as f:
                feature_importance = json.load(f)

            return feature_importance

        except Exception as e:
            logger.debug(f"Could not load feature importance: {e}")
            # Return empty list if not available
            return []

    def _convert_param_types(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Convert string parameters to appropriate types.

        Args:
            params: Dict of string parameters

        Returns:
            Dict with properly typed parameters
        """
        typed_params = {}

        for key, value in params.items():
            # Try to convert to int
            try:
                typed_params[key] = int(value)
                continue
            except (ValueError, TypeError):
                pass

            # Try to convert to float
            try:
                typed_params[key] = float(value)
                continue
            except (ValueError, TypeError):
                pass

            # Try to convert to bool
            if value.lower() in ['true', 'false']:
                typed_params[key] = value.lower() == 'true'
                continue

            # Keep as string
            typed_params[key] = value

        return typed_params

    def _get_display_name(self, model_name: str) -> str:
        """
        Convert model name to display-friendly format.

        Args:
            model_name: Original model name

        Returns:
            Display-friendly name
        """
        name_mapping = {
            "tokyo_rent_lgbm": "Tokyo Rent Predictor (LightGBM)",
            "tokyo_passed_rent_model": "Tokyo Rent Predictor (Linear)",
        }

        return name_mapping.get(model_name, model_name.replace("_", " ").title())

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> bool:
        """
        Transition a model version to a different stage.

        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Target stage (Production, Staging, Archived)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True  # Archive previous production versions
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True

        except Exception as e:
            logger.error(f"Error transitioning model stage: {str(e)}")
            return False

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of registered model information
        """
        try:
            models = self.client.search_registered_models()

            model_list = []
            for model in models:
                model_info = {
                    "name": model.name,
                    "latest_version": max([int(v.version) for v in model.latest_versions]) if model.latest_versions else None,
                    "description": model.description,
                    "tags": dict(model.tags) if hasattr(model, 'tags') else {}
                }
                model_list.append(model_info)

            return model_list

        except Exception as e:
            logger.error(f"Error listing registered models: {str(e)}")
            return []

    def get_all_active_models_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for all active models in the system.

        Returns:
            Dict containing metadata for challenger and passed models
        """
        result = {
            "challenger": None,
            "passed": None
        }

        # Try to get challenger model (LightGBM)
        try:
            challenger_metadata = self.get_production_model_metadata("tokyo_rent_lgbm")
            result["challenger"] = challenger_metadata
        except Exception as e:
            logger.error(f"Could not get challenger model metadata: {e}")

        # Try to get passed model (Linear)
        try:
            passed_metadata = self.get_production_model_metadata("tokyo_passed_rent_model")
            result["passed"] = passed_metadata
        except Exception as e:
            logger.error(f"Could not get passed model metadata: {e}")

        return result

    def log_feature_importance(self, run_id: str, feature_names: List[str],
                              importance_values: List[float]) -> bool:
        """
        Log feature importance to MLflow.

        Args:
            run_id: MLflow run ID
            feature_names: List of feature names
            importance_values: List of importance values

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create feature importance list
            feature_importance = [
                {"name": name, "importance": float(value)}
                for name, value in zip(feature_names, importance_values)
            ]

            # Sort by importance (descending)
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            # Save to temporary file
            temp_file = f"/tmp/feature_importance_{run_id}.json"
            with open(temp_file, 'w') as f:
                json.dump(feature_importance, f, indent=2)

            # Log as artifact
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(temp_file, "feature_importance.json")

            # Clean up
            os.remove(temp_file)

            logger.info(f"Logged feature importance for run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Error logging feature importance: {str(e)}")
            return False
