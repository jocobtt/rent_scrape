"""
Data Lineage Service

Tracks dataset versions and their lineage through the training pipeline.
Integrates with MLflow to formally attach datasets to training runs using:
  - mlflow.log_input() / PandasDataset for formal dataset registration
  - SHA-256 content hashing for stable dataset fingerprinting
  - JSON manifests stored as MLflow artifacts for human inspection
  - Run tags (dataset.hash / dataset.name / dataset.source) for O(1) lookup

No new infrastructure required — runs on top of the existing MLflow SQLite setup.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Content-addressed store for versioned training datasets
DATASET_STORE_DIR = Path(os.path.dirname(__file__)).parent / "dataset_versions"


class DataLineageService:
    """Central service for dataset hashing, versioning, and MLflow lineage tracking."""

    def __init__(self, tracking_uri: Optional[str] = None):
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(uri)
        self.client = MlflowClient()
        DATASET_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Hashing                                                             #
    # ------------------------------------------------------------------ #

    def hash_dataframe(self, df: pd.DataFrame) -> str:
        """
        Compute a stable SHA-256 hash of a DataFrame.

        Columns are sorted before serialisation so the hash is invariant to
        column insertion order (HuggingFace vs CSV load paths may differ).

        Returns:
            64-character lowercase hex digest.
        """
        csv_bytes = (
            df.reindex(sorted(df.columns), axis=1)
            .to_csv(index=False)
            .encode("utf-8")
        )
        return hashlib.sha256(csv_bytes).hexdigest()

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_versioned_dataset(
        self,
        df: pd.DataFrame,
        source_description: str,
        dataset_hash: Optional[str] = None,
        kind: str = "snapshot",
    ) -> Dict[str, Any]:
        """
        Persist a DataFrame to a content-addressed directory with a manifest.

        Structure created::

            api/dataset_versions/
                {hash[:12]}/
                    data.csv       ← the training data
                    manifest.json  ← metadata

        Idempotent: if a directory for this hash already exists, the existing
        manifest is returned without overwriting anything.

        Args:
            df:                 DataFrame to persist.
            source_description: Human-readable label, e.g.
                                "HuggingFace jbrazzy/tokyo_rent split=train"
                                "Suumo scrape 2026-02-21 pages=0-50".
            dataset_hash:       Pre-computed hash (avoids double hashing).
            kind:               "scrape" for a cleaned scrape that training may use as
                                its input (see services.data_cleaning), or "snapshot"
                                (default) for lineage copies of training frames.

        Returns:
            Manifest dict with keys: hash, hash_short, created_at,
            source_description, n_rows, n_cols, columns, data_path,
            manifest_path.
        """
        if dataset_hash is None:
            dataset_hash = self.hash_dataframe(df)

        hash_short = dataset_hash[:12]
        version_dir = DATASET_STORE_DIR / hash_short
        manifest_path = version_dir / "manifest.json"
        data_path = version_dir / "data.csv"

        # Idempotent: return existing manifest if already stored
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)

        version_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(data_path, index=False)

        manifest = {
            "hash": dataset_hash,
            "hash_short": hash_short,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_description": source_description,
            "kind": kind,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "data_path": str(data_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Dataset persisted: hash=%s rows=%d path=%s",
            hash_short, len(df), data_path,
        )
        return manifest

    # ------------------------------------------------------------------ #
    #  MLflow integration                                                  #
    # ------------------------------------------------------------------ #

    def log_dataset_to_run(
        self,
        df: pd.DataFrame,
        run_id: str,
        source_description: str,
        source_type: str = "local",
        dataset_name: str = "tokyo_rent_training",
        context: str = "training",
        dataset_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Attach a dataset to an MLflow run using mlflow.log_input() and set tags.

        Performs the following steps in order:
          1. Hash the DataFrame (or accept pre-computed hash).
          2. Persist to content-addressed storage (idempotent).
          3. Build the appropriate DatasetSource subclass.
          4. Create a PandasDataset and call mlflow.log_input().
          5. Log the manifest JSON as an MLflow artifact.
          6. Set three run tags via MlflowClient.

        Args:
            df:                 Full training DataFrame (post-cleaning, pre-split).
            run_id:             Active MLflow run ID.
            source_description: Human-readable source label.
            source_type:        "huggingface" | "local" | "scraped".
            dataset_name:       Name registered in MLflow for this dataset.
            context:            MLflow input context ("training" or "evaluation").
            dataset_hash:       Pre-computed hash.

        Returns:
            Manifest dict.
        """
        if dataset_hash is None:
            dataset_hash = self.hash_dataframe(df)

        manifest = self.save_versioned_dataset(df, source_description, dataset_hash)

        # Build appropriate DatasetSource
        try:
            source = self._build_source(source_type, source_description, manifest["data_path"])
        except Exception as e:
            logger.warning("Could not build DatasetSource (%s); using None: %s", source_type, e)
            source = None

        # Log via mlflow.log_input inside the run
        try:
            from mlflow.data.pandas_dataset import PandasDataset

            dataset_obj = PandasDataset(
                df=df,
                source=source,
                name=dataset_name,
                digest=dataset_hash[:8],
            )

            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_input(dataset_obj, context=context)
                mlflow.log_artifact(manifest["manifest_path"])

        except Exception as e:
            logger.warning("mlflow.log_input failed (non-fatal): %s", e)

        # Set searchable tags on the run regardless of whether log_input succeeded
        for tag_key, tag_val in [
            ("dataset.hash", dataset_hash),
            ("dataset.name", dataset_name),
            ("dataset.source", source_description),
        ]:
            try:
                self.client.set_tag(run_id, tag_key, tag_val)
            except Exception as e:
                logger.warning("Failed to set tag %s: %s", tag_key, e)

        logger.info(
            "Dataset lineage logged to run %s (hash=%s)", run_id, dataset_hash[:12]
        )
        return manifest

    def _build_source(self, source_type: str, source_description: str, data_path: str):
        """Build the appropriate MLflow DatasetSource for the given type."""
        if source_type == "huggingface":
            try:
                from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
                return HuggingFaceDatasetSource(path="jbrazzy/tokyo_rent", split="train")
            except ImportError:
                pass

        # For local/scraped files use HTTP source with file:// URI
        try:
            from mlflow.data.http_dataset_source import HTTPDatasetSource
            return HTTPDatasetSource(url=f"file://{data_path}")
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------ #
    #  Lineage queries                                                     #
    # ------------------------------------------------------------------ #

    def get_lineage_for_model(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve dataset lineage for a registered model version.

        Looks up the MLflow run linked to the model version, reads the three
        dataset tags, and loads the manifest artifact for row/column counts.

        Args:
            model_name: Registered model name, e.g. "tokyo_rent_lgbm".
            version:    Specific version string. If None, uses the latest
                        Production-stage version (falls back to latest any-stage).

        Returns:
            Dict with keys: model_name, model_version, run_id, stage, dataset, inputs.
            dataset is None for models trained before this feature was added.

        Raises:
            ValueError: If no versions of the model exist in the registry.
        """
        model_version = self._resolve_model_version(model_name, version)

        run = self.client.get_run(model_version.run_id)
        tags = run.data.tags

        dataset_hash = tags.get("dataset.hash")

        dataset_info = None
        if dataset_hash:
            dataset_info = {
                "hash": dataset_hash,
                "hash_short": dataset_hash[:12],
                "name": tags.get("dataset.name", "unknown"),
                "source": tags.get("dataset.source", "unknown"),
                "n_rows": None,
                "n_cols": None,
                "columns": [],
                "created_at": None,
                "manifest_available": False,
            }

            # Enrich with manifest if available
            manifest = self._load_manifest_for_hash(dataset_hash)
            if manifest:
                dataset_info.update({
                    "n_rows": manifest.get("n_rows"),
                    "n_cols": manifest.get("n_cols"),
                    "columns": manifest.get("columns", []),
                    "created_at": manifest.get("created_at"),
                    "manifest_available": True,
                })

        # Collect raw DatasetInput objects
        inputs = []
        try:
            run_inputs = run.inputs.dataset_inputs
            for di in run_inputs:
                inputs.append({
                    "name": di.dataset.name if di.dataset else None,
                    "digest": di.dataset.digest if di.dataset else None,
                    "source_type": di.dataset.source_type if di.dataset else None,
                    "context": [t.key for t in di.tags] if di.tags else [],
                })
        except Exception:
            pass

        return {
            "model_name": model_name,
            "model_version": model_version.version,
            "run_id": model_version.run_id,
            "stage": model_version.current_stage,
            "dataset": dataset_info,
            "inputs": inputs,
        }

    def compare_model_datasets(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """
        Compare the datasets used to train two versions of a model.

        Args:
            model_name: Registered model name.
            version_a, version_b: Version strings to compare.

        Returns:
            Dict with same_dataset bool, hash comparison, and diff details.
        """
        lineage_a = self.get_lineage_for_model(model_name, version_a)
        lineage_b = self.get_lineage_for_model(model_name, version_b)

        ds_a = lineage_a.get("dataset") or {}
        ds_b = lineage_b.get("dataset") or {}

        hash_a = ds_a.get("hash")
        hash_b = ds_b.get("hash")

        same_dataset = bool(hash_a and hash_b and hash_a == hash_b)

        cols_a = set(ds_a.get("columns") or [])
        cols_b = set(ds_b.get("columns") or [])

        rows_a = ds_a.get("n_rows")
        rows_b = ds_b.get("n_rows")

        diff = {
            "n_rows_delta": (rows_b - rows_a) if (rows_a is not None and rows_b is not None) else None,
            "n_cols_delta": (ds_b.get("n_cols", 0) - ds_a.get("n_cols", 0)) if ds_a.get("n_cols") else None,
            "columns_added": sorted(cols_b - cols_a),
            "columns_removed": sorted(cols_a - cols_b),
            "source_changed": ds_a.get("source") != ds_b.get("source"),
            "date_a": ds_a.get("created_at"),
            "date_b": ds_b.get("created_at"),
        }

        return {
            "model_name": model_name,
            "version_a": version_a,
            "version_b": version_b,
            "same_dataset": same_dataset,
            "hash_a": hash_a,
            "hash_b": hash_b,
            "diff": diff,
            "lineage_a": lineage_a,
            "lineage_b": lineage_b,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_model_version(self, model_name: str, version: Optional[str]):
        """Return the MlflowClient model version object for the given name/version."""
        if version is not None:
            return self.client.get_model_version(model_name, version)

        # Prefer Production stage, fall back to any latest version
        prod = self.client.get_latest_versions(model_name, stages=["Production"])
        if prod:
            return prod[0]

        all_versions = self.client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            raise ValueError(f"No versions found for model '{model_name}'")

        return max(all_versions, key=lambda v: int(v.version))

    def _load_manifest_for_hash(self, dataset_hash: str) -> Optional[Dict[str, Any]]:
        """Load manifest.json from the content-addressed store for a given hash."""
        hash_short = dataset_hash[:12]
        manifest_path = DATASET_STORE_DIR / hash_short / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.debug("Could not read manifest for %s: %s", hash_short, e)
        return None
