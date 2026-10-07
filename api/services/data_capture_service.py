"""
Data Capture Service for Prediction Logging

This service logs predictions and input features to enable drift monitoring and
model performance analysis over time.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PredictionLog(BaseModel):
    """Schema for a single prediction log entry"""

    timestamp: str
    model_type: str
    dataset_version: Optional[str] = None  # Short SHA-256 hash (12 chars) of training dataset
    sqr_m: float
    rei_price: Optional[float] = None
    shikikin: Optional[float] = None
    maintenence_price: float
    year_built: float
    floor: float
    eki_walk: float
    prediction: float
    actual_value: Optional[float] = None


class DataCaptureService:
    """Service for capturing and storing prediction data"""

    def __init__(
        self,
        storage_dir: str = "./prediction_logs",
        buffer_size: int = 100,
        auto_flush: bool = True,
    ):
        """
        Initialize the data capture service

        Args:
            storage_dir: Directory to store prediction logs
            buffer_size: Number of predictions to buffer before flushing to disk
            auto_flush: Whether to automatically flush buffer when size is reached
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.auto_flush = auto_flush

        # In-memory buffer for predictions
        self._buffer: List[Dict] = []

        # Current log file path
        self.current_log_file = self._get_current_log_file()

        logger.info(
            f"Initialized DataCaptureService with storage at {self.storage_dir}"
        )

    def log_prediction(
        self,
        model_type: str,
        input_features: Dict[str, float],
        prediction: float,
        actual_value: Optional[float] = None,
        dataset_version: Optional[str] = None,
    ) -> None:
        """
        Log a single prediction

        Args:
            model_type: Type of model used (passed, challenger, etc.)
            input_features: Dictionary of input features
            prediction: Model prediction
            actual_value: Actual target value if available
            dataset_version: Short SHA-256 hash (12 chars) of the training
                             dataset used to train the serving model. Enables
                             drift analysis segmented by training data version.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "dataset_version": dataset_version,
            "sqr_m": input_features.get("sqr_m"),
            "rei_price": input_features.get("rei_price"),
            "shikikin": input_features.get("shikikin"),
            "maintenence_price": input_features.get("maintenence_price"),
            "year_built": input_features.get("year_built"),
            "floor": input_features.get("floor"),
            "eki_walk": input_features.get("eki_walk"),
            "prediction": prediction,
            "actual_value": actual_value,
        }

        self._buffer.append(log_entry)

        # Auto-flush if buffer is full
        if self.auto_flush and len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> int:
        """
        Flush buffered predictions to disk

        Returns:
            Number of predictions flushed
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)

        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self._buffer)

            # Append to current log file
            if self.current_log_file.exists():
                # Append to existing file
                existing_df = pd.read_csv(self.current_log_file)
                df = pd.concat([existing_df, df], ignore_index=True)

            # Save to CSV
            df.to_csv(self.current_log_file, index=False)

            logger.info(f"Flushed {count} predictions to {self.current_log_file}")

            # Clear buffer
            self._buffer.clear()

            return count

        except Exception as e:
            logger.error(f"Error flushing predictions: {e}")
            raise

    def get_recent_predictions(
        self,
        n: Optional[int] = None,
        model_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve recent predictions

        Args:
            n: Number of most recent predictions to return
            model_type: Filter by model type (passed, challenger, etc.)
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)

        Returns:
            DataFrame of predictions
        """
        # Flush current buffer first
        if self._buffer:
            self.flush()

        # Load all prediction logs
        df = self._load_all_logs()

        if df.empty:
            logger.warning("No prediction logs found")
            return df

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Apply filters
        if model_type:
            df = df[df["model_type"] == model_type]

        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df["timestamp"] >= start]

        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df["timestamp"] <= end]

        # Sort by timestamp descending
        df = df.sort_values("timestamp", ascending=False)

        # Limit to n most recent
        if n:
            df = df.head(n)

        logger.info(f"Retrieved {len(df)} predictions")
        return df

    def get_predictions_with_actuals(self) -> pd.DataFrame:
        """
        Get only predictions that have actual values recorded

        Returns:
            DataFrame of predictions with actual values
        """
        df = self.get_recent_predictions()

        if df.empty:
            return df

        # Filter to only rows with actual values
        df = df[df["actual_value"].notna()]

        logger.info(f"Found {len(df)} predictions with actual values")
        return df

    def update_actual_value(
        self,
        timestamp: str,
        actual_value: float,
    ) -> bool:
        """
        Update a prediction log with the actual value

        Args:
            timestamp: Timestamp of the prediction to update
            actual_value: Actual target value

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Flush buffer first
            if self._buffer:
                self.flush()

            # Load all logs
            df = self._load_all_logs()

            if df.empty:
                logger.warning("No prediction logs found")
                return False

            # Find the prediction by timestamp
            mask = df["timestamp"] == timestamp

            if not mask.any():
                logger.warning(f"No prediction found with timestamp {timestamp}")
                return False

            # Update actual value
            df.loc[mask, "actual_value"] = actual_value

            # Save back to file
            # Determine which file the prediction belongs to
            log_file = self._find_log_file_for_timestamp(timestamp)
            if log_file:
                df_file = df[df["timestamp"] == timestamp]
                df_file.to_csv(log_file, index=False)
                logger.info(f"Updated actual value for prediction at {timestamp}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating actual value: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Get statistics about captured predictions

        Returns:
            Dictionary with statistics
        """
        df = self._load_all_logs()

        if df.empty:
            return {
                "total_predictions": 0,
                "predictions_with_actuals": 0,
                "models": {},
            }

        stats = {
            "total_predictions": len(df),
            "predictions_with_actuals": df["actual_value"].notna().sum(),
            "models": df["model_type"].value_counts().to_dict(),
            "date_range": {
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
            },
        }

        return stats

    def _load_all_logs(self) -> pd.DataFrame:
        """Load all prediction log files into a single DataFrame"""
        log_files = list(self.storage_dir.glob("predictions_*.csv"))

        if not log_files:
            return pd.DataFrame()

        dfs = []
        for file in log_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load log file {file}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def _get_current_log_file(self) -> Path:
        """Get the current log file path (one per day)"""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.storage_dir / f"predictions_{date_str}.csv"

    def _find_log_file_for_timestamp(self, timestamp: str) -> Optional[Path]:
        """Find which log file contains a given timestamp"""
        try:
            dt = pd.to_datetime(timestamp)
            date_str = dt.strftime("%Y%m%d")
            log_file = self.storage_dir / f"predictions_{date_str}.csv"

            if log_file.exists():
                return log_file
        except Exception as e:
            logger.error(f"Error finding log file for timestamp {timestamp}: {e}")

        return None

    def rotate_logs(self, days_to_keep: int = 30) -> int:
        """
        Remove old log files

        Args:
            days_to_keep: Number of days of logs to keep

        Returns:
            Number of files deleted
        """
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)
        deleted_count = 0

        for log_file in self.storage_dir.glob("predictions_*.csv"):
            if log_file.stat().st_mtime < cutoff_date:
                log_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted old log file: {log_file}")

        logger.info(f"Rotated logs: deleted {deleted_count} files")
        return deleted_count
