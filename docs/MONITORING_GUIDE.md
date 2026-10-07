# Evidently AI Drift Monitoring Integration Guide

This guide explains how to use the Evidently AI drift monitoring integration for the Tokyo Rent Predictor project.

## Overview

The monitoring system provides:

- **Data Drift Detection**: Monitors changes in input feature distributions
- **Prediction Drift**: Tracks changes in model output distributions
- **Data Quality**: Checks for missing values, outliers, and data issues
- **Model Performance**: Tracks prediction accuracy when actual values are available
- **Automated Logging**: All predictions are automatically logged for monitoring

## Installation

The integration is already set up! Just install dependencies:

```bash
cd api
uv sync  # or pip install -e .
```

This will install:
- `evidently>=0.4.0` - Drift monitoring library
- `plotly>=5.0.0` - Visualization dependency

## Architecture

### Core Components

1. **DriftMonitoringService** (`services/drift_monitoring_service.py`)
   - Generates drift reports
   - Runs automated drift tests
   - Compares current data against reference/baseline data

2. **DataCaptureService** (`services/data_capture_service.py`)
   - Logs all predictions automatically
   - Stores data in CSV format (one file per day)
   - Provides retrieval and filtering capabilities

3. **MonitoringReportsManager** (`services/monitoring_reports.py`)
   - High-level API for report generation
   - Coordinates drift monitoring and data capture
   - Provides comparison between models

4. **MonitoringDashboard** (`services/monitoring_dashboard.py`)
   - Creates time-series dashboards
   - Model comparison visualizations
   - Workspace management

### Data Flow

```
Prediction Request
    ↓
/predict or /challenger_predict endpoint
    ↓
Model makes prediction
    ↓
DataCaptureService logs: features + prediction + timestamp
    ↓
Data stored in ./prediction_logs/predictions_YYYYMMDD.csv
    ↓
Monitoring endpoints analyze logged data vs reference data
    ↓
Evidently generates reports in ./monitoring_reports/
```

## API Endpoints

### 1. Get Monitoring Summary

```bash
GET /monitoring/summary
```

Returns statistics about predictions and recent reports.

**Example:**
```bash
curl http://localhost:8000/monitoring/summary
```

**Response:**
```json
{
  "prediction_statistics": {
    "total_predictions": 1523,
    "predictions_with_actuals": 45,
    "models": {
      "passed": 821,
      "challenger": 702
    },
    "date_range": {
      "start": "2025-01-15",
      "end": "2025-01-22"
    }
  },
  "recent_reports": {
    "data_drift": {
      "path": "./monitoring_reports/data_drift_report_20250122_143052.html",
      "created": "2025-01-22T14:30:52",
      "size_kb": 245.3
    },
    "data_quality": {...},
    "model_performance": {...}
  }
}
```

### 2. Check for Data Drift

```bash
GET /monitoring/drift?days_back=7&model_type=challenger&drift_threshold=0.3
```

**Parameters:**
- `days_back` (default: 7): Number of days of prediction data to analyze
- `model_type` (optional): Filter by "passed" or "challenger"
- `drift_threshold` (default: 0.3): Maximum allowed share of drifted features

**Example:**
```bash
curl "http://localhost:8000/monitoring/drift?days_back=7&model_type=challenger"
```

**Response:**
```json
{
  "status": "success",
  "tests_passed": false,
  "drift_detected": true,
  "summary": {
    "total_features": 7,
    "drifted_features": 3,
    "drift_share": 0.43,
    "dataset_drift": true
  },
  "report_path": "./monitoring_reports/data_drift_report_20250122_143052.html",
  "timestamp": "2025-01-22T14:30:52"
}
```

### 3. Generate All Monitoring Reports

```bash
POST /monitoring/reports/generate
```

**Body:**
```json
{
  "days_back": 7,
  "model_type": "challenger"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/monitoring/reports/generate" \
  -H "Content-Type: application/json" \
  -d '{"days_back": 7}'
```

**Response:**
```json
{
  "status": "success",
  "reports": {
    "data_drift": "./monitoring_reports/data_drift_report_20250122_143052.html",
    "data_quality": "./monitoring_reports/data_quality_report_20250122_143053.html",
    "model_performance": null
  },
  "message": "Reports generated successfully"
}
```

Open the HTML files in your browser to view interactive reports!

### 4. Compare Models Drift

```bash
GET /monitoring/compare_models?days_back=7
```

**Example:**
```bash
curl "http://localhost:8000/monitoring/compare_models?days_back=7"
```

**Response:**
```json
{
  "status": "success",
  "comparison": {
    "passed_model": {
      "predictions": 821,
      "drift": {
        "total_features": 7,
        "drifted_features": 2,
        "drift_share": 0.29,
        "dataset_drift": false
      }
    },
    "challenger_model": {
      "predictions": 702,
      "drift": {
        "total_features": 7,
        "drifted_features": 3,
        "drift_share": 0.43,
        "dataset_drift": true
      }
    }
  },
  "days_analyzed": 7
}
```

### 5. Get Prediction Statistics

```bash
GET /monitoring/predictions/stats
```

**Example:**
```bash
curl http://localhost:8000/monitoring/predictions/stats
```

## Using the Python API Directly

### Example: Generate Drift Report

```python
from services.monitoring_reports import MonitoringReportsManager

# Initialize manager
manager = MonitoringReportsManager()

# Run drift detection
result = manager.run_drift_detection(
    days_back=7,
    model_type="challenger",
    drift_threshold=0.3
)

print(f"Drift detected: {result['drift_detected']}")
print(f"Report saved to: {result['report_path']}")
```

### Example: Access Prediction Logs

```python
from services.data_capture_service import DataCaptureService

# Initialize service
data_capture = DataCaptureService()

# Get recent predictions
recent = data_capture.get_recent_predictions(n=100)
print(f"Retrieved {len(recent)} predictions")

# Get predictions with actual values
with_actuals = data_capture.get_predictions_with_actuals()
print(f"Predictions with ground truth: {len(with_actuals)}")

# Get statistics
stats = data_capture.get_statistics()
print(stats)
```

### Example: Manual Report Generation

```python
from services.drift_monitoring_service import DriftMonitoringService, load_reference_data
import pandas as pd

# Load reference data (training data)
reference = load_reference_data()

# Initialize drift service
drift_service = DriftMonitoringService(reference_data=reference)

# Load current predictions (from logs or API)
current_data = pd.read_csv("prediction_logs/predictions_20250122.csv")

# Generate drift report
report, metrics = drift_service.generate_data_drift_report(
    current_data=current_data,
    save_html=True,
    save_json=True
)

print("Report generated!")
```

## Monitoring Workflow

### Daily Monitoring

1. **Check drift automatically:**
```bash
curl http://localhost:8000/monitoring/drift?days_back=1
```

2. **Review HTML reports:**
```bash
open monitoring_reports/data_drift_report_*.html
```

3. **Compare models:**
```bash
curl http://localhost:8000/monitoring/compare_models
```

### Weekly Deep Dive

```bash
# Generate comprehensive reports
curl -X POST http://localhost:8000/monitoring/reports/generate \
  -H "Content-Type: application/json" \
  -d '{"days_back": 7}'

# Check model comparison
curl http://localhost:8000/monitoring/compare_models?days_back=7
```

### Setting Up Alerts

You can set up automated alerts based on drift detection:

```python
import requests
from datetime import datetime

def check_drift_and_alert():
    response = requests.get("http://localhost:8000/monitoring/drift?days_back=1")
    result = response.json()

    if result["drift_detected"]:
        # Send alert (email, Slack, PagerDuty, etc.)
        print(f"ALERT: Drift detected!")
        print(f"Drifted features: {result['summary']['drifted_features']}")
        print(f"Drift share: {result['summary']['drift_share']:.2%}")

        # Could integrate with:
        # - Email (SMTP)
        # - Slack webhook
        # - PagerDuty
        # - CloudWatch alarms

check_drift_and_alert()
```

## Integration with MLflow

The monitoring system works alongside your existing MLflow integration:

```python
from services.mlflow_service import MLflowService
from services.monitoring_reports import MonitoringReportsManager

mlflow_service = MLflowService()
monitoring_manager = MonitoringReportsManager()

# Get current model metadata from MLflow
model_metadata = mlflow_service.get_production_model_metadata("tokyo_rent_lgbm")
print(f"Model version: {model_metadata['version']}")

# Check if current production model is drifting
drift_result = monitoring_manager.run_drift_detection(days_back=7)

if drift_result["drift_detected"]:
    print("Consider retraining the model!")
    # Trigger retraining via API
    # POST /retrain
```

## Directory Structure

```
api/
├── services/
│   ├── drift_monitoring_service.py      # Core drift monitoring
│   ├── data_capture_service.py          # Prediction logging
│   ├── monitoring_reports.py            # High-level report API
│   └── monitoring_dashboard.py          # Dashboard generation
├── prediction_logs/                     # Logged predictions (CSV files)
│   ├── predictions_20250121.csv
│   ├── predictions_20250122.csv
│   └── ...
├── monitoring_reports/                  # Generated reports (HTML/JSON)
│   ├── data_drift_report_20250122_143052.html
│   ├── data_quality_report_20250122_143053.html
│   └── ...
└── monitoring_workspace/                # Evidently workspace (dashboards)
    └── dashboard_data_20250122.json
```

## Configuration

### Change Storage Directories

Edit `model_api.py`:

```python
# Initialize services
data_capture_service = DataCaptureService(
    storage_dir="./custom_prediction_logs",  # Change here
    buffer_size=100
)

monitoring_manager = MonitoringReportsManager(
    reports_dir="./custom_reports",          # Change here
    predictions_dir="./custom_prediction_logs"
)
```

### Adjust Buffer Size

The data capture service buffers predictions before writing to disk:

```python
data_capture_service = DataCaptureService(
    buffer_size=50,      # Flush after 50 predictions (default: 100)
    auto_flush=True      # Automatically flush when buffer is full
)
```

### Set Custom Drift Thresholds

```python
# In your code or API calls
drift_result = manager.run_drift_detection(
    days_back=7,
    drift_threshold=0.2  # More strict (default: 0.3 = 30% of features)
)
```

## Best Practices

1. **Regular Monitoring**: Check drift at least weekly
2. **Baseline Updates**: Retrain and update reference data periodically
3. **Log Actuals**: Update predictions with actual values when available for performance tracking
4. **Alert Thresholds**: Start with 30% drift threshold, adjust based on your use case
5. **Report Review**: Review HTML reports in detail when drift is detected
6. **Model Comparison**: Always compare both models to choose the best performer

## Troubleshooting

### No predictions available for monitoring

**Issue**: API returns "No prediction data available"

**Solution**:
- Make sure you've made some predictions via `/predict` or `/challenger_predict`
- Check that `prediction_logs/` directory exists and has CSV files
- Verify buffer has been flushed: `data_capture_service.flush()`

### Reference data not loading

**Issue**: "Failed to load reference data"

**Solution**:
- Ensure you have internet connection (loads from Huggingface)
- Or provide local CSV: `load_reference_data("/path/to/training_data.csv")`

### Reports not generating

**Issue**: HTML reports are empty or error in generation

**Solution**:
- Need at least 10-20 predictions for meaningful reports
- Ensure prediction logs have all required feature columns
- Check logs for detailed error messages

## Advanced: Custom Monitoring Metrics

You can extend the monitoring system with custom metrics:

```python
from evidently.metrics import ColumnDriftMetric
from services.drift_monitoring_service import DriftMonitoringService

drift_service = DriftMonitoringService(reference_data=reference)

# Check drift for specific column
drift_info = drift_service.check_column_drift(
    current_data=current_predictions,
    column_name="sqr_m"
)

print(f"Drift detected for sqr_m: {drift_info['drift_detected']}")
print(f"Drift score: {drift_info['drift_score']}")
```

## Next Steps

1. **Automated Retraining**: Set up automated retraining when drift exceeds threshold
2. **Slack Integration**: Send drift alerts to Slack channel
3. **Performance Tracking**: Collect actual values and track MAE/RMSE over time
4. **A/B Testing**: Use drift monitoring to validate A/B test results
5. **CI/CD Integration**: Add drift checks to your deployment pipeline

## Resources

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Drift Detection Guide](https://www.evidentlyai.com/blog/ml-monitoring-data-drift-detection-tutorial)
- [MLOps Best Practices](https://ml-ops.org/)

## Support

For issues or questions:
- Check logs in the API output
- Review generated HTML reports for detailed insights
- Open an issue in the project repository
