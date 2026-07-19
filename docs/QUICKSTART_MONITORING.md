# Quick Start: Evidently AI Monitoring

Get started with drift monitoring in 5 minutes!

## Prerequisites

- API running on `http://localhost:8000`
- Python 3.12+ with dependencies installed

## Step 1: Start the API

```bash
cd api
python model_api.py
```

## Step 2: Run the Demo Script

```bash
python example_monitoring.py
```

This will:
1. Make sample predictions (automatically logged)
2. Show prediction statistics
3. Check for data drift
4. Generate monitoring reports

## Step 3: View Reports

Open the generated HTML reports in your browser:

```bash
open monitoring_reports/data_drift_report_*.html
```

## Step 4: Try the API Directly

### Check Monitoring Summary

```bash
curl http://localhost:8000/monitoring/summary | jq
```

### Check for Drift

```bash
curl "http://localhost:8000/monitoring/drift?days_back=7" | jq
```

### Generate Reports

```bash
curl -X POST "http://localhost:8000/monitoring/reports/generate" \
  -H "Content-Type: application/json" \
  -d '{"days_back": 7}' | jq
```

### Compare Models

```bash
curl "http://localhost:8000/monitoring/compare_models?days_back=7" | jq
```

## Step 5: Make More Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sqr_m": 30.0,
    "rei_price": 60000,
    "shikikin": 120000,
    "maintenence_price": 6000,
    "year_built": 8,
    "floor": 4,
    "eki_walk": 6
  }' | jq
```

All predictions are automatically logged for monitoring!

## Understanding the Reports

### Data Drift Report
- Shows which features have drifted
- Compares current vs training data distributions
- Provides drift scores for each feature

### Data Quality Report
- Checks for missing values
- Detects outliers
- Validates data types

### Model Performance Report
- Tracks prediction accuracy (requires actual values)
- Shows MAE, RMSE, MAPE metrics
- Highlights performance degradation

## Key Directories

```
api/
├── prediction_logs/          # Logged predictions (CSV)
├── monitoring_reports/       # Generated reports (HTML/JSON)
└── monitoring_workspace/     # Dashboard data
```

## What Gets Logged?

Every prediction automatically logs:
- Timestamp
- Model type (passed/challenger)
- All 7 input features
- Prediction value
- Actual value (when available)

## Monitoring Workflow

### Daily
```bash
# Quick drift check
curl "http://localhost:8000/monitoring/drift?days_back=1"
```

### Weekly
```bash
# Generate comprehensive reports
curl -X POST "http://localhost:8000/monitoring/reports/generate" \
  -d '{"days_back": 7}'

# Compare models
curl "http://localhost:8000/monitoring/compare_models?days_back=7"
```

## Setting Drift Thresholds

```bash
# Stricter threshold (20% of features can drift)
curl "http://localhost:8000/monitoring/drift?drift_threshold=0.2"

# More lenient (50% of features can drift)
curl "http://localhost:8000/monitoring/drift?drift_threshold=0.5"
```

## Troubleshooting

**"No prediction data available"**
- Make some predictions first via `/predict` or `/challenger_predict`

**"Not enough data for dashboard"**
- Need at least 10-20 predictions per day
- Run `python example_monitoring.py` to generate sample data

**Reports not opening**
- Check the path in the API response
- Use absolute path: `open /full/path/to/report.html`

## Next Steps

1. Read the full guide: [MONITORING_GUIDE.md](./MONITORING_GUIDE.md)
2. Set up automated drift alerts
3. Integrate with your CI/CD pipeline
4. Add Slack/email notifications

## Getting Help

- Full documentation: `MONITORING_GUIDE.md`
- Evidently docs: https://docs.evidentlyai.com/
- Project issues: GitHub repository
