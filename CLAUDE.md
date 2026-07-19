# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tokyo Apartment Rent Predictor - A machine learning system that predicts rental prices for apartments in Tokyo using data scraped from Suumo. The project features a complete MLOps pipeline with automated model promotion, A/B testing, drift monitoring, and model registry management.

**Key Technologies:** FastAPI, LightGBM, scikit-learn, PyTorch, TabPFN, TabNet, MLflow, Evidently AI, Docker

## Development Commands

### API Development

```bash
# Navigate to API directory first
cd api

# Install dependencies
uv sync

# Start the API server (http://localhost:8000)
./start_api.sh
# or manually:
uv run model_api.py

# Run tests (starts/stops API automatically)
./run_tests.sh

# View API documentation (after starting server)
# Open http://localhost:8000/docs
```

### Model Training

```bash
cd api

# Train LightGBM challenger model
# (Auto-promotes to production if metrics meet criteria)
python models/challenger_model.py

# Train linear regression passed model
# (Auto-promotes to production if metrics meet criteria)
python models/reg_model.py

# Train pretrained tabular deep learning models
# (TabPFN and TabNet - state-of-the-art for tabular data)
python models/tabular_pretrained_model.py --model all  # Train both
python models/tabular_pretrained_model.py --model tabpfn  # Only TabPFN
python models/tabular_pretrained_model.py --model tabnet  # Only TabNet
```

### MLflow & Monitoring

```bash
# Start entire stack with Docker (MLflow + API)
docker-compose up -d

# Access MLflow UI at http://localhost:5001
# API will be at http://localhost:8000

# Run scheduled tasks (promotion checks, drift monitoring)
cd api
nohup python scheduled_tasks.py > scheduled_tasks.log 2>&1 &

# Monitor scheduled task logs
tail -f api/scheduled_tasks.log
```

### Testing MLflow Integration

```bash
cd api
python test_mlflow_integration.py
```

## Architecture

### High-Level System Flow

```
Data Scraping (Suumo) → Model Training → MLflow Registry
                                              ↓
                            Auto-Promotion Service (metric checks)
                                              ↓
                            Production Deployment (FastAPI)
                                              ↓
                            Drift Monitoring (Evidently AI)
                                              ↓
                            Auto-Retrain Service (if drift detected)
```

### Critical Service Relationships

**Auto-Promotion Pipeline:**
- Training scripts (`models/challenger_model.py`, `models/reg_model.py`) → Auto-check promotion criteria after training
- `services/auto_promotion_service.py` → Validates metrics (RMSE < 17500, R² > 0.88, MAPE < 9%) + compares with production model (requires ≥2% improvement)
- `services/auto_retrain_service.py` → Monitors drift/performance degradation → Triggers retraining → Checks promotion
- `scheduled_tasks.py` → Runs automated promotion checks (daily 2AM), drift checks (hourly), and monitoring reports (daily 3AM)

**A/B Testing Flow:**
- `services/ab_testing_service.py` → Manages experiments, traffic splitting, statistical analysis (t-tests, Mann-Whitney U, Cohen's d)
- `services/ab_mlflow_integration.py` → Logs A/B test results to MLflow, auto-promotes winner to Production stage
- Traffic assignment supports: percentage-based, hash-based (consistent user experience), and shadow mode

**Drift Monitoring Flow:**
- `services/data_capture_service.py` → Logs ALL predictions to CSV (buffered writes, one file per day)
- `services/drift_monitoring_service.py` → Uses Evidently AI to detect data drift, prediction drift, data quality issues
- `services/monitoring_reports.py` → Orchestrates report generation (HTML + JSON outputs)
- Auto-retrain triggered when drift exceeds thresholds OR production model performance degrades

**Model Registry:**

- Models registered in MLflow with names:
  - `tokyo_rent_lgbm` (LightGBM)
  - `tokyo_passed_rent_model` (Linear/Ridge/Lasso)
  - `tokyo_rent_tabpfn` (TabPFN - pretrained for tabular data)
  - `tokyo_rent_tabnet` (TabNet - Google's interpretable architecture)
  - `tokyo_rent_torch` (Custom PyTorch neural networks)
- Stages: None → Staging → Production → Archived
- Auto-promotion transitions directly from None → Production (if criteria met)

### Directory Structure

```
api/
├── model_api.py              # Main FastAPI app (40+ endpoints)
├── scheduled_tasks.py        # Automated tasks (promotion, drift, monitoring)
├── models/                   # Model training scripts & artifacts
│   ├── challenger_model.py   # LightGBM training + auto-promotion
│   ├── reg_model.py          # Linear/Ridge/Lasso training + auto-promotion
│   ├── tabular_pretrained_model.py  # TabPFN & TabNet training + auto-promotion
│   ├── torch_model.py        # Custom PyTorch neural network architectures
│   ├── transformer_model.py  # Transformer-based models for tabular data
│   ├── challenger-model.joblib
│   └── passed-model.joblib
├── services/                 # Business logic layer
│   ├── auto_promotion_service.py   # Metric-based promotion decisions
│   ├── auto_retrain_service.py     # Drift-triggered retraining
│   ├── ab_testing_service.py       # A/B test management & statistics
│   ├── ab_mlflow_integration.py    # A/B test MLflow logging
│   ├── drift_monitoring_service.py # Evidently AI drift detection
│   ├── data_capture_service.py     # Prediction logging
│   ├── monitoring_reports.py       # Report orchestration
│   ├── monitoring_dashboard.py     # Time-series visualizations
│   ├── mlflow_service.py           # MLflow registry operations
│   ├── prediction_service.py       # Prediction logic
│   └── training_service.py         # Model training orchestration
├── utils/
│   └── scrape_data.py        # Suumo web scraping
└── tests/
    └── test_api.py

data_scrape/                  # Standalone scraping scripts
tf/                          # Terraform (GCP deployment)
```

## Model Features

**Input Schema (all predictions):**
- `sqr_m`: Apartment size in square meters
- `rei_price`: Key money (礼金) in 万円 (10,000 yen units)
- `shikikin`: Security deposit (敷金) in 万円
- `maintenence_price`: Monthly maintenance fee in 万円
- `year_built`: Years since construction
- `floor`: Floor number
- `eki_walk`: Walking time to nearest station (minutes)

**Output:** Predicted monthly rent in 万円

## MLflow Integration Details

**Tracking URI:**
- Docker: `http://mlflow:5000` (internal network)
- Local: `http://localhost:5001` (external access)
- Legacy code may reference `file://s3_path` (needs updating)

**What Gets Logged (Both Models):**
- Metrics: RMSE, MAE, R², MAPE, MSE
- Parameters: All training hyperparameters
- Artifacts: Trained model, feature_importance.json
- Tags: model_type, training_date
- Model description with key metrics

**Auto-Promotion Criteria:**
```python
# LightGBM (tokyo_rent_lgbm)
RMSE < 17,500
R² > 0.88
MAPE < 9.0%
Improvement over production ≥ 2%

# Linear models (tokyo_passed_rent_model)
RMSE < 20,000
R² > 0.85
MAPE < 10.0%
Improvement over production ≥ 2%

# Deep Learning Models (Pretrained Tabular)

# TabPFN (tokyo_rent_tabpfn) - Pretrained for tabular data
RMSE < 15,000  # Higher expectations for pretrained models
R² > 0.90
MAPE < 8.0%
Improvement over production ≥ 2%

# TabNet (tokyo_rent_tabnet) - Google's interpretable architecture
RMSE < 16,000
R² > 0.89
MAPE < 8.5%
Improvement over production ≥ 2%

# Custom PyTorch (tokyo_rent_torch)
RMSE < 16,000
R² > 0.90
MAPE < 8.0%
Improvement over production ≥ 2%
```

## Important API Endpoints

### Core Prediction
- `POST /predict` - Production model prediction
- `POST /challenger_predict` - Challenger model prediction
- `POST /compare_predictions` - Compare both models

### Model Management
- `POST /retrain` - Retrain models with fresh scraped data
- `POST /models/{model_name}/promote/{version}` - Manual promotion
- `POST /models/check-promotion` - Check all models for promotion eligibility
- `POST /models/{model_name}/rollback` - Rollback to previous version
- `GET /models/{model_name}/promotion-status` - Get promotion status

### A/B Testing
- `POST /ab-test/create` - Create experiment
- `POST /ab-test/predict` - Get prediction with traffic splitting
- `POST /ab-test/analyze/{experiment_name}` - Analyze results (statistical tests)
- `POST /ab-test/promote-winner/{experiment_name}` - Promote winning model

### Monitoring & Drift
- `POST /monitoring/generate-drift-report` - Generate drift report
- `GET /monitoring/drift-summary` - Get drift detection summary
- `POST /retrain/check-and-execute` - Check drift and retrain if needed
- `POST /retrain/force` - Force immediate retraining

### MLflow Metadata
- `GET /model/metadata?model_name=tokyo_rent_lgbm` - Get model metadata
- `GET /model/metadata/all` - Get all models metadata

## Development Workflows

### Adding New Model Features

1. Update training scripts in `models/` with new feature engineering
2. Retrain both models: `python models/challenger_model.py && python models/reg_model.py`
3. Models auto-promote if they meet criteria (RMSE, R², MAPE, improvement %)
4. Check MLflow UI to verify registration: http://localhost:5001
5. Use `/compare_predictions` endpoint to test both models

### Running A/B Tests

1. Create experiment: `POST /ab-test/create` with control/treatment models and traffic split
2. Route production traffic through: `POST /ab-test/predict`
3. Let it run (minimum samples required for statistical significance)
4. Analyze results: `POST /ab-test/analyze/{experiment_name}` (gets t-test, Cohen's d, confidence intervals)
5. Promote winner: `POST /ab-test/promote-winner/{experiment_name}` (auto-transitions in MLflow)

### Monitoring Model Performance

1. Predictions are automatically logged by `data_capture_service.py` (buffered, daily files)
2. Drift reports generated daily at 3AM via `scheduled_tasks.py`
3. Manual generation: `POST /monitoring/generate-drift-report`
4. Check drift summary: `GET /monitoring/drift-summary`
5. If drift detected, auto-retrain triggered (hourly checks via scheduled tasks)

### Manual Model Promotion

```bash
# Get current production model metadata
curl http://localhost:8000/model/metadata?model_name=tokyo_rent_lgbm

# Check if new models are eligible for promotion
curl -X POST http://localhost:8000/models/check-promotion

# Manually promote a specific version (with safety checks)
curl -X POST http://localhost:8000/models/tokyo_rent_lgbm/promote/3

# Force promotion (skip metric checks)
curl -X POST http://localhost:8000/models/tokyo_rent_lgbm/promote/3?force=true

# Rollback if needed
curl -X POST http://localhost:8000/models/tokyo_rent_lgbm/rollback
```

## Environment Setup

**Required Environment Variables (Docker):**
```bash
# MLflow
MLFLOW_PORT=5001              # External port (default: 5001 to avoid macOS AirPlay conflict)
MLFLOW_TRACKING_URI=http://mlflow:5000  # Internal Docker network

# API
API_PORT=8000
PORT=8000
HOST=0.0.0.0
```

**Local Development (No Docker):**
```bash
# Create .env file in api/ directory
MLFLOW_TRACKING_URI=http://localhost:5001
```

## Automated Tasks Schedule

**Scheduled Tasks Runner:** `python scheduled_tasks.py` (run as background service)

- **Daily 2:00 AM:** Check all models for promotion eligibility
- **Hourly at :00:** Check for drift/performance degradation → Auto-retrain if needed
- **Daily 3:00 AM:** Generate monitoring reports (drift, data quality, model performance)
- **Hourly at :30:** Health check for production models

## Key Implementation Notes

**Auto-Promotion Logic:**
- Training scripts call `AutoPromotionService.check_model_for_promotion()` after training
- Checks absolute thresholds (RMSE, R², MAPE) AND compares with current production model
- Requires ≥2% improvement over production to promote (configurable in `auto_promotion_service.py`)
- Archives old production model when promoting new one

**A/B Testing Traffic Assignment:**
- Hash-based: `hash(user_id) % 100 < treatment_traffic_percent` → Consistent assignment
- Random: Random assignment per request → Use for anonymous users
- Shadow mode: Both models predict, only control returned → Safe testing

**Drift Detection Triggers:**
- Data drift: Distribution shifts in input features (statistical tests)
- Prediction drift: Model output distribution changes
- Performance degradation: Production model metrics decline
- Threshold exceeded → Auto-retrain triggered → New model auto-promoted if better

**Model Versioning:**
- MLflow automatically increments version on each registration
- Stage transitions: `None` → `Production` (auto-promotion) or `Staging` → `Production` (manual)
- Version history preserved (can view all past versions in MLflow UI)

**Data Capture:**
- All predictions logged to `prediction_logs/predictions_YYYY-MM-DD.csv`
- Buffered writes (default: 100 predictions before flush)
- Used for drift monitoring and model performance tracking
- Ground truth can be added later via `/monitoring/update-ground-truth`

## Documentation References

- [API_README.md](docs/API_README.md) - Detailed API documentation
- [MLFLOW_IMPLEMENTATION.md](docs/MLFLOW_IMPLEMENTATION.md) - MLflow setup and usage
- [AUTO_PROMOTION_GUIDE.md](docs/AUTO_PROMOTION_GUIDE.md) - Automated promotion pipeline guide
- [AB_TESTING_GUIDE.md](docs/AB_TESTING_GUIDE.md) - A/B testing framework guide
- [MONITORING_GUIDE.md](docs/MONITORING_GUIDE.md) - Evidently AI drift monitoring guide
- [QUICKSTART_MONITORING.md](docs/QUICKSTART_MONITORING.md) - 5-minute monitoring quick start

## Common Troubleshooting

**"Model not found in MLflow registry"**
- Train the model at least once: `python models/challenger_model.py`
- Check MLFLOW_TRACKING_URI is correctly set
- Verify MLflow server is running: `curl http://localhost:5001/health`

**"No production model found"**
- Models need to meet promotion criteria OR be manually promoted
- Check promotion status: `curl http://localhost:8000/models/tokyo_rent_lgbm/promotion-status`
- Manually promote if needed: `POST /models/{model_name}/promote/{version}`

**Docker port conflicts**
- MLflow default port changed to 5001 (macOS AirPlay uses 5000)
- Change MLFLOW_PORT in `.env` if 5001 is also in use

**Scheduled tasks not running**
- Ensure `scheduled_tasks.py` is running: `ps aux | grep scheduled_tasks`
- Check logs: `tail -f api/scheduled_tasks.log`
- Manually restart: `cd api && python scheduled_tasks.py &`
