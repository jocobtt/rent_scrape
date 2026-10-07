# MLflow Model Versioning Implementation

## Overview

This document describes the MLflow integration for model versioning, metadata tracking, and serving in the Tokyo Rent Predictor API.

## Architecture

### Components

1. **MLflowService** (`services/mlflow_service.py`)
   - Handles MLflow model registry operations
   - Retrieves model metadata from MLflow tracking server
   - Manages model versioning and staging

2. **Model Training Scripts**
   - `models/challenger_model.py` - LightGBM model training with MLflow logging
   - `models/reg_model.py` - Linear regression models training with MLflow logging

3. **API Endpoints** (`model_api.py`)
   - `/model/metadata` - Get metadata for a specific model
   - `/model/metadata/all` - Get metadata for all active models

## Setup

### 1. MLflow Tracking URI

Set the MLflow tracking URI in your environment or in the code:

```python
# Option 1: Environment variable
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

# Option 2: In code (models/challenger_model.py, models/reg_model.py)
mlflow.set_tracking_uri("your_tracking_uri")
```

For local development, MLflow will use `file:///tmp/mlruns` by default.

### 2. Model Registration

Models are automatically registered to MLflow when trained:

- **Challenger Model**: Registered as `tokyo_rent_lgbm`
- **Passed Model**: Registered as `tokyo_passed_rent_model`

## API Endpoints

### GET /model/metadata

Get metadata for a specific model from MLflow.

**Query Parameters:**
- `model_name` (optional): Model name (default: "tokyo_rent_lgbm")
  - Options: "tokyo_rent_lgbm" or "tokyo_passed_rent_model"
- `version` (optional): Specific version number (default: latest production version)

**Example Requests:**

```bash
# Get latest production version of challenger model
curl http://localhost:8000/model/metadata

# Get specific model by name
curl http://localhost:8000/model/metadata?model_name=tokyo_passed_rent_model

# Get specific version
curl http://localhost:8000/model/metadata?model_name=tokyo_rent_lgbm&version=2
```

**Response Format:**

```json
{
  "name": "Tokyo Rent Predictor (LightGBM)",
  "version": "3",
  "run_id": "abc123def456",
  "status": "active",
  "last_updated": "2025-10-18T10:30:00Z",
  "description": "LightGBM model trained on Tokyo apartment data. RMSE: 0.1234, R2: 0.8765",
  "metrics": {
    "rmse": 0.1234,
    "mae": 0.0987,
    "r2_score": 0.8765,
    "mape": 12.34,
    "mse": 0.1234
  },
  "params": {
    "n_estimators": 100000,
    "max_depth": 8,
    "learning_rate": 0.005,
    "num_leaves": 128,
    "boosting_type": "gbdt"
  },
  "feature_importance": [
    {
      "name": "sqr_m",
      "importance": 0.45
    },
    {
      "name": "ku_name_shibuya",
      "importance": 0.25
    }
  ],
  "model_name": "tokyo_rent_lgbm",
  "stage": "Production"
}
```

### GET /model/metadata/all

Get metadata for all active models (challenger and passed).

**Example Request:**

```bash
curl http://localhost:8000/model/metadata/all
```

**Response Format:**

```json
{
  "models": {
    "challenger": {
      "name": "Tokyo Rent Predictor (LightGBM)",
      "version": "3",
      "metrics": { ... },
      "params": { ... }
    },
    "passed": {
      "name": "Tokyo Rent Predictor (Linear)",
      "version": "2",
      "metrics": { ... },
      "params": { ... }
    }
  },
  "timestamp": "2025-10-18T12:00:00"
}
```

## Model Training with MLflow

### Training the Challenger Model (LightGBM)

```python
from models.challenger_model import load_data, train_and_log_model

# Load data
X_train, X_test, y_train, y_test = load_data("data/tokyo_model.csv")

# Define parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'max_depth': 8,
    'num_leaves': 128,
    'max_bin': 512,
    'num_iterations': 100000
}

# Train and log to MLflow
model = train_and_log_model(X_train, X_test, y_train, y_test, params)
```

**What gets logged:**
- Model artifact (LightGBM model)
- Parameters (all training parameters)
- Metrics: MSE, RMSE, MAE, R2 Score, MAPE
- Feature importance (JSON artifact)
- Model description

### Training the Passed Model (Linear Regression)

```python
from models.reg_model import load_data, train_model, evaluate_model, log_mlflow

# Load data
X_train, X_test, y_train, y_test = load_data()

# Train models
models = train_model(X_train, y_train, alpha=1.0)

# Find best model
best_mse = float('inf')
best_model = None
best_model_name = ""

for model_name, model in models.items():
    mse = evaluate_model(model, X_test, y_test)
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_model_name = model_name

# Log to MLflow
if best_mse < 4:
    log_mlflow(best_model_name, best_model, X_train, X_test, y_test,
               mlflow_name="reg_passed_model", best_mse=best_mse)
```

**What gets logged:**
- Model artifact (scikit-learn model)
- Parameters (alpha, test_size, random_state, model_type)
- Metrics: MSE, RMSE, MAE, R2 Score, MAPE
- Feature importance (coefficients as JSON artifact)
- Model description

## MLflow Service Usage

### Get Production Model Metadata

```python
from services import MLflowService

mlflow_service = MLflowService()

# Get latest production version
metadata = mlflow_service.get_production_model_metadata("tokyo_rent_lgbm")

print(f"Model: {metadata['name']}")
print(f"Version: {metadata['version']}")
print(f"RMSE: {metadata['metrics']['rmse']}")
```

### Get Specific Model Version

```python
metadata = mlflow_service.get_model_metadata_by_version(
    model_name="tokyo_rent_lgbm",
    version="2"
)
```

### Transition Model Stage

```python
# Promote a model to production
success = mlflow_service.transition_model_stage(
    model_name="tokyo_rent_lgbm",
    version="3",
    stage="Production"
)
```

## Model Retraining

When you retrain models via the `/retrain` endpoint, the new models are automatically:
1. Trained with the fresh data
2. Logged to MLflow with all metrics and artifacts
3. Registered in the MLflow model registry
4. Saved locally as `.joblib` files

```python
# Via API
POST /retrain
{
  "url": "https://example.com/listings",
  "wait_time_min": 1,
  "wait_time_max": 5,
  "pages": [0, 50],
  "model_to_retrain": ["challenger", "passed"]
}
```

## Feature Importance

Feature importance is logged as a JSON artifact for both models:

**Format:**
```json
[
  {
    "name": "sqr_m",
    "importance": 0.45
  },
  {
    "name": "ku_name_shibuya",
    "importance": 0.25
  }
]
```

- **LightGBM**: Uses built-in feature importance (`model.feature_importances_`)
- **Linear Models**: Uses absolute values of coefficients as importance

## Metrics Tracked

All models track these metrics:

1. **RMSE** (Root Mean Squared Error) - Primary metric
2. **MAE** (Mean Absolute Error)
3. **R2 Score** (Coefficient of Determination)
4. **MAPE** (Mean Absolute Percentage Error)
5. **MSE** (Mean Squared Error)

## Model Registry

### Model Names

- `tokyo_rent_lgbm` - LightGBM challenger model
- `tokyo_passed_rent_model` - Linear regression passed model

### Model Stages

- **None**: Default stage when first registered
- **Staging**: Model being tested
- **Production**: Active production model
- **Archived**: Deprecated model

The API defaults to returning models in "Production" stage, falling back to "None" stage if no production model exists.

## Integration with Frontend

### Example: Fetch Model Metadata in React

```javascript
const ModelInfo = () => {
  const [metadata, setMetadata] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/model/metadata')
      .then(res => res.json())
      .then(data => setMetadata(data))
      .catch(err => console.error(err));
  }, []);

  if (!metadata) return <div>Loading...</div>;

  return (
    <div>
      <h2>{metadata.name}</h2>
      <p>Version: {metadata.version}</p>
      <p>RMSE: {metadata.metrics.rmse.toFixed(4)}</p>
      <p>R² Score: {metadata.metrics.r2_score.toFixed(4)}</p>

      <h3>Top Features</h3>
      <ul>
        {metadata.feature_importance.slice(0, 5).map(f => (
          <li key={f.name}>{f.name}: {f.importance.toFixed(4)}</li>
        ))}
      </ul>
    </div>
  );
};
```

## Troubleshooting

### Issue: "Model not found in MLflow registry"

**Solution:**
1. Ensure MLflow tracking URI is correctly configured
2. Train the model at least once to register it
3. Check that the model name matches exactly

### Issue: Feature importance not showing

**Solution:**
1. Ensure the model was trained with the updated training scripts
2. Check that the feature_importance.json artifact was logged
3. Retrain the model if it was trained before this implementation

### Issue: MLflow connection errors

**Solution:**
1. Verify MLflow server is running (if using remote server)
2. Check tracking URI configuration
3. Ensure network connectivity to MLflow server
4. For local development, MLflow will create local directories automatically

## Best Practices

1. **Always log feature importance** when training models
2. **Use descriptive model descriptions** with key metrics
3. **Transition models to Production stage** explicitly
4. **Archive old versions** when promoting new ones
5. **Monitor metrics trends** across versions
6. **Use semantic versioning** for model descriptions when possible

## Future Enhancements

Potential improvements to consider:

1. **Model Comparison Endpoint** - Compare metrics across versions
2. **A/B Testing Support** - Serve different models to different users
3. **Model Performance Monitoring** - Track prediction accuracy over time
4. **Automated Model Promotion** - Auto-promote models based on metrics
5. **Model Rollback** - Quickly revert to previous version
6. **Experiment Tracking** - Track hyperparameter tuning experiments
7. **Model Lineage** - Track data and code versions used for training

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
