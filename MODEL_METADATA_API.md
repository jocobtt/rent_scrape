# Model Metadata API Documentation

## Overview

This document describes the API endpoint needed to support the `ModelInfo` component, which displays ML model metadata from MLflow.

## Endpoint

```
GET /model/metadata
```

## Response Format

The endpoint should return a JSON object with the following structure:

```json
{
  "name": "Tokyo Rent Predictor",
  "version": "1.2.3",
  "run_id": "abc123def456ghi789jkl012mno345pq",
  "status": "active",
  "last_updated": "2025-10-15T10:30:00Z",
  "description": "Random Forest model trained on 10,000+ Tokyo apartment listings",
  "metrics": {
    "rmse": 0.1234,
    "mae": 0.0987,
    "r2_score": 0.8765,
    "mape": 12.34
  },
  "params": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
  },
  "feature_importance": [
    {
      "name": "sqr_m",
      "importance": 0.45
    },
    {
      "name": "ku_name",
      "importance": 0.25
    },
    {
      "name": "eki_walk",
      "importance": 0.15
    },
    {
      "name": "age",
      "importance": 0.10
    },
    {
      "name": "floor",
      "importance": 0.05
    }
  ]
}
```

## Field Descriptions

### Required Fields

- **`name`** (string): The name of the model
- **`version`** (string): Model version number (semantic versioning recommended)
- **`status`** (string): One of: `"active"`, `"staging"`, `"archived"`

### Optional Fields

- **`run_id`** (string): MLflow run ID for traceability
- **`last_updated`** (string): ISO 8601 timestamp of last model update
- **`description`** (string): Human-readable description of the model
- **`metrics`** (object): Performance metrics from MLflow
  - Common metrics: `rmse`, `mae`, `r2_score`, `mape`, `accuracy`, etc.
- **`params`** (object): Model hyperparameters
- **`feature_importance`** (array): List of features and their importance scores
  - Each item should have `name` (string) and `importance` (number between 0 and 1)

## Example MLflow Integration (Python/Flask)

Here's an example of how to implement this endpoint using MLflow:

```python
from flask import Flask, jsonify
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)

@app.route('/model/metadata', methods=['GET'])
def get_model_metadata():
    try:
        client = MlflowClient()

        # Get the latest production model version
        model_name = "tokyo-rent-predictor"
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])

        if not latest_versions:
            return jsonify({"error": "No model found in production"}), 404

        model_version = latest_versions[0]
        run = client.get_run(model_version.run_id)

        # Extract metrics
        metrics = {
            "rmse": run.data.metrics.get("rmse"),
            "mae": run.data.metrics.get("mae"),
            "r2_score": run.data.metrics.get("r2_score"),
            "mape": run.data.metrics.get("mape"),
        }

        # Extract parameters
        params = dict(run.data.params)

        # Get feature importance (assuming it's logged as an artifact)
        feature_importance = []
        try:
            import json
            artifact_path = f"runs:/{model_version.run_id}/feature_importance.json"
            local_path = mlflow.artifacts.download_artifacts(artifact_path)
            with open(local_path, 'r') as f:
                feature_importance = json.load(f)
        except Exception as e:
            print(f"Could not load feature importance: {e}")

        # Build response
        response = {
            "name": model_name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "status": model_version.current_stage.lower(),
            "last_updated": model_version.last_updated_timestamp,
            "description": model_version.description or "Tokyo apartment rent prediction model",
            "metrics": metrics,
            "params": params,
            "feature_importance": feature_importance
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Usage in Frontend

### Basic Usage (Model.jsx)

Add the component to your Model page to show model info:

```jsx
import ModelInfo from '../components/ModelInfo';

// In your Model component:
<div className="model">
  <ModelInfo />
  {/* Rest of your prediction form */}
</div>
```

### Compact Usage

Use the compact prop for a collapsible version:

```jsx
<ModelInfo compact={true} />
```

## Error Handling

If the API endpoint is unavailable or returns an error, the component will gracefully display an error message. The prediction functionality will continue to work independently.

## Caching Recommendations

Consider implementing caching on the backend:
- Cache the model metadata response for 5-15 minutes
- Invalidate cache when a new model is deployed
- Use Redis or similar for distributed caching

## Security Considerations

- Ensure the endpoint doesn't expose sensitive information
- Consider rate limiting the endpoint
- Add authentication if model metadata is considered proprietary
