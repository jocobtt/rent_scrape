# Tokyo Rent Predictor API

A FastAPI-based REST API for predicting Tokyo apartment rental prices using machine learning models.

## Features

- **Two Model Types**: 
  - Passed Model (production): Traditional ML model using sklearn
  - Challenger Model: LightGBM-based model for experimentation
- **Multiple Endpoints**: Prediction, comparison, health checks, and model retraining
- **Real-time Data Scraping**: Integration with web scraping for model retraining
- **Model Comparison**: Side-by-side comparison of both models

## Quick Start

### Prerequisites

- Python 3.12+
- UV package manager (or pip)

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Ensure model files exist:
   - `models/passed-model.joblib`
   - `models/challenger-model.joblib`

3. Start the API:
```bash
uv run model_api.py
# or use the startup script:
./start_api.sh
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Core Endpoints

#### `GET /`
Welcome message

#### `GET /health`
Health check with model loading status
```json
{
  "status": "ok",
  "models_loaded": {
    "challenger_model": true,
    "passed_model": true
  },
  "timestamp": "2025-10-13T..."
}
```

#### `GET /model_info`
Get information about loaded models
```json
{
  "models": {
    "passed_model": {
      "loaded": true,
      "type": "sklearn.linear_model._base.LinearRegression"
    },
    "challenger_model": {
      "loaded": true,
      "type": "lightgbm.sklearn.LGBMRegressor"
    }
  },
  "api_version": "1.0.0",
  "supported_endpoints": [...]
}
```

### Prediction Endpoints

#### `POST /predict`
Get prediction from the passed (production) model

**Request Body:**
```json
{
  "sqr_m": 25.0,
  "rei_price": 5.0,
  "shikikin": 10.0,
  "maintenence_price": 2.0,
  "year_built": 10.0,
  "floor": 3.0,
  "eki_walk": 5.0
}
```

**Response:**
```json
{
  "prediction": 8.5,
  "model_type": "passed",
  "status": "success"
}
```

#### `POST /challenger_predict`
Get prediction from the challenger model

Same request format as `/predict`, returns prediction from LightGBM model.

#### `POST /compare_predictions`
Compare predictions from both models

**Response:**
```json
{
  "predictions": {
    "passed_model": 8.5,
    "challenger_model": 8.2,
    "difference": 0.3,
    "percentage_difference": 3.53
  },
  "input_data": {...},
  "status": "success"
}
```

### Model Management

#### `POST /retrain`
Retrain specified models with new data

**Request Body:**
```json
{
  "url": "https://suumo.jp/jj/chintai/ichiran/FR301FC001/...",
  "wait_time_min": 1,
  "wait_time_max": 5,
  "pages": [0, 50],
  "model_to_retrain": ["challenger", "passed"]
}
```

## Input Data Schema

All prediction endpoints expect the following parameters:

- `sqr_m` (float): Square meters of the apartment
- `rei_price` (float): Key money/礼金 in 万円 (10,000 yen units)
- `shikikin` (float): Security deposit/敷金 in 万円
- `maintenence_price` (float): Monthly maintenance fee in 万円
- `year_built` (float): Years since construction
- `floor` (float): Floor number
- `eki_walk` (float): Walking time to nearest station in minutes

## Testing

Run the test script to validate all endpoints:
```bash
# Run tests manually (make sure API is running first)
uv run tests/test_api.py

# Or use the automated test runner (starts/stops API automatically)
./run_tests.sh
```

## Error Handling

All endpoints return structured error responses:
```json
{
  "detail": "Error message",
  "status_code": 500
}
```

## Project Structure

```
api/
├── model_api.py              # Main FastAPI application
├── models/                   # Model definitions and artifacts
│   ├── __init__.py
│   ├── challenger_model.py   # LightGBM model training
│   ├── reg_model.py         # Sklearn model training
│   ├── challenger-model.joblib
│   └── passed-model.joblib
├── services/                 # Business logic services
│   ├── __init__.py
│   ├── prediction_service.py # Prediction logic
│   ├── training_service.py  # Model training logic
│   └── model_loader.py      # Model loading utilities
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── scrape_data.py       # Data scraping utilities
├── tests/                    # Test files
│   └── test_api.py
├── start_api.sh             # Startup script
├── run_tests.sh             # Test runner script
└── API_README.md            # This file
```

## Development

### Adding New Endpoints

1. Define your endpoint function with proper type hints
2. Add error handling with try/catch blocks
3. Include documentation in the docstring
4. Add tests to `tests/test_api.py`
5. Use the service classes for business logic

### Adding New Services

1. Create new service class in `services/` directory
2. Add imports to `services/__init__.py`
3. Initialize service in `model_api.py`
4. Use dependency injection pattern

### Model Updates

Models are loaded once at startup using the `ModelLoader` service and cached in `app.state`. The `/retrain` endpoint uses the `TrainingService` to update both saved files and in-memory models.

## Deployment

The API is configured for deployment with:
- Docker support (see `Dockerfile`)
- Environment variable configuration
- Health check endpoints for load balancers
- CORS middleware for web frontend integration

## Environment Variables

- `PORT`: API port (default: 8000)
- `HOST`: API host (default: 0.0.0.0)

## Logging

The API uses Python's logging module. Logs include:
- Request/response information
- Model loading status
- Error details
- Performance metrics