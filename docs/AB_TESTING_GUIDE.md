# A/B Testing Framework Guide

## Overview

This comprehensive A/B testing framework goes **far beyond** simple model comparison. It provides:

### What Makes This A/B Testing Framework Complete

#### 1. **Statistical Rigor** ✅
- **T-tests** for statistical significance
- **Mann-Whitney U tests** (non-parametric alternative)
- **Cohen's d** effect size calculation
- **Confidence intervals** (95% default, configurable)
- **P-value** calculations
- **Minimum effect size** thresholds (practical significance)

#### 2. **Traffic Management** ✅
- **Percentage-based splitting** (e.g., 90% control, 10% treatment)
- **Hash-based assignment** (consistent user experience with user_id)
- **Shadow mode** (both models predict, only control returned to user)
- **Manual override** for testing
- **Gradual rollout** support

#### 3. **Experiment Lifecycle** ✅
- **Create** → **Start** → **Running** → **Analyze** → **Promote/Stop**
- **Pause/Resume** capability
- **Date range** restrictions
- **Status tracking** (draft, running, paused, completed, winner_selected)

#### 4. **MLflow Integration** ✅
- **Automatic experiment logging** to MLflow
- **Metrics tracking** (MAE, RMSE, MAPE, R²)
- **Parameter logging** (traffic split, confidence level, etc.)
- **Statistical test results** logged as metrics
- **Automated model promotion** to Production stage in Model Registry
- **Experiment comparison** across multiple A/B tests

#### 5. **Comprehensive Analysis** ✅
- **Sample size validation** (minimum samples required)
- **Control vs Treatment metrics**
- **Percent improvement** calculation
- **Statistical AND practical significance**
- **Winner recommendation** with confidence level
- **Actionable recommendations** (promote, continue, discard)

#### 6. **Production Features** ✅
- **Latency tracking** (compare model performance)
- **Debug information** (see both predictions)
- **Separate logging** (A/B test logs + monitoring logs)
- **Daily log files** (easy to manage)
- **Ground truth integration** (update with actual values)
- **Markdown reports** for stakeholders

## Key Differences from Simple Comparison

| Feature | Simple Comparison | This A/B Framework |
|---------|------------------|-------------------|
| Traffic splitting | ❌ Manual | ✅ Automatic with % control |
| Statistical tests | ❌ None | ✅ T-test, Mann-Whitney U, Cohen's d |
| Confidence intervals | ❌ No | ✅ Yes, configurable |
| Shadow mode | ❌ No | ✅ Yes, test without user impact |
| Experiment tracking | ❌ Manual | ✅ Automatic in MLflow |
| Winner selection | ❌ Manual eyeballing | ✅ Automated with statistical criteria |
| Sample size validation | ❌ No | ✅ Yes, min samples enforced |
| Effect size | ❌ No | ✅ Yes, practical significance |
| User consistency | ❌ No | ✅ Yes, hash-based assignment |
| Production promotion | ❌ Manual | ✅ Automated via MLflow |
| Reporting | ❌ None | ✅ Markdown reports |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│            A/B Testing Request Flow                      │
└─────────────────────────────────────────────────────────┘

POST /ab-test/predict
    │
    ▼
┌─────────────────────────┐
│ 1. Get Experiment Config│
│   - Traffic split       │
│   - Shadow mode         │
│   - Min samples         │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 2. Assign Model         │
│   Based on:             │
│   - Traffic %           │
│   - User hash           │
│   - Random selection    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 3. Make Predictions     │
│   BOTH models:          │
│   - Control (passed)    │
│   - Treatment (chall.)  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 4. Select Return Value  │
│   Shadow mode:          │
│     Always control      │
│   Normal mode:          │
│     Assigned model      │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 5. Log Everything       │
│   - Both predictions    │
│   - Assignment          │
│   - Latency             │
│   - Input features      │
└─────────────────────────┘
    │
    ▼
Storage: ab_results/experiment_name_YYYYMMDD.jsonl
```

## Quick Start

### 1. Create an A/B Test

```bash
curl -X POST "http://localhost:8000/ab-test/create" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "challenger_vs_passed_jan2025",
    "control_model": "passed",
    "treatment_model": "challenger",
    "traffic_split": 0.1,
    "shadow_mode": true,
    "min_samples": 100,
    "confidence_level": 0.95,
    "min_effect_size": 0.05
  }'
```

**Parameters Explained:**
- `traffic_split: 0.1` → 10% get treatment, 90% get control
- `shadow_mode: true` → Both predict, only control returned (safe testing)
- `min_samples: 100` → Need 100 samples before statistical tests
- `confidence_level: 0.95` → 95% confidence (p < 0.05)
- `min_effect_size: 0.05` → Must be 5%+ improvement to be meaningful

### 2. Start the Experiment

```bash
curl -X POST "http://localhost:8000/ab-test/challenger_vs_passed_jan2025/start"
```

### 3. Make Predictions (Traffic is Split Automatically)

```bash
curl -X POST "http://localhost:8000/ab-test/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "challenger_vs_passed_jan2025",
    "input_data": {
      "sqr_m": 30.0,
      "rei_price": 60000,
      "shikikin": 120000,
      "maintenence_price": 6000,
      "year_built": 8,
      "floor": 4,
      "eki_walk": 6
    },
    "user_id": "user_12345"
  }'
```

**Response:**
```json
{
  "prediction": 85000.0,
  "status": "success",
  "experiment": {
    "name": "challenger_vs_passed_jan2025",
    "assigned_model": "treatment",
    "returned_model": "control",
    "shadow_mode": true
  },
  "debug_info": {
    "control_prediction": 85000.0,
    "treatment_prediction": 83500.0,
    "difference": 1500.0,
    "latency_ms": 45.2
  }
}
```

Notice: Assigned to treatment, but control returned (shadow mode)!

### 4. Analyze Results

```bash
curl "http://localhost:8000/ab-test/challenger_vs_passed_jan2025/analyze"
```

**Response:**
```json
{
  "status": "success",
  "analysis": {
    "experiment_name": "challenger_vs_passed_jan2025",
    "sample_sizes": {
      "total_predictions": 1250,
      "predictions_with_actuals": 150,
      "control_group": 135,
      "treatment_group": 15
    },
    "control_metrics": {
      "mae": 12500.5,
      "rmse": 18200.3,
      "mape": 8.4,
      "r2": 0.89
    },
    "treatment_metrics": {
      "mae": 11800.2,
      "rmse": 17100.5,
      "mape": 7.9,
      "r2": 0.91
    },
    "statistical_tests": {
      "t_test": {
        "t_statistic": 2.45,
        "p_value": 0.015,
        "significant": true
      },
      "effect_size": {
        "cohens_d": 0.45,
        "interpretation": "small"
      },
      "confidence_intervals": {
        "control": {"lower": 11800, "upper": 13200},
        "treatment": {"lower": 10900, "upper": 12700}
      },
      "mean_errors": {
        "control": 12500.5,
        "treatment": 11800.2,
        "difference": 700.3,
        "percent_improvement": 5.6
      }
    },
    "recommendation": {
      "winner": "treatment",
      "confidence": "high",
      "message": "Treatment model is statistically significantly better (5.6% improvement in MAE)",
      "is_statistically_significant": true,
      "is_practically_significant": true,
      "percent_improvement": 5.6,
      "recommendation_action": "promote_treatment_to_production"
    },
    "mlflow_run_id": "abc123def456"
  }
}
```

### 5. Promote Winner

```bash
curl -X POST "http://localhost:8000/ab-test/challenger_vs_passed_jan2025/promote?model_registry_name=tokyo_rent_lgbm"
```

## Advanced Features

### Shadow Mode Testing

Test challenger safely without affecting users:

```bash
# Create with shadow_mode: true
curl -X POST "http://localhost:8000/ab-test/create" \
  -d '{"experiment_name": "safe_test", "shadow_mode": true, "traffic_split": 0.5}'
```

- Both models always predict
- Users always get control model's prediction
- System logs both predictions
- Analyze performance without risk

### Gradual Rollout

Start small and increase traffic:

```bash
# Week 1: 5% traffic
curl -X POST "http://localhost:8000/ab-test/create" \
  -d '{"experiment_name": "gradual_rollout", "traffic_split": 0.05}'

# Week 2: Increase to 20%
# Stop experiment, create new one with 0.20

# Week 3: Analyze and promote if successful
```

### User-Consistent Assignment

Same user gets same model:

```python
# User always gets the same model
response = requests.post("/ab-test/predict", json={
    "experiment_name": "test",
    "input_data": {...},
    "user_id": "user_123"  # Hash-based assignment
})
```

### Force Model for Testing

Override assignment for debugging:

```bash
# Force control model
curl -X POST "/ab-test/predict" \
  -d '{
    "experiment_name": "test",
    "input_data": {...},
    "force_model": "control"
  }'

# Force treatment model
curl -X POST "/ab-test/predict" \
  -d '{"force_model": "treatment", ...}'
```

## Statistical Analysis Deep Dive

### What Gets Tested

1. **T-Test (Parametric)**
   - Assumes normal distribution
   - Tests mean difference
   - Returns t-statistic and p-value
   - Significant if p < (1 - confidence_level)

2. **Mann-Whitney U (Non-parametric)**
   - No distribution assumption
   - Tests if distributions differ
   - More robust for non-normal data
   - Alternative to t-test

3. **Cohen's d (Effect Size)**
   - Standardized mean difference
   - Independent of sample size
   - Interpretation:
     - < 0.2: negligible
     - 0.2-0.5: small
     - 0.5-0.8: medium
     - > 0.8: large

4. **Confidence Intervals**
   - Range where true mean likely lies
   - Default 95% CI
   - Non-overlapping CIs = significant difference

### Winner Selection Logic

A treatment model wins if ALL of:

1. **Statistically Significant**: p-value < 0.05 (95% confidence)
2. **Practically Significant**: Improvement >= min_effect_size
3. **Treatment Better**: Lower MAE than control

If only statistically significant but not practically → **Keep control**
If only practically but not statistically → **Collect more data**

### Sample Size Requirements

- Minimum 10 samples per group for statistical tests
- Configurable `min_samples` before analysis starts
- More samples = more statistical power
- Rule of thumb: 100+ samples for reliable results

## Integration with Existing Features

### Works with Drift Monitoring

```python
# Drift monitoring tracks all models
# A/B testing determines which to use

# Check drift while A/B testing
curl "/monitoring/drift?model_type=challenger"

# A/B test helps validate drift isn't affecting treatment
curl "/ab-test/my_experiment/analyze"
```

### Works with MLflow

```python
# A/B test results auto-logged to MLflow
# View in MLflow UI under "ab_test_{experiment_name}"

# Promotion updates Model Registry
# Winning model moved to "Production" stage
```

### Works with Prediction Logging

```python
# Every A/B test prediction logged twice:
# 1. A/B test logs (for analysis)
# 2. Data capture logs (for monitoring)

# Access both:
ab_results = pd.read_json("ab_results/experiment_YYYYMMDD.jsonl", lines=True)
monitoring_logs = pd.read_csv("prediction_logs/predictions_YYYYMMDD.csv")
```

## API Reference

### Create Experiment
`POST /ab-test/create`

### Start Experiment
`POST /ab-test/{experiment_name}/start`

### Stop Experiment
`POST /ab-test/{experiment_name}/stop`

### List Experiments
`GET /ab-test/list`

### Get Experiment Details
`GET /ab-test/{experiment_name}`

### Make A/B Test Prediction
`POST /ab-test/predict`

### Analyze Experiment
`GET /ab-test/{experiment_name}/analyze?days_back=7`

### Promote Winner
`POST /ab-test/{experiment_name}/promote?model_registry_name=tokyo_rent_lgbm&auto_promote=false`

### Get Markdown Report
`GET /ab-test/{experiment_name}/report`

## Best Practices

### 1. Start with Shadow Mode

```bash
# Phase 1: Shadow mode (no user impact)
{"shadow_mode": true, "traffic_split": 0.5}

# Phase 2: Small % real traffic
{"shadow_mode": false, "traffic_split": 0.05}

# Phase 3: Gradual increase
{"shadow_mode": false, "traffic_split": 0.2}
```

### 2. Set Appropriate Thresholds

```python
# Conservative (require strong evidence)
min_samples = 200
confidence_level = 0.99  # 99% confidence
min_effect_size = 0.10   # 10% improvement required

# Standard (balanced)
min_samples = 100
confidence_level = 0.95  # 95% confidence
min_effect_size = 0.05   # 5% improvement required

# Aggressive (faster decisions)
min_samples = 50
confidence_level = 0.90  # 90% confidence
min_effect_size = 0.03   # 3% improvement required
```

### 3. Monitor Continuously

```bash
# Daily checks
curl "/ab-test/my_experiment/analyze?days_back=1"

# Weekly comprehensive
curl "/ab-test/my_experiment/analyze"

# Check for early signals
# Stop if treatment clearly worse
```

### 4. Document Decisions

```bash
# Get markdown report
curl "/ab-test/my_experiment/report" > experiment_report.md

# Share with stakeholders
# Include in git repo
# Reference in PRs
```

### 5. Clean Up Completed Experiments

```bash
# Stop after conclusion
curl -X POST "/ab-test/my_experiment/stop"

# Keep logs for historical analysis
# Archive results to S3/GCS
```

## Troubleshooting

**"Insufficient data"**
- Need minimum samples (default 100)
- Need actuals for statistical tests
- Solution: Collect more predictions with ground truth

**"Not statistically significant"**
- Difference exists but not strong enough
- Solution: Collect more data OR accept control model

**"Significant but not meaningful"**
- Statistically different but < min_effect_size
- Solution: Keep control, difference too small to matter

**"Low confidence recommendation"**
- Not enough evidence for promotion
- Solution: Continue experiment or keep control

## Next Steps

1. **Set up automated analysis**: Cron job to check experiments daily
2. **Alert on completion**: Send Slack message when min_samples reached
3. **Dashboard**: Build real-time A/B test dashboard
4. **Multi-armed bandit**: Extend to dynamic traffic allocation
5. **Stratification**: Split by user segments (geography, price range)

## Resources

- [Statistical Testing in A/B Tests](https://en.wikipedia.org/wiki/A/B_testing)
- [Cohen's d Effect Size](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Practical Significance](https://en.wikipedia.org/wiki/Practical_significance)
