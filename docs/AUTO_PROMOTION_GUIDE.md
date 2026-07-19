# Automated Model Promotion Pipeline - Complete Guide

## Overview

You now have a **complete automated model promotion pipeline** that handles the entire model lifecycle from training to production deployment!

## What Was Implemented (The Missing 50%)

### 1. Auto-Promotion Service ✅
**File:** `services/auto_promotion_service.py` (600+ lines)

**Features:**
- Metric-based promotion criteria (RMSE, R², MAPE thresholds)
- Automatic comparison with production model
- Safety checks before promotion
- Manual promotion with force override
- Automatic rollback capability
- Promotion status tracking

### 2. Auto-Retrain Service ✅
**File:** `services/auto_retrain_service.py` (300+ lines)

**Features:**
- Drift-triggered retraining
- Performance degradation detection
- Automatic retraining execution
- Post-retrain auto-promotion check
- Configurable thresholds

### 3. Training Script Integration ✅
**Files:** `models/challenger_model.py`, `models/reg_model.py`

**Added:**
- Automatic promotion check after training
- Visual feedback on promotion decisions
- Integration with promotion service

### 4. API Endpoints ✅
**File:** `model_api.py` (6 new endpoints)

- `POST /models/{model_name}/promote/{version}` - Manual promotion
- `POST /models/check-promotion` - Check all models for promotion
- `GET /models/{model_name}/promotion-status` - Get promotion status
- `POST /models/{model_name}/rollback` - Rollback to previous version
- `POST /retrain/check-and-execute` - Check and retrain if needed
- `POST /retrain/force` - Force immediate retraining

### 5. Scheduled Tasks ✅
**File:** `scheduled_tasks.py`

**Schedules:**
- Daily at 2:00 AM: Check for promotable models
- Hourly: Check for drift and performance degradation
- Daily at 3:00 AM: Generate monitoring reports
- Hourly at :30: Health check for production models

## Complete Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│         AUTOMATED MODEL PROMOTION PIPELINE                    │
└──────────────────────────────────────────────────────────────┘

1. TRAINING ← YOU HAD THIS
   │
   ├─ python models/challenger_model.py
   ├─ Train model with data
   ├─ Log metrics to MLflow
   └─ Register to Model Registry (Stage: None)
   │
   ▼
2. AUTO-PROMOTION CHECK ← NEW! (Training Scripts)
   │
   ├─ Check absolute thresholds:
   │  ├─ RMSE < 17,500?
   │  ├─ R² > 0.88?
   │  └─ MAPE < 9%?
   │
   ├─ Compare with production:
   │  ├─ Is it better?
   │  └─ At least 2% improvement?
   │
   ├─ IF ALL PASS:
   │  └─ ✅ AUTO-PROMOTE TO PRODUCTION!
   │
   └─ ELSE:
      └─ ℹ️  Model stays in "None" stage
   │
   ▼
3. API-TRIGGERED PROMOTION ← NEW! (Manual Override)
   │
   ├─ POST /models/{name}/promote/{version}
   ├─ With safety checks OR
   └─ Force promotion (skip checks)
   │
   ▼
4. SCHEDULED PROMOTION CHECK ← NEW! (Daily Automation)
   │
   ├─ Daily at 2:00 AM
   ├─ Check ALL registered models
   ├─ Promote qualified models
   └─ Send notifications
   │
   ▼
5. DRIFT MONITORING ← YOU HAD THIS
   │
   ├─ Hourly drift checks
   ├─ Performance monitoring
   └─ Data quality checks
   │
   ▼
6. AUTO-RETRAINING ← NEW! (Drift Response)
   │
   ├─ IF drift > 30% OR performance degraded > 15%:
   │  ├─ Trigger retraining
   │  ├─ Train new models
   │  └─ Check if new models should be promoted
   │
   └─ Loop back to step 2
   │
   ▼
7. ROLLBACK ← NEW! (Safety Net)
   │
   ├─ POST /models/{name}/rollback
   ├─ Restore previous production version
   └─ Instant recovery from bad deployments
```

## Quick Start

### 1. Install Dependencies

```bash
cd api
uv sync
# Adds: scipy>=1.10.0, schedule>=1.2.0
```

### 2. Train a Model (Auto-Promotion Enabled!)

```bash
python models/challenger_model.py
```

**What happens:**
1. Model trains
2. Logs to MLflow
3. **NEW:** Automatically checks promotion criteria
4. **NEW:** Promotes to Production if qualified!

**Example output:**
```
============================================================
🎉 MODEL AUTO-PROMOTED TO PRODUCTION!
============================================================
Model: tokyo_rent_lgbm v5
Reason: Passed all promotion criteria
Metrics: RMSE=17200.00, R²=0.8950, MAPE=8.50%
============================================================
```

### 3. Check Promotion Status

```bash
curl http://localhost:8000/models/tokyo_rent_lgbm/promotion-status | jq
```

**Response:**
```json
{
  "status": "success",
  "info": {
    "model_name": "tokyo_rent_lgbm",
    "total_versions": 5,
    "latest_by_stage": {
      "Production": {
        "version": "5",
        "metrics": {
          "rmse": 17200,
          "r2_score": 0.895,
          "mape": 8.5
        }
      },
      "None": {
        "version": "4",
        "metrics": {...}
      }
    },
    "promotion_criteria": {
      "max_rmse": 17500,
      "min_r2": 0.88,
      "max_mape": 9.0,
      "min_improvement_percent": 2.0
    }
  }
}
```

### 4. Manual Promotion

```bash
# Promote specific version with safety checks
curl -X POST "http://localhost:8000/models/tokyo_rent_lgbm/promote/4"

# Force promotion (skip safety checks)
curl -X POST "http://localhost:8000/models/tokyo_rent_lgbm/promote/4?force=true"
```

### 5. Check and Retrain Based on Drift

```bash
# Dry run (check only)
curl -X POST "http://localhost:8000/retrain/check-and-execute?dry_run=true" | jq

# Execute (actually retrain if needed)
curl -X POST "http://localhost:8000/retrain/check-and-execute?dry_run=false" | jq
```

### 6. Start Scheduled Tasks

```bash
# Run in foreground
python scheduled_tasks.py

# Run in background
nohup python scheduled_tasks.py > scheduled_tasks.log 2>&1 &
```

## Promotion Criteria (Configurable!)

### Default Criteria

**Challenger Model (LightGBM):**
```python
PromotionCriteria(
    max_rmse=17500,      # Must be < 17,500
    min_r2=0.88,         # Must be > 0.88
    max_mape=9.0,        # Must be < 9%
    min_improvement_percent=2.0,  # Must be 2% better than production
    require_better_than_production=True
)
```

**Passed Model (Linear):**
```python
PromotionCriteria(
    max_rmse=19000,
    min_r2=0.85,
    max_mape=10.0,
    min_improvement_percent=2.0,
    require_better_than_production=True
)
```

### Customize Criteria

```python
from services.auto_promotion_service import AutoPromotionService, PromotionCriteria

auto_promotion = AutoPromotionService()

# Set custom criteria
custom_criteria = PromotionCriteria(
    model_name="tokyo_rent_lgbm",
    max_rmse=16000,  # More strict!
    min_r2=0.90,
    max_mape=8.0,
    min_improvement_percent=5.0,  # Require 5% improvement
)

auto_promotion.set_criteria("tokyo_rent_lgbm", custom_criteria)
```

## Complete API Reference

### Promotion Endpoints

#### 1. Manual Promote
```bash
POST /models/{model_name}/promote/{version}?force=false
```

#### 2. Check All Models
```bash
POST /models/check-promotion?dry_run=true
```

#### 3. Get Status
```bash
GET /models/{model_name}/promotion-status
```

#### 4. Rollback
```bash
POST /models/{model_name}/rollback
```

### Retraining Endpoints

#### 5. Auto-Retrain Check
```bash
POST /retrain/check-and-execute?days_back=7&dry_run=true
```

#### 6. Force Retrain
```bash
POST /retrain/force?model_name=tokyo_rent_lgbm
```

## Example Workflows

### Workflow 1: New Model Training

```bash
# 1. Train model
python models/challenger_model.py

# Output shows:
# ✅ MODEL AUTO-PROMOTED TO PRODUCTION!

# 2. Verify promotion
curl http://localhost:8000/models/tokyo_rent_lgbm/promotion-status

# 3. Model is now serving traffic!
```

### Workflow 2: Safe Model Testing

```bash
# 1. Train with modified criteria to prevent auto-promotion
# Edit promotion criteria in auto_promotion_service.py

# 2. Train model
python models/challenger_model.py

# Output shows:
# ℹ️  Model NOT promoted: RMSE not better than production

# 3. Manually promote when ready
curl -X POST "http://localhost:8000/models/tokyo_rent_lgbm/promote/5"
```

### Workflow 3: Drift-Triggered Pipeline

```bash
# 1. Scheduled task detects drift
# (Runs hourly automatically)

# 2. Auto-retrain triggers
# - Retrains both models
# - New versions registered

# 3. Auto-promotion checks new models
# - Compares with production
# - Promotes if better

# 4. All automatic!
```

### Workflow 4: Rollback After Bad Deployment

```bash
# Oh no! New model is causing issues

# Quick rollback
curl -X POST "http://localhost:8000/models/tokyo_rent_lgbm/rollback"

# Response:
# ✅ Rolled back tokyo_rent_lgbm to v4

# Production restored instantly!
```

## Scheduled Tasks Details

### What Runs Automatically

#### Daily at 2:00 AM - Promotion Check
```
Checks ALL models for promotion eligibility
Promotes qualified models automatically
Logs results to scheduled_tasks.log
```

#### Hourly at :00 - Drift Check
```
Checks last hour of predictions for drift
Checks performance degradation
Triggers retraining if needed
Auto-promotes new models if better
```

#### Daily at 3:00 AM - Monitoring Reports
```
Generates comprehensive drift reports
Saves to monitoring_reports/
Available for review
```

#### Hourly at :30 - Health Check
```
Checks production model metrics
Monitors for issues
Logs health status
```

### View Scheduled Task Logs

```bash
tail -f scheduled_tasks.log
```

## Safety Features

### 1. Absolute Thresholds
Models must meet minimum quality standards before promotion.

### 2. Comparison with Production
New models must be **better** than current production (configurable %).

### 3. Dry Run Mode
Test promotion logic without actually promoting:
```bash
curl -X POST "/models/check-promotion?dry_run=true"
```

### 4. Manual Override
Force promotion when needed (e.g., hotfixes):
```bash
curl -X POST "/models/{name}/promote/{version}?force=true"
```

### 5. Instant Rollback
Revert to previous version in seconds:
```bash
curl -X POST "/models/{name}/rollback"
```

### 6. Minimum Time Between Retrains
Prevents thrashing (default: 1 day between retrains).

## Monitoring & Alerts

### Check What's Running

```bash
# See which models are in production
curl http://localhost:8000/models/tokyo_rent_lgbm/promotion-status
curl http://localhost:8000/models/tokyo_passed_rent_model/promotion-status
```

### View Recent Promotions

```bash
# Check promotion logs
grep "PROMOTED" scheduled_tasks.log

# Check MLflow UI
# Models → {model_name} → Versions → Stage changes
```

### Set Up Alerts

Add to `scheduled_tasks.py`:
```python
def send_slack_alert(message):
    # Your Slack webhook integration
    pass

# In run_promotion_check():
if result["models_promoted"]:
    send_slack_alert(f"🎉 Promoted: {result['models_promoted']}")
```

## Customization

### Adjust Promotion Criteria

Edit `services/auto_promotion_service.py`:
```python
self.criteria = {
    "tokyo_rent_lgbm": PromotionCriteria(
        max_rmse=16000,  # Change this
        min_r2=0.90,      # And this
        # ...
    )
}
```

### Adjust Retraining Triggers

Edit `services/auto_retrain_service.py`:
```python
def __init__(self):
    self.drift_threshold = 0.2  # More sensitive (was 0.3)
    self.performance_degradation_threshold = 0.10  # More sensitive (was 0.15)
```

### Change Schedule

Edit `scheduled_tasks.py`:
```python
# Run every 4 hours instead of hourly
schedule.every(4).hours.do(run_drift_check)

# Run at different time
schedule.every().day.at("01:00").do(run_promotion_check)
```

## Troubleshooting

### Models not auto-promoting after training

**Check:**
1. Does model meet absolute criteria? (RMSE, R², MAPE)
2. Is it better than production by min_improvement_percent?
3. Check training script output for reason

**Solution:**
```python
# See exact criteria
curl http://localhost:8000/models/{name}/promotion-status
```

### Scheduled tasks not running

**Check:**
```bash
# Is script running?
ps aux | grep scheduled_tasks

# Check logs
tail scheduled_tasks.log
```

**Solution:**
```bash
# Restart
pkill -f scheduled_tasks
nohup python scheduled_tasks.py > scheduled_tasks.log 2>&1 &
```

### Retraining not triggering

**Check drift thresholds:**
```bash
curl "http://localhost:8000/retrain/check-and-execute?dry_run=true"
```

## Summary: What You Now Have

### Before (What You Had)
- ✅ MLflow model registry
- ✅ Model versioning
- ✅ A/B test promotion
- ❌ No training → production automation
- ❌ No scheduled checks
- ❌ No drift-triggered retraining

### After (Complete Pipeline!)
- ✅ MLflow model registry
- ✅ Model versioning
- ✅ A/B test promotion
- ✅ **Training → Production automation**
- ✅ **Scheduled promotion checks**
- ✅ **Drift-triggered retraining**
- ✅ **Manual promotion API**
- ✅ **Rollback capability**
- ✅ **Health monitoring**

## Next Steps

1. **Test the pipeline:**
```bash
python models/challenger_model.py
```

2. **Start scheduled tasks:**
```bash
nohup python scheduled_tasks.py > scheduled_tasks.log 2>&1 &
```

3. **Monitor:**
```bash
tail -f scheduled_tasks.log
```

4. **Customize criteria** based on your needs

5. **Set up alerts** (Slack, email, etc.)

---

**You now have a 100% complete automated model promotion pipeline!** 🎉
