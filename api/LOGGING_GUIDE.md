# Comprehensive Logging Guide

## Overview

The Tokyo Rent Predictor API now includes **production-grade logging** with:

- ✅ **Structured JSON logging** (easy parsing for log aggregation)
- ✅ **Request/Response tracking** (automatic request ID generation)
- ✅ **Performance monitoring** (timing for all operations)
- ✅ **Log rotation** (automatic file management)
- ✅ **Context propagation** (request IDs flow through all logs)
- ✅ **Sensitive data filtering** (automatic redaction)
- ✅ **Environment-aware** (colored console for dev, JSON for production)

---

## Quick Start

### View Logs

```bash
cd api

# View all logs (tail -f for live updates)
tail -f logs/app.log

# View errors only
tail -f logs/error.log

# View API access logs
tail -f logs/access.log
```

### Log Files Structure

```
api/logs/
├── app.log          # All application logs (rotates at 10MB)
├── app.log.1        # Backup 1
├── app.log.2        # Backup 2
├── error.log        # Errors only (rotates at 10MB)
├── access.log       # API requests (rotates daily)
├── access.log.2025-12-14
└── access.log.2025-12-13
```

---

## Log Formats

### Development (Colored Console)

```
[2025-12-15 10:30:45] [INFO    ] [model_api] POST /predict - 200
[2025-12-15 10:30:46] [ERROR   ] [prediction_service] Model prediction failed
```

### Production (JSON)

```json
{
  "timestamp": "2025-12-15T10:30:45.123456Z",
  "level": "INFO",
  "logger": "model_api",
  "message": "POST /predict - 200",
  "module": "model_api",
  "function": "predict_main",
  "line": 197,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/predict",
  "status_code": 200,
  "duration_ms": 45.23,
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0..."
}
```

**Benefits of JSON:**
- Easy to parse with tools like `jq`, `grep`, or log aggregators
- Structured for Elasticsearch, Splunk, Datadog
- Can filter by any field
- Machine-readable for automated alerts

---

## Configuration

### Environment Variables

```bash
# Set environment (development, staging, production)
export ENVIRONMENT=production

# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export LOG_LEVEL=INFO

# Logs are in JSON format when ENVIRONMENT=production
# Otherwise colored console format
```

### Programmatic Configuration

Edit [config/logging_config.py](config/logging_config.py) to customize:

```python
setup_logging(
    environment="production",      # production, staging, development
    log_level="INFO",               # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_dir="logs",                 # Log directory
    enable_console=True,            # Console output
    enable_file=True,               # File output
    enable_json=True                # JSON format (vs colored console)
)
```

---

## Using Logging in Code

### Basic Logging

```python
from config.logging_config import get_logger

logger = get_logger(__name__)

# Different log levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error!")
```

### Structured Logging with Context

```python
logger.info(
    "User prediction request",
    extra={
        "request_id": request_id,
        "user_id": "user123",
        "model_name": "LightGBM",
        "prediction": 125.5,
        "duration_ms": 42.3
    }
)
```

**Output (JSON):**
```json
{
  "timestamp": "2025-12-15T10:30:45Z",
  "level": "INFO",
  "message": "User prediction request",
  "request_id": "550e8400...",
  "user_id": "user123",
  "model_name": "LightGBM",
  "prediction": 125.5,
  "duration_ms": 42.3
}
```

### Performance Logging

Use the `PerformanceLogger` context manager to automatically time operations:

```python
from config.logging_config import PerformanceLogger, get_logger

logger = get_logger(__name__)

with PerformanceLogger("database_query", logger, table="users"):
    result = db.query("SELECT * FROM users")
    # Automatically logs duration when block exits
```

**Output:**
```
Starting: database_query
Completed: database_query (duration_ms: 123.45, status: success, table: users)
```

### Prediction Logging

Use the helper function for consistent prediction logging:

```python
from config.logging_config import log_prediction

log_prediction(
    logger=logger,
    model_name="LightGBM",
    input_data={"sqr_m": 50, "floor": 5, ...},
    prediction=125.5,
    duration_ms=45.23,
    request_id=request_id
)
```

### Request Context Logging

Automatically include request information:

```python
from middleware.logging_middleware import log_with_request_context

log_with_request_context(
    logger,
    "info",
    "Processing prediction",
    request=request,
    model="LightGBM",
    prediction=125.5
)
```

---

## Automatic Request/Response Logging

### Request Logging Middleware

**All API requests are automatically logged** with:
- Request ID (auto-generated UUID)
- HTTP method and path
- Query parameters
- Client IP and User-Agent
- Response status code
- Request duration

**Example Log:**
```json
{
  "timestamp": "2025-12-15T10:30:45Z",
  "level": "INFO",
  "message": "POST /predict - 200",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/predict",
  "query_params": null,
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "status_code": 200,
  "duration_ms": 45.23,
  "event_type": "request.completed"
}
```

### Request Body Logging

**Selected endpoints automatically log request bodies** (with sensitive data redacted):

Logged endpoints:
- `/predict`
- `/challenger_predict`
- `/compare_predictions`
- `/retrain`
- `/ab-test/predict`

**Sensitive fields are automatically redacted:**
- password
- token
- api_key
- secret
- authorization
- credit_card

**Example:**
```json
{
  "message": "Request body for /predict",
  "request_id": "550e8400...",
  "request_body": {
    "sqr_m": 50,
    "floor": 5,
    "api_key": "***REDACTED***"
  }
}
```

### Request ID Tracking

Every request gets a unique ID that flows through all logs:

```bash
# Find all logs for a specific request
grep "550e8400-e29b-41d4-a716-446655440000" logs/app.log

# Or with jq (if JSON format)
cat logs/app.log | jq 'select(.request_id == "550e8400...")'
```

**Request ID is also returned in response headers:**
```
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
```

---

## Log Levels

### When to Use Each Level

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Detailed debugging info | Variable values, function entry/exit |
| **INFO** | General information | Request processed, model prediction made |
| **WARNING** | Unexpected but handled | Rate limit approaching, slow query |
| **ERROR** | Error occurred | Exception caught, prediction failed |
| **CRITICAL** | System failure | Database down, model file missing |

### Setting Log Level

```bash
# Development - see everything
export LOG_LEVEL=DEBUG

# Production - important events only
export LOG_LEVEL=INFO

# Only errors and critical
export LOG_LEVEL=ERROR
```

---

## Log Rotation

### Automatic Rotation

**App Logs (Size-based):**
- Rotates when file reaches 10 MB
- Keeps 10 backup files
- Oldest automatically deleted

**Access Logs (Time-based):**
- Rotates daily at midnight
- Keeps 30 days of history
- Automatically cleaned up

### Manual Rotation

```bash
# Force log rotation
logrotate -f /path/to/logrotate.conf

# Or simply move/rename
mv logs/app.log logs/app.log.old
# App will create new app.log automatically
```

---

## Querying Logs

### With `grep` (Text Format)

```bash
# Find all errors
grep "ERROR" logs/app.log

# Find specific request ID
grep "550e8400-e29b-41d4-a716-446655440000" logs/app.log

# Find all predictions
grep "Prediction made" logs/app.log

# Find slow requests (> 1 second = 1000ms)
grep "duration_ms" logs/app.log | grep -E "duration_ms\":\s*[0-9]{4,}"
```

### With `jq` (JSON Format)

```bash
# All errors
cat logs/app.log | jq 'select(.level == "ERROR")'

# Requests slower than 1 second
cat logs/app.log | jq 'select(.duration_ms > 1000)'

# Predictions for specific model
cat logs/app.log | jq 'select(.model_name == "LightGBM")'

# Group by status code
cat logs/app.log | jq '.status_code' | sort | uniq -c

# Average request duration
cat logs/access.log | jq '.duration_ms' | awk '{sum+=$1; count++} END {print sum/count}'
```

### With Log Aggregation Tools

**Elasticsearch + Kibana:**
```json
POST /logs/_search
{
  "query": {
    "bool": {
      "must": [
        {"term": {"level": "ERROR"}},
        {"range": {"duration_ms": {"gte": 1000}}}
      ]
    }
  }
}
```

**Datadog:**
```
status:error duration:>1000ms
```

**Splunk:**
```
index=app_logs level=ERROR duration_ms>1000
```

---

## Monitoring & Alerts

### Setting Up Alerts

**1. Log to File → Monitor File**

```bash
# Check for errors in last minute
tail -n 1000 logs/error.log | grep "$(date -u +%Y-%m-%dT%H:%M)" | wc -l

# Alert if > 10 errors per minute
if [ $(tail -n 1000 logs/error.log | wc -l) -gt 10 ]; then
    echo "High error rate!" | mail -s "Alert" admin@example.com
fi
```

**2. Send Logs to Cloud Logging**

```python
# Google Cloud Logging
import google.cloud.logging

client = google.cloud.logging.Client()
client.setup_logging()

# Logs automatically sent to Cloud Logging
logger.error("This goes to GCP")
```

**3. Use Log Aggregation**

- **Elasticsearch:** Parse JSON logs, visualize in Kibana
- **Datadog:** Automatic parsing of JSON logs
- **Splunk:** Index JSON logs for real-time monitoring

---

## Best Practices

### ✅ DO

1. **Use structured logging with context:**
   ```python
   logger.info("Prediction made", extra={"model": "LightGBM", "prediction": 125.5})
   ```

2. **Log at appropriate levels:**
   - INFO for important business events
   - WARNING for potential issues
   - ERROR for failures

3. **Include request_id for tracing:**
   ```python
   logger.error("Prediction failed", extra={"request_id": request_id})
   ```

4. **Use performance logging for slow operations:**
   ```python
   with PerformanceLogger("model_inference", logger):
       prediction = model.predict(X)
   ```

5. **Log exceptions with context:**
   ```python
   try:
       result = risky_operation()
   except Exception as e:
       logger.error("Operation failed", exc_info=True, extra={"user_id": user_id})
   ```

### ❌ DON'T

1. **Don't log sensitive data:**
   ```python
   # Bad
   logger.info(f"User password: {password}")

   # Good
   logger.info("User authenticated", extra={"user_id": user_id})
   ```

2. **Don't log in tight loops:**
   ```python
   # Bad
   for i in range(10000):
       logger.debug(f"Processing {i}")

   # Good
   logger.info(f"Processing {len(items)} items")
   ```

3. **Don't use string formatting in log calls:**
   ```python
   # Bad (formatted even if not logged)
   logger.debug(f"Value: {expensive_calculation()}")

   # Good (only evaluated if logged)
   logger.debug("Value: %s", expensive_calculation())
   ```

4. **Don't catch exceptions without logging:**
   ```python
   # Bad
   try:
       risky()
   except:
       pass

   # Good
   try:
       risky()
   except Exception as e:
       logger.error("Operation failed", exc_info=True)
   ```

---

## Example: Complete Request Flow

### Request Comes In

```json
{
  "timestamp": "2025-12-15T10:30:45.000Z",
  "level": "DEBUG",
  "message": "POST /predict",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "request.started",
  "method": "POST",
  "path": "/predict",
  "client_ip": "192.168.1.100"
}
```

### Request Body Logged

```json
{
  "timestamp": "2025-12-15T10:30:45.010Z",
  "level": "DEBUG",
  "message": "Request body for /predict",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "request_body": {
    "sqr_m": 50,
    "floor": 5,
    "eki_walk": 8
  }
}
```

### Prediction Made

```json
{
  "timestamp": "2025-12-15T10:30:45.045Z",
  "level": "INFO",
  "message": "Prediction made with LightGBM",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "prediction",
  "model_name": "LightGBM",
  "prediction": 125.5,
  "duration_ms": 35.23
}
```

### Request Completed

```json
{
  "timestamp": "2025-12-15T10:30:45.050Z",
  "level": "INFO",
  "message": "POST /predict - 200",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "request.completed",
  "status_code": 200,
  "duration_ms": 50.12
}
```

**All tied together by `request_id`!**

---

## Troubleshooting

### Logs Not Appearing

1. Check log level:
   ```bash
   echo $LOG_LEVEL
   # Should be DEBUG or INFO
   ```

2. Check log directory exists:
   ```bash
   ls -la logs/
   ```

3. Check file permissions:
   ```bash
   chmod 755 logs/
   ```

### JSON Format Not Working

Set environment variable:
```bash
export ENVIRONMENT=production
# Restart API
```

### Request IDs Not in Logs

Make sure middleware is registered:
```python
app.add_middleware(RequestLoggingMiddleware)
```

### Performance Impact

**Logging is async and buffered:**
- Console: ~0.1ms per log
- File: ~0.01ms per log (buffered)
- JSON: Same as above (just different format)

**Tips to reduce impact:**
- Use INFO level in production (not DEBUG)
- Avoid logging in tight loops
- Use lazy evaluation (`logger.debug("Value: %s", value)`)

---

## Files Created

1. **[config/logging_config.py](config/logging_config.py)** - Centralized logging configuration
2. **[middleware/logging_middleware.py](middleware/logging_middleware.py)** - Request/response logging
3. **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** - This documentation

---

## Summary

Your API now has **production-grade logging** with:

- ✅ Automatic request/response tracking
- ✅ Unique request IDs for tracing
- ✅ Structured JSON logging for easy parsing
- ✅ Automatic log rotation
- ✅ Sensitive data redaction
- ✅ Performance timing
- ✅ Multiple log levels
- ✅ Colored console for development
- ✅ Environment-aware configuration

**Start the API and check logs:**
```bash
cd api
./start_api.sh

# In another terminal
tail -f logs/app.log
```

**Make a request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sqr_m": 50, "floor": 5, ...}'
```

**See the logs!** Every request is tracked with timing, status, and context.

🎯 **Your API is now production-ready with comprehensive logging!**
