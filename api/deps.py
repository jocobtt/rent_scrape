"""Dependencies shared by every router: rate limiter, admin guard, rate-limit tiers."""
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def require_admin(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Guard for state-changing admin routes (training, promotion, rollback, cleanup).

    Requires the ADMIN_API_KEY env var to be set and sent as the `X-API-Key`
    header. If the key is not configured the routes are disabled (fail closed).
    """
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoints are disabled: ADMIN_API_KEY is not configured",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )

# Rate limit configurations (requests per time window)
PREDICTION_RATE_LIMIT = "100/minute"  # Core prediction endpoints
TRAINING_RATE_LIMIT = "5/hour"        # Expensive training operations
MONITORING_RATE_LIMIT = "20/minute"   # Monitoring and reporting
DEFAULT_RATE_LIMIT = "200/minute"     # Default for other endpoints
