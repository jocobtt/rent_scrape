"""
Request/Response Logging Middleware

Logs all API requests and responses with:
- Request ID generation and tracking
- Request/response timing
- Status codes
- Error tracking
- User context (when available)
"""

import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable
import json

# Get loggers
logger = logging.getLogger(__name__)
access_logger = logging.getLogger("api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses

    Adds request_id to all requests and logs:
    - Request method, path, query params
    - Response status code
    - Request duration
    - Client IP
    - User agent
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Store request_id in request state for use in endpoints
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Prepare request context
        request_context = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else None,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }

        # Log request
        logger.debug(
            f"{request.method} {request.url.path}",
            extra={
                **request_context,
                "event_type": "request.started"
            }
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Prepare response context
            response_context = {
                **request_context,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "event_type": "request.completed"
            }

            # Add request_id to response headers
            response.headers["X-Request-ID"] = request_id

            # Log based on status code
            if response.status_code >= 500:
                logger.error(
                    f"{request.method} {request.url.path} - {response.status_code}",
                    extra=response_context
                )
            elif response.status_code >= 400:
                logger.warning(
                    f"{request.method} {request.url.path} - {response.status_code}",
                    extra=response_context
                )
            else:
                logger.info(
                    f"{request.method} {request.url.path} - {response.status_code}",
                    extra=response_context
                )

            # Log to access log (always INFO level)
            access_logger.info(
                f'{request_context["client_ip"]} - "{request.method} {request.url.path}" {response.status_code}',
                extra=response_context
            )

            return response

        except Exception as e:
            # Calculate duration even on error
            duration_ms = (time.time() - start_time) * 1000

            # Log exception
            logger.error(
                f"{request.method} {request.url.path} - Exception: {str(e)}",
                extra={
                    **request_context,
                    "duration_ms": round(duration_ms, 2),
                    "event_type": "request.failed",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )

            # Re-raise the exception to let FastAPI handle it
            raise


class RequestBodyLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log request bodies (with sensitive data filtering)

    Only logs for specific endpoints and filters sensitive fields
    """

    # Fields to redact from logs
    SENSITIVE_FIELDS = {
        'password', 'token', 'api_key', 'secret', 'authorization',
        'credit_card', 'ssn', 'auth'
    }

    # Endpoints to log request bodies for
    LOG_BODY_PATHS = {
        '/predict',
        '/challenger_predict',
        '/compare_predictions',
        '/retrain',
        '/ab-test/predict'
    }

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if we should log body for this path
        if request.url.path not in self.LOG_BODY_PATHS:
            return await call_next(request)

        # Only log bodies for POST/PUT/PATCH
        if request.method not in ['POST', 'PUT', 'PATCH']:
            return await call_next(request)

        # Read and log request body
        try:
            body_bytes = await request.body()

            if body_bytes:
                try:
                    body_json = json.loads(body_bytes.decode('utf-8'))
                    # Redact sensitive fields
                    filtered_body = self._filter_sensitive_data(body_json)

                    logger.debug(
                        f"Request body for {request.url.path}",
                        extra={
                            "request_id": getattr(request.state, 'request_id', None),
                            "path": request.url.path,
                            "request_body": filtered_body,
                            "event_type": "request.body"
                        }
                    )
                except json.JSONDecodeError:
                    # Not JSON, log as text (truncated)
                    body_text = body_bytes.decode('utf-8', errors='replace')
                    logger.debug(
                        f"Request body for {request.url.path} (non-JSON)",
                        extra={
                            "request_id": getattr(request.state, 'request_id', None),
                            "path": request.url.path,
                            "body_preview": body_text[:200],
                            "event_type": "request.body"
                        }
                    )

            # Create new request with body
            # This is necessary because we consumed the body stream
            async def receive():
                return {"type": "http.request", "body": body_bytes}

            request._receive = receive

        except Exception as e:
            logger.warning(
                f"Failed to log request body: {str(e)}",
                extra={"request_id": getattr(request.state, 'request_id', None)}
            )

        return await call_next(request)

    def _filter_sensitive_data(self, data: dict) -> dict:
        """Recursively filter sensitive fields from dict"""
        if not isinstance(data, dict):
            return data

        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_FIELDS):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value

        return filtered


def get_request_id(request: Request) -> str:
    """
    Get request ID from request state

    Args:
        request: FastAPI Request object

    Returns:
        Request ID string
    """
    return getattr(request.state, 'request_id', 'unknown')


def log_with_request_context(
    logger: logging.Logger,
    level: str,
    message: str,
    request: Request = None,
    **extra
):
    """
    Log message with request context

    Args:
        logger: Logger instance
        level: Log level (info, warning, error, etc.)
        message: Log message
        request: Optional FastAPI Request object
        **extra: Additional context fields
    """
    context = extra.copy()

    if request:
        context["request_id"] = get_request_id(request)
        context["path"] = request.url.path
        context["method"] = request.method

    log_func = getattr(logger, level.lower())
    log_func(message, extra=context)
