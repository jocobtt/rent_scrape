"""
Centralized Logging Configuration

Provides structured logging with:
- JSON formatting for easy parsing
- Request ID tracking
- Log rotation
- Different log levels for different environments
- Performance tracking
- Structured context fields
"""

import logging
import logging.handlers
import sys
import json
import os
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging

    Outputs logs in JSON format with consistent fields:
    - timestamp
    - level
    - logger
    - message
    - request_id (if available)
    - user_id (if available)
    - extra context fields
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""

        # Base log structure
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }

        # Add extra context fields
        # These are added via logger.info("msg", extra={"request_id": "123", ...})
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName',
                'relativeCreated', 'thread', 'threadName', 'exc_info',
                'exc_text', 'stack_info', 'getMessage'
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console output for development
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format: [TIMESTAMP] [LEVEL] [logger] message
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"{color}[{timestamp}] [{record.levelname:8}] [{record.name}]{reset} {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    environment: str = None,
    log_level: str = None,
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False
) -> None:
    """
    Configure application logging

    Args:
        environment: Environment name (development, staging, production)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_console: Enable console output
        enable_file: Enable file output
        enable_json: Use JSON formatting (recommended for production)
    """

    # Determine environment and log level
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO" if environment == "production" else "DEBUG")

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler (with colors in development)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        if enable_json or environment == "production":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ColoredConsoleFormatter())

        root_logger.addHandler(console_handler)

    # File handlers (with rotation)
    if enable_file:
        # General application log (with rotation)
        app_log_file = log_path / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)  # Capture all logs to file
        app_handler.setFormatter(JSONFormatter() if enable_json else ColoredConsoleFormatter())
        root_logger.addHandler(app_handler)

        # Error log (errors only)
        error_log_file = log_path / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter() if enable_json else ColoredConsoleFormatter())
        root_logger.addHandler(error_handler)

        # Access log (for API requests)
        access_log_file = log_path / "access.log"
        access_handler = logging.handlers.TimedRotatingFileHandler(
            access_log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        access_handler.setLevel(logging.INFO)
        access_handler.setFormatter(JSONFormatter())

        # Add to specific logger for access logs
        access_logger = logging.getLogger("api.access")
        access_logger.addHandler(access_handler)
        access_logger.propagate = False  # Don't propagate to root logger

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    # Log configuration
    root_logger.info(
        "Logging configured",
        extra={
            "environment": environment,
            "log_level": log_level,
            "log_dir": str(log_path.absolute()),
            "console_enabled": enable_console,
            "file_enabled": enable_file,
            "json_format": enable_json
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Performance logging helpers

class PerformanceLogger:
    """
    Context manager for logging performance metrics

    Usage:
        with PerformanceLogger("database_query", logger):
            # Your code here
            result = db.query(...)
    """

    def __init__(self, operation_name: str, logger: logging.Logger, **context):
        self.operation_name = operation_name
        self.logger = logger
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(
            f"Starting: {self.operation_name}",
            extra={"operation": self.operation_name, **self.context}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000

        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "duration_ms": round(duration_ms, 2),
                    "status": "success",
                    **self.context
                }
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "duration_ms": round(duration_ms, 2),
                    "status": "error",
                    "error_type": exc_type.__name__ if exc_type else None,
                    **self.context
                },
                exc_info=(exc_type, exc_val, exc_tb)
            )

        return False  # Don't suppress exceptions


def log_prediction(
    logger: logging.Logger,
    model_name: str,
    input_data: Dict[str, Any],
    prediction: float,
    duration_ms: float,
    request_id: str = None
) -> None:
    """
    Log prediction event with structured data

    Args:
        logger: Logger instance
        model_name: Name of the model used
        input_data: Input features
        prediction: Predicted value
        duration_ms: Prediction duration in milliseconds
        request_id: Optional request ID for tracing
    """
    logger.info(
        f"Prediction made with {model_name}",
        extra={
            "event_type": "prediction",
            "model_name": model_name,
            "prediction": prediction,
            "duration_ms": round(duration_ms, 2),
            "request_id": request_id,
            "input_features": input_data
        }
    )


def log_model_event(
    logger: logging.Logger,
    event_type: str,
    model_name: str,
    version: str = None,
    metrics: Dict[str, float] = None,
    **kwargs
) -> None:
    """
    Log model lifecycle events (training, promotion, rollback, etc.)

    Args:
        logger: Logger instance
        event_type: Type of event (training, promotion, rollback, etc.)
        model_name: Name of the model
        version: Model version
        metrics: Model metrics
        **kwargs: Additional context
    """
    log_data = {
        "event_type": f"model.{event_type}",
        "model_name": model_name,
    }

    if version:
        log_data["version"] = version
    if metrics:
        log_data["metrics"] = metrics

    log_data.update(kwargs)

    logger.info(f"Model event: {event_type}", extra=log_data)
