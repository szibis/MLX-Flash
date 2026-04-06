"""Structured logging configuration for MLX-Flash Python components.

Produces JSON or text logs with consistent fields:
  - timestamp (ISO 8601)
  - level
  - component (python-worker, python-server)
  - worker_port (when applicable)
  - message + structured fields

Usage:
    from mlx_flash_compress.log_config import setup_logging
    logger = setup_logging(component="python-worker", port=8081, json_format=True)
    logger.info("Model loaded", extra={"model": "Qwen3-30B", "load_time_s": 4.2})
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line — compatible with Loki, Vector, Datadog, ELK."""

    def __init__(self, component: str = "python", worker_port: int = 0):
        super().__init__()
        self.component = component
        self.worker_port = worker_port

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "component": self.component,
            "target": record.name,
            "message": record.getMessage(),
        }
        if self.worker_port:
            entry["worker_port"] = self.worker_port

        # Merge extra fields (from logger.info("msg", extra={...}))
        for key in ("model", "load_time_s", "tokens", "tok_per_s", "pressure",
                     "memory_gb", "port", "action", "error", "session_id",
                     "request_id", "latency_ms", "status_code"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text format with component prefix — matches Rust tracing text output."""

    def __init__(self, component: str = "python", worker_port: int = 0):
        super().__init__()
        self.component = component
        self.worker_port = worker_port

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%.fZ")
        level = record.levelname.upper().ljust(5)
        port_str = f" :{self.worker_port}" if self.worker_port else ""
        msg = record.getMessage()

        # Append structured extras inline
        extras = []
        for key in ("model", "load_time_s", "tokens", "tok_per_s", "pressure",
                     "memory_gb", "port", "action", "error"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        extra_str = " " + " ".join(extras) if extras else ""

        return f"{ts} {level} {self.component}{port_str}: {msg}{extra_str}"


def setup_logging(
    component: str = "python-worker",
    port: int = 0,
    json_format: bool = False,
    log_file: str = None,
    level: str = "INFO",
) -> logging.Logger:
    """Configure structured logging for a Python component.

    Args:
        component: Component name (python-worker, python-server)
        port: Worker port number (added to every log line)
        json_format: True for JSON output, False for human-readable text
        log_file: Optional file path (also logs to stdout)
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance.
    """
    # Allow env override
    json_format = json_format or os.environ.get("MLX_FLASH_LOG_FORMAT", "").lower() == "json"
    log_file = log_file or os.environ.get("MLX_FLASH_LOG_FILE")
    level = os.environ.get("MLX_FLASH_LOG_LEVEL", level).upper()

    logger = logging.getLogger("mlx_flash")
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.handlers.clear()

    if json_format:
        formatter = JsonFormatter(component=component, worker_port=port)
    else:
        formatter = TextFormatter(component=component, worker_port=port)

    # Always log to stdout (cloud-native)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Optionally also log to file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
