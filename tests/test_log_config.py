"""Tests for structured logging configuration."""
import json
import logging

from mlx_flash_compress.log_config import (
    JsonFormatter,
    TextFormatter,
    setup_logging,
)


class TestJsonFormatter:
    def test_formats_as_json(self):
        fmt = JsonFormatter(component="test-worker", worker_port=8081)
        record = logging.LogRecord(
            name="mlx_flash", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        line = fmt.format(record)
        data = json.loads(line)
        assert data["level"] == "info"
        assert data["component"] == "test-worker"
        assert data["worker_port"] == 8081
        assert data["message"] == "hello world"
        assert "timestamp" in data

    def test_includes_extra_fields(self):
        fmt = JsonFormatter(component="worker")
        record = logging.LogRecord(
            name="mlx_flash", level=logging.INFO, pathname="", lineno=0,
            msg="loaded", args=(), exc_info=None,
        )
        record.model = "Qwen3-30B"
        record.load_time_s = 4.2
        line = fmt.format(record)
        data = json.loads(line)
        assert data["model"] == "Qwen3-30B"
        assert data["load_time_s"] == 4.2

    def test_no_port_when_zero(self):
        fmt = JsonFormatter(component="test", worker_port=0)
        record = logging.LogRecord(
            name="mlx_flash", level=logging.WARNING, pathname="", lineno=0,
            msg="warn", args=(), exc_info=None,
        )
        line = fmt.format(record)
        data = json.loads(line)
        assert "worker_port" not in data


class TestTextFormatter:
    def test_formats_as_text(self):
        fmt = TextFormatter(component="rust-proxy", worker_port=0)
        record = logging.LogRecord(
            name="mlx_flash", level=logging.INFO, pathname="", lineno=0,
            msg="listening", args=(), exc_info=None,
        )
        line = fmt.format(record)
        assert "INFO" in line
        assert "rust-proxy" in line
        assert "listening" in line

    def test_includes_port(self):
        fmt = TextFormatter(component="worker", worker_port=8081)
        record = logging.LogRecord(
            name="mlx_flash", level=logging.INFO, pathname="", lineno=0,
            msg="ready", args=(), exc_info=None,
        )
        line = fmt.format(record)
        assert ":8081" in line

    def test_appends_extras(self):
        fmt = TextFormatter(component="worker")
        record = logging.LogRecord(
            name="mlx_flash", level=logging.INFO, pathname="", lineno=0,
            msg="loaded", args=(), exc_info=None,
        )
        record.model = "test-model"
        line = fmt.format(record)
        assert "model=test-model" in line


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging(component="test", json_format=False)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "mlx_flash"

    def test_json_format_handler(self):
        logger = setup_logging(component="test-json", json_format=True)
        assert any(
            isinstance(h.formatter, JsonFormatter) for h in logger.handlers
        )

    def test_text_format_handler(self):
        logger = setup_logging(component="test-text", json_format=False)
        assert any(
            isinstance(h.formatter, TextFormatter) for h in logger.handlers
        )
