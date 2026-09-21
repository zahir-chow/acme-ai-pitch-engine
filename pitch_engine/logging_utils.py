"""Structured observability and logging utilities for unattended batch execution (Part 3)."""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom contextual fields
        for field in ("job_id", "frame_index", "scene_id", "event_type", "latency_ms"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(
    job_id: str = "default",
    debug: bool = False,
    json_format: bool = False
) -> logging.Logger:
    """Configures the root logger with appropriate formatting for CLI or containerized execution."""
    logger = logging.getLogger("pitch_engine")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Avoid duplicate handlers if re-configured
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
    else:
        fmt = f"[%(asctime)s] [{job_id}] [%(levelname)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger.addHandler(handler)
    logger.propagate = False
    return logger
