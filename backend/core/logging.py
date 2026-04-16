"""
KG-MAG — Structured logging configuration.
Uses structlog for JSON-structured logs in production,
pretty console output in development.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", environment: str = "development") -> None:
    """
    Configure structlog + stdlib logging.
    Call once at application startup.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if environment == "production":
        # JSON output for log aggregators (Datadog, CloudWatch, etc.)
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        formatter = logging.Formatter("%(message)s")
    else:
        # Human-readable console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        formatter = logging.Formatter("%(message)s")

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),  # ← FIXED: Use stdlib logger factory
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib so third-party libs (uvicorn, httpx) integrate
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Silence noisy libraries
    for lib in ["httpx", "httpcore", "faiss", "sentence_transformers"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)