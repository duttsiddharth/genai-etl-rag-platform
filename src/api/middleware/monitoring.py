"""Request logging + metrics middleware.

Persona: MLOps/DevOps Engineer.
"""
from __future__ import annotations

import logging
import time

from fastapi import Request

from src.monitoring.logger import get_correlation_id, new_correlation_id
from src.monitoring.metrics import HTTP_REQUESTS_TOTAL, REQUEST_LATENCY_SECONDS

logger = logging.getLogger("documind.api.access")


async def request_logging_middleware(request: Request, call_next):
    new_correlation_id()
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    HTTP_REQUESTS_TOTAL.labels(method=request.method, path=request.url.path, status=str(response.status_code)).inc()
    REQUEST_LATENCY_SECONDS.labels(path=request.url.path).observe(duration)

    logger.info(
        "api.request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration * 1000, 1),
            "correlation_id": get_correlation_id(),
        },
    )
    response.headers["X-Correlation-Id"] = get_correlation_id()
    return response
