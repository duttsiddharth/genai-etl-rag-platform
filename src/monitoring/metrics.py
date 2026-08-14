"""Prometheus metrics for the API and GenAI pipeline.

Persona: MLOps/DevOps Engineer.
JD requirement covered: "Champion DevOps and MLOps practices focusing on
... AI model monitoring" and "monitoring and logging solutions to ensure
the performance and reliability of AI models."
"""
from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dependency missing
    PROMETHEUS_AVAILABLE = False

if PROMETHEUS_AVAILABLE:
    HTTP_REQUESTS_TOTAL = Counter(
        "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
    )
    REQUEST_LATENCY_SECONDS = Histogram(
        "request_latency_seconds", "Request latency in seconds", ["path"]
    )
    RETRIEVAL_HIT_RATE = Gauge(
        "retrieval_hit_rate", "Most recent offline evaluation hit-rate@k for retrieval"
    )
    TOKENS_USED_TOTAL = Counter(
        "tokens_used_total", "Approximate tokens consumed by generation calls"
    )
    AGENT_STEPS_USED = Histogram(
        "agent_steps_used", "Number of steps used per agent run"
    )
    LOW_CONFIDENCE_ANSWERS_TOTAL = Counter(
        "low_confidence_answers_total", "Count of RAG answers with an empty/low retrieval score"
    )

    def render_latest() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST

else:
    # Dependency-free no-op fallbacks so the app still boots without the
    # optional `prometheus-client` package installed.
    class _NoopMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    HTTP_REQUESTS_TOTAL = _NoopMetric()
    REQUEST_LATENCY_SECONDS = _NoopMetric()
    RETRIEVAL_HIT_RATE = _NoopMetric()
    TOKENS_USED_TOTAL = _NoopMetric()
    AGENT_STEPS_USED = _NoopMetric()
    LOW_CONFIDENCE_ANSWERS_TOTAL = _NoopMetric()

    def render_latest() -> tuple[bytes, str]:
        return b"# prometheus_client not installed\n", "text/plain"
