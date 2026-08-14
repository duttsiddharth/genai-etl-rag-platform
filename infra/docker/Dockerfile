# DocuMind AI — multi-stage container build
# Persona: MLOps/DevOps Engineer
# JD requirement covered: "Previous experience leveraging tools like
# Docker, Kubernetes, and Git to build and manage AI pipelines."

# ---- Builder stage ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_PROVIDER=local-hashing \
    VECTOR_STORE=numpy \
    LLM_PROVIDER=stub \
    CLOUD_PROVIDER=local

COPY src ./src
COPY data/sample_docs ./data/sample_docs
COPY scripts ./scripts

RUN mkdir -p /app/data/processed /app/artifacts/finetuned && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
