import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("EMBEDDING_PROVIDER", "local-hashing")
os.environ.setdefault("VECTOR_STORE", "numpy")
os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("API_KEY", "demo-local-key")

from src.api.main import app  # noqa: E402

HEADERS = {"X-API-Key": "demo-local-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "vector_store" in body


def test_ingest_requires_api_key(client):
    resp = client.post("/ingest", json={"force": True})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_ingest_sample_corpus(client):
    resp = client.post("/ingest", json={"force": True}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents_processed"] >= 4
    assert body["chunks_indexed"] > 0


def test_query_returns_grounded_answer_with_citations(client):
    # Ensure corpus is ingested first (module-scoped client, tests run in file order).
    client.post("/ingest", json={"force": True}, headers=HEADERS)

    resp = client.post(
        "/query",
        json={"question": "What is the default hybrid retrieval alpha?", "k": 4},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert len(body["citations"]) >= 1
    assert body["retrieval_strategy"] == "hybrid"


def test_query_rejects_empty_question(client):
    resp = client.post("/query", json={"question": ""}, headers=HEADERS)
    assert resp.status_code == 422


def test_agent_run_completes_within_step_budget(client):
    client.post("/ingest", json={"force": True}, headers=HEADERS)

    resp = client.post(
        "/agent/run",
        json={
            "goal": "Compare the rollback procedure and the vector store timeout incident, then summarize.",
            "max_steps": 6,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["steps_used"] <= body["max_steps"]
    assert body["steps"][-1]["action"] == "finish"
    assert body["answer"]


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
