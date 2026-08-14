"""Pydantic request/response schemas for the API layer.

Persona: GenAI Developer. See docs/04_api_specification.md for the
human-readable contract these mirror.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_paths: list[str] | None = Field(
        default=None,
        description="Explicit files to ingest. If omitted, ingests everything under the configured source directory.",
    )
    force: bool = Field(default=False, description="Re-ingest even if the checksum is unchanged.")


class ManifestEntryOut(BaseModel):
    document_id: str
    source_path: str
    checksum: str
    chunk_count: int


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_indexed: int
    skipped_unchanged: int
    manifest: list[dict]
    duration_seconds: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0, description="Hybrid fusion weight; None uses server default.")


class CitationOut(BaseModel):
    chunk_id: str
    score: float
    source: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieval_strategy: str
    latency_ms: float


class AgentRunRequest(BaseModel):
    goal: str = Field(min_length=1)
    max_steps: int = Field(default=6, ge=1, le=20)


class AgentStepOut(BaseModel):
    step: int
    action: str
    args: dict
    observation: str | None


class AgentRunResponse(BaseModel):
    answer: str
    steps: list[AgentStepOut]
    steps_used: int
    max_steps: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    documents_indexed: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
