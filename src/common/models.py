"""Shared domain models used across the ETL, GenAI, and API layers.

Kept dependency-free (stdlib only) so every layer can import this module
without pulling in heavy ML dependencies.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


def checksum(text: str) -> str:
    """Deterministic short checksum used for change detection during ingestion."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Document:
    """A normalized document produced by the ETL pipeline, ready for chunking."""

    document_id: str
    source_path: str
    source_type: str  # "pdf" | "html" | "txt" | "json"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    extracted_at: float = field(default_factory=time.time)

    @property
    def content_checksum(self) -> str:
        return checksum(self.text)


@dataclass
class Chunk:
    """A retrieval-sized unit of a Document, ready to be embedded and indexed."""

    chunk_id: str
    document_id: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single scored retrieval candidate returned by the hybrid retriever."""

    chunk: Chunk
    dense_score: float
    lexical_score: float
    fused_score: float


@dataclass
class ManifestEntry:
    """One row of the ingestion manifest — the audit trail for ETL loads."""

    document_id: str
    source_path: str
    checksum: str
    chunk_count: int
    ingested_at: float = field(default_factory=time.time)
