"""Chunking strategies for GenAI ingestion.

Persona: GenAI Developer.
JD requirement covered: "Apply expertise in generative AI concepts
including document storage, chunking, vector databases, RAG
implementation..."

Three strategies are implemented, selectable via config:
  - fixed:      naive fixed-size character windows with overlap
  - recursive:  splits on paragraph -> sentence -> word boundaries, packing
                greedily up to chunk_size (this is the default; mirrors the
                widely-used "recursive character splitter" pattern)
  - semantic:   groups consecutive sentences until an embedding-similarity
                breakpoint is crossed, so chunk boundaries fall on topic
                shifts rather than arbitrary character counts
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from src.common.models import Chunk, Document

logger = logging.getLogger("documind.genai.chunking")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass
class ChunkingConfig:
    strategy: str = "recursive"  # "fixed" | "recursive" | "semantic"
    chunk_size: int = 500
    overlap: int = 50
    semantic_similarity_threshold: float = 0.55


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def chunk_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs -> sentences -> words into <=chunk_size chunks,
    carrying `overlap` characters of trailing context into the next chunk."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            sentences = _split_sentences(para)
            if not sentences:
                # No sentence boundaries found (e.g. very long token) — fall back to word split.
                sentences = para.split(" ")
            units.extend(sentences)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # carry the tail of the previous chunk forward as overlap context
                tail = current[-overlap:] if overlap else ""
                current = f"{tail} {unit}".strip()
            else:
                # a single unit longer than chunk_size — hard split it
                chunks.extend(chunk_fixed(unit, chunk_size, overlap))
                current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_semantic(
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    similarity_threshold: float = 0.55,
    max_chunk_size: int = 1200,
) -> list[str]:
    """Group consecutive sentences into a chunk until embedding similarity
    to the running chunk centroid drops below `similarity_threshold`, or the
    chunk would exceed `max_chunk_size` characters — whichever comes first.

    This demonstrates topic-boundary-aware chunking, one of the "innovative
    generative AI techniques" areas called out in the JD (hybrid RAG
    optimization starts with good chunk boundaries).
    """
    import numpy as np

    sentences = []
    for para in [p for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]:
        sentences.extend(_split_sentences(para))
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    vectors = np.array(embed_fn(sentences))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit_vectors = vectors / norms

    chunks: list[str] = []
    current_sentences = [sentences[0]]
    current_centroid = unit_vectors[0]

    for i in range(1, len(sentences)):
        sim = float(np.dot(current_centroid, unit_vectors[i]))
        candidate_text = " ".join(current_sentences + [sentences[i]])
        if sim >= similarity_threshold and len(candidate_text) <= max_chunk_size:
            current_sentences.append(sentences[i])
            # running centroid update
            n = len(current_sentences)
            current_centroid = (current_centroid * (n - 1) + unit_vectors[i]) / n
            current_centroid = current_centroid / (np.linalg.norm(current_centroid) or 1e-8)
        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_centroid = unit_vectors[i]

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None, embed_fn: Callable | None = None):
        self.config = config or ChunkingConfig()
        self.embed_fn = embed_fn  # required only for strategy == "semantic"

    def split(self, document: Document) -> list[Chunk]:
        if self.config.strategy == "fixed":
            raw_chunks = chunk_fixed(document.text, self.config.chunk_size, self.config.overlap)
        elif self.config.strategy == "recursive":
            raw_chunks = chunk_recursive(document.text, self.config.chunk_size, self.config.overlap)
        elif self.config.strategy == "semantic":
            if self.embed_fn is None:
                raise ValueError("semantic chunking requires an embed_fn")
            raw_chunks = chunk_semantic(
                document.text, self.embed_fn, self.config.semantic_similarity_threshold
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")

        chunks = [
            Chunk(
                chunk_id=f"{document.document_id}::chunk_{i}",
                document_id=document.document_id,
                text=text,
                position=i,
                metadata={"source": document.metadata.get("filename", document.source_path)},
            )
            for i, text in enumerate(raw_chunks)
        ]
        logger.info(
            "chunk.complete",
            extra={
                "document_id": document.document_id,
                "strategy": self.config.strategy,
                "chunk_count": len(chunks),
            },
        )
        return chunks
