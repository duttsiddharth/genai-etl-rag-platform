"""Vector store abstraction.

Persona: GenAI Developer.
JD requirement covered: "vector databases" under GenAI concepts.

`NumpyVectorStore` is a dependency-free, cosine-similarity vector index
backed by an in-memory matrix with JSON-lines persistence — used as the
default so the reference implementation runs anywhere. `ChromaVectorStore`
wraps the popular open-source `chromadb` package and is used automatically
when it's installed and `VECTOR_STORE=chroma` is set, demonstrating the
pluggable interface the architecture doc describes (§6, "Vector store").
Both satisfy the same `VectorStore` interface, so `HybridRetriever` and
the ETL pipeline are indifferent to which backend is active.
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from src.common.models import Chunk

logger = logging.getLogger("documind.genai.vector_store")


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, cosine_similarity) pairs, highest similarity first."""
        ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class NumpyVectorStore(VectorStore):
    name = "numpy"

    def __init__(self, persist_path: str | Path = "data/processed/vector_store.jsonl"):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        with self.persist_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                chunk = Chunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    text=row["text"],
                    position=row["position"],
                    metadata=row.get("metadata", {}),
                )
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = row["vector"]

    def _persist_append(self, chunk: Chunk, vector: list[float]) -> None:
        with self.persist_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "position": chunk.position,
                        "metadata": chunk.metadata,
                        "vector": vector,
                    }
                )
                + "\n"
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vector
            self._persist_append(chunk, vector)
        logger.info("vector_store.upsert", extra={"count": len(chunks), "backend": self.name})

    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        import numpy as np

        if not self._vectors:
            return []
        ids = list(self._vectors.keys())
        matrix = np.array([self._vectors[i] for i in ids])
        q = np.array(query_vector)

        matrix_norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(q)
        denom = matrix_norms * (q_norm or 1e-8)
        denom[denom == 0] = 1e-8
        sims = (matrix @ q) / denom

        top_idx = np.argsort(-sims)[:k]
        return [(self._chunks[ids[i]], float(sims[i])) for i in top_idx]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)


class ChromaVectorStore(VectorStore):
    """Adapter around the `chromadb` open-source vector database."""

    name = "chroma"

    def __init__(self, persist_dir: str | Path = "data/processed/chroma", collection_name: str = "documind"):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb not installed. `pip install chromadb`, or set VECTOR_STORE=numpy.") from exc
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(collection_name)
        self._chunk_cache: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"document_id": c.document_id, "position": c.position, **c.metadata} for c in chunks],
        )
        for c in chunks:
            self._chunk_cache[c.chunk_id] = c
        logger.info("vector_store.upsert", extra={"count": len(chunks), "backend": self.name})

    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        result = self._collection.query(query_embeddings=[query_vector], n_results=k)
        pairs = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            chunk = Chunk(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                text=text,
                position=meta.get("position", 0),
                metadata=meta,
            )
            similarity = 1.0 - dist  # chroma default distance is cosine distance
            pairs.append((chunk, similarity))
        return pairs

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunk_cache.values())

    def count(self) -> int:
        return self._collection.count()


def get_vector_store(name: str | None = None) -> VectorStore:
    backend = name or os.getenv("VECTOR_STORE", "numpy")
    if backend == "chroma":
        return ChromaVectorStore()
    if backend == "numpy":
        return NumpyVectorStore()
    raise ValueError(f"Unknown VECTOR_STORE '{backend}'. Options: ['numpy', 'chroma']")
