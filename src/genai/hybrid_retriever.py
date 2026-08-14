"""Hybrid (dense + lexical) retrieval with score fusion.

Persona: GenAI Developer.
JD requirement covered: "Conduct research and experiments to explore
innovative generative AI techniques such as ... hybrid RAG optimization
... " This module is the direct implementation of that line item — see
`docs/11_rag_evaluation_report.md` for the experiment that motivates the
default fusion weight.

Fusion formula: fused_score = alpha * dense_score + (1 - alpha) * lexical_score
- alpha = 1.0  -> pure dense/semantic retrieval
- alpha = 0.0  -> pure BM25 lexical retrieval
- 0 < alpha < 1 -> hybrid (default 0.6, tuned empirically — see eval report)
"""
from __future__ import annotations

import logging
import re

from src.common.models import Chunk, RetrievalResult
from src.genai.embeddings import EmbeddingProvider
from src.genai.vector_store import VectorStore

logger = logging.getLogger("documind.genai.hybrid_retriever")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class BM25Index:
    """Thin wrapper around `rank_bm25.BM25Okapi`, rebuilt from the vector
    store's chunk set. Kept separate from the vector store because lexical
    indexes and vector indexes have very different update/rebuild
    characteristics in production systems."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._chunk_ids = [c.chunk_id for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank_bm25 not installed. `pip install rank-bm25`.") from exc
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores, self._chunks), key=lambda x: -x[1])[:k]
        return [(chunk, float(score)) for (_id, score, chunk) in ranked]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        alpha: float = 0.6,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.alpha = alpha

    def _bm25_index(self) -> BM25Index:
        # Rebuilt per-call for simplicity/correctness in this reference
        # implementation; a production system would incrementally update
        # or periodically rebuild this off the hot path.
        return BM25Index(self.vector_store.all_chunks())

    def retrieve(self, query: str, k: int = 4, alpha: float | None = None) -> list[RetrievalResult]:
        alpha = self.alpha if alpha is None else alpha
        candidate_pool = max(k * 3, 10)

        query_vector = self.embedding_provider.embed_query(query)
        dense_hits = self.vector_store.search(query_vector, candidate_pool)
        dense_scores = {chunk.chunk_id: score for chunk, score in dense_hits}
        dense_chunks = {chunk.chunk_id: chunk for chunk, _ in dense_hits}

        lexical_hits: list[tuple[Chunk, float]] = []
        if alpha < 1.0:
            lexical_hits = self._bm25_index().search(query, candidate_pool)
        lexical_scores = {chunk.chunk_id: score for chunk, score in lexical_hits}
        for chunk, _ in lexical_hits:
            dense_chunks.setdefault(chunk.chunk_id, chunk)

        norm_dense = _minmax_normalize(dense_scores)
        norm_lexical = _minmax_normalize(lexical_scores)

        all_ids = set(norm_dense) | set(norm_lexical)
        fused: list[RetrievalResult] = []
        for cid in all_ids:
            d = norm_dense.get(cid, 0.0)
            lex = norm_lexical.get(cid, 0.0)
            fused_score = alpha * d + (1 - alpha) * lex
            fused.append(
                RetrievalResult(
                    chunk=dense_chunks[cid],
                    dense_score=dense_scores.get(cid, 0.0),
                    lexical_score=lexical_scores.get(cid, 0.0),
                    fused_score=fused_score,
                )
            )

        fused.sort(key=lambda r: -r.fused_score)
        results = fused[:k]
        logger.info(
            "hybrid_retriever.retrieve",
            extra={"query": query, "alpha": alpha, "candidates": len(all_ids), "returned": len(results)},
        )
        return results
