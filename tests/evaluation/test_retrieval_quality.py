"""RAG evaluation harness: dense-only vs. lexical-only vs. hybrid retrieval.

Persona: QA / Evaluation Engineer, with GenAI Developer.
Produces `artifacts/eval/retrieval_quality_report.json`, which is read
verbatim into `docs/11_rag_evaluation_report.md`.

Metrics: hit-rate@k (did the expected chunk appear in the top-k) and MRR
(mean reciprocal rank of the first relevant chunk).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.etl.extract import extract
from src.etl.transform import transform
from src.genai.chunking import Chunker, ChunkingConfig
from src.genai.embeddings import HashingEmbeddingProvider
from src.genai.hybrid_retriever import HybridRetriever
from src.genai.vector_store import NumpyVectorStore

SAMPLE_DOCS_DIR = Path("data/sample_docs")


@dataclass
class EvalQuery:
    question: str
    expected_source_contains: str  # substring expected in the winning chunk's source filename


EVAL_QUERIES = [
    EvalQuery("What is the default hybrid retrieval alpha value?", "genai_platform_overview"),
    EvalQuery("What does the /health 503 response usually mean?", "incident_response_faq"),
    EvalQuery("How do I roll back a bad Kubernetes deployment?", "cloud_deployment_guide"),
    EvalQuery("What caused ticket TCK-1042?", "support_tickets_export"),
    EvalQuery("What Python version should a new developer use to set up their environment?", "onboarding_guide"),
    EvalQuery("Which cloud service hosts the API on Azure?", "cloud_deployment_guide"),
    EvalQuery("What happens when the agent hits its max_steps limit?", "incident_response_faq"),
    EvalQuery("What regex issue was found in the PII scrubber?", "support_tickets_export"),
]


def _build_index() -> tuple[NumpyVectorStore, HashingEmbeddingProvider]:
    embedder = HashingEmbeddingProvider(dimensions=256)
    store = NumpyVectorStore(persist_path="artifacts/eval/eval_vector_store.jsonl")
    # start clean each run
    store._chunks.clear()
    store._vectors.clear()
    Path(store.persist_path).write_text("")

    chunker = Chunker(ChunkingConfig(strategy="recursive", chunk_size=400, overlap=40))
    for path in sorted(SAMPLE_DOCS_DIR.iterdir()):
        if path.suffix.lower() not in {".txt", ".html", ".json"}:
            continue
        doc = transform(extract(path))
        chunks = chunker.split(doc)
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert(chunks, vectors)
    return store, embedder


def _hit_rate_and_mrr(retriever: HybridRetriever, alpha: float, k: int = 5) -> tuple[float, float]:
    hits = 0
    reciprocal_ranks = []
    for q in EVAL_QUERIES:
        results = retriever.retrieve(q.question, k=k, alpha=alpha)
        rank = None
        for i, r in enumerate(results, start=1):
            if q.expected_source_contains in r.chunk.metadata.get("source", ""):
                rank = i
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    hit_rate = hits / len(EVAL_QUERIES)
    mrr = sum(reciprocal_ranks) / len(EVAL_QUERIES)
    return hit_rate, mrr


def test_hybrid_retrieval_matches_or_beats_single_strategy():
    store, embedder = _build_index()
    retriever = HybridRetriever(store, embedder, alpha=0.6)

    dense_hit5, dense_mrr = _hit_rate_and_mrr(retriever, alpha=1.0, k=5)
    lexical_hit5, lexical_mrr = _hit_rate_and_mrr(retriever, alpha=0.0, k=5)
    hybrid_hit5, hybrid_mrr = _hit_rate_and_mrr(retriever, alpha=0.6, k=5)

    dense_hit1, _ = _hit_rate_and_mrr(retriever, alpha=1.0, k=1)
    lexical_hit1, _ = _hit_rate_and_mrr(retriever, alpha=0.0, k=1)
    hybrid_hit1, _ = _hit_rate_and_mrr(retriever, alpha=0.6, k=1)

    # Alpha sweep, for the tuning narrative in the eval report.
    sweep = {}
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        hr, mrr = _hit_rate_and_mrr(retriever, alpha=a, k=5)
        sweep[str(a)] = {"hit_rate_at_5": hr, "mrr": round(mrr, 4)}

    report = {
        "eval_query_count": len(EVAL_QUERIES),
        "dense_only": {"alpha": 1.0, "hit_rate_at_1": dense_hit1, "hit_rate_at_5": dense_hit5, "mrr": round(dense_mrr, 4)},
        "lexical_only": {"alpha": 0.0, "hit_rate_at_1": lexical_hit1, "hit_rate_at_5": lexical_hit5, "mrr": round(lexical_mrr, 4)},
        "hybrid": {"alpha": 0.6, "hit_rate_at_1": hybrid_hit1, "hit_rate_at_5": hybrid_hit5, "mrr": round(hybrid_mrr, 4)},
        "alpha_sweep_hit_rate_at_5": sweep,
    }
    Path("artifacts/eval").mkdir(parents=True, exist_ok=True)
    Path("artifacts/eval/retrieval_quality_report.json").write_text(json.dumps(report, indent=2))

    # The core claim under test: hybrid should not be worse than the best
    # single-strategy baseline on this labeled query set.
    assert hybrid_hit5 >= max(dense_hit5, lexical_hit5) - 1e-9
