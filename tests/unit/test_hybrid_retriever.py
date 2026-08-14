from src.common.models import Chunk
from src.genai.embeddings import HashingEmbeddingProvider
from src.genai.hybrid_retriever import HybridRetriever, _minmax_normalize
from src.genai.vector_store import NumpyVectorStore


def test_minmax_normalize_handles_constant_scores():
    scores = {"a": 5.0, "b": 5.0}
    normalized = _minmax_normalize(scores)
    assert normalized == {"a": 0.0, "b": 0.0}


def test_minmax_normalize_scales_to_unit_range():
    scores = {"a": 0.0, "b": 5.0, "c": 10.0}
    normalized = _minmax_normalize(scores)
    assert normalized["a"] == 0.0
    assert normalized["c"] == 1.0
    assert 0.0 < normalized["b"] < 1.0


def test_hybrid_retriever_alpha_extremes(tmp_path):
    embedder = HashingEmbeddingProvider(dimensions=128)
    store = NumpyVectorStore(persist_path=tmp_path / "vs.jsonl")

    chunks = [
        Chunk(chunk_id="c1", document_id="d1", text="hybrid retrieval fuses dense and lexical scores", position=0),
        Chunk(chunk_id="c2", document_id="d1", text="the incident runbook covers rollback procedures", position=1),
        Chunk(chunk_id="c3", document_id="d1", text="onboarding guide for new developers joining the team", position=2),
    ]
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)

    retriever = HybridRetriever(store, embedder, alpha=0.6)

    # Pure lexical (alpha=0.0) must return the chunk containing the exact keyword "rollback".
    lexical_results = retriever.retrieve("rollback procedures", k=1, alpha=0.0)
    assert lexical_results[0].chunk.chunk_id == "c2"

    # Default hybrid retrieval should also surface the same clearly-relevant chunk first.
    hybrid_results = retriever.retrieve("rollback procedures", k=1)
    assert hybrid_results[0].chunk.chunk_id == "c2"


def test_hybrid_retriever_returns_empty_on_empty_store(tmp_path):
    embedder = HashingEmbeddingProvider(dimensions=64)
    store = NumpyVectorStore(persist_path=tmp_path / "empty.jsonl")
    retriever = HybridRetriever(store, embedder)
    assert retriever.retrieve("anything", k=3) == []
