import numpy as np

from src.genai.embeddings import HashingEmbeddingProvider, get_embedding_provider


def test_hashing_embedding_is_deterministic():
    provider = HashingEmbeddingProvider(dimensions=128)
    v1 = provider.embed(["hello world"])[0]
    v2 = provider.embed(["hello world"])[0]
    assert v1 == v2


def test_hashing_embedding_is_normalized():
    provider = HashingEmbeddingProvider(dimensions=128)
    vec = np.array(provider.embed(["some longer text with several tokens repeated tokens tokens"])[0])
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_similar_texts_score_higher_than_dissimilar():
    provider = HashingEmbeddingProvider(dimensions=256)
    a, b, c = provider.embed(
        [
            "hybrid retrieval combines dense and lexical search",
            "hybrid search fuses dense embeddings with lexical bm25 scores",
            "the weather today is sunny with a chance of rain",
        ]
    )
    a, b, c = np.array(a), np.array(b), np.array(c)
    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))
    assert sim_ab > sim_ac


def test_get_embedding_provider_default_is_hashing():
    provider = get_embedding_provider()
    assert provider.name == "local-hashing"


def test_get_embedding_provider_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        get_embedding_provider("not-a-real-provider")
