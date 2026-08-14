from src.common.models import Document
from src.genai.chunking import Chunker, ChunkingConfig, chunk_fixed, chunk_recursive


def test_chunk_fixed_respects_size_and_overlap():
    text = "x" * 2000
    chunks = chunk_fixed(text, chunk_size=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)
    # 2000 chars, step=450 -> ceil(2000/450) chunks roughly
    assert len(chunks) >= 4


def test_chunk_fixed_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_fixed("hello world", chunk_size=10, overlap=10)


def test_chunk_recursive_preserves_content():
    text = "First paragraph sentence one. Sentence two.\n\nSecond paragraph sentence.\n\nThird paragraph."
    chunks = chunk_recursive(text, chunk_size=60, overlap=10)
    assert len(chunks) >= 1
    joined = " ".join(chunks)
    assert "First paragraph" in joined
    assert "Third paragraph" in joined


def test_chunker_produces_chunk_objects_with_ids():
    doc = Document(document_id="doc1", source_path="doc1.txt", source_type="txt", text="A" * 1200)
    chunker = Chunker(ChunkingConfig(strategy="fixed", chunk_size=400, overlap=40))
    chunks = chunker.split(doc)
    assert len(chunks) > 1
    assert all(c.chunk_id.startswith("doc1::chunk_") for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunker_unknown_strategy_raises():
    import pytest

    doc = Document(document_id="doc1", source_path="doc1.txt", source_type="txt", text="hello")
    chunker = Chunker(ChunkingConfig(strategy="not-a-strategy"))
    with pytest.raises(ValueError):
        chunker.split(doc)
