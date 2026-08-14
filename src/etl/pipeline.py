"""ETL orchestration pipeline: extract -> transform -> load -> chunk -> embed -> index.

Persona: GenAI Developer.
This is the top-level entry point the API's /ingest route and the demo
script both call. Each stage is logged with duration so the whole run is
observable end-to-end (NFR-5).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.common.models import Document
from src.etl.extract import extract
from src.etl.load import ManifestStore, ProcessedStore, load
from src.etl.transform import transform
from src.genai.chunking import Chunker, ChunkingConfig
from src.genai.embeddings import EmbeddingProvider, get_embedding_provider
from src.genai.vector_store import VectorStore, get_vector_store

logger = logging.getLogger("documind.etl.pipeline")


@dataclass
class IngestSummary:
    documents_processed: int = 0
    chunks_indexed: int = 0
    skipped_unchanged: int = 0
    manifest: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0


class ETLPipeline:
    def __init__(
        self,
        manifest_path: str | Path | None = None,
        processed_dir: str | Path | None = None,
        chunking_config: ChunkingConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ):
        # DOCUMIND_DATA_DIR lets a deployment target with a read-only
        # filesystem (e.g. a serverless function, where only /tmp is
        # writable) redirect ETL output without touching call sites. Falls
        # back to /tmp automatically when VERCEL is set (Vercel always sets
        # this in the function runtime) so the path is correct even if the
        # deployment forgot to configure DOCUMIND_DATA_DIR explicitly.
        default_data_dir = "/tmp/documind_data" if os.getenv("VERCEL") else "data/processed"
        data_dir = os.getenv("DOCUMIND_DATA_DIR", default_data_dir)
        manifest_path = manifest_path or f"{data_dir}/manifest.jsonl"
        processed_dir = processed_dir or data_dir

        self.manifest_store = ManifestStore(manifest_path)
        self.processed_store = ProcessedStore(processed_dir)
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.chunker = Chunker(
            chunking_config or ChunkingConfig(),
            embed_fn=self.embedding_provider.embed,
        )
        self.vector_store = vector_store or get_vector_store()

    def run(self, source_paths: list[str | Path], force: bool = False) -> IngestSummary:
        start = time.time()
        summary = IngestSummary()

        for path in source_paths:
            t0 = time.time()
            doc: Document = extract(path)
            doc = transform(doc)

            if not force and self.manifest_store.already_ingested(
                doc.document_id, doc.metadata.get("checksum", "")
            ):
                logger.info("pipeline.skip_unchanged", extra={"document_id": doc.document_id})
                summary.skipped_unchanged += 1
                continue

            chunks = self.chunker.split(doc)
            texts = [c.text for c in chunks]
            vectors = self.embedding_provider.embed(texts) if texts else []
            self.vector_store.upsert(chunks, vectors)

            entry = load(doc, len(chunks), self.manifest_store, self.processed_store)
            summary.documents_processed += 1
            summary.chunks_indexed += len(chunks)
            summary.manifest.append(entry.__dict__)

            logger.info(
                "pipeline.document_complete",
                extra={"document_id": doc.document_id, "seconds": round(time.time() - t0, 3)},
            )

        summary.duration_seconds = round(time.time() - start, 3)
        logger.info(
            "pipeline.run_complete",
            extra={
                "documents_processed": summary.documents_processed,
                "chunks_indexed": summary.chunks_indexed,
                "seconds": summary.duration_seconds,
            },
        )
        return summary

    def run_directory(self, source_dir: str | Path, force: bool = False) -> IngestSummary:
        from src.etl.extract import SUPPORTED_EXTENSIONS

        source_dir = Path(source_dir)
        paths = [p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        return self.run(paths, force=force)
