"""Load stage of the ETL pipeline.

Persona: GenAI Developer.
Writes normalized documents to a processed-data store and maintains an
append-only ingestion manifest — the audit trail referenced in
NFR-3 ("Reliability") and BR-3 ("auditable — every answer traceable").
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.common.models import Document, ManifestEntry

logger = logging.getLogger("documind.etl.load")


class ManifestStore:
    """Append-only JSON-lines manifest of every document ever ingested."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.manifest_path.touch()

    def record(self, entry: ManifestEntry) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__) + "\n")
        logger.info("load.manifest_recorded", extra={"document_id": entry.document_id})

    def read_all(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        entries = []
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def already_ingested(self, document_id: str, checksum: str) -> bool:
        """Idempotency check: skip re-ingesting an unchanged document."""
        for entry in self.read_all():
            if entry["document_id"] == document_id and entry["checksum"] == checksum:
                return True
        return False


class ProcessedStore:
    """Writes the cleaned document text to a local 'processed' area.

    In a cloud deployment this would delegate to the `src.cloud` provider
    interface (S3 / Blob Storage / GCS) instead of local disk — see
    `src/cloud/README` note in `03_architecture_design.md` §7.
    """

    def __init__(self, processed_dir: str | Path):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def write(self, document: Document) -> Path:
        out_path = self.processed_dir / f"{document.document_id}.json"
        out_path.write_text(
            json.dumps(
                {
                    "document_id": document.document_id,
                    "source_path": document.source_path,
                    "source_type": document.source_type,
                    "text": document.text,
                    "metadata": document.metadata,
                    "extracted_at": document.extracted_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_path


def load(
    document: Document,
    chunk_count: int,
    manifest_store: ManifestStore,
    processed_store: ProcessedStore,
) -> ManifestEntry:
    processed_store.write(document)
    entry = ManifestEntry(
        document_id=document.document_id,
        source_path=document.source_path,
        checksum=document.metadata.get("checksum", ""),
        chunk_count=chunk_count,
        ingested_at=time.time(),
    )
    manifest_store.record(entry)
    logger.info(
        "load.complete",
        extra={"document_id": document.document_id, "chunk_count": chunk_count},
    )
    return entry
