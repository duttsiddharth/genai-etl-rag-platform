"""Extract stage of the ETL pipeline.

Persona: GenAI Developer.
JD requirement covered: "Design and develop efficient, maintainable, and
reusable Python scripts for data extraction, transformation, and loading
(ETL) in GenAI applications."

Supports PDF, HTML, TXT, and JSON (ticket-export style) sources behind a
single `extract()` dispatch function so the pipeline stays format-agnostic.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.common.models import Document

logger = logging.getLogger("documind.etl.extract")

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".json"}


class UnsupportedSourceError(ValueError):
    pass


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup  # optional dependency

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except ImportError:
        # Fallback: lightweight regex-based tag stripper so extraction never
        # hard-fails just because an optional parsing library isn't installed.
        logger.warning("beautifulsoup4 not installed; using regex HTML stripper fallback")
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return text


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to extract PDF sources. Install with `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_json_ticket(path: Path) -> str:
    """Extract text from a ticket/record export where relevant fields are
    joined into a single text blob (subject, description, resolution, etc.)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]
    text_parts = []
    preferred_fields = ("title", "subject", "summary", "description", "body", "resolution", "notes")
    for record in records:
        if not isinstance(record, dict):
            text_parts.append(str(record))
            continue
        fields_present = [str(record[f]) for f in preferred_fields if f in record and record[f]]
        if fields_present:
            text_parts.append("\n".join(fields_present))
        else:
            # No recognized field names — fall back to flattening the whole record.
            text_parts.append(" | ".join(f"{k}: {v}" for k, v in record.items()))
    return "\n\n".join(text_parts)


_DISPATCH = {
    ".txt": ("txt", _extract_txt),
    ".html": ("html", _extract_html),
    ".htm": ("html", _extract_html),
    ".pdf": ("pdf", _extract_pdf),
    ".json": ("json", _extract_json_ticket),
}


def extract(source_path: str | Path) -> Document:
    """Extract a single source file into a raw (pre-transform) Document."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    ext = path.suffix.lower()
    if ext not in _DISPATCH:
        raise UnsupportedSourceError(
            f"Unsupported source extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    source_type, extractor_fn = _DISPATCH[ext]
    logger.info("extract.start", extra={"source_path": str(path), "source_type": source_type})
    text = extractor_fn(path)
    document_id = path.stem
    doc = Document(
        document_id=document_id,
        source_path=str(path),
        source_type=source_type,
        text=text,
        metadata={"filename": path.name, "extension": ext},
    )
    logger.info(
        "extract.complete",
        extra={"source_path": str(path), "chars_extracted": len(text)},
    )
    return doc


def extract_many(source_paths: list[str | Path]) -> list[Document]:
    documents = []
    for p in source_paths:
        try:
            documents.append(extract(p))
        except (UnsupportedSourceError, FileNotFoundError) as exc:
            logger.error("extract.failed", extra={"source_path": str(p), "error": str(exc)})
            raise
    return documents
