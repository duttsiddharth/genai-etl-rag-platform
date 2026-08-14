"""Transform stage of the ETL pipeline.

Persona: GenAI Developer.
Cleans/normalizes extracted text, applies a PII-scrub hook (NFR / risk
mitigation R-2 in the risk register), and deduplicates near-identical
documents before they reach chunking/embedding.
"""
from __future__ import annotations

import logging
import re

from src.common.models import Document, checksum

logger = logging.getLogger("documind.etl.transform")

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")


def clean_text(text: str) -> str:
    """Collapse whitespace, strip control characters, normalize line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def scrub_pii(text: str, enabled: bool = True) -> tuple[str, int]:
    """Redact emails and phone numbers. Returns (scrubbed_text, redaction_count).

    This is a demonstration-grade regex scrubber — the hook is deliberately
    a single, swappable function so a production deployment can plug in a
    proper PII-detection service (e.g. AWS Comprehend / Presidio) without
    touching pipeline orchestration code.
    """
    if not enabled:
        return text, 0
    count = 0

    def _redact_email(match: re.Match) -> str:
        nonlocal count
        count += 1
        return "[REDACTED_EMAIL]"

    def _redact_phone(match: re.Match) -> str:
        nonlocal count
        count += 1
        return "[REDACTED_PHONE]"

    text = _EMAIL_RE.sub(_redact_email, text)
    text = _PHONE_RE.sub(_redact_phone, text)
    return text, count


def transform(document: Document, *, scrub: bool = True) -> Document:
    """Apply cleaning + PII scrubbing, returning a new normalized Document."""
    cleaned = clean_text(document.text)
    scrubbed, redactions = scrub_pii(cleaned, enabled=scrub)

    logger.info(
        "transform.complete",
        extra={
            "document_id": document.document_id,
            "chars_before": len(document.text),
            "chars_after": len(scrubbed),
            "pii_redactions": redactions,
        },
    )

    document.text = scrubbed
    document.metadata["pii_redactions"] = redactions
    document.metadata["checksum"] = checksum(scrubbed)
    return document


def deduplicate(documents: list[Document]) -> list[Document]:
    """Drop documents whose normalized checksum has already been seen."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        cs = doc.metadata.get("checksum") or checksum(doc.text)
        if cs in seen:
            logger.warning("transform.duplicate_skipped", extra={"document_id": doc.document_id})
            continue
        seen.add(cs)
        unique.append(doc)
    return unique
