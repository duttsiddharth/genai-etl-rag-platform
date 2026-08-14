from src.common.models import Document
from src.etl.transform import clean_text, deduplicate, scrub_pii, transform


def test_clean_text_collapses_whitespace_and_control_chars():
    raw = "Hello\x00\x01   world\t\t\ffoo\r\n\r\n\r\nbar"
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "  " not in cleaned
    assert "Hello world foo" in cleaned


def test_scrub_pii_redacts_email_and_phone():
    text = "Contact John at john.doe@example.com or 416-555-0182 for details."
    scrubbed, count = scrub_pii(text)
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert count == 2
    assert "john.doe@example.com" not in scrubbed


def test_scrub_pii_disabled_is_noop():
    text = "Email me at a@b.com"
    scrubbed, count = scrub_pii(text, enabled=False)
    assert scrubbed == text
    assert count == 0


def test_transform_sets_checksum_metadata():
    doc = Document(document_id="d1", source_path="d1.txt", source_type="txt", text="Hello   world")
    result = transform(doc)
    assert result.metadata["checksum"]
    assert result.text == "Hello world"


def test_deduplicate_drops_repeat_checksums():
    d1 = transform(Document(document_id="a", source_path="a.txt", source_type="txt", text="same content"))
    d2 = transform(Document(document_id="b", source_path="b.txt", source_type="txt", text="same content"))
    d3 = transform(Document(document_id="c", source_path="c.txt", source_type="txt", text="different content"))
    unique = deduplicate([d1, d2, d3])
    assert len(unique) == 2
    assert {d.document_id for d in unique} == {"a", "c"}
