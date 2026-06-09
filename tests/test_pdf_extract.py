from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.pdf.extract import extract_text_from_pdf, validate_pdf_bytes


def _make_text_pdf(text: str) -> bytes:
    """Build a minimal PDF with embedded text using reportlab-free approach."""
    # Use pypdf to merge: write text via a simple content stream PDF
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET"
    content_bytes = content.encode("latin-1", errors="replace")
    content_len = len(content_bytes)

    header = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    )
    stream_header = f"4 0 obj<</Length {content_len}>>stream\n".encode("ascii")
    footer = (
        b"\nendstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000261 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n420\n%%EOF\n"
    )
    return header + stream_header + content_bytes + footer


def test_validate_pdf_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_pdf_bytes(b"")


def test_extract_text_from_valid_pdf(monkeypatch) -> None:
    monkeypatch.setenv("MIN_EXTRACTED_CHARS", "10")
    from app.config import get_settings

    get_settings.cache_clear()

    long_text = "This research paper studies neural networks and their applications. " * 5
    pdf_bytes = _make_text_pdf(long_text)
    result = extract_text_from_pdf(pdf_bytes)

    assert result.page_count >= 1
    assert "neural networks" in result.text.lower()
    assert result.title


def test_extract_text_rejects_insufficient_text(monkeypatch) -> None:
    monkeypatch.setenv("MIN_EXTRACTED_CHARS", "500")
    from app.config import get_settings

    get_settings.cache_clear()

    pdf_bytes = _make_text_pdf("short")
    with pytest.raises(ValueError, match="Could not extract enough text"):
        extract_text_from_pdf(pdf_bytes)


def test_extract_text_rejects_too_many_pages(monkeypatch) -> None:
    monkeypatch.setenv("MAX_PDF_PAGES", "1")
    monkeypatch.setenv("MIN_EXTRACTED_CHARS", "5")
    from app.config import get_settings

    get_settings.cache_clear()

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "x" * 600

    with patch("app.pdf.extract.PdfReader") as mock_reader:
        instance = mock_reader.return_value
        instance.pages = [mock_page, mock_page]
        instance.metadata = None

        with pytest.raises(ValueError, match="maximum allowed"):
            extract_text_from_pdf(b"%PDF-fake")
