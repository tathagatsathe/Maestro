from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

from app.config import get_settings


@dataclass(frozen=True)
class ExtractedPaper:
    title: str
    text: str
    page_count: int


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def validate_pdf_bytes(file_bytes: bytes) -> None:
    settings = get_settings()
    max_bytes = settings.max_pdf_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(
            f"PDF exceeds maximum size of {settings.max_pdf_size_mb} MB"
        )
    if len(file_bytes) == 0:
        raise ValueError("PDF file is empty")


def extract_text_from_pdf(file_bytes: bytes) -> ExtractedPaper:
    """Extract text from a PDF byte stream with validation."""
    validate_pdf_bytes(file_bytes)
    settings = get_settings()

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError("Invalid or corrupted PDF file") from exc

    page_count = len(reader.pages)
    if page_count > settings.max_pdf_pages:
        raise ValueError(
            f"PDF has {page_count} pages; maximum allowed is {settings.max_pdf_pages}"
        )

    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)

    raw_text = "\n\n".join(pages)
    text = _normalize_whitespace(raw_text)

    if len(text) < settings.min_extracted_chars:
        raise ValueError(
            "Could not extract enough text from PDF. "
            "Scanned or image-only PDFs without OCR are not supported."
        )

    title = _extract_title(reader, text)
    return ExtractedPaper(title=title, text=text, page_count=page_count)


def _extract_title(reader: PdfReader, text: str) -> str:
    if reader.metadata:
        meta_title = reader.metadata.get("/Title")
        if meta_title and str(meta_title).strip():
            return _normalize_whitespace(str(meta_title))[:512]

    first_line = text.split(". ", 1)[0].strip()
    if first_line:
        return first_line[:512]
    return "Untitled research paper"
