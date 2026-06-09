from __future__ import annotations

import uuid
from pathlib import Path

UPLOADS_DIR = Path("uploads")


def run_upload_dir(run_id: uuid.UUID) -> Path:
    return UPLOADS_DIR / str(run_id)


def save_run_artifacts(
    run_id: uuid.UUID,
    *,
    pdf_bytes: bytes,
    extracted_text: str,
) -> Path:
    """Persist uploaded PDF and extracted text for a run."""
    directory = run_upload_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)

    pdf_path = directory / "paper.pdf"
    text_path = directory / "extracted.txt"

    pdf_path.write_bytes(pdf_bytes)
    text_path.write_text(extracted_text, encoding="utf-8")
    return directory


def load_extracted_text(run_id: uuid.UUID) -> str:
    text_path = run_upload_dir(run_id) / "extracted.txt"
    if not text_path.exists():
        raise FileNotFoundError(
            f"Extracted text not found for run {run_id}"
        )
    return text_path.read_text(encoding="utf-8")
