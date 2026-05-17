from __future__ import annotations

import re
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def slugify_task_name(task: str, *, max_length: int = 80) -> str:
    text = task.strip()
    if not text:
        return "report"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    text = re.sub(r"_+", "_", text).strip("._")
    if not text:
        return "report"
    if len(text) > max_length:
        text = text[:max_length].rstrip("._")
    return text


def report_filename(task: str, quality_score: float) -> str:
    score = max(0.0, min(1.0, quality_score))
    slug = slugify_task_name(task)
    return f"{slug}_{score:.2f}.md"


def save_report(
    *,
    task: str,
    content: str,
    quality_score: float,
    output_dir: Path | None = None,
) -> Path:
    dest = output_dir or OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / report_filename(task, quality_score)
    path.write_text(content, encoding="utf-8")
    return path
