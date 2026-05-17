from __future__ import annotations

from pathlib import Path

from app.output import report_filename, save_report, slugify_task_name


def test_slugify_task_name() -> None:
    assert slugify_task_name("Analyze LangGraph patterns") == "Analyze_LangGraph_patterns"
    assert slugify_task_name("  ") == "report"
    assert slugify_task_name('bad/name:chars?') == "badnamechars"


def test_report_filename_uses_task_and_score() -> None:
    name = report_filename("Analyze LangGraph patterns", 0.85)
    assert name == "Analyze_LangGraph_patterns_0.85.md"


def test_report_filename_clamps_score() -> None:
    assert report_filename("My task", 1.5) == "My_task_1.00.md"
    assert report_filename("My task", -0.1) == "My_task_0.00.md"


def test_save_report_writes_content_without_score_header(tmp_path: Path) -> None:
    path = save_report(
        task="Quarterly review",
        content="# Report\n\nBody",
        quality_score=0.72,
        output_dir=tmp_path,
    )
    assert path == tmp_path / "Quarterly_review_0.72.md"
    assert path.read_text(encoding="utf-8") == "# Report\n\nBody"
