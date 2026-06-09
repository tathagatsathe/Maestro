from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.observability.metrics import EVAL_PASS_RATE

logger = logging.getLogger(__name__)

GOLDEN_TASKS_PATH = Path(__file__).parent / "golden_tasks.json"


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    details: str


def load_golden_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or GOLDEN_TASKS_PATH
    return json.loads(source.read_text(encoding="utf-8"))


def evaluate_output(task: dict[str, Any], output: str) -> EvalResult:
    task_id = str(task["id"])
    min_len = int(task.get("min_output_length", 0))
    required_terms = list(task.get("required_terms", []))

    issues: list[str] = []
    if len(output.strip()) < min_len:
        issues.append(f"output length {len(output.strip())} < {min_len}")

    lower = output.lower()
    for term in required_terms:
        if term.lower() not in lower:
            issues.append(f"missing required term: {term}")

    passed = not issues
    details = "ok" if passed else "; ".join(issues)
    return EvalResult(task_id=task_id, passed=passed, details=details)


async def run_eval_with_outputs(
    outputs: dict[str, str],
    *,
    tasks_path: Path | None = None,
) -> list[EvalResult]:
    tasks = load_golden_tasks(tasks_path)
    results: list[EvalResult] = []
    for task in tasks:
        task_id = str(task["id"])
        output = outputs.get(task_id, "")
        results.append(evaluate_output(task, output))
    return results


def record_eval_metrics(results: list[EvalResult]) -> float:
    if not results:
        EVAL_PASS_RATE.set(0.0)
        return 0.0
    rate = sum(1 for r in results if r.passed) / len(results)
    EVAL_PASS_RATE.set(rate)
    return rate


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": [
            {"task_id": r.task_id, "passed": r.passed, "details": r.details}
            for r in results
        ],
    }
