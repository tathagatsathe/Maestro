from __future__ import annotations

from typing import Annotated, TypedDict


def merge_messages(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    return (left or []) + (right or [])


class ExplainState(TypedDict, total=False):
    paper_text: str
    paper_title: str
    paper_brief: str
    draft: str
    critique: str
    quality_score: float
    retry_count: int
    final_output: str
    current_agent: str
    messages: Annotated[list[dict], merge_messages]
