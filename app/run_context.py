from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class RunContext:
    run_id: uuid.UUID
    input_tokens: int = 0
    output_tokens: int = 0

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def estimated_cost_usd(self) -> float:
        settings = get_settings()
        input_cost = (self.input_tokens / 1_000_000) * settings.llm_input_cost_per_million
        output_cost = (self.output_tokens / 1_000_000) * settings.llm_output_cost_per_million
        return round(input_cost + output_cost, 6)


_current_run: ContextVar[RunContext | None] = ContextVar("current_run", default=None)


def set_run_context(ctx: RunContext) -> None:
    _current_run.set(ctx)


def get_run_context() -> RunContext | None:
    return _current_run.get()


def clear_run_context() -> None:
    _current_run.set(None)
