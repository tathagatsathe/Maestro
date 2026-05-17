from __future__ import annotations

from types import SimpleNamespace

from app.observability.langsmith_tracing import extract_token_usage


def test_extract_token_usage_from_usage_metadata() -> None:
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 80,
            "total_tokens": 200,
        },
        response_metadata={},
    )
    assert extract_token_usage(response) == {
        "input_tokens": 120,
        "output_tokens": 80,
        "total_tokens": 200,
    }


def test_extract_token_usage_from_response_metadata() -> None:
    response = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "usage": {"prompt_tokens": 50, "completion_tokens": 25},
        },
    )
    assert extract_token_usage(response) == {
        "input_tokens": 50,
        "output_tokens": 25,
        "total_tokens": 75,
    }


def test_extract_token_usage_defaults_to_zero() -> None:
    response = SimpleNamespace(usage_metadata=None, response_metadata={})
    assert extract_token_usage(response) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
