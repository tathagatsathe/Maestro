from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.models import Run, RunStatus


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://agent:agentpass@localhost:5432/agentdb",
    )
    from app.config import get_settings

    get_settings.cache_clear()

    mock_run = Run(
        id=uuid.uuid4(),
        task="test task",
        status=RunStatus.PENDING.value,
    )

    mock_session = AsyncMock()
    mock_repo = AsyncMock()
    mock_repo.create = AsyncMock(return_value=mock_run)
    mock_repo.get = AsyncMock(return_value=mock_run)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_queue = AsyncMock()
    mock_queue.enqueue = AsyncMock()

    import app.main as main_module

    with (
        patch.object(main_module, "init_db", new=AsyncMock()),
        patch.object(main_module, "get_session_factory", return_value=mock_factory),
        patch.object(main_module, "RunRepository", return_value=mock_repo),
        patch.object(main_module, "JobQueue", return_value=mock_queue),
        patch.object(main_module, "check_rate_limit", new=AsyncMock()),
    ):
        with TestClient(main_module.app) as client:
            yield client, mock_run, mock_repo, mock_queue


def test_health_no_auth(api_client) -> None:
    client, *_ = api_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_start_run_requires_api_key(api_client) -> None:
    client, *_ = api_client
    response = client.post("/run", json={"task": "hello"})
    assert response.status_code == 422


def test_start_run_invalid_api_key(api_client) -> None:
    client, *_ = api_client
    response = client.post(
        "/run",
        json={"task": "hello"},
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 401


def test_start_run_success(api_client) -> None:
    client, mock_run, _repo, mock_queue = api_client
    response = client.post(
        "/run",
        json={"task": "hello"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == str(mock_run.id)
    assert body["status"] == "pending"
    mock_queue.enqueue.assert_awaited_once()


def test_get_run_success(api_client) -> None:
    client, mock_run, mock_repo, _ = api_client
    completed = Run(
        id=mock_run.id,
        task="hello",
        status=RunStatus.COMPLETED.value,
        quality_score=0.9,
        retry_count=1,
        input_tokens=100,
        output_tokens=200,
        estimated_cost_usd=0.01,
    )
    mock_repo.get = AsyncMock(return_value=completed)

    response = client.get(
        f"/run/{mock_run.id}",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["quality_score"] == 0.9
