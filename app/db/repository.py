from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, Run, RunStatus, RunType


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, task: str) -> Run:
        run = Run(task=task, status=RunStatus.PENDING.value)
        self._session.add(run)
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def create_explain_run(self, source_filename: str) -> Run:
        run = Run(
            task=f"Explain research paper: {source_filename}",
            run_type=RunType.EXPLAIN_PAPER.value,
            source_filename=source_filename,
            status=RunStatus.PENDING.value,
        )
        self._session.add(run)
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def get(self, run_id: uuid.UUID) -> Run | None:
        return await self._session.get(Run, run_id)

    async def update_status(self, run_id: uuid.UUID, status: RunStatus) -> None:
        await self._session.execute(
            update(Run).where(Run.id == run_id).values(status=status.value)
        )
        await self._session.commit()

    async def complete(
        self,
        run_id: uuid.UUID,
        *,
        final_output: str,
        quality_score: float,
        retry_count: int,
        output_path: str | None,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        await self._session.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(
                status=RunStatus.COMPLETED.value,
                final_output=final_output,
                quality_score=quality_score,
                retry_count=retry_count,
                output_path=output_path,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost_usd,
                completed_at=datetime.now(timezone.utc),
            )
        )
        await self._session.commit()

    async def fail(self, run_id: uuid.UUID, error: str) -> None:
        await self._session.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(
                status=RunStatus.FAILED.value,
                error=error,
                completed_at=datetime.now(timezone.utc),
            )
        )
        await self._session.commit()

    async def add_tokens(
        self, run_id: uuid.UUID, input_tokens: int, output_tokens: int
    ) -> None:
        run = await self.get(run_id)
        if run is None:
            return
        run.input_tokens += input_tokens
        run.output_tokens += output_tokens
        await self._session.commit()


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_document(
        self, title: str, source: str | None, chunks: list[tuple[str, list[float]]]
    ) -> Document:
        doc = Document(title=title, source=source)
        self._session.add(doc)
        await self._session.flush()
        for content, embedding in chunks:
            self._session.add(
                Chunk(document_id=doc.id, content=content, embedding=embedding)
            )
        await self._session.commit()
        await self._session.refresh(doc)
        return doc

    async def similarity_search(
        self, embedding: list[float], k: int = 3
    ) -> list[tuple[str, str, float]]:
        distance = Chunk.embedding.cosine_distance(embedding)
        stmt = (
            select(Chunk.content, Document.title, distance.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .order_by(distance)
            .limit(k)
        )
        result = await self._session.execute(stmt)
        rows = result.all()
        return [(row[0], row[1], float(row[2])) for row in rows]
