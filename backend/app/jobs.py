"""In-process background ingestion jobs with observable progress."""

from __future__ import annotations

import asyncio
import uuid

from app.ingestion import IngestionService
from app.schemas import DocumentInput, IngestResponse


class IngestionJobs:
    """Keeps the local API responsive while long graph extraction runs.

    A durable queue can replace this registry when deploying multiple API workers.
    """

    def __init__(self, ingestion: IngestionService) -> None:
        self.ingestion = ingestion
        self._jobs: dict[str, IngestResponse] = {}
        self._lock = asyncio.Lock()

    async def submit(self, documents: list[DocumentInput]) -> IngestResponse:
        job_id = str(uuid.uuid4())
        response = IngestResponse(job_id=job_id, status="accepted", phase="queued")
        async with self._lock:
            self._jobs[job_id] = response
        asyncio.create_task(self._run(job_id, documents), name=f"ingestion:{job_id}")
        return response

    async def get(self, job_id: str) -> IngestResponse | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def _update(self, job_id: str, **changes: object) -> None:
        async with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)

    async def _run(self, job_id: str, documents: list[DocumentInput]) -> None:
        async def report(phase: str, processed: int, total: int) -> None:
            await self._update(job_id, status="processing", phase=phase, graph_children_processed=processed, graph_children_total=total)

        try:
            result = await self.ingestion.ingest(documents, report_progress=report)
            await self._update(job_id, **result.model_dump(exclude={"job_id"}), status="completed", phase="completed")
        except Exception as error:
            await self._update(job_id, status="failed", phase="failed", warnings=[f"Ingestion failed: {type(error).__name__}"])
