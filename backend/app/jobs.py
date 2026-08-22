"""In-process background ingestion jobs with observable progress."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack

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
        # Without a strong reference the event loop may garbage-collect a
        # running ingestion task mid-flight.
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # Ingestion replaces a source in place, so two concurrent uploads of the
        # same file would each prune the other's freshly written rows.
        self._source_locks: dict[str, asyncio.Lock] = {}

    async def submit(self, documents: list[DocumentInput], force: bool = False) -> IngestResponse:
        job_id = str(uuid.uuid4())
        response = IngestResponse(job_id=job_id, status="accepted", phase="queued")
        async with self._lock:
            self._jobs[job_id] = response
        task = asyncio.create_task(self._run(job_id, documents, force), name=f"ingestion:{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return response

    async def _source_lock(self, source_id: str) -> asyncio.Lock:
        async with self._lock:
            return self._source_locks.setdefault(source_id, asyncio.Lock())

    async def get(self, job_id: str) -> IngestResponse | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> IngestResponse | None:
        """Cancel a queued or running job without affecting other ingestions."""
        async with self._lock:
            response = self._jobs.get(job_id)
            if response is None:
                return None
            if response.status in {"completed", "failed", "cancelled"}:
                return response
            task = self._tasks.get(job_id)
            if task is not None:
                task.cancel()
            cancelled = response.model_copy(
                update={
                    "status": "cancelled",
                    "phase": "cancelled",
                    "warnings": ["Indexing was cancelled by the user"],
                }
            )
            self._jobs[job_id] = cancelled
            return cancelled

    async def _update(self, job_id: str, **changes: object) -> None:
        async with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)

    async def _run(self, job_id: str, documents: list[DocumentInput], force: bool = False) -> None:
        async def report(phase: str, processed: int, total: int) -> None:
            await self._update(
                job_id,
                status="processing",
                phase=phase,
                graph_children_processed=processed,
                graph_children_total=total,
            )

        try:
            async with AsyncExitStack() as sources:
                # Sorted so two jobs touching the same pair cannot deadlock.
                for source_id in sorted({document.source_id for document in documents}):
                    await sources.enter_async_context(await self._source_lock(source_id))
                result = await self.ingestion.ingest(documents, report_progress=report, force=force)
            completed = result.model_dump(exclude={"job_id"})
            completed.update(status="completed", phase="completed")
            await self._update(job_id, **completed)
        except asyncio.CancelledError:
            await self._update(
                job_id,
                status="cancelled",
                phase="cancelled",
                warnings=["Indexing was cancelled by the user"],
            )
            raise
        except Exception as error:
            await self._update(
                job_id,
                status="failed",
                phase="failed",
                warnings=[f"Ingestion failed: {type(error).__name__}"],
            )
