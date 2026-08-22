import asyncio

from app.jobs import IngestionJobs
from app.schemas import DocumentInput, IngestResponse


class ReportingIngestion:
    async def ingest(self, documents, report_progress, force=False):
        await report_progress("graph", 1, 2)
        await asyncio.sleep(0)
        return IngestResponse(
            job_id="service-generated",
            status="completed",
            documents_indexed=len(documents),
            child_chunks_indexed=2,
            graph_relationships_indexed=3,
            phase="completed",
            graph_children_processed=2,
            graph_children_total=2,
        )


def test_background_job_reports_then_completes() -> None:
    async def run() -> IngestResponse:
        jobs = IngestionJobs(ReportingIngestion())
        submitted = await jobs.submit([DocumentInput(source_id="test", content="content")])
        for _ in range(10):
            status = await jobs.get(submitted.job_id)
            if status and status.status == "completed":
                return status
            await asyncio.sleep(0)
        raise AssertionError("job did not complete")

    result = asyncio.run(run())
    assert result.phase == "completed"
    assert result.child_chunks_indexed == 2
    assert result.graph_relationships_indexed == 3


def test_running_job_can_be_cancelled() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    class BlockingIngestion:
        async def ingest(self, documents, report_progress, force=False):
            started.set()
            await finish.wait()
            return IngestResponse(job_id="service-generated", status="completed")

    async def run() -> IngestResponse:
        jobs = IngestionJobs(BlockingIngestion())
        submitted = await jobs.submit([DocumentInput(source_id="test", content="content")])
        await started.wait()
        cancelled = await jobs.cancel(submitted.job_id)
        assert cancelled is not None
        await asyncio.sleep(0)
        result = await jobs.get(submitted.job_id)
        assert result is not None
        return result

    result = asyncio.run(run())
    assert result.status == "cancelled"
    assert result.phase == "cancelled"
