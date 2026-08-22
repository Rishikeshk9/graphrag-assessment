"""Ingestion orchestration: chunk, embed, extract graph facts, persist provenance."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.chunking import ChildChunk, HierarchicalChunker, HierarchicalDocument
from app.embeddings import EmbeddingProvider
from app.graph import GraphExtractor, GraphFact, GraphStore
from app.schemas import DocumentInput, IngestResponse
from app.vector_store import VectorStore


@dataclass(frozen=True)
class IngestionResult:
    documents_indexed: int
    parent_chunks_indexed: int
    child_chunks_indexed: int


class IngestionService:
    def __init__(
        self,
        chunker: HierarchicalChunker,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        graph_extractor: GraphExtractor | None = None,
        graph_store: GraphStore | None = None,
        extraction_concurrency: int = 4,
    ) -> None:
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.graph_extractor = graph_extractor
        self.graph_store = graph_store
        self.extraction_concurrency = max(1, extraction_concurrency)

    async def ingest(
        self,
        documents: list[DocumentInput],
        report_progress: Callable[[str, int, int], Awaitable[None]] | None = None,
        force: bool = False,
    ) -> IngestResponse:
        warnings: list[str] = []
        pending: list[tuple[HierarchicalDocument, str, str | None]] = []
        skipped = 0

        for document_input in documents:
            content = document_input.content or ""
            content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            document = self.chunker.chunk(document_input.source_id, content)
            if not force and await self._is_already_indexed(document, content_sha256):
                skipped += 1
                warnings.append(
                    f"{document.source_id}: content unchanged, reusing the existing index"
                )
                continue
            pending.append((document, content_sha256, document_input.file_sha256))

        child_total = sum(len(document.children) for document, _, _ in pending)
        parent_total = 0
        processed_children = 0
        relationship_total = 0
        graph_failures = 0
        stale_vectors_removed = 0
        stale_relationships_removed = 0
        if report_progress:
            await report_progress("chunking", 0, child_total)

        for document, content_sha256, file_sha256 in pending:
            ingest_run_id = str(uuid.uuid4())
            if report_progress:
                await report_progress("embedding", processed_children, child_total)
            parent_vectors = await self.embeddings.embed([parent.text for parent in document.parents])
            child_vectors = await self.embeddings.embed([child.text for child in document.children])
            await self.vector_store.upsert_document(
                document,
                parent_vectors,
                child_vectors,
                ingest_run_id=ingest_run_id,
                content_sha256=content_sha256,
                file_sha256=file_sha256,
            )
            stale_vectors_removed += await self.vector_store.prune_document(
                document.source_id, keep_ingest_run_id=ingest_run_id
            )

            if self.graph_extractor is not None and self.graph_store is not None:
                extracted, failures, extraction_warnings = await self._extract_document_facts(
                    document.children, processed_children, child_total, report_progress
                )
                graph_failures += failures
                warnings.extend(extraction_warnings)
                relationship_total += await self.graph_store.upsert_facts(
                    extracted, ingest_run_id=ingest_run_id
                )
                # A run that extracted nothing only because every chunk errored
                # must not be allowed to delete the previous good extraction.
                if extracted or not failures:
                    stale_relationships_removed += await self.graph_store.prune_document(
                        document.source_id, keep_ingest_run_id=ingest_run_id
                    )
                else:
                    warnings.append(
                        f"{document.source_id}: kept the previous graph, this run extracted nothing"
                    )

            parent_total += len(document.parents)
            processed_children += len(document.children)

        return IngestResponse(
            job_id=str(uuid.uuid4()),
            status="completed",
            documents_indexed=len(pending),
            documents_skipped=skipped,
            parent_chunks_indexed=parent_total,
            child_chunks_indexed=child_total,
            graph_relationships_indexed=relationship_total,
            graph_extraction_failures=graph_failures,
            stale_vectors_removed=stale_vectors_removed,
            stale_relationships_removed=stale_relationships_removed,
            warnings=warnings,
            phase="completed",
            graph_children_processed=child_total,
            graph_children_total=child_total,
        )

    async def _is_already_indexed(self, document: HierarchicalDocument, content_sha256: str) -> bool:
        """Report whether this exact content is already fully present as vectors.

        Comparing a stored hash alone would wrongly skip a document whose vectors
        were dropped out of band, so the chunk count has to agree as well.
        """
        indexed = await self.vector_store.count_indexed_children(document.source_id, content_sha256)
        return indexed == len(document.children)

    async def source_holding_file(self, file_sha256: str) -> str | None:
        """Name the already-indexed source for these bytes, so upload can skip early.

        The text hash is the stronger duplicate check, because two files with
        different bytes can extract to identical text. This one only exists to
        avoid re-running text extraction on a byte-identical re-upload.
        """
        return await self.vector_store.find_source_by_file(file_sha256)

    async def _extract_document_facts(
        self,
        children: list[ChildChunk],
        offset: int,
        child_total: int,
        report_progress: Callable[[str, int, int], Awaitable[None]] | None,
    ) -> tuple[list[GraphFact], int, list[str]]:
        """Extract facts with bounded concurrency so a long PDF is not serialized.

        A failed chunk degrades that chunk only; the rest of the document still
        contributes graph facts.
        """
        assert self.graph_extractor is not None
        semaphore = asyncio.Semaphore(self.extraction_concurrency)
        completed = 0
        progress_lock = asyncio.Lock()

        async def extract(child: ChildChunk) -> list[GraphFact] | Exception:
            nonlocal completed
            async with semaphore:
                try:
                    return await self.graph_extractor.extract(child)
                except Exception as error:
                    return error
                finally:
                    async with progress_lock:
                        completed += 1
                        if report_progress:
                            await report_progress("graph", offset + completed, child_total)

        results = await asyncio.gather(*(extract(child) for child in children))

        facts: list[GraphFact] = []
        failures = 0
        warnings: list[str] = []
        for child, result in zip(children, results, strict=True):
            if isinstance(result, Exception):
                failures += 1
                warnings.append(
                    f"Graph extraction skipped for child {child.index}: {type(result).__name__}"
                )
            else:
                facts.extend(result)
        return facts, failures, warnings
