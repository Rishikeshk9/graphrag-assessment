"""Stage 2 ingestion orchestration: chunk, embed, and persist provenance."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.chunking import HierarchicalChunker
from app.embeddings import EmbeddingProvider
from app.graph import GraphExtractor, GraphStore
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
    ) -> None:
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.graph_extractor = graph_extractor
        self.graph_store = graph_store

    async def ingest(
        self,
        documents: list[DocumentInput],
        report_progress: Callable[[str, int, int], Awaitable[None]] | None = None,
    ) -> IngestResponse:
        parent_total = 0
        child_total = 0
        relationship_total = 0
        graph_failures = 0
        warnings: list[str] = []
        for document_input in documents:
            document = self.chunker.chunk(document_input.source_id, document_input.content or "")
            if report_progress:
                await report_progress("chunking", child_total, child_total + len(document.children))
            parent_vectors = await self.embeddings.embed([parent.text for parent in document.parents])
            child_vectors = await self.embeddings.embed([child.text for child in document.children])
            await self.vector_store.upsert_document(document, parent_vectors, child_vectors)
            if report_progress:
                await report_progress("embedding", child_total, child_total + len(document.children))
            if self.graph_extractor is not None and self.graph_store is not None:
                for child_index, child in enumerate(document.children, start=1):
                    try:
                        facts = await self.graph_extractor.extract(child)
                        relationship_total += await self.graph_store.upsert_facts(facts)
                    except Exception as error:
                        graph_failures += 1
                        warnings.append(
                            f"Graph extraction skipped for child {child.index}: "
                            f"{type(error).__name__}"
                        )
                    if report_progress:
                        await report_progress("graph", child_total + child_index, child_total + len(document.children))
            parent_total += len(document.parents)
            child_total += len(document.children)

        return IngestResponse(
            job_id=str(uuid.uuid4()),
            status="completed",
            documents_indexed=len(documents),
            parent_chunks_indexed=parent_total,
            child_chunks_indexed=child_total,
            graph_relationships_indexed=relationship_total,
            graph_extraction_failures=graph_failures,
            warnings=warnings,
            phase="completed",
            graph_children_processed=child_total,
            graph_children_total=child_total,
        )
