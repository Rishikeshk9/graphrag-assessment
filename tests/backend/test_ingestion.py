import asyncio

from app.chunking import HierarchicalChunker
from app.graph import GraphFact
from app.ingestion import IngestionService
from app.schemas import DocumentInput


def chunker() -> HierarchicalChunker:
    return HierarchicalChunker(parent_size=12, parent_overlap=2, child_size=6, child_overlap=1)


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 1.0, 0.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    """Models Qdrant closely enough to observe replacement: points keyed by id."""

    def __init__(self) -> None:
        self.documents = []
        self.points: dict[str, dict[str, str]] = {}
        self.prunes: list[tuple[str, str | None]] = []

    async def upsert_document(
        self,
        document,
        parent_vectors,
        child_vectors,
        *,
        ingest_run_id,
        content_sha256,
        file_sha256=None,
    ) -> None:
        self.documents.append((document, parent_vectors, child_vectors))
        for chunk in [*document.parents, *document.children]:
            self.points[chunk.id] = {
                "source_id": document.source_id,
                "ingest_run_id": ingest_run_id,
                "document_sha256": content_sha256,
                "file_sha256": file_sha256 or "",
                "chunk_type": "child" if hasattr(chunk, "parent_id") else "parent",
            }

    async def find_source_by_file(self, file_sha256: str) -> str | None:
        for payload in self.points.values():
            if file_sha256 and payload["file_sha256"] == file_sha256:
                return payload["source_id"]
        return None

    async def prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int:
        self.prunes.append((source_id, keep_ingest_run_id))
        stale = [
            point_id
            for point_id, payload in self.points.items()
            if payload["source_id"] == source_id
            and payload["ingest_run_id"] != keep_ingest_run_id
        ]
        for point_id in stale:
            del self.points[point_id]
        return len(stale)

    async def count_indexed_children(self, source_id: str, content_sha256: str) -> int:
        return sum(
            1
            for payload in self.points.values()
            if payload["source_id"] == source_id
            and payload["document_sha256"] == content_sha256
            and payload["chunk_type"] == "child"
        )


class FailingGraphExtractor:
    async def extract(self, child):
        raise RuntimeError("model unavailable")


class ConcurrencyProbeExtractor:
    """Records how many extractions overlap so the semaphore stays observable."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def extract(self, child):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return [
            GraphFact(
                source="Acme",
                source_type="Company",
                target="Beta",
                target_type="Company",
                relationship_type="ACQUIRED",
                evidence=child.text,
                source_id=child.source_id,
                parent_chunk_id=child.parent_id,
                child_chunk_id=child.id,
            )
        ]


class FakeGraphStore:
    """Mirrors the canonical-edge model: one edge per triple, many evidence spans."""

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.edges: dict[tuple[str, str, str], list[dict]] = {}
        self.prunes: list[tuple[str, str | None]] = []

    async def upsert_facts(self, facts, *, ingest_run_id: str):
        self.batches.append(len(facts))
        for fact in facts:
            spans = self.edges.setdefault((fact.source, fact.relationship_type, fact.target), [])
            existing = next(
                (
                    span
                    for span in spans
                    if span["child_chunk_id"] == fact.child_chunk_id
                    and span["evidence"] == fact.evidence
                ),
                None,
            )
            if existing is not None:
                existing["run"] = ingest_run_id
            else:
                spans.append(
                    {
                        "source_id": fact.source_id,
                        "child_chunk_id": fact.child_chunk_id,
                        "evidence": fact.evidence,
                        "run": ingest_run_id,
                    }
                )
        return len(facts)

    async def prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int:
        removed = 0
        for key, spans in list(self.edges.items()):
            kept = [
                span
                for span in spans
                if not (
                    span["source_id"] == source_id
                    and (
                        keep_ingest_run_id is None
                        or span["run"] is None
                        or span["run"] != keep_ingest_run_id
                    )
                )
            ]
            removed += len(spans) - len(kept)
            if kept:
                self.edges[key] = kept
            else:
                del self.edges[key]
        self.prunes.append((source_id, keep_ingest_run_id))
        return removed


def test_ingestion_indexes_parent_and_child_vectors() -> None:
    store = FakeVectorStore()
    service = IngestionService(
        chunker=chunker(), embeddings=FakeEmbeddings(), vector_store=store
    )

    response = asyncio.run(
        service.ingest(
            [DocumentInput(source_id="guide", content="one two three four five six seven eight nine")]
        )
    )

    document, parent_vectors, child_vectors = store.documents[0]
    assert response.status == "completed"
    assert response.documents_indexed == 1
    assert response.parent_chunks_indexed == len(document.parents)
    assert response.child_chunks_indexed == len(document.children)
    assert len(parent_vectors) == len(document.parents)
    assert len(child_vectors) == len(document.children)


def test_ingestion_indexes_vectors_when_graph_extraction_fails() -> None:
    service = IngestionService(
        chunker=chunker(),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
        graph_extractor=FailingGraphExtractor(),
        graph_store=FakeGraphStore(),
    )

    response = asyncio.run(
        service.ingest([DocumentInput(source_id="guide", content="one two three four")])
    )

    assert response.status == "completed"
    assert response.graph_extraction_failures == response.child_chunks_indexed
    assert response.graph_relationships_indexed == 0
    assert response.warnings


def test_graph_extraction_runs_with_bounded_concurrency() -> None:
    extractor = ConcurrencyProbeExtractor()
    store = FakeGraphStore()
    service = IngestionService(
        chunker=HierarchicalChunker(parent_size=200, parent_overlap=0, child_size=6, child_overlap=0),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
        graph_extractor=extractor,
        graph_store=store,
        extraction_concurrency=3,
    )
    content = " ".join(f"word-{index}" for index in range(60))

    response = asyncio.run(service.ingest([DocumentInput(source_id="guide", content=content)]))

    assert response.child_chunks_indexed > 3
    assert 1 < extractor.peak <= 3
    assert response.graph_relationships_indexed == response.child_chunks_indexed


def replaceable_service(
    vectors: FakeVectorStore, graph: FakeGraphStore, extractor=None
) -> IngestionService:
    return IngestionService(
        chunker=chunker(),
        embeddings=FakeEmbeddings(),
        vector_store=vectors,
        graph_extractor=extractor or ConcurrencyProbeExtractor(),
        graph_store=graph,
    )


def test_reingesting_unchanged_content_is_skipped_entirely() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)
    document = DocumentInput(source_id="guide", content="one two three four five six")

    first = asyncio.run(service.ingest([document]))
    points_after_first = dict(vectors.points)
    second = asyncio.run(service.ingest([document]))
    # A skipped document is not re-stamped at all, so payloads must be identical.

    assert first.documents_indexed == 1 and first.documents_skipped == 0
    assert second.documents_indexed == 0 and second.documents_skipped == 1
    assert second.warnings == ["guide: content unchanged, reusing the existing index"]
    assert vectors.points == points_after_first


def test_forced_reingest_replaces_rather_than_accumulates() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)
    document = DocumentInput(source_id="guide", content="one two three four five six")

    asyncio.run(service.ingest([document]))
    graph.edges[("Acme", "HALLUCINATED", "Ghost")] = [
        {"source_id": "guide", "child_chunk_id": "stale", "evidence": "made up", "run": "older"}
    ]
    points_before = set(vectors.points)
    real_edges_before = len(graph.edges) - 1

    response = asyncio.run(service.ingest([document], force=True))

    assert set(vectors.points) == points_before
    assert response.stale_relationships_removed == 1
    assert len(graph.edges) == real_edges_before
    assert ("Acme", "HALLUCINATED", "Ghost") not in graph.edges


def test_unstamped_rows_from_an_older_schema_are_pruned() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)
    document = DocumentInput(source_id="guide", content="one two three four five six")
    graph.edges[("Acme", "LEGACY", "Ghost")] = [
        {"source_id": "guide", "child_chunk_id": "old", "evidence": "unstamped", "run": None}
    ]

    response = asyncio.run(service.ingest([document]))

    assert response.stale_relationships_removed == 1
    assert ("Acme", "LEGACY", "Ghost") not in graph.edges


def test_editing_a_document_drops_the_chunks_it_no_longer_has() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)
    long_text = " ".join(f"word-{index}" for index in range(40))

    asyncio.run(service.ingest([DocumentInput(source_id="guide", content=long_text)]))
    stale_ids = set(vectors.points)
    response = asyncio.run(
        service.ingest([DocumentInput(source_id="guide", content="a much shorter body")])
    )

    assert response.stale_vectors_removed > 0
    assert not stale_ids & set(vectors.points)
    assert {payload["source_id"] for payload in vectors.points.values()} == {"guide"}


def test_replacing_one_document_leaves_another_untouched() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)
    keep = DocumentInput(source_id="keeper", content="alpha beta gamma delta")

    asyncio.run(service.ingest([keep, DocumentInput(source_id="guide", content="one two three")]))
    keeper_points = {
        point_id
        for point_id, payload in vectors.points.items()
        if payload["source_id"] == "keeper"
    }
    keeper_spans = {
        key
        for key, spans in graph.edges.items()
        if any(span["source_id"] == "keeper" for span in spans)
    }
    asyncio.run(service.ingest([DocumentInput(source_id="guide", content="totally new body")]))

    assert keeper_points <= set(vectors.points)
    assert keeper_spans <= set(graph.edges)


def test_a_fully_failed_extraction_keeps_the_previous_graph() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)
    document = DocumentInput(source_id="guide", content="one two three four five six")

    asyncio.run(service.ingest([document]))
    survivors = {key: list(spans) for key, spans in graph.edges.items()}
    broken = replaceable_service(vectors, graph, extractor=FailingGraphExtractor())

    response = asyncio.run(broken.ingest([document], force=True))

    assert response.graph_extraction_failures > 0
    assert graph.edges == survivors
    assert any("kept the previous graph" in warning for warning in response.warnings)


def test_one_fact_stated_by_two_documents_is_a_single_edge() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)

    asyncio.run(
        service.ingest(
            [
                DocumentInput(source_id="paper-a", content="Acme acquired Beta in 2024."),
                DocumentInput(source_id="paper-b", content="Acme purchased Beta last year."),
            ]
        )
    )

    edge = graph.edges[("Acme", "ACQUIRED", "Beta")]
    assert len(graph.edges) == 1
    assert {span["source_id"] for span in edge} == {"paper-a", "paper-b"}


def test_deleting_one_source_keeps_a_fact_another_source_still_supports() -> None:
    vectors, graph = FakeVectorStore(), FakeGraphStore()
    service = replaceable_service(vectors, graph)
    asyncio.run(
        service.ingest(
            [
                DocumentInput(source_id="paper-a", content="Acme acquired Beta in 2024."),
                DocumentInput(source_id="paper-b", content="Acme purchased Beta last year."),
            ]
        )
    )

    asyncio.run(graph.prune_document("paper-a", keep_ingest_run_id=None))
    surviving = graph.edges[("Acme", "ACQUIRED", "Beta")]

    assert {span["source_id"] for span in surviving} == {"paper-b"}

    asyncio.run(graph.prune_document("paper-b", keep_ingest_run_id=None))
    assert graph.edges == {}


def test_ingestion_reports_one_total_for_multiple_documents() -> None:
    service = IngestionService(
        chunker=chunker(),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
        graph_extractor=FailingGraphExtractor(),
        graph_store=FakeGraphStore(),
    )
    updates: list[tuple[str, int, int]] = []

    async def report(phase: str, processed: int, total: int) -> None:
        updates.append((phase, processed, total))

    response = asyncio.run(
        service.ingest(
            [
                DocumentInput(source_id="first", content="one two three four"),
                DocumentInput(source_id="second", content="five six seven eight"),
            ],
            report_progress=report,
        )
    )

    total = response.child_chunks_indexed
    assert updates[0] == ("chunking", 0, total)
    assert {reported_total for _, _, reported_total in updates} == {total}
    assert updates[-1] == ("graph", total, total)
