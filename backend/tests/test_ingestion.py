import asyncio

from app.chunking import HierarchicalChunker
from app.ingestion import IngestionService
from app.schemas import DocumentInput


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 1.0, 0.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents = []

    async def upsert_document(self, document, parent_vectors, child_vectors) -> None:
        self.documents.append((document, parent_vectors, child_vectors))


class FailingGraphExtractor:
    async def extract(self, child):
        raise RuntimeError("model unavailable")


class FakeGraphStore:
    async def upsert_facts(self, facts):
        return len(facts)


def test_ingestion_indexes_parent_and_child_vectors() -> None:
    store = FakeVectorStore()
    service = IngestionService(
        chunker=HierarchicalChunker(parent_size=8, parent_overlap=2, child_size=3, child_overlap=1),
        embeddings=FakeEmbeddings(),
        vector_store=store,
    )

    response = asyncio.run(
        service.ingest(
            [DocumentInput(source_id="guide", content="one two three four five six seven eight nine")]
        )
    )

    assert response.status == "completed"
    assert response.documents_indexed == 1
    assert response.parent_chunks_indexed == 2
    assert response.child_chunks_indexed == 5
    document, parent_vectors, child_vectors = store.documents[0]
    assert len(parent_vectors) == len(document.parents)
    assert len(child_vectors) == len(document.children)


def test_ingestion_indexes_vectors_when_graph_extraction_fails() -> None:
    service = IngestionService(
        chunker=HierarchicalChunker(parent_size=8, parent_overlap=2, child_size=3, child_overlap=1),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
        graph_extractor=FailingGraphExtractor(),
        graph_store=FakeGraphStore(),
    )

    response = asyncio.run(
        service.ingest([DocumentInput(source_id="guide", content="one two three four")])
    )

    assert response.status == "completed"
    assert response.child_chunks_indexed == 2
    assert response.graph_extraction_failures == 2
    assert response.graph_relationships_indexed == 0


def test_ingestion_reports_one_total_for_multiple_documents() -> None:
    service = IngestionService(
        chunker=HierarchicalChunker(parent_size=8, parent_overlap=2, child_size=3, child_overlap=1),
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

    assert response.child_chunks_indexed == 4
    assert updates[0] == ("chunking", 0, 4)
    assert {total for _, _, total in updates} == {4}
    assert updates[-1] == ("graph", 4, 4)
