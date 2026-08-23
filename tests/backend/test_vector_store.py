import asyncio
from unittest.mock import Mock

from qdrant_client import QdrantClient

from app import vector_store
from app.chunking import HierarchicalChunker
from app.vector_store import QdrantVectorStore


def build_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        url="http://unused",
        children_collection="children_test",
        parents_collection="parents_test",
        client=client,
    )


def test_qdrant_cloud_api_key_is_passed_to_client(monkeypatch) -> None:
    client_factory = Mock()
    monkeypatch.setattr(vector_store, "QdrantClient", client_factory)

    QdrantVectorStore(
        url="https://cluster.cloud.qdrant.io",
        children_collection="children_test",
        parents_collection="parents_test",
        api_key="cloud-key",
    )

    client_factory.assert_called_once_with(
        url="https://cluster.cloud.qdrant.io", api_key="cloud-key"
    )


def index(store: QdrantVectorStore, source_id: str, text: str, run: str, digest: str):
    document = HierarchicalChunker(
        parent_size=8, parent_overlap=2, child_size=3, child_overlap=1
    ).chunk(source_id, text)
    asyncio.run(
        store.upsert_document(
            document,
            [[1.0, 0.0, 0.0] for _ in document.parents],
            [[0.0, 1.0, 0.0] for _ in document.children],
            ingest_run_id=run,
            content_sha256=digest,
        )
    )
    return document


def test_qdrant_records_child_to_parent_provenance() -> None:
    client = QdrantClient(location=":memory:")
    store = build_store(client)

    document = index(store, "policy", "one two three four five six seven eight nine", "run-1", "d1")

    first_child = client.retrieve("children_test", ids=[document.children[0].id])[0]
    parent = client.retrieve("parents_test", ids=[document.parents[0].id])[0]
    assert first_child.payload["parent_id"] == document.parents[0].id
    assert first_child.payload["source_id"] == "policy"
    assert first_child.payload["ingest_run_id"] == "run-1"
    assert parent.payload["chunk_type"] == "parent"


def test_pruning_removes_only_the_superseded_points_of_one_source() -> None:
    client = QdrantClient(location=":memory:")
    store = build_store(client)
    keeper = index(store, "keeper", "alpha beta gamma delta epsilon", "run-1", "d1")
    index(store, "policy", "one two three four five six seven eight nine", "run-1", "d1")

    replacement = index(store, "policy", "a totally different body of text", "run-2", "d2")
    removed = asyncio.run(store.prune_document("policy", keep_ingest_run_id="run-2"))

    surviving = {str(point.id) for point in client.scroll("children_test", limit=100)[0]}
    assert removed > 0
    assert {child.id for child in replacement.children} <= surviving
    assert {child.id for child in keeper.children} <= surviving
    assert asyncio.run(store.count_indexed_children("policy", "d1")) == 0
    assert asyncio.run(store.count_indexed_children("policy", "d2")) == len(replacement.children)


def test_deleting_a_document_keeps_nothing_for_that_source() -> None:
    client = QdrantClient(location=":memory:")
    store = build_store(client)
    keeper = index(store, "keeper", "alpha beta gamma delta epsilon", "run-1", "d1")
    index(store, "policy", "one two three four five six seven eight nine", "run-1", "d1")

    asyncio.run(store.prune_document("policy", keep_ingest_run_id=None))

    surviving = {str(point.id) for point in client.scroll("children_test", limit=100)[0]}
    assert surviving == {child.id for child in keeper.children}
