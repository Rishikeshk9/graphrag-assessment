import asyncio

from qdrant_client import QdrantClient

from app.chunking import HierarchicalChunker
from app.vector_store import QdrantVectorStore


def test_qdrant_records_child_to_parent_provenance() -> None:
    document = HierarchicalChunker(
        parent_size=8, parent_overlap=2, child_size=3, child_overlap=1
    ).chunk("policy", "one two three four five six seven eight nine")
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(
        url="http://unused",
        children_collection="children_test",
        parents_collection="parents_test",
        client=client,
    )
    parent_vectors = [[1.0, 0.0, 0.0] for _ in document.parents]
    child_vectors = [[0.0, 1.0, 0.0] for _ in document.children]

    asyncio.run(store.upsert_document(document, parent_vectors, child_vectors))

    first_child = client.retrieve("children_test", ids=[document.children[0].id])[0]
    parent = client.retrieve("parents_test", ids=[document.parents[0].id])[0]
    assert first_child.payload["parent_id"] == document.parents[0].id
    assert first_child.payload["source_id"] == "policy"
    assert parent.payload["chunk_type"] == "parent"
