import asyncio

from app.graph import GraphFact
from app.retrieval import RetrievalService
from app.schemas import RetrievalRequest
from app.vector_store import StoredParent, VectorHit


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3]]


class FakeVectorStore:
    def __init__(self) -> None:
        self.searches = 0

    async def search_children(self, vector, limit):
        self.searches += 1
        return [
            VectorHit("child-1", "parent-1", "source", "matching child one", 0.9),
            VectorHit("child-2", "parent-1", "source", "matching child two", 0.8),
        ]

    async def get_parents(self, parent_ids):
        return [StoredParent("parent-1", "source", "full parent context")]


class FakeGraphStore:
    async def traverse(self, query, hops, limit):
        assert hops == 2
        return [
            GraphFact(
                source="Microsoft", source_type="Company",
                target="Activision Blizzard", target_type="Company",
                relationship_type="ACQUIRED", evidence="Microsoft acquired Activision Blizzard.",
                source_id="source", parent_chunk_id="parent-1", child_chunk_id="child-1",
            )
        ]


def test_retrieval_fuses_child_hits_parent_context_and_graph_facts() -> None:
    service = RetrievalService(FakeEmbeddings(), FakeVectorStore(), FakeGraphStore())
    response = asyncio.run(service.retrieve(RetrievalRequest(query="Who acquired Activision Blizzard?")))

    assert len(response.child_citations) == 2
    assert response.parent_contexts[0].matching_child_chunk_ids == ["child-1", "child-2"]
    assert response.graph_triples[0].predicate == "ACQUIRED"


def test_subgraph_returns_visualization_ready_nodes_and_edges() -> None:
    embeddings, vector_store = FakeEmbeddings(), FakeVectorStore()
    service = RetrievalService(embeddings, vector_store, FakeGraphStore())

    nodes, edges = asyncio.run(service.subgraph(RetrievalRequest(query="Microsoft")))

    assert {node["id"] for node in nodes} == {"Microsoft", "Activision Blizzard"}
    assert nodes[0]["type"] == "Company"
    assert edges[0]["label"] == "ACQUIRED"
    assert edges[0]["child_chunk_id"] == "child-1"
    # Visualization is graph-only: no embedding call and no vector search.
    assert (embeddings.calls, vector_store.searches) == (0, 0)
