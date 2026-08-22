import asyncio

from app.graph import GraphFact
from app.retrieval import RetrievalService
from app.schemas import RetrievalRequest
from app.vector_store import StoredParent, VectorHit


class FakeEmbeddings:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3]]


class FakeVectorStore:
    async def search_children(self, vector, limit):
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
                source="Microsoft", source_type="ORGANIZATION",
                target="Activision Blizzard", target_type="ORGANIZATION",
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
    service = RetrievalService(FakeEmbeddings(), FakeVectorStore(), FakeGraphStore())
    nodes, edges = asyncio.run(service.subgraph(RetrievalRequest(query="Microsoft")))

    assert {node["id"] for node in nodes} == {"Microsoft", "Activision Blizzard"}
    assert edges[0]["label"] == "ACQUIRED"
