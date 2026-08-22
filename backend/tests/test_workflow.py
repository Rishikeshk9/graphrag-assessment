import asyncio

from app.schemas import Citation, ParentContext, RetrievalRequest, RetrievalResponse
from app.workflow import GraphRAGWorkflow


class FakeRetrieval:
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return RetrievalResponse(query=request.query, child_citations=[], parent_contexts=[], graph_triples=[])


def test_langgraph_workflow_returns_retrieval_response() -> None:
    response = asyncio.run(GraphRAGWorkflow(FakeRetrieval()).retrieve(RetrievalRequest(query="test", graph_hops=3)))
    assert response.query == "test"


class MismatchedProvenanceRetrieval:
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return RetrievalResponse(
            query=request.query,
            child_citations=[
                Citation(
                    parent_chunk_id="parent-1",
                    child_chunk_id="child-1",
                    source_id="source",
                    excerpt="citation text absent from its parent",
                )
            ],
            parent_contexts=[
                ParentContext(
                    parent_chunk_id="parent-1",
                    source_id="source",
                    text="A different parent context.",
                    matching_child_chunk_ids=["child-1"],
                )
            ],
            graph_triples=[],
        )


def test_langgraph_workflow_withholds_mismatched_provenance() -> None:
    response = asyncio.run(
        GraphRAGWorkflow(MismatchedProvenanceRetrieval()).retrieve(
            RetrievalRequest(query="test")
        )
    )

    assert response.child_citations == []
    assert response.parent_contexts == []
