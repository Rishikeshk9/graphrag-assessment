import asyncio

from app.schemas import Citation, GraphTriple, ParentContext, RetrievalRequest, RetrievalResponse
from app.workflow import GraphRAGWorkflow


class FakeRetrieval:
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return RetrievalResponse(
            query=request.query, child_citations=[], parent_contexts=[], graph_triples=[]
        )


def test_langgraph_workflow_returns_retrieval_response() -> None:
    response = asyncio.run(
        GraphRAGWorkflow(FakeRetrieval()).retrieve(RetrievalRequest(query="test", graph_hops=3))
    )
    assert response.query == "test"


class PartiallyMismatchedRetrieval:
    """One citation matches its parent, one does not."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return RetrievalResponse(
            query=request.query,
            child_citations=[
                Citation(
                    parent_chunk_id="parent-1",
                    child_chunk_id="child-1",
                    source_id="source",
                    excerpt="citation text absent from its parent",
                ),
                Citation(
                    parent_chunk_id="parent-2",
                    child_chunk_id="child-2",
                    source_id="source",
                    excerpt="grounded excerpt",
                ),
            ],
            parent_contexts=[
                ParentContext(
                    parent_chunk_id="parent-1",
                    source_id="source",
                    text="A different parent context.",
                    matching_child_chunk_ids=["child-1"],
                ),
                ParentContext(
                    parent_chunk_id="parent-2",
                    source_id="source",
                    text="Context that contains the grounded excerpt verbatim.",
                    matching_child_chunk_ids=["child-2"],
                ),
            ],
            graph_triples=[
                GraphTriple(
                    subject="Acme",
                    predicate="ACQUIRED",
                    object="Beta",
                    source_parent_chunk_id="parent-2",
                    source_child_chunk_id="child-2",
                    source_id="source",
                    evidence="Acme acquired Beta.",
                )
            ],
        )


def test_workflow_drops_only_the_ungrounded_citation() -> None:
    response = asyncio.run(
        GraphRAGWorkflow(PartiallyMismatchedRetrieval()).retrieve(RetrievalRequest(query="test"))
    )

    assert [citation.child_chunk_id for citation in response.child_citations] == ["child-2"]
    assert [parent.parent_chunk_id for parent in response.parent_contexts] == ["parent-2"]
    # Graph facts carry their own extraction-time evidence check.
    assert len(response.graph_triples) == 1
