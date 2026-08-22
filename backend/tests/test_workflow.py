import asyncio

from app.schemas import RetrievalRequest, RetrievalResponse
from app.workflow import GraphRAGWorkflow


class FakeRetrieval:
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return RetrievalResponse(query=request.query, child_citations=[], parent_contexts=[], graph_triples=[])


def test_langgraph_workflow_returns_retrieval_response() -> None:
    response = asyncio.run(GraphRAGWorkflow(FakeRetrieval()).retrieve(RetrievalRequest(query="test", graph_hops=3)))
    assert response.query == "test"
