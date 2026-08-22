"""LangGraph orchestration for GraphRAG evidence gathering and verification."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.retrieval import RetrievalService
from app.schemas import RetrievalRequest, RetrievalResponse


class EvidenceState(TypedDict, total=False):
    request: RetrievalRequest
    planned_request: RetrievalRequest
    response: RetrievalResponse
    provenance_verified: bool


class GraphRAGWorkflow:
    """Plans a bounded traversal, retrieves evidence, then verifies provenance.

    The deterministic planning step makes the workflow inspectable and safe while
    still providing a LangGraph seam for later LLM tool-selection or reranking.
    """

    def __init__(self, retrieval: RetrievalService) -> None:
        self.retrieval = retrieval
        graph = StateGraph(EvidenceState)
        graph.add_node("plan_multi_hop", self._plan_multi_hop)
        graph.add_node("retrieve_evidence", self._retrieve_evidence)
        graph.add_node("verify_provenance", self._verify_provenance)
        graph.add_edge(START, "plan_multi_hop")
        graph.add_edge("plan_multi_hop", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "verify_provenance")
        graph.add_edge("verify_provenance", END)
        self._app = graph.compile()

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        state = await self._app.ainvoke({"request": request})
        return state["response"]

    @staticmethod
    async def _plan_multi_hop(state: EvidenceState) -> dict[str, RetrievalRequest]:
        request = state["request"]
        # Preserve explicit caller controls while bounding traversal cost.
        return {"planned_request": request.model_copy(update={"graph_hops": min(request.graph_hops, 3)})}

    async def _retrieve_evidence(self, state: EvidenceState) -> dict[str, RetrievalResponse]:
        return {"response": await self.retrieval.retrieve(state["planned_request"])}

    @staticmethod
    async def _verify_provenance(state: EvidenceState) -> dict[str, object]:
        """Drop only the evidence that cannot be traced, never the whole set.

        A single mismatched citation used to blank every source, parent, and
        triple, which turned one bad chunk into an unanswerable question.
        """
        response = state["response"]
        parent_text = {item.parent_chunk_id: item.text.casefold() for item in response.parent_contexts}

        verified_citations = [
            citation
            for citation in response.child_citations
            if citation.excerpt.casefold() in parent_text.get(citation.parent_chunk_id, "")
        ]
        kept_parent_ids = {citation.parent_chunk_id for citation in verified_citations}
        verified_parents = [
            parent.model_copy(
                update={
                    "matching_child_chunk_ids": [
                        citation.child_chunk_id
                        for citation in verified_citations
                        if citation.parent_chunk_id == parent.parent_chunk_id
                        and citation.child_chunk_id is not None
                    ]
                }
            )
            for parent in response.parent_contexts
            if parent.parent_chunk_id in kept_parent_ids
        ]
        # Graph facts carry their own verbatim evidence check at extraction time,
        # so they survive a vector-side mismatch.
        return {
            "provenance_verified": len(verified_citations) == len(response.child_citations),
            "response": response.model_copy(
                update={
                    "child_citations": verified_citations,
                    "parent_contexts": verified_parents,
                }
            ),
        }
