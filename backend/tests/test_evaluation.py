import asyncio

from fastapi.testclient import TestClient

from app.main import create_app
from app.evaluation import EvaluationService
from app.schemas import Citation, EvaluationRequest, GraphTriple, ParentContext, RetrievalResponse


class FakeRetrieval:
    async def retrieve(self, request):
        return RetrievalResponse(
            query=request.query,
            child_citations=[Citation(parent_chunk_id="parent-1", child_chunk_id="child-1", source_id="microsoft-email", excerpt="Microsoft acquired Activision Blizzard.")],
            parent_contexts=[ParentContext(parent_chunk_id="parent-1", source_id="microsoft-email", text="The announcement confirms Microsoft acquired Activision Blizzard.", matching_child_chunk_ids=["child-1"])],
            graph_triples=[GraphTriple(subject="Microsoft", predicate="ACQUIRED", object="Activision Blizzard", source_parent_chunk_id="parent-1", source_child_chunk_id="child-1", source_id="microsoft-email", evidence="Microsoft acquired Activision Blizzard.")],
        )


def test_evaluation_scores_source_graph_and_provenance() -> None:
    result = asyncio.run(
        EvaluationService(FakeRetrieval()).evaluate(
            EvaluationRequest.model_validate(
                {"cases": [{"id": "acquisition", "query": "Who acquired Activision Blizzard?", "expected_source_ids": ["microsoft-email"], "expected_graph_triples": [["Microsoft", "ACQUIRED", "Activision Blizzard"]]}]}
            )
        )
    )
    case = result.cases[0]
    assert case.source_recall == 1.0
    assert case.graph_recall == 1.0
    assert case.citation_grounding_rate == 1.0
    assert case.passed is True


def test_evaluation_api_returns_metrics(monkeypatch) -> None:
    monkeypatch.setattr("app.api.v1.evaluation.get_retrieval_service", lambda: FakeRetrieval())
    response = TestClient(create_app()).post(
        "/api/v1/evaluate",
        json={
            "cases": [
                {
                    "id": "acquisition",
                    "query": "Who acquired Activision Blizzard?",
                    "expected_source_ids": ["microsoft-email"],
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["mean_source_recall"] == 1.0
