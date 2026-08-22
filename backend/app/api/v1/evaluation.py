"""Repeatable GraphRAG evaluation endpoint."""

from fastapi import APIRouter

from app.api.v1.retrieval import get_retrieval_service
from app.evaluation import EvaluationService
from app.schemas import EvaluationRequest, EvaluationResponse

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("", response_model=EvaluationResponse, summary="Evaluate retrieval, graph facts, and provenance")
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    return await EvaluationService(get_retrieval_service()).evaluate(request)
