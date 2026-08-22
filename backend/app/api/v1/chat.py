from functools import lru_cache

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.v1.retrieval import get_retrieval_service
from app.chat import ChatService, OllamaTextGenerator
from app.config import get_settings
from app.schemas import ChatRequest
from app.workflow import GraphRAGWorkflow

router = APIRouter(tags=["chat"])


@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        retrieval=GraphRAGWorkflow(get_retrieval_service()),
        generator=OllamaTextGenerator(settings.ollama_url, settings.ollama_chat_model),
    )


@router.post("/chat", summary="Stream a grounded GraphRAG answer via SSE")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        get_chat_service().stream_events(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
