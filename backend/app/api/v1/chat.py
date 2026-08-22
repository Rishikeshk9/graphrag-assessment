from functools import lru_cache

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.v1.retrieval import get_retrieval_service
from app.chat import ChatService, OllamaTextGenerator, OpenRouterTextGenerator
from app.config import get_settings
from app.schemas import ChatRequest
from app.workflow import GraphRAGWorkflow

router = APIRouter(tags=["chat"])


@lru_cache
def get_chat_service(provider: str) -> ChatService:
    settings = get_settings()
    if provider == "openrouter":
        generator = OpenRouterTextGenerator(
            settings.openrouter_api_key,
            settings.openrouter_chat_model,
            base_url=settings.openrouter_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )
    else:
        generator = OllamaTextGenerator(settings.ollama_url, settings.ollama_chat_model)
    return ChatService(
        retrieval=GraphRAGWorkflow(get_retrieval_service(provider)),
        generator=generator,
    )


@router.post("/chat", summary="Stream a grounded GraphRAG answer via SSE")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        get_chat_service(request.llm_provider).stream_events(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
