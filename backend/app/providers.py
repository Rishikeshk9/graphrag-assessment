"""Factories for model providers and their compatible vector collections."""

from app.config import Settings
from app.embeddings import EmbeddingProvider, OllamaEmbeddingProvider, OpenRouterEmbeddingProvider
from app.schemas import LLMProvider
from app.vector_store import QdrantVectorStore


def embedding_provider_for(settings: Settings, provider: LLMProvider) -> EmbeddingProvider:
    if provider == "openrouter":
        return OpenRouterEmbeddingProvider(
            settings.openrouter_api_key,
            settings.openrouter_embedding_model,
            base_url=settings.openrouter_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )
    return OllamaEmbeddingProvider(settings.ollama_url, settings.ollama_embedding_model)


def vector_store_for(settings: Settings, provider: LLMProvider) -> QdrantVectorStore:
    """Keep models with different vector dimensions in separate collections."""
    suffix = "" if provider == "local" else "_openrouter"
    return QdrantVectorStore(
        settings.qdrant_url,
        f"{settings.qdrant_children_collection}{suffix}",
        f"{settings.qdrant_parents_collection}{suffix}",
        api_key=settings.qdrant_api_key,
    )
