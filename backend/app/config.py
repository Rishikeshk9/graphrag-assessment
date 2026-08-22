from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "GraphRAG Assessment API"
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default=["http://127.0.0.1:5173", "http://localhost:5173"],
        validation_alias="CORS_ORIGINS",
    )
    qdrant_url: str = "http://localhost:6333"
    qdrant_children_collection: str = "graphrag_children"
    qdrant_parents_collection: str = "graphrag_parents"
    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "qwen3-embedding"
    ollama_graph_model: str = "qwen3:4b"
    # Qwen2.5 Instruct reliably emits final-answer tokens without exposing a
    # reasoning stream; Qwen3 remains the structured graph-extraction model.
    ollama_chat_model: str = "qwen2.5:7b-instruct"
    # OpenRouter is selected per chat or ingestion request by the UI switch.
    # Keep model names separate because graph extraction needs JSON output while
    # answer generation benefits from a longer natural-language response.
    openrouter_chat_model: str = "stealth/ox-alpha"
    # Graph extraction can use a stronger hosted structured-output model while
    # embeddings and answer generation remain local. An API key is required
    # only when this provider is selected.
    graph_extraction_provider: Literal["ollama", "openrouter"] = "ollama"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_model: str = "nvidia/nemotron-3-embed-1b:free"
    openrouter_graph_model: str = "stealth/ox-alpha"
    # Ox Alpha supports JSON-object output but not strict JSON Schema routing.
    # Parsed output is still validated locally against GraphExtraction.
    openrouter_graph_response_format: Literal["json_schema", "json_object"] = "json_object"
    openrouter_site_url: str = "http://127.0.0.1:5173"
    openrouter_app_name: str = "GraphRAG Assessment"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    # Local-only default; override with NEO4J_PASSWORD outside development.
    neo4j_password: str = "local-rag-password"  # noqa: S105
    graph_namespace: str = "graphrag-assessment"
    parent_chunk_tokens: int = Field(default=1_000, ge=100, le=2_000)
    parent_chunk_overlap_tokens: int = Field(default=100, ge=0, le=500)
    child_chunk_tokens: int = Field(default=200, ge=50, le=500)
    child_chunk_overlap_tokens: int = Field(default=40, ge=0, le=100)
    graph_extraction_concurrency: int = Field(default=4, ge=1, le=16)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
