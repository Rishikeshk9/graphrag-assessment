from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "GraphRAG Assessment API"
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    api_v1_prefix: str = "/api/v1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_children_collection: str = "graphrag_children"
    qdrant_parents_collection: str = "graphrag_parents"
    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "qwen3-embedding"
    ollama_graph_model: str = "qwen3:4b"
    # Qwen2.5 Instruct reliably emits final-answer tokens without exposing a
    # reasoning stream; Qwen3 remains the structured graph-extraction model.
    ollama_chat_model: str = "qwen2.5:7b-instruct"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "local-rag-password"
    graph_namespace: str = "graphrag-assessment"
    parent_chunk_tokens: int = Field(default=1_000, ge=100, le=2_000)
    parent_chunk_overlap_tokens: int = Field(default=100, ge=0, le=500)
    child_chunk_tokens: int = Field(default=200, ge=50, le=500)
    child_chunk_overlap_tokens: int = Field(default=40, ge=0, le=100)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
