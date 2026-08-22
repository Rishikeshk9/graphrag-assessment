"""Embedding-provider interface for local Ollama and OpenRouter models."""

from __future__ import annotations

from typing import Protocol

from app.http_client import model_client
from app.http_retry import post_with_retry


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingProvider:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = await model_client.get()
        response = await post_with_retry(
            client,
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return embeddings


class OpenRouterEmbeddingProvider:
    """Generate embeddings through OpenRouter's OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "",
        app_name: str = "GraphRAG Assessment",
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter embeddings")
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        client = await model_client.get()
        response = await post_with_retry(
            client,
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": self.model, "input": texts, "encoding_format": "float"},
        )
        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("OpenRouter returned an invalid embedding response")

        ordered = sorted(data, key=lambda item: item.get("index", -1))
        embeddings = [item.get("embedding") for item in ordered]
        if not all(
            isinstance(vector, list)
            and vector
            and all(isinstance(value, int | float) for value in vector)
            for vector in embeddings
        ):
            raise RuntimeError("OpenRouter returned invalid embedding vectors")
        return embeddings
