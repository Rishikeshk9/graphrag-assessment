"""Embedding-provider interface and Ollama implementation."""

from __future__ import annotations

from typing import Protocol

import httpx

from app.http_retry import post_with_retry


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingProvider:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await post_with_retry(
                client,
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return embeddings
