import asyncio

import httpx
import pytest

from app.embeddings import OpenRouterEmbeddingProvider


def test_openrouter_embedding_provider_uses_embeddings_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = request.content
        captured["authorization"] = request.headers["authorization"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def fake_get() -> httpx.AsyncClient:
        return client

    monkeypatch.setattr("app.embeddings.model_client.get", fake_get)
    provider = OpenRouterEmbeddingProvider("test-key", "example/embed", base_url="https://example.test/v1")

    async def run() -> list[list[float]]:
        try:
            return await provider.embed(["first", "second"])
        finally:
            await client.aclose()

    assert asyncio.run(run()) == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["authorization"] == "Bearer test-key"
    assert b'"input":["first","second"]' in captured["payload"]


def test_openrouter_embedding_provider_requires_key() -> None:
    provider = OpenRouterEmbeddingProvider("", "example/embed")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        asyncio.run(provider.embed(["text"]))
