"""Shared pooled HTTP clients for local model calls.

Opening a client per request cost a fresh connection for every chunk during
ingestion. A pooled client is created lazily inside the running loop and closed
by the application lifespan.
"""

from __future__ import annotations

import asyncio

import httpx

DEFAULT_LIMITS = httpx.Limits(max_connections=16, max_keepalive_connections=8)


class SharedAsyncClient:
    def __init__(self, timeout: float, limits: httpx.Limits = DEFAULT_LIMITS) -> None:
        self.timeout = timeout
        self.limits = limits
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(timeout=self.timeout, limits=self.limits)
        return self._client

    async def aclose(self) -> None:
        async with self._lock:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
            self._client = None


model_client = SharedAsyncClient(timeout=120.0)
streaming_client = SharedAsyncClient(timeout=300.0)


async def close_shared_clients() -> None:
    await model_client.aclose()
    await streaming_client.aclose()
