"""Small retry policy for transient local-model startup or overload errors."""

from __future__ import annotations

import asyncio

import httpx


async def post_with_retry(client: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
    last_error: httpx.HTTPError | None = None
    for attempt in range(3):
        try:
            response = await client.post(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            last_error = error
            client_error = (
                isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500
            )
            if attempt == 2 or client_error:
                raise
            await asyncio.sleep(0.4 * (2**attempt))
    raise RuntimeError("unreachable") from last_error
