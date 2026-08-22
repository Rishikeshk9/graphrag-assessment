import asyncio
import json

import httpx

from app.chat import (
    ChatService,
    OpenRouterTextGenerator,
    build_generation_messages,
    condense_query,
)
from app.schemas import (
    ChatRequest,
    ChatTurn,
    Citation,
    GraphTriple,
    ParentContext,
    RetrievalResponse,
)


class FakeRetrievalService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, request):
        self.queries.append(request.query)
        return RetrievalResponse(
            query=request.query,
            child_citations=[
                Citation(
                    parent_chunk_id="p1",
                    child_chunk_id="c1",
                    source_id="doc",
                    excerpt="Acme acquired Beta.",
                )
            ],
            parent_contexts=[
                ParentContext(
                    parent_chunk_id="p1",
                    source_id="doc",
                    text="Acme acquired Beta.",
                    matching_child_chunk_ids=["c1"],
                )
            ],
            graph_triples=[
                GraphTriple(
                    subject="Acme",
                    predicate="ACQUIRED",
                    object="Beta",
                    source_parent_chunk_id="p1",
                    source_child_chunk_id="c1",
                    source_id="doc",
                    evidence="Acme acquired Beta.",
                )
            ],
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def stream(self, messages):
        self.messages = messages
        assert messages[0]["role"] == "system"
        assert "[S1]" in messages[-1]["content"]
        assert "[G1]" in messages[-1]["content"]
        yield "Acme acquired Beta [S1] [G1]."


def collect(service, request):
    async def run():
        return [event async for event in service.stream_events(request)]

    return asyncio.run(run())


def test_chat_streams_evidence_before_answer_tokens() -> None:
    service = ChatService(FakeRetrievalService(), FakeGenerator())

    events = collect(service, ChatRequest(query="Who acquired Beta?"))

    assert events[0].startswith("event: sources")
    assert events[1].startswith("event: parents")
    assert events[2].startswith("event: graph")
    assert "Acme acquired Beta" in events[3]
    assert events[-1].startswith("event: done")


def test_history_is_replayed_to_the_model_and_used_for_retrieval() -> None:
    retrieval = FakeRetrievalService()
    generator = FakeGenerator()
    request = ChatRequest(
        query="What did it pay?",
        history=[
            ChatTurn(role="user", content="Who acquired Activision Blizzard?"),
            ChatTurn(role="assistant", content="Microsoft did [S1]."),
        ],
    )

    collect(ChatService(retrieval, generator), request)

    assert "Activision Blizzard" in retrieval.queries[0]
    assert [message["role"] for message in generator.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_condense_query_is_a_no_op_without_history() -> None:
    assert condense_query("Who acquired Beta?", []) == "Who acquired Beta?"


def test_generation_citations_match_the_displayed_child_excerpts() -> None:
    messages = build_generation_messages(
        RetrievalResponse(
            query="Who is Phil?",
            child_citations=[
                Citation(
                    parent_chunk_id="p1",
                    child_chunk_id="c1",
                    source_id="email",
                    excerpt="Phil thanks the team.",
                ),
                Citation(
                    parent_chunk_id="p1",
                    child_chunk_id="c2",
                    source_id="email",
                    excerpt="From: Phil Spencer.",
                ),
                Citation(
                    parent_chunk_id="p1",
                    child_chunk_id="c3",
                    source_id="email",
                    excerpt="Bobby Kotick reports directly to me.",
                ),
            ],
            parent_contexts=[
                ParentContext(
                    parent_chunk_id="p1",
                    source_id="email",
                    text="From: Phil Spencer. Bobby Kotick reports directly to me.",
                    matching_child_chunk_ids=["c1", "c2", "c3"],
                )
            ],
            graph_triples=[],
        ),
        "Who is Phil?",
        [],
    )

    prompt = messages[-1]["content"]
    assert "[P1] Source: email" in prompt
    first_excerpt = "[S1] Source: email; parent: p1\nExact retrieved excerpt: Phil thanks the team."
    third_excerpt = (
        "[S3] Source: email; parent: p1\n"
        "Exact retrieved excerpt: Bobby Kotick reports directly to me."
    )
    assert first_excerpt in prompt
    assert third_excerpt in prompt
    assert "Preserve the direction" in messages[0]["content"]


def test_openrouter_generator_streams_openai_compatible_tokens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"Grounded"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":" answer."}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def get_client() -> httpx.AsyncClient:
        return client

    monkeypatch.setattr("app.chat.streaming_client.get", get_client)
    async def collect_tokens() -> list[str]:
        generator = OpenRouterTextGenerator(
            "test-key", "example/model", site_url="https://example.test"
        )
        tokens = [
            token
            async for token in generator.stream([{"role": "user", "content": "hello"}])
        ]
        await client.aclose()
        return tokens

    tokens = asyncio.run(collect_tokens())

    assert "".join(tokens) == "Grounded answer."
    assert captured["payload"] == {
        "model": "example/model",
        "stream": True,
        "temperature": 0.1,
        "max_tokens": 320,
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "low", "exclude": True},
    }
    assert captured["headers"]["authorization"] == "Bearer test-key"
