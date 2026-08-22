import asyncio

from app.chat import ChatService
from app.schemas import Citation, GraphTriple, ParentContext, RetrievalResponse


class FakeRetrievalService:
    async def retrieve(self, request):
        return RetrievalResponse(
            query=request.query,
            child_citations=[Citation(parent_chunk_id="p1", child_chunk_id="c1", source_id="doc", excerpt="Acme acquired Beta.")],
            parent_contexts=[ParentContext(parent_chunk_id="p1", source_id="doc", text="Acme acquired Beta.", matching_child_chunk_ids=["c1"])],
            graph_triples=[GraphTriple(subject="Acme", predicate="ACQUIRED", object="Beta", source_parent_chunk_id="p1", source_child_chunk_id="c1", source_id="doc", evidence="Acme acquired Beta.")],
        )


class FakeGenerator:
    async def stream(self, system_prompt, user_prompt):
        assert "[S1]" in user_prompt
        assert "[G1]" in user_prompt
        yield "Acme acquired Beta [S1] [G1]."


def test_chat_streams_evidence_before_answer_tokens() -> None:
    service = ChatService(FakeRetrievalService(), FakeGenerator())

    events = asyncio.run(_collect(service))

    assert events[0].startswith("event: sources")
    assert events[1].startswith("event: parents")
    assert events[2].startswith("event: graph")
    assert 'Acme acquired Beta' in events[3]
    assert events[-1].startswith("event: done")


async def _collect(service):
    from app.schemas import ChatRequest

    return [event async for event in service.stream_events(ChatRequest(query="Who acquired Beta?"))]
