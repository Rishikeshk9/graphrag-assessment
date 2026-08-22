from functools import lru_cache

from fastapi import APIRouter, Query

from app.config import get_settings
from app.graph import Neo4jGraphStore
from app.providers import embedding_provider_for, vector_store_for
from app.retrieval import RetrievalService
from app.schemas import LLMProvider, RetrievalRequest, RetrievalResponse, SubgraphResponse

router = APIRouter(tags=["retrieval"])


@lru_cache
def get_retrieval_service(provider: LLMProvider = "local") -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        embeddings=embedding_provider_for(settings, provider),
        vector_store=vector_store_for(settings, provider),
        graph_store=Neo4jGraphStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.graph_namespace,
        ),
    )


@router.post("/retrieve", response_model=RetrievalResponse, summary="Retrieve GraphRAG evidence")
async def retrieve(
    request: RetrievalRequest,
    llm_provider: LLMProvider = Query(default="local", description="Embedding provider"),
) -> RetrievalResponse:
    return await get_retrieval_service(llm_provider).retrieve(request)


@router.get("/graph/subgraph", response_model=SubgraphResponse, summary="Return query-relevant graph JSON")
async def subgraph(
    query: str = Query(min_length=1, max_length=4_000),
    graph_hops: int = Query(default=2, ge=0, le=3),
    graph_limit: int = Query(default=20, ge=1, le=50),
    llm_provider: LLMProvider = Query(default="local", description="Embedding provider"),
) -> SubgraphResponse:
    nodes, edges = await get_retrieval_service(llm_provider).subgraph(
        RetrievalRequest(query=query, graph_hops=graph_hops, graph_limit=graph_limit)
    )
    return SubgraphResponse(nodes=nodes, edges=edges)
