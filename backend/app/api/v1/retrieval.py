from functools import lru_cache

from fastapi import APIRouter, Query

from app.config import get_settings
from app.embeddings import OllamaEmbeddingProvider
from app.graph import Neo4jGraphStore
from app.retrieval import RetrievalService
from app.schemas import RetrievalRequest, RetrievalResponse, SubgraphResponse
from app.vector_store import QdrantVectorStore

router = APIRouter(tags=["retrieval"])


@lru_cache
def get_retrieval_service() -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        embeddings=OllamaEmbeddingProvider(settings.ollama_url, settings.ollama_embedding_model),
        vector_store=QdrantVectorStore(
            settings.qdrant_url,
            settings.qdrant_children_collection,
            settings.qdrant_parents_collection,
        ),
        graph_store=Neo4jGraphStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.graph_namespace,
        ),
    )


@router.post("/retrieve", response_model=RetrievalResponse, summary="Retrieve GraphRAG evidence")
async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    return await get_retrieval_service().retrieve(request)


@router.get("/graph/subgraph", response_model=SubgraphResponse, summary="Return query-relevant graph JSON")
async def subgraph(
    query: str = Query(min_length=1, max_length=4_000),
    graph_hops: int = Query(default=2, ge=0, le=3),
    graph_limit: int = Query(default=20, ge=1, le=50),
) -> SubgraphResponse:
    nodes, edges = await get_retrieval_service().subgraph(
        RetrievalRequest(query=query, graph_hops=graph_hops, graph_limit=graph_limit)
    )
    return SubgraphResponse(nodes=nodes, edges=edges)
