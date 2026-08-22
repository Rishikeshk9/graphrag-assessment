from functools import lru_cache
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.chunking import HierarchicalChunker
from app.config import get_settings
from app.document_loader import DocumentExtractionError, extract_pdf_document, file_sha256
from app.graph import GraphExtractor, Neo4jGraphStore, OllamaGraphExtractor, OpenRouterGraphExtractor
from app.ingestion import IngestionService
from app.jobs import IngestionJobs
from app.providers import embedding_provider_for, vector_store_for
from app.schemas import (
    ClearKnowledgeBaseResponse,
    DeleteDocumentResponse,
    IngestRequest,
    IngestResponse,
    KnowledgeBaseUsageResponse,
    LLMProvider,
)

router = APIRouter(tags=["ingestion"])


@lru_cache
def get_ingestion_service(provider: LLMProvider = "local") -> IngestionService:
    settings = get_settings()
    graph_extractor: GraphExtractor
    if provider == "openrouter":
        graph_extractor = OpenRouterGraphExtractor(
            settings.openrouter_api_key,
            settings.openrouter_graph_model,
            base_url=settings.openrouter_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
            response_format=settings.openrouter_graph_response_format,
        )
    else:
        graph_extractor = OllamaGraphExtractor(
            settings.ollama_url, settings.ollama_graph_model
        )
    return IngestionService(
        chunker=HierarchicalChunker(
            parent_size=settings.parent_chunk_tokens,
            parent_overlap=settings.parent_chunk_overlap_tokens,
            child_size=settings.child_chunk_tokens,
            child_overlap=settings.child_chunk_overlap_tokens,
        ),
        embeddings=embedding_provider_for(settings, provider),
        vector_store=vector_store_for(settings, provider),
        graph_extractor=graph_extractor,
        graph_store=Neo4jGraphStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.graph_namespace,
            graph_extractor.extractor_name,
        ),
        extraction_concurrency=settings.graph_extraction_concurrency,
    )


@lru_cache
def get_ingestion_jobs(provider: LLMProvider = "local") -> IngestionJobs:
    return IngestionJobs(get_ingestion_service(provider))


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest inline documents with parent-child chunking",
)
async def ingest(request: IngestRequest) -> IngestResponse:
    return await get_ingestion_jobs(request.llm_provider).submit(request.documents, force=request.force)


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a text-based PDF",
)
async def ingest_pdf(
    file: Annotated[UploadFile, File(description="Text-selectable PDF document")],
    force: Annotated[bool, Query(description="Re-index even if the content is unchanged")] = False,
    llm_provider: Annotated[LLMProvider, Query(description="Provider for graph extraction")] = "local",
) -> IngestResponse:
    filename = file.filename or "uploaded-document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")
    payload = await file.read()
    if not force:
        indexed_as = await get_ingestion_service(llm_provider).source_holding_file(file_sha256(payload))
        if indexed_as is not None:
            return IngestResponse(
                job_id=str(uuid4()),
                status="completed",
                phase="completed",
                documents_skipped=1,
                warnings=[f"{indexed_as}: identical file already indexed, nothing to do"],
            )
    try:
        document = extract_pdf_document(filename, payload)
    except DocumentExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return await get_ingestion_jobs(llm_provider).submit([document], force=force)


@router.delete(
    "/documents/{source_id}",
    response_model=DeleteDocumentResponse,
    summary="Remove a document's chunks and graph relationships",
)
async def delete_document(source_id: str) -> DeleteDocumentResponse:
    service = get_ingestion_service()
    vectors_removed = await service.vector_store.prune_document(source_id, keep_ingest_run_id=None)
    relationships_removed = 0
    if service.graph_store is not None:
        relationships_removed = await service.graph_store.prune_document(
            source_id, keep_ingest_run_id=None
        )
    return DeleteDocumentResponse(
        source_id=source_id,
        vectors_removed=vectors_removed,
        relationships_removed=relationships_removed,
    )


@router.delete(
    "/knowledge-base",
    response_model=ClearKnowledgeBaseResponse,
    summary="Clear all vectors and graph facts in this knowledge base",
)
async def clear_knowledge_base() -> ClearKnowledgeBaseResponse:
    local_service = get_ingestion_service("local")
    openrouter_service = get_ingestion_service("openrouter")
    vectors_removed = await local_service.vector_store.clear()
    vectors_removed += await openrouter_service.vector_store.clear()
    relationships_removed = 0
    entities_removed = 0
    if local_service.graph_store is not None:
        relationships_removed, entities_removed = await local_service.graph_store.clear()
    return ClearKnowledgeBaseResponse(
        vectors_removed=vectors_removed,
        relationships_removed=relationships_removed,
        entities_removed=entities_removed,
    )


@router.get(
    "/knowledge-base/usage",
    response_model=KnowledgeBaseUsageResponse,
    summary="Show current vector and graph record counts",
)
async def knowledge_base_usage() -> KnowledgeBaseUsageResponse:
    local_service = get_ingestion_service("local")
    openrouter_service = get_ingestion_service("openrouter")
    local_parents, local_children = await local_service.vector_store.usage()
    router_parents, router_children = await openrouter_service.vector_store.usage()
    entities = 0
    relationships = 0
    if local_service.graph_store is not None:
        entities, relationships = await local_service.graph_store.usage()
    return KnowledgeBaseUsageResponse(
        qdrant_parent_vectors=local_parents + router_parents,
        qdrant_child_vectors=local_children + router_children,
        neo4j_entities=entities,
        neo4j_relationships=relationships,
    )


@router.get("/ingest/{job_id}", response_model=IngestResponse, summary="Get background ingestion progress")
async def ingestion_status(job_id: str) -> IngestResponse:
    result = await get_ingestion_jobs("local").get(job_id)
    if result is None:
        result = await get_ingestion_jobs("openrouter").get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return result


@router.post("/ingest/{job_id}/cancel", response_model=IngestResponse, summary="Cancel background ingestion")
async def cancel_ingestion(job_id: str) -> IngestResponse:
    result = await get_ingestion_jobs("local").cancel(job_id)
    if result is None:
        result = await get_ingestion_jobs("openrouter").cancel(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return result
