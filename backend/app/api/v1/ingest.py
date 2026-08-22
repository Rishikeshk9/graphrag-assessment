from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.chunking import HierarchicalChunker
from app.config import get_settings
from app.document_loader import DocumentExtractionError, extract_pdf_document
from app.embeddings import OllamaEmbeddingProvider
from app.graph import Neo4jGraphStore, OllamaGraphExtractor
from app.ingestion import IngestionService
from app.jobs import IngestionJobs
from app.schemas import IngestRequest, IngestResponse
from app.vector_store import QdrantVectorStore

router = APIRouter(tags=["ingestion"])


@lru_cache
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        chunker=HierarchicalChunker(
            parent_size=settings.parent_chunk_tokens,
            parent_overlap=settings.parent_chunk_overlap_tokens,
            child_size=settings.child_chunk_tokens,
            child_overlap=settings.child_chunk_overlap_tokens,
        ),
        embeddings=OllamaEmbeddingProvider(
            settings.ollama_url, settings.ollama_embedding_model
        ),
        vector_store=QdrantVectorStore(
            settings.qdrant_url,
            settings.qdrant_children_collection,
            settings.qdrant_parents_collection,
        ),
        graph_extractor=OllamaGraphExtractor(
            settings.ollama_url, settings.ollama_graph_model
        ),
        graph_store=Neo4jGraphStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.graph_namespace,
        ),
    )


@lru_cache
def get_ingestion_jobs() -> IngestionJobs:
    return IngestionJobs(get_ingestion_service())


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest inline documents with parent-child chunking",
)
async def ingest(request: IngestRequest) -> IngestResponse:
    return await get_ingestion_jobs().submit(request.documents)


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a text-based PDF",
)
async def ingest_pdf(
    file: Annotated[UploadFile, File(description="Text-selectable PDF document")],
) -> IngestResponse:
    filename = file.filename or "uploaded-document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")
    payload = await file.read()
    try:
        document = extract_pdf_document(filename, payload)
    except DocumentExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return await get_ingestion_jobs().submit([document])


@router.get("/ingest/{job_id}", response_model=IngestResponse, summary="Get background ingestion progress")
async def ingestion_status(job_id: str) -> IngestResponse:
    result = await get_ingestion_jobs().get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return result
