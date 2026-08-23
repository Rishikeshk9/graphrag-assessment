"""Stable request and response contracts shared by API stages."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

LLMProvider = Literal["local", "openrouter"]


class ServiceStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    status: ServiceStatus
    service: str
    timestamp: datetime
    dependencies: dict[str, ServiceStatus] = Field(default_factory=dict)


class DocumentInput(BaseModel):
    """One source to hierarchically chunk and index."""

    source_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_url: HttpUrl | None = None
    content: Annotated[str | None, Field(min_length=1)] = None
    file_sha256: str | None = Field(
        default=None,
        description="Hash of the original bytes, when the content came from an uploaded file",
    )
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def requires_inline_content(self) -> "DocumentInput":
        if self.content is None:
            raise ValueError("Stage 2 ingestion requires inline document content")
        return self


class IngestRequest(BaseModel):
    documents: Annotated[list[DocumentInput], Field(min_length=1, max_length=100)]
    force: bool = Field(
        default=False,
        description="Re-index even when the stored content hash already matches",
    )
    llm_provider: LLMProvider = "local"


class IngestResponse(BaseModel):
    job_id: str
    status: Literal["accepted", "processing", "completed", "failed", "cancelled"]
    documents_indexed: int = 0
    documents_skipped: int = 0
    parent_chunks_indexed: int = 0
    child_chunks_indexed: int = 0
    graph_relationships_indexed: int = 0
    graph_extraction_failures: int = 0
    stale_vectors_removed: int = 0
    stale_relationships_removed: int = 0
    warnings: list[str] = Field(default_factory=list)
    phase: Literal["queued", "chunking", "embedding", "graph", "completed", "failed", "cancelled"] = "queued"
    graph_children_processed: int = 0
    graph_children_total: int = 0


class DeleteDocumentResponse(BaseModel):
    source_id: str
    vectors_removed: int
    relationships_removed: int


class KnowledgeBaseDocument(BaseModel):
    source_id: str
    providers: list[LLMProvider] = Field(default_factory=list)
    parent_vectors: int = 0
    child_vectors: int = 0
    file_sha256: str | None = None


class KnowledgeBaseDocumentsResponse(BaseModel):
    documents: list[KnowledgeBaseDocument]


class ClearKnowledgeBaseResponse(BaseModel):
    vectors_removed: int
    relationships_removed: int
    entities_removed: int


class KnowledgeBaseUsageResponse(BaseModel):
    """Counts belonging to the current GraphRAG workspace only."""

    qdrant_parent_vectors: int
    qdrant_child_vectors: int
    neo4j_entities: int
    neo4j_relationships: int


class ChatTurn(BaseModel):
    """One prior exchange, supplied by the client so the API stays stateless."""

    role: Literal["user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=8_000)]


class ChatRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=4_000)]
    conversation_id: str | None = None
    history: Annotated[list[ChatTurn], Field(max_length=20)] = Field(default_factory=list)
    child_limit: Annotated[int, Field(ge=1, le=20)] = 6
    graph_hops: Annotated[int, Field(ge=0, le=3)] = 2
    graph_limit: Annotated[int, Field(ge=1, le=50)] = 20
    llm_provider: LLMProvider = "local"


class ModelProvidersResponse(BaseModel):
    """Provider capabilities exposed to the browser without exposing secrets."""

    default_provider: LLMProvider
    openrouter_configured: bool
    embedding_provider: LLMProvider


class Citation(BaseModel):
    parent_chunk_id: str
    child_chunk_id: str | None = None
    source_id: str
    excerpt: str


class GraphTriple(BaseModel):
    """One canonical fact. The scalar provenance is the primary supporting span.

    A claim stated by several chunks or several documents stays one triple; the
    list fields enumerate everything that backs it.
    """

    subject: str
    predicate: str
    object: str
    source_parent_chunk_id: str
    source_child_chunk_id: str
    source_id: str
    evidence: str
    source_ids: list[str] = Field(default_factory=list)
    supporting_child_chunk_ids: list[str] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=4_000)]
    child_limit: Annotated[int, Field(ge=1, le=20)] = 6
    graph_hops: Annotated[int, Field(ge=0, le=3)] = 2
    graph_limit: Annotated[int, Field(ge=1, le=50)] = 20


class ParentContext(BaseModel):
    parent_chunk_id: str
    source_id: str
    text: str
    matching_child_chunk_ids: list[str]


class RetrievalResponse(BaseModel):
    query: str
    child_citations: list[Citation]
    parent_contexts: list[ParentContext]
    graph_triples: list[GraphTriple]


class SubgraphResponse(BaseModel):
    nodes: list[dict[str, str]]
    edges: list[dict[str, str]]


class EvaluationCase(BaseModel):
    """A deterministic retrieval-quality check for one known question."""

    id: Annotated[str, Field(min_length=1, max_length=128)]
    query: Annotated[str, Field(min_length=1, max_length=4_000)]
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_graph_triples: list[tuple[str, str, str]] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    cases: Annotated[list[EvaluationCase], Field(min_length=1, max_length=100)]
    child_limit: Annotated[int, Field(ge=1, le=20)] = 6
    graph_hops: Annotated[int, Field(ge=0, le=3)] = 2
    graph_limit: Annotated[int, Field(ge=1, le=50)] = 20


class EvaluationCaseResult(BaseModel):
    id: str
    query: str
    retrieved_source_ids: list[str]
    source_recall: float
    graph_recall: float
    citation_grounding_rate: float
    passed: bool


class EvaluationResponse(BaseModel):
    cases: list[EvaluationCaseResult]
    mean_source_recall: float
    mean_graph_recall: float
    mean_citation_grounding_rate: float
    pass_rate: float
