"""Stable request and response contracts shared by API stages."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


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
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def requires_inline_content(self) -> "DocumentInput":
        if self.content is None:
            raise ValueError("Stage 2 ingestion requires inline document content")
        return self


class IngestRequest(BaseModel):
    documents: Annotated[list[DocumentInput], Field(min_length=1, max_length=100)]


class IngestResponse(BaseModel):
    job_id: str
    status: Literal["accepted", "processing", "completed", "failed"]
    documents_indexed: int = 0
    parent_chunks_indexed: int = 0
    child_chunks_indexed: int = 0
    graph_relationships_indexed: int = 0
    graph_extraction_failures: int = 0
    warnings: list[str] = Field(default_factory=list)
    phase: Literal["queued", "chunking", "embedding", "graph", "completed", "failed"] = "queued"
    graph_children_processed: int = 0
    graph_children_total: int = 0


class ChatRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=4_000)]
    conversation_id: str | None = None
    child_limit: Annotated[int, Field(ge=1, le=20)] = 6
    graph_hops: Annotated[int, Field(ge=0, le=3)] = 2
    graph_limit: Annotated[int, Field(ge=1, le=50)] = 20


class Citation(BaseModel):
    parent_chunk_id: str
    child_chunk_id: str | None = None
    source_id: str
    excerpt: str


class GraphTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    source_parent_chunk_id: str
    source_child_chunk_id: str
    source_id: str
    evidence: str


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
