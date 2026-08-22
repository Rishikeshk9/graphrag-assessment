"""Qdrant persistence for parent context and searchable child chunks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.chunking import HierarchicalDocument


@dataclass(frozen=True)
class VectorHit:
    child_chunk_id: str
    parent_chunk_id: str
    source_id: str
    text: str
    score: float


@dataclass(frozen=True)
class StoredParent:
    parent_chunk_id: str
    source_id: str
    text: str


class VectorStore(Protocol):
    async def upsert_document(
        self,
        document: HierarchicalDocument,
        parent_vectors: list[list[float]],
        child_vectors: list[list[float]],
        *,
        ingest_run_id: str,
        content_sha256: str,
        file_sha256: str | None = None,
    ) -> None: ...

    async def prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int: ...

    async def clear(self) -> int: ...

    async def usage(self) -> tuple[int, int]: ...

    async def count_indexed_children(self, source_id: str, content_sha256: str) -> int: ...

    async def find_source_by_file(self, file_sha256: str) -> str | None: ...

    async def search_children(self, query_vector: list[float], limit: int) -> list[VectorHit]: ...

    async def get_parents(self, parent_chunk_ids: list[str]) -> list[StoredParent]: ...


class QdrantVectorStore:
    """Stores only child chunks for retrieval and parent chunks for context expansion."""

    def __init__(
        self,
        url: str,
        children_collection: str,
        parents_collection: str,
        client: QdrantClient | None = None,
    ) -> None:
        self.client = client or QdrantClient(url=url)
        self.children_collection = children_collection
        self.parents_collection = parents_collection

    def _ensure_collection(self, name: str, vector_size: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            self.client.create_payload_index(name, "source_id", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(name, "parent_id", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(name, "ingest_run_id", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(name, "document_sha256", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(name, "file_sha256", PayloadSchemaType.KEYWORD)

    def _upsert_document(
        self,
        document: HierarchicalDocument,
        parent_vectors: list[list[float]],
        child_vectors: list[list[float]],
        ingest_run_id: str,
        content_sha256: str,
        file_sha256: str | None,
    ) -> None:
        if len(parent_vectors) != len(document.parents):
            raise ValueError("parent vectors must match parent chunks")
        if len(child_vectors) != len(document.children):
            raise ValueError("child vectors must match child chunks")
        if not child_vectors:
            raise ValueError("cannot persist a document with no child chunks")
        dimension = len(child_vectors[0])
        if dimension == 0 or any(len(vector) != dimension for vector in [*parent_vectors, *child_vectors]):
            raise ValueError("all vectors must share a non-zero dimension")

        self._ensure_collection(self.parents_collection, dimension)
        self._ensure_collection(self.children_collection, dimension)
        self.client.upsert(
            collection_name=self.parents_collection,
            points=[
                PointStruct(
                    id=parent.id,
                    vector=vector,
                    payload={
                        "chunk_type": "parent",
                        "source_id": parent.source_id,
                        "parent_id": parent.id,
                        "parent_index": parent.index,
                        "text": parent.text,
                        "token_count": parent.token_count,
                        "content_sha256": parent.content_sha256,
                        "ingest_run_id": ingest_run_id,
                        "document_sha256": content_sha256,
                        "file_sha256": file_sha256 or "",
                    },
                )
                for parent, vector in zip(document.parents, parent_vectors, strict=True)
            ],
            wait=True,
        )
        self.client.upsert(
            collection_name=self.children_collection,
            points=[
                PointStruct(
                    id=child.id,
                    vector=vector,
                    payload={
                        "chunk_type": "child",
                        "source_id": child.source_id,
                        "parent_id": child.parent_id,
                        "parent_index": child.parent_index,
                        "child_index": child.index,
                        "text": child.text,
                        "token_count": child.token_count,
                        "content_sha256": child.content_sha256,
                        "ingest_run_id": ingest_run_id,
                        "document_sha256": content_sha256,
                        "file_sha256": file_sha256 or "",
                    },
                )
                for child, vector in zip(document.children, child_vectors, strict=True)
            ],
            wait=True,
        )

    async def upsert_document(
        self,
        document: HierarchicalDocument,
        parent_vectors: list[list[float]],
        child_vectors: list[list[float]],
        *,
        ingest_run_id: str,
        content_sha256: str,
        file_sha256: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._upsert_document,
            document,
            parent_vectors,
            child_vectors,
            ingest_run_id,
            content_sha256,
            file_sha256,
        )

    def _prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int:
        """Drop points for a source that the given run did not write.

        Re-ingesting identical content reuses the same point IDs, so those rows
        are overwritten and re-stamped rather than pruned. Only chunks that no
        longer exist in the document are removed. Passing no run keeps nothing,
        which is how a document is deleted outright.
        """
        stale = Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))],
            must_not=(
                [FieldCondition(key="ingest_run_id", match=MatchValue(value=keep_ingest_run_id))]
                if keep_ingest_run_id is not None
                else None
            ),
        )
        removed = 0
        for collection in (self.parents_collection, self.children_collection):
            if not self.client.collection_exists(collection):
                continue
            removed += self.client.count(collection, count_filter=stale, exact=True).count
            self.client.delete(collection, points_selector=FilterSelector(filter=stale), wait=True)
        return removed

    async def prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int:
        return await asyncio.to_thread(self._prune_document, source_id, keep_ingest_run_id)

    def _clear(self) -> int:
        removed = 0
        for collection in (self.parents_collection, self.children_collection):
            if self.client.collection_exists(collection):
                removed += self.client.count(collection, exact=True).count
                self.client.delete_collection(collection)
        return removed

    async def clear(self) -> int:
        """Delete this application's parent and child vector collections."""
        return await asyncio.to_thread(self._clear)

    def _usage(self) -> tuple[int, int]:
        """Return parent and child point counts without creating collections."""
        parent_count = (
            self.client.count(self.parents_collection, exact=True).count
            if self.client.collection_exists(self.parents_collection)
            else 0
        )
        child_count = (
            self.client.count(self.children_collection, exact=True).count
            if self.client.collection_exists(self.children_collection)
            else 0
        )
        return int(parent_count), int(child_count)

    async def usage(self) -> tuple[int, int]:
        return await asyncio.to_thread(self._usage)

    def _count_indexed_children(self, source_id: str, content_sha256: str) -> int:
        if not self.client.collection_exists(self.children_collection):
            return 0
        return self.client.count(
            self.children_collection,
            count_filter=Filter(
                must=[
                    FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                    FieldCondition(key="document_sha256", match=MatchValue(value=content_sha256)),
                ]
            ),
            exact=True,
        ).count

    async def count_indexed_children(self, source_id: str, content_sha256: str) -> int:
        return await asyncio.to_thread(self._count_indexed_children, source_id, content_sha256)

    def _find_source_by_file(self, file_sha256: str) -> str | None:
        """Name the source already holding these exact bytes, if any."""
        if not file_sha256 or not self.client.collection_exists(self.children_collection):
            return None
        points, _ = self.client.scroll(
            self.children_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="file_sha256", match=MatchValue(value=file_sha256))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return str(points[0].payload["source_id"]) if points else None

    async def find_source_by_file(self, file_sha256: str) -> str | None:
        return await asyncio.to_thread(self._find_source_by_file, file_sha256)

    def _search_children(self, query_vector: list[float], limit: int) -> list[VectorHit]:
        response = self.client.query_points(
            collection_name=self.children_collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorHit(
                child_chunk_id=str(point.id),
                parent_chunk_id=str(point.payload["parent_id"]),
                source_id=str(point.payload["source_id"]),
                text=str(point.payload["text"]),
                score=float(point.score),
            )
            for point in response.points
        ]

    async def search_children(self, query_vector: list[float], limit: int) -> list[VectorHit]:
        return await asyncio.to_thread(self._search_children, query_vector, limit)

    def _get_parents(self, parent_chunk_ids: list[str]) -> list[StoredParent]:
        if not parent_chunk_ids:
            return []
        points = self.client.retrieve(
            collection_name=self.parents_collection,
            ids=parent_chunk_ids,
            with_payload=True,
            with_vectors=False,
        )
        by_id = {
            str(point.id): StoredParent(
                parent_chunk_id=str(point.id),
                source_id=str(point.payload["source_id"]),
                text=str(point.payload["text"]),
            )
            for point in points
        }
        return [by_id[parent_id] for parent_id in parent_chunk_ids if parent_id in by_id]

    async def get_parents(self, parent_chunk_ids: list[str]) -> list[StoredParent]:
        return await asyncio.to_thread(self._get_parents, parent_chunk_ids)
