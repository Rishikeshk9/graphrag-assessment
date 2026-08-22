"""GraphRAG evidence retrieval: child-vector search, parent expansion, graph traversal."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from app.embeddings import EmbeddingProvider
from app.graph import GraphFact, GraphStore
from app.schemas import Citation, GraphTriple, ParentContext, RetrievalRequest, RetrievalResponse
from app.vector_store import StoredParent, VectorHit, VectorStore


class RetrievalService:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        graph_store: GraphStore,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.graph_store = graph_store

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        query_vector = (await self.embeddings.embed([request.query]))[0]
        child_hits, graph_facts = await asyncio.gather(
            self.vector_store.search_children(query_vector, request.child_limit),
            self.graph_store.traverse(request.query, request.graph_hops, request.graph_limit),
        )
        parent_ids = list(dict.fromkeys(hit.parent_chunk_id for hit in child_hits))
        parents = await self.vector_store.get_parents(parent_ids)
        return RetrievalResponse(
            query=request.query,
            child_citations=[self._citation(hit) for hit in child_hits],
            parent_contexts=self._parent_contexts(parents, child_hits),
            graph_triples=[self._triple(fact) for fact in graph_facts],
        )

    @staticmethod
    def _citation(hit: VectorHit) -> Citation:
        return Citation(
            parent_chunk_id=hit.parent_chunk_id,
            child_chunk_id=hit.child_chunk_id,
            source_id=hit.source_id,
            excerpt=hit.text,
        )

    @staticmethod
    def _parent_contexts(parents: list[StoredParent], hits: list[VectorHit]) -> list[ParentContext]:
        child_ids: dict[str, list[str]] = defaultdict(list)
        for hit in hits:
            child_ids[hit.parent_chunk_id].append(hit.child_chunk_id)
        return [
            ParentContext(
                parent_chunk_id=parent.parent_chunk_id,
                source_id=parent.source_id,
                text=parent.text,
                matching_child_chunk_ids=child_ids[parent.parent_chunk_id],
            )
            for parent in parents
        ]

    @staticmethod
    def _triple(fact: GraphFact) -> GraphTriple:
        return GraphTriple(
            subject=fact.source,
            predicate=fact.relationship_type,
            object=fact.target,
            source_parent_chunk_id=fact.parent_chunk_id,
            source_child_chunk_id=fact.child_chunk_id,
            source_id=fact.source_id,
            evidence=fact.evidence,
            source_ids=list(fact.source_ids),
            supporting_child_chunk_ids=list(fact.supporting_child_chunk_ids),
        )

    async def subgraph(self, request: RetrievalRequest) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Traverse the graph only: visualization never needs the vector path."""
        facts = await self.graph_store.traverse(
            request.query, request.graph_hops, request.graph_limit
        )
        nodes: dict[str, dict[str, str]] = {}
        edges: list[dict[str, str]] = []
        for index, fact in enumerate(facts):
            nodes.setdefault(
                fact.source, {"id": fact.source, "label": fact.source, "type": fact.source_type}
            )
            nodes.setdefault(
                fact.target, {"id": fact.target, "label": fact.target, "type": fact.target_type}
            )
            edges.append(
                {
                    "id": f"{index}:{fact.child_chunk_id}",
                    "source": fact.source,
                    "target": fact.target,
                    "label": fact.relationship_type,
                    "evidence": fact.evidence,
                    "source_id": fact.source_id,
                    "parent_chunk_id": fact.parent_chunk_id,
                    "child_chunk_id": fact.child_chunk_id,
                    "supporting_sources": str(len(set(fact.source_ids))),
                    "supporting_chunks": str(len(fact.supporting_child_chunk_ids)),
                }
            )
        return list(nodes.values()), edges
