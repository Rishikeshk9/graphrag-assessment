"""Source-grounded, typed graph extraction and Neo4j persistence."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from app.chunking import ChildChunk, normalize_text


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    entity_type: str = Field(default="ENTITY", min_length=1, max_length=64)


class ExtractedRelationship(BaseModel):
    source: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=160)
    relationship_type: str = Field(min_length=1, max_length=64)
    evidence: str = Field(min_length=1, max_length=500)


class GraphExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=30)
    relationships: list[ExtractedRelationship] = Field(default_factory=list, max_length=50)


@dataclass(frozen=True)
class GraphFact:
    source: str
    source_type: str
    target: str
    target_type: str
    relationship_type: str
    evidence: str
    source_id: str
    parent_chunk_id: str
    child_chunk_id: str


class GraphExtractor(Protocol):
    async def extract(self, child: ChildChunk) -> list[GraphFact]: ...


class GraphStore(Protocol):
    async def upsert_facts(self, facts: list[GraphFact]) -> int: ...

    async def traverse(self, query: str, hops: int, limit: int) -> list[GraphFact]: ...


EXTRACTION_SYSTEM_PROMPT = """You extract a factual knowledge graph from a supplied text fragment.
Return JSON matching the supplied schema. Extract only explicit entities and relationships supported by
an exact evidence quote from the fragment. Use short UPPER_SNAKE_CASE relationship types such as
ACQUIRED, EMPLOYS, LOCATED_IN, OWNS, or PART_OF. Do not infer missing facts. If no relationships
are explicit, return empty arrays."""

RELATIONSHIP_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def canonical_name(value: str) -> str:
    return normalize_text(value).casefold()


def normalized_relationship_type(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper()).strip("_")
    if not RELATIONSHIP_PATTERN.fullmatch(normalized):
        return "RELATED_TO"
    return normalized


def grounded_facts(extraction: GraphExtraction, child: ChildChunk) -> list[GraphFact]:
    """Reject model output unless it can be directly attributed to the child chunk."""
    entities = {canonical_name(entity.name): entity for entity in extraction.entities}
    child_text = normalize_text(child.text).casefold()
    accepted: list[GraphFact] = []
    seen: set[tuple[str, str, str, str]] = set()

    for relationship in extraction.relationships:
        source = entities.get(canonical_name(relationship.source))
        target = entities.get(canonical_name(relationship.target))
        evidence = normalize_text(relationship.evidence)
        key = (
            canonical_name(relationship.source),
            canonical_name(relationship.target),
            normalized_relationship_type(relationship.relationship_type),
            evidence.casefold(),
        )
        if source is None or target is None or evidence.casefold() not in child_text or key in seen:
            continue
        seen.add(key)
        accepted.append(
            GraphFact(
                source=normalize_text(source.name),
                source_type=normalized_relationship_type(source.entity_type),
                target=normalize_text(target.name),
                target_type=normalized_relationship_type(target.entity_type),
                relationship_type=key[2],
                evidence=evidence,
                source_id=child.source_id,
                parent_chunk_id=child.parent_id,
                child_chunk_id=child.id,
            )
        )
    return accepted


class OllamaGraphExtractor:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def extract(self, child: ChildChunk) -> list[GraphFact]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": GraphExtraction.model_json_schema(),
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"TEXT FRAGMENT:\n{child.text}"},
                    ],
                },
            )
            response.raise_for_status()

        content = response.json().get("message", {}).get("content", "")
        try:
            extraction = GraphExtraction.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("Ollama did not return a valid graph extraction") from error
        return grounded_facts(extraction, child)


class Neo4jGraphStore:
    """Writes logically isolated assessment facts into Neo4j with chunk provenance."""

    def __init__(self, uri: str, user: str, password: str, namespace: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.namespace = namespace
        self._schema_ready = False

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_identity IF NOT EXISTS "
                "FOR (entity:Entity) REQUIRE (entity.namespace, entity.canonical_name) IS UNIQUE"
            ).consume()
        self._schema_ready = True

    def _upsert_facts(self, facts: list[GraphFact]) -> int:
        if not facts:
            return 0
        self._ensure_schema()
        with self.driver.session() as session:
            for fact in facts:
                query = f"""
                MERGE (source:Entity {{namespace: $namespace, canonical_name: $source_canonical_name}})
                SET source.name = $source_name, source.entity_type = $source_type
                MERGE (target:Entity {{namespace: $namespace, canonical_name: $target_canonical_name}})
                SET target.name = $target_name, target.entity_type = $target_type
                MERGE (source)-[relation:{fact.relationship_type} {{
                    namespace: $namespace,
                    child_chunk_id: $child_chunk_id,
                    evidence: $evidence
                }}]->(target)
                SET relation.source_id = $source_id,
                    relation.parent_chunk_id = $parent_chunk_id,
                    relation.extractor = 'ollama-structured-v1'
                """
                session.run(
                    query,
                    namespace=self.namespace,
                    source_canonical_name=canonical_name(fact.source),
                    source_name=fact.source,
                    source_type=fact.source_type,
                    target_canonical_name=canonical_name(fact.target),
                    target_name=fact.target,
                    target_type=fact.target_type,
                    source_id=fact.source_id,
                    parent_chunk_id=fact.parent_chunk_id,
                    child_chunk_id=fact.child_chunk_id,
                    evidence=fact.evidence,
                ).consume()
        return len(facts)

    async def upsert_facts(self, facts: list[GraphFact]) -> int:
        return await asyncio.to_thread(self._upsert_facts, facts)

    def _traverse(self, query: str, hops: int, limit: int) -> list[GraphFact]:
        if hops == 0:
            return []
        terms = [term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]+", query) if len(term) > 2]
        if not terms:
            return []
        # `hops` is validated at the API boundary before becoming Cypher syntax.
        cypher = f"""
        MATCH (seed:Entity {{namespace: $namespace}})
        WHERE any(term IN $terms WHERE toLower(seed.name) CONTAINS term)
        MATCH path=(seed)-[*1..{hops}]-(connected:Entity {{namespace: $namespace}})
        UNWIND relationships(path) AS relation
        WITH DISTINCT startNode(relation) AS source, relation, endNode(relation) AS target
        WHERE relation.namespace = $namespace
        RETURN source.name AS source,
               source.entity_type AS source_type,
               type(relation) AS relationship_type,
               target.name AS target,
               target.entity_type AS target_type,
               relation.evidence AS evidence,
               relation.source_id AS source_id,
               relation.parent_chunk_id AS parent_chunk_id,
               relation.child_chunk_id AS child_chunk_id
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = session.run(cypher, namespace=self.namespace, terms=terms, limit=limit)
            return [GraphFact(**record.data()) for record in records]

    async def traverse(self, query: str, hops: int, limit: int) -> list[GraphFact]:
        return await asyncio.to_thread(self._traverse, query, hops, limit)
