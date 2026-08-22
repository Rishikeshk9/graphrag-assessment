"""Source-grounded, typed graph extraction and Neo4j persistence."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Protocol

from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from app.chunking import ChildChunk, normalize_text
from app.http_client import model_client
from app.http_retry import post_with_retry


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    entity_type: str = Field(default="Entity", min_length=1, max_length=64)


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
    """One statement plus the chunk that supports it.

    Extraction produces one of these per evidence span. Persistence collapses
    them onto a single canonical edge per (subject, predicate, object), so the
    same claim found in five chunks is one edge with five supporting spans. On
    the way back out the scalar fields carry the primary span and the tuples
    carry everything that backs the claim.
    """

    source: str
    source_type: str
    target: str
    target_type: str
    relationship_type: str
    evidence: str
    source_id: str
    parent_chunk_id: str
    child_chunk_id: str
    source_ids: tuple[str, ...] = ()
    supporting_child_chunk_ids: tuple[str, ...] = ()


class GraphExtractor(Protocol):
    extractor_name: str

    async def extract(self, child: ChildChunk) -> list[GraphFact]: ...


class GraphStore(Protocol):
    async def upsert_facts(self, facts: list[GraphFact], *, ingest_run_id: str) -> int: ...

    async def prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int: ...

    async def clear(self) -> tuple[int, int]: ...

    async def usage(self) -> tuple[int, int]: ...

    async def traverse(self, query: str, hops: int, limit: int) -> list[GraphFact]: ...


EXTRACTION_SYSTEM_PROMPT = """You extract a factual knowledge graph from a supplied text fragment.
Return JSON matching the supplied schema. Extract only explicit entities and relationships supported by
an exact evidence quote from the fragment. Use short UPPER_SNAKE_CASE relationship types such as
ACQUIRED, EMPLOYS, LOCATED_IN, OWNS, or PART_OF, and PascalCase entity types such as Company, Person,
Product, Place, or Organization. A relationship's evidence quote must contain the written names of
both endpoints. Never emit a pronoun such as "me", "we", "it", or "they" as an entity, and do not
resolve one into an entity. Do not infer missing facts. If no relationships are explicit, return empty
arrays."""

RELATIONSHIP_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,63}$")
DEFAULT_ENTITY_LABEL = "Entity"
MAX_SEED_TERMS = 12
# A query phrase may name a node more specifically than the node itself
# ("Northwind Traders" should still reach "Northwind"), but short names match
# far too eagerly in that direction.
MIN_CONTAINED_SEED_NAME = 4
PRONOUN_ENTITY_NAMES = frozenset(
    {
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
    }
)

# Traversal seeds are matched against entity names, so query words that carry no
# entity signal only add noise and slow the Cypher scan down.
STOPWORDS = frozenset(
    """about after all also and any are was were what when where which who whom why with
    been being between both but can did does doing during each few for from had has have
    how into its itself more most not now off once only other our out over own same she
    should some such than that the their them then there these they this those through too
    under until very will would you your his her him not don just how did tell give show
    list explain describe summarize summary please one ones another other others thing
    things something anything everything""".split()
)


def canonical_name(value: str) -> str:
    return normalize_text(value).casefold()


def normalized_relationship_type(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper()).strip("_")
    if not RELATIONSHIP_PATTERN.fullmatch(normalized):
        return "RELATED_TO"
    return normalized


def normalized_entity_label(value: str) -> str:
    """Map a model-supplied entity type onto a safe PascalCase Neo4j label."""
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value.strip()) if part]
    normalized = "".join(part[:1].upper() + part[1:].lower() for part in parts)
    if not LABEL_PATTERN.fullmatch(normalized):
        return DEFAULT_ENTITY_LABEL
    return normalized


def _deduplicated(terms: list[str], already_seen: set[str]) -> list[str]:
    unique: list[str] = []
    for term in terms:
        folded = term.casefold()
        if folded and folded not in already_seen:
            already_seen.add(folded)
            unique.append(folded)
    return unique[:MAX_SEED_TERMS]


def seed_term_tiers(query: str) -> list[list[str]]:
    """Split seeds into a precise tier and a recall tier.

    Bare keywords are only worth matching when nothing named in the query hits
    the graph. Mixing them in unconditionally lets a filler word like "company"
    substring-match an unrelated node such as "Company Strategy Series" and drag
    a whole disconnected component into the answer's subgraph.
    """
    phrases = [normalize_text(match) for match in re.findall(r'"([^"]{2,80})"', query)]
    remainder = re.sub(r'"[^"]*"', " ", query)
    proper_nouns = [
        normalize_text(match)
        for match in re.findall(r"\b[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*", remainder)
        if len(normalize_text(match)) > 2 and canonical_name(match) not in STOPWORDS
    ]
    keywords = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]+", remainder)
        if len(word) > 2 and word.casefold() not in STOPWORDS
    ]
    seen: set[str] = set()
    named = _deduplicated([*phrases, *proper_nouns], seen)
    fallback = _deduplicated(keywords, seen)
    return [tier for tier in (named, fallback) if tier]


def seed_terms(query: str) -> list[str]:
    """Every seed the traversal may use, most precise first."""
    return [term for tier in seed_term_tiers(query) for term in tier]


def _display_name(entity: ExtractedEntity | None, fallback: str) -> str:
    return normalize_text(entity.name if entity is not None else fallback)


def _entity_label(entity: ExtractedEntity | None) -> str:
    if entity is None:
        return DEFAULT_ENTITY_LABEL
    return normalized_entity_label(entity.entity_type)


def _endpoints_are_named(
    source: str,
    target: str,
    evidence: str,
    child_text: str,
) -> bool:
    if source in PRONOUN_ENTITY_NAMES or target in PRONOUN_ENTITY_NAMES:
        return False
    if source not in child_text or target not in child_text:
        return False
    return source in evidence and target in evidence


def grounded_facts(extraction: GraphExtraction, child: ChildChunk) -> list[GraphFact]:
    """Reject model output unless it can be directly attributed to the child chunk.

    A quote that merely exists in the chunk is not evidence of the relation the
    model attached it to: small extractors happily pair a real sentence with an
    invented triple. Both endpoints must therefore be named in the child and in
    the evidence quote. Pronouns are never valid graph endpoints.

    The entity list is only consulted for typing. Models routinely return a
    correct relationship while forgetting to repeat its endpoints there, and
    grounding the endpoint names against the chunk is the stronger check anyway.
    """
    entities = {canonical_name(entity.name): entity for entity in extraction.entities}
    child_text = normalize_text(child.text).casefold()
    accepted: list[GraphFact] = []
    seen: set[tuple[str, str, str, str]] = set()

    for relationship in extraction.relationships:
        source_key = canonical_name(relationship.source)
        target_key = canonical_name(relationship.target)
        source = entities.get(source_key)
        target = entities.get(target_key)
        evidence = normalize_text(relationship.evidence)
        key = (
            source_key,
            target_key,
            normalized_relationship_type(relationship.relationship_type),
            evidence.casefold(),
        )
        if source_key == target_key or key in seen:
            continue
        if evidence.casefold() not in child_text:
            continue
        if not _endpoints_are_named(
            source_key,
            target_key,
            evidence.casefold(),
            child_text,
        ):
            continue
        seen.add(key)
        accepted.append(
            GraphFact(
                source=_display_name(source, relationship.source),
                source_type=_entity_label(source),
                target=_display_name(target, relationship.target),
                target_type=_entity_label(target),
                relationship_type=key[2],
                evidence=evidence,
                source_id=child.source_id,
                parent_chunk_id=child.parent_id,
                child_chunk_id=child.id,
            )
        )
    return accepted


class OllamaGraphExtractor:
    extractor_name = "ollama-structured-v1"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def extract(self, child: ChildChunk) -> list[GraphFact]:
        client = await model_client.get()
        response = await post_with_retry(
            client,
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

        content = response.json().get("message", {}).get("content", "")
        try:
            extraction = GraphExtraction.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("Ollama did not return a valid graph extraction") from error
        return grounded_facts(extraction, child)


class OpenRouterGraphExtractor:
    """OpenRouter-backed schema-constrained graph extraction.

    The graph remains source-grounded locally after model output is parsed, so a
    stronger hosted model improves recall without being allowed to persist an
    unsupported edge. OpenRouter exposes an OpenAI-compatible chat endpoint.
    """

    extractor_name = "openrouter-structured-v1"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "",
        app_name: str = "GraphRAG Assessment",
        response_format: str = "json_schema",
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.site_url = site_url
        self.app_name = app_name
        self.response_format = response_format

    async def extract(self, child: ChildChunk) -> list[GraphFact]:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when GRAPH_EXTRACTION_PROVIDER=openrouter"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        schema = GraphExtraction.model_json_schema()
        response_format: dict[str, object]
        user_content = f"TEXT FRAGMENT:\n{child.text}"
        if self.response_format == "json_object":
            # Some models offer JSON mode but not strict server-enforced schemas.
            # Show the schema in-context and retain Pydantic validation before
            # local evidence grounding.
            response_format = {"type": "json_object"}
            user_content += (
                "\n\nOUTPUT JSON SCHEMA (return this object only, without markdown):\n"
                + json.dumps(schema)
            )
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "graph_extraction",
                    "strict": True,
                    "schema": schema,
                },
            }
        response = await post_with_retry(
            await model_client.get(),
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 1_200,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "response_format": response_format,
                # Ox Alpha requires reasoning. Its trace is omitted while the
                # low effort keeps enough completion budget for graph JSON.
                "reasoning": {"effort": "low", "exclude": True},
                # Route only to providers that can honour the declared format.
                "provider": {"require_parameters": True},
            },
        )
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter did not return a text graph extraction")
        try:
            extraction = GraphExtraction.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("OpenRouter did not return a valid graph extraction") from error
        return grounded_facts(extraction, child)


class Neo4jGraphStore:
    """Writes logically isolated assessment facts into Neo4j with chunk provenance."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        namespace: str,
        extractor_name: str = "ollama-structured-v1",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.namespace = namespace
        self.extractor_name = extractor_name
        self._schema_ready = False

    def close(self) -> None:
        self.driver.close()

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_identity IF NOT EXISTS "
                "FOR (entity:Entity) REQUIRE (entity.namespace, entity.canonical_name) IS UNIQUE"
            ).consume()
        self._schema_ready = True

    def _upsert_facts(self, facts: list[GraphFact], ingest_run_id: str) -> int:
        if not facts:
            return 0
        self._ensure_schema()
        with self.driver.session() as session:
            for fact in facts:
                # Labels and relationship types cannot be parameterized in Cypher.
                # Both are normalized above to a strict identifier pattern first.
                source_label = _label_clause("source", fact.source_type)
                target_label = _label_clause("target", fact.target_type)
                query = f"""
                MERGE (source:Entity {{namespace: $namespace, canonical_name: $source_canonical_name}})
                SET source.name = $source_name, source.entity_type = $source_type{source_label}
                MERGE (target:Entity {{namespace: $namespace, canonical_name: $target_canonical_name}})
                SET target.name = $target_name, target.entity_type = $target_type{target_label}
                MERGE (source)-[relation:{fact.relationship_type} {{namespace: $namespace}}]->(target)
                ON CREATE SET relation.evidence = [],
                              relation.child_chunk_ids = [],
                              relation.parent_chunk_ids = [],
                              relation.source_ids = [],
                              relation.ingest_run_ids = []
                WITH relation,
                     [index IN range(0, size(relation.source_ids) - 1)
                      WHERE NOT (relation.child_chunk_ids[index] = $child_chunk_id
                                 AND relation.evidence[index] = $evidence)] AS kept
                SET relation.evidence =
                        [index IN kept | relation.evidence[index]] + [$evidence],
                    relation.child_chunk_ids =
                        [index IN kept | relation.child_chunk_ids[index]] + [$child_chunk_id],
                    relation.parent_chunk_ids =
                        [index IN kept | relation.parent_chunk_ids[index]] + [$parent_chunk_id],
                    relation.source_ids =
                        [index IN kept | relation.source_ids[index]] + [$source_id],
                    relation.ingest_run_ids =
                        [index IN kept | relation.ingest_run_ids[index]] + [$ingest_run_id],
                    relation.extractor = $extractor_name
                """
                session.run(
                    query,
                    namespace=self.namespace,
                    ingest_run_id=ingest_run_id,
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
                    extractor_name=self.extractor_name,
                ).consume()
        return len(facts)

    async def upsert_facts(self, facts: list[GraphFact], *, ingest_run_id: str) -> int:
        return await asyncio.to_thread(self._upsert_facts, facts, ingest_run_id)

    def _prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int:
        """Withdraw a source's support for its claims, keeping shared facts.

        An edge is not owned by a document, it is *supported* by one. So
        pruning removes this source's evidence spans and only deletes the edge
        once nothing supports it any more: if another document independently
        stated the same thing, the fact survives with that document's
        provenance. Entities follow the same rule one level up and are removed
        only once no relationship touches them.
        """
        # `NULL <> 'run'` is NULL rather than true in Cypher, so spans written
        # before stamping existed have to be matched explicitly or they would
        # survive every prune.
        withdraw_evidence = """
        MATCH ()-[relation]->()
        WHERE relation.namespace = $namespace AND $source_id IN relation.source_ids
        WITH relation, size(relation.source_ids) AS before,
             [index IN range(0, size(relation.source_ids) - 1)
              WHERE NOT (relation.source_ids[index] = $source_id
                         AND ($keep_ingest_run_id IS NULL
                              OR relation.ingest_run_ids[index] IS NULL
                              OR relation.ingest_run_ids[index] <> $keep_ingest_run_id))] AS kept
        SET relation.evidence = [index IN kept | relation.evidence[index]],
            relation.child_chunk_ids = [index IN kept | relation.child_chunk_ids[index]],
            relation.parent_chunk_ids = [index IN kept | relation.parent_chunk_ids[index]],
            relation.source_ids = [index IN kept | relation.source_ids[index]],
            relation.ingest_run_ids = [index IN kept | relation.ingest_run_ids[index]]
        RETURN sum(before - size(kept)) AS removed
        """
        unsupported_relationships = """
        MATCH ()-[relation]->()
        WHERE relation.namespace = $namespace
          AND (size(coalesce(relation.source_ids, [])) = 0
               OR (relation.source_ids IS NULL AND relation.source_id = $source_id))
        WITH collect(relation) AS dead
        FOREACH (relation IN dead | DELETE relation)
        RETURN size(dead) AS removed
        """
        orphan_entities = """
        MATCH (entity:Entity {namespace: $namespace})
        WHERE NOT (entity)--()
        WITH collect(entity) AS orphans
        FOREACH (entity IN orphans | DELETE entity)
        RETURN size(orphans) AS removed
        """
        with self.driver.session() as session:
            removed = session.run(
                withdraw_evidence,
                namespace=self.namespace,
                source_id=source_id,
                keep_ingest_run_id=keep_ingest_run_id,
            ).single()["removed"]
            session.run(
                unsupported_relationships, namespace=self.namespace, source_id=source_id
            ).consume()
            session.run(orphan_entities, namespace=self.namespace).consume()
        return int(removed or 0)

    async def prune_document(self, source_id: str, keep_ingest_run_id: str | None) -> int:
        return await asyncio.to_thread(self._prune_document, source_id, keep_ingest_run_id)

    def _clear(self) -> tuple[int, int]:
        """Delete graph facts and entities belonging to this application's namespace."""
        with self.driver.session() as session:
            relationships = session.run(
                "MATCH ()-[relation]->() WHERE relation.namespace = $namespace "
                "RETURN count(relation) AS count",
                namespace=self.namespace,
            ).single()["count"]
            entities = session.run(
                "MATCH (entity:Entity {namespace: $namespace}) RETURN count(entity) AS count",
                namespace=self.namespace,
            ).single()["count"]
            session.run(
                "MATCH ()-[relation]->() WHERE relation.namespace = $namespace DELETE relation",
                namespace=self.namespace,
            ).consume()
            session.run(
                "MATCH (entity:Entity {namespace: $namespace}) DELETE entity",
                namespace=self.namespace,
            ).consume()
        return int(relationships), int(entities)

    async def clear(self) -> tuple[int, int]:
        return await asyncio.to_thread(self._clear)

    def _usage(self) -> tuple[int, int]:
        """Return entity and relationship counts for this app's graph namespace."""
        with self.driver.session() as session:
            entities = session.run(
                "MATCH (entity:Entity {namespace: $namespace}) RETURN count(entity) AS count",
                namespace=self.namespace,
            ).single()["count"]
            relationships = session.run(
                "MATCH ()-[relation]->() WHERE relation.namespace = $namespace "
                "RETURN count(relation) AS count",
                namespace=self.namespace,
            ).single()["count"]
        return int(entities), int(relationships)

    async def usage(self) -> tuple[int, int]:
        return await asyncio.to_thread(self._usage)

    def _traverse(self, query: str, hops: int, limit: int) -> list[GraphFact]:
        if hops == 0:
            return []
        for terms in seed_term_tiers(query):
            facts = self._traverse_terms(terms, hops, limit)
            if facts:
                return facts
        return []

    def _traverse_terms(self, terms: list[str], hops: int, limit: int) -> list[GraphFact]:
        # `hops` is validated at the API boundary before becoming Cypher syntax.
        cypher = f"""
        MATCH (seed:Entity {{namespace: $namespace}})
        WHERE any(term IN $terms WHERE toLower(seed.name) CONTAINS term
                  OR (size(seed.name) >= $min_contained_name AND term CONTAINS toLower(seed.name)))
        MATCH path=(seed)-[*1..{hops}]-(:Entity {{namespace: $namespace}})
        UNWIND range(0, size(relationships(path)) - 1) AS hop
        WITH relationships(path)[hop] AS relation, hop
        WHERE relation.namespace = $namespace
        WITH relation, min(hop) AS distance
        ORDER BY distance
        LIMIT $limit
        RETURN startNode(relation).name AS source,
               coalesce(startNode(relation).entity_type, 'Entity') AS source_type,
               type(relation) AS relationship_type,
               endNode(relation).name AS target,
               coalesce(endNode(relation).entity_type, 'Entity') AS target_type,
               relation.evidence[0] AS evidence,
               relation.source_ids[0] AS source_id,
               relation.parent_chunk_ids[0] AS parent_chunk_id,
               relation.child_chunk_ids[0] AS child_chunk_id,
               relation.source_ids AS source_ids,
               relation.child_chunk_ids AS supporting_child_chunk_ids
        """
        with self.driver.session() as session:
            records = session.run(
                cypher,
                namespace=self.namespace,
                terms=terms,
                limit=limit,
                min_contained_name=MIN_CONTAINED_SEED_NAME,
            )
            return [_fact_from_record(record.data()) for record in records]

    async def traverse(self, query: str, hops: int, limit: int) -> list[GraphFact]:
        return await asyncio.to_thread(self._traverse, query, hops, limit)


def _fact_from_record(record: dict[str, object]) -> GraphFact:
    fields = dict(record)
    for name in ("source_ids", "supporting_child_chunk_ids"):
        fields[name] = tuple(fields.get(name) or ())
    return GraphFact(**fields)  # type: ignore[arg-type]


def _label_clause(alias: str, entity_type: str) -> str:
    """Return a `SET` fragment adding a typed label next to the base `:Entity`."""
    if entity_type == DEFAULT_ENTITY_LABEL:
        return ""
    return f", {alias}:{entity_type}"
