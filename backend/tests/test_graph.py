from app.chunking import ChildChunk
from app.graph import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphExtraction,
    grounded_facts,
    normalized_relationship_type,
)


def child(text: str) -> ChildChunk:
    return ChildChunk(
        id="child-1",
        parent_id="parent-1",
        source_id="report",
        parent_index=0,
        index=0,
        text=text,
        token_count=len(text.split()),
        content_sha256="hash",
    )


def test_grounded_facts_preserve_child_and_parent_provenance() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Acme", entity_type="Organization"),
            ExtractedEntity(name="Beta Labs", entity_type="Organization"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Acme",
                target="Beta Labs",
                relationship_type="acquired",
                evidence="Acme acquired Beta Labs in 2024.",
            )
        ],
    )

    facts = grounded_facts(extraction, child("Acme acquired Beta Labs in 2024."))

    assert len(facts) == 1
    assert facts[0].relationship_type == "ACQUIRED"
    assert facts[0].child_chunk_id == "child-1"
    assert facts[0].parent_chunk_id == "parent-1"


def test_ungrounded_or_unknown_entity_relations_are_rejected() -> None:
    extraction = GraphExtraction(
        entities=[ExtractedEntity(name="Acme")],
        relationships=[
            ExtractedRelationship(
                source="Acme",
                target="Missing Entity",
                relationship_type="OWNS",
                evidence="Acme owns everything.",
            )
        ],
    )

    assert grounded_facts(extraction, child("Acme owns nothing.")) == []
    assert normalized_relationship_type("owns a") == "OWNS_A"
