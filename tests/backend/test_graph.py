from app.chunking import ChildChunk
from app.graph import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphExtraction,
    _label_clause,
    grounded_facts,
    normalized_entity_label,
    normalized_relationship_type,
    seed_term_tiers,
    seed_terms,
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


def test_entity_types_become_pascal_case_labels_not_relationship_types() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Acme", entity_type="public company"),
            ExtractedEntity(name="Beta Labs", entity_type="STARTUP"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Acme",
                target="Beta Labs",
                relationship_type="ACQUIRED",
                evidence="Acme acquired Beta Labs.",
            )
        ],
    )

    fact = grounded_facts(extraction, child("Acme acquired Beta Labs."))[0]

    assert fact.source_type == "PublicCompany"
    assert fact.target_type == "Startup"


def test_unusable_entity_types_fall_back_to_the_base_label() -> None:
    assert normalized_entity_label("???") == "Entity"
    assert normalized_entity_label("Person") == "Person"


def test_label_clause_is_valid_cypher_and_omitted_for_the_base_label() -> None:
    assert _label_clause("source", "Company") == ", source:Company"
    assert _label_clause("target", "Entity") == ""


def test_relations_quoting_text_outside_the_chunk_are_rejected() -> None:
    extraction = GraphExtraction(
        entities=[ExtractedEntity(name="Acme")],
        relationships=[
            ExtractedRelationship(
                source="Acme",
                target="Beta Labs",
                relationship_type="OWNS",
                evidence="Acme owns everything.",
            )
        ],
    )

    assert grounded_facts(extraction, child("Acme owns nothing.")) == []
    assert normalized_relationship_type("owns a") == "OWNS_A"


def test_an_undeclared_endpoint_is_kept_with_the_base_label() -> None:
    """Models often return a sound relation while forgetting to list its entities."""
    extraction = GraphExtraction(
        entities=[ExtractedEntity(name="Acme", entity_type="Company")],
        relationships=[
            ExtractedRelationship(
                source="Acme",
                target="Beta Labs",
                relationship_type="ACQUIRED",
                evidence="Acme acquired Beta Labs.",
            )
        ],
    )

    fact = grounded_facts(extraction, child("Acme acquired Beta Labs."))[0]

    assert (fact.source, fact.source_type) == ("Acme", "Company")
    assert (fact.target, fact.target_type) == ("Beta Labs", "Entity")


def test_a_real_quote_paired_with_an_unrelated_relation_is_rejected() -> None:
    """The quote exists in the chunk but names neither endpoint, so it proves nothing."""
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Activision Blizzard", entity_type="Organization"),
            ExtractedEntity(name="Game Pass", entity_type="Product"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Activision Blizzard",
                target="Game Pass",
                relationship_type="OWNS",
                evidence="we will have more to share in the coming months.",
            )
        ],
    )
    text = (
        "We begin the work to make Activision Blizzard's library available in Game Pass "
        "and other platforms - we will have more to share in the coming months."
    )

    assert grounded_facts(extraction, child(text)) == []


def test_pronoun_resolved_relationships_are_rejected() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Bobby Kotick", entity_type="Person"),
            ExtractedEntity(name="Activision Blizzard", entity_type="Organization"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Bobby Kotick",
                target="Activision Blizzard",
                relationship_type="REPORTS_TO",
                evidence=(
                    "Bobby Kotick has agreed to remain in his role through the end "
                    "of 2023, reporting directly to me."
                ),
            )
        ],
    )
    text = (
        "Bobby Kotick has agreed to remain in his role through the end of 2023, "
        "reporting directly to me. Activision Blizzard will continue operating separately."
    )

    assert grounded_facts(extraction, child(text)) == []


def test_pronouns_can_never_be_persisted_as_graph_entities() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="me", entity_type="Person"),
            ExtractedEntity(name="Bobby Kotick", entity_type="Person"),
        ],
        relationships=[
            ExtractedRelationship(
                source="me",
                target="Bobby Kotick",
                relationship_type="REPORTS_TO",
                evidence="Bobby Kotick is reporting directly to me.",
            )
        ],
    )

    assert grounded_facts(
        extraction, child("Bobby Kotick is reporting directly to me.")
    ) == []


def test_platform_mentions_do_not_become_part_of_relationships() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Activision Blizzard", entity_type="Organization"),
            ExtractedEntity(name="mobile", entity_type="Platform"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Activision Blizzard",
                target="mobile",
                relationship_type="PART_OF",
                evidence="across new platforms from mobile to cloud streaming",
            )
        ],
    )
    text = (
        "Activision Blizzard will continue to build games across new platforms "
        "from mobile to cloud streaming."
    )

    assert grounded_facts(extraction, child(text)) == []


def test_pronoun_resolved_visits_are_rejected() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Phil", entity_type="Person"),
            ExtractedEntity(name="Activision", entity_type="Organization"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Phil",
                target="Activision",
                relationship_type="VISITS",
                evidence="Over the coming weeks, I'll be visiting the Activision offices.",
            )
        ],
    )
    text = "Phil Spencer wrote: Over the coming weeks, I'll be visiting the Activision offices."

    assert grounded_facts(extraction, child(text)) == []


def test_relationship_requires_both_endpoints_in_its_evidence() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Microsoft", entity_type="Company"),
            ExtractedEntity(name="Activision Blizzard", entity_type="Company"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Microsoft",
                target="Activision Blizzard",
                relationship_type="ACQUIRED",
                evidence="We have completed the acquisition of Activision Blizzard.",
            )
        ],
    )
    text = (
        "Microsoft CEO Satya Nadella announced: We have completed the acquisition "
        "of Activision Blizzard."
    )

    assert grounded_facts(extraction, child(text)) == []


def test_an_endpoint_missing_from_the_chunk_is_rejected() -> None:
    extraction = GraphExtraction(
        entities=[
            ExtractedEntity(name="Acme", entity_type="Company"),
            ExtractedEntity(name="Beta Labs", entity_type="Company"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Acme",
                target="Beta Labs",
                relationship_type="ACQUIRED",
                evidence="Acme announced a deal today.",
            )
        ],
    )

    assert grounded_facts(extraction, child("Acme announced a deal today.")) == []


def test_self_referential_relations_are_rejected() -> None:
    extraction = GraphExtraction(
        entities=[ExtractedEntity(name="Acme", entity_type="Company")],
        relationships=[
            ExtractedRelationship(
                source="Acme",
                target="acme",
                relationship_type="OWNS",
                evidence="Acme owns Acme.",
            )
        ],
    )

    assert grounded_facts(extraction, child("Acme owns Acme.")) == []


def test_generic_keywords_are_held_back_until_named_seeds_miss() -> None:
    named, fallback = seed_term_tiers("Who acquired Activision Blizzard and who was the company's CEO?")

    assert named == ["activision blizzard", "ceo"]
    assert "company" in fallback
    assert "company" not in named


def test_seed_term_tiers_collapse_to_one_tier_without_keywords() -> None:
    assert seed_term_tiers('About "Game Pass"') == [["game pass"]]


def test_seed_terms_prefer_entities_over_question_words() -> None:
    terms = seed_terms("Who acquired Activision Blizzard?")

    assert terms[0] == "activision blizzard"
    assert "who" not in terms
    assert "the" not in terms


def test_seed_terms_honour_quoted_phrases() -> None:
    assert seed_terms('Tell me about "Game Pass" revenue')[0] == "game pass"


def test_seed_terms_are_empty_for_a_stopword_only_query() -> None:
    assert seed_terms("what about the other one") == []
