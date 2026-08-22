from app.chunking import HierarchicalChunker, token_count, words_for_tokens


def test_hierarchical_chunking_preserves_parent_child_provenance() -> None:
    content = " ".join(f"token-{index}" for index in range(25))
    chunker = HierarchicalChunker(
        parent_size=10, parent_overlap=2, child_size=4, child_overlap=1
    )

    document = chunker.chunk("handbook", content)

    parent_ids = {parent.id for parent in document.parents}
    assert len(document.parents) > 1
    assert all(parent.token_count <= 10 for parent in document.parents)
    assert all(child.token_count <= 4 for child in document.children)
    assert {child.parent_id for child in document.children} == parent_ids
    assert document.children[0].parent_index == 0


def test_every_child_is_contained_in_its_parent_context() -> None:
    content = "\n\n".join(
        " ".join(f"word{paragraph}-{index}" for index in range(40)) for paragraph in range(4)
    )
    document = HierarchicalChunker().chunk("manual", content)
    parents = {parent.id: parent.text for parent in document.parents}

    assert all(child.text in parents[child.parent_id] for child in document.children)


def test_paragraph_boundaries_are_respected_when_they_fit() -> None:
    paragraphs = ["Microsoft acquired Activision Blizzard.", "Phil Spencer sent the email."]
    document = HierarchicalChunker(child_size=40, child_overlap=0).chunk(
        "email", "\n\n".join(paragraphs)
    )

    assert all(
        any(child.text.startswith(paragraph.split()[0]) for paragraph in paragraphs)
        for child in document.children
    )


def test_token_budget_accounts_for_subword_tokens() -> None:
    # Ten whitespace words are more than ten LLM tokens, so the word budget for a
    # token budget must be strictly smaller than the token count.
    assert words_for_tokens(1_000) < 1_000
    assert token_count("one two three four") > 4


def test_chunk_ids_are_stable_for_identical_input() -> None:
    chunker = HierarchicalChunker(parent_size=6, parent_overlap=1, child_size=3, child_overlap=1)

    first = chunker.chunk("source-a", "alpha beta gamma delta epsilon zeta eta")
    second = chunker.chunk("source-a", "alpha  beta gamma delta epsilon zeta eta")

    assert [parent.id for parent in first.parents] == [parent.id for parent in second.parents]
    assert [child.id for child in first.children] == [child.id for child in second.children]


def test_blank_content_is_rejected() -> None:
    try:
        HierarchicalChunker().chunk("empty", "   \n\n  ")
    except ValueError as error:
        assert "blank" in str(error)
    else:  # pragma: no cover - guard against silent acceptance
        raise AssertionError("blank content must raise")
