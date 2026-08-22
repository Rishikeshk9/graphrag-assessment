from app.chunking import HierarchicalChunker, token_count


def test_hierarchical_chunking_preserves_parent_child_provenance() -> None:
    content = " ".join(f"token-{index}" for index in range(25))
    chunker = HierarchicalChunker(
        parent_size=10, parent_overlap=2, child_size=4, child_overlap=1
    )

    document = chunker.chunk("handbook", content)

    assert len(document.parents) == 3
    assert all(parent.token_count <= 10 for parent in document.parents)
    assert all(child.token_count <= 4 for child in document.children)
    assert {child.parent_id for child in document.children} == {
        parent.id for parent in document.parents
    }
    assert document.children[0].parent_index == 0
    assert token_count(document.children[0].text) == 4


def test_chunk_ids_are_stable_for_identical_input() -> None:
    chunker = HierarchicalChunker(parent_size=6, parent_overlap=1, child_size=3, child_overlap=1)

    first = chunker.chunk("source-a", "alpha beta gamma delta epsilon zeta eta")
    second = chunker.chunk("source-a", "alpha  beta gamma delta epsilon zeta eta")

    assert [parent.id for parent in first.parents] == [parent.id for parent in second.parents]
    assert [child.id for child in first.children] == [child.id for child in second.children]

