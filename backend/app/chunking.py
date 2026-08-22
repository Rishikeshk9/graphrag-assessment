"""Deterministic small-to-big document chunking with stable provenance IDs."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\S+")
SPACE_PATTERN = re.compile(r"\s+")
ID_NAMESPACE = uuid.UUID("7f5554b6-73d9-48d3-a3ee-c3b7cb12fb03")


@dataclass(frozen=True)
class ParentChunk:
    id: str
    source_id: str
    index: int
    text: str
    token_count: int
    content_sha256: str


@dataclass(frozen=True)
class ChildChunk:
    id: str
    parent_id: str
    source_id: str
    parent_index: int
    index: int
    text: str
    token_count: int
    content_sha256: str


@dataclass(frozen=True)
class HierarchicalDocument:
    source_id: str
    parents: list[ParentChunk]
    children: list[ChildChunk]


def normalize_text(text: str) -> str:
    return SPACE_PATTERN.sub(" ", text).strip()


def token_count(text: str) -> int:
    """Return a predictable whitespace-token approximation for chunk budgeting."""
    return len(TOKEN_PATTERN.findall(text))


def _stable_id(kind: str, source_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
    return str(uuid.uuid5(ID_NAMESPACE, f"{kind}:{source_id}:{index}:{digest}"))


def _sliding_windows(words: list[str], size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("chunk overlap must be non-negative and smaller than chunk size")

    windows: list[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        windows.append(" ".join(window))
        if start + size >= len(words):
            break
    return windows


class HierarchicalChunker:
    """Creates searchable child chunks that preserve a larger parent context."""

    def __init__(
        self,
        parent_size: int = 1_000,
        parent_overlap: int = 100,
        child_size: int = 200,
        child_overlap: int = 40,
    ) -> None:
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, source_id: str, content: str) -> HierarchicalDocument:
        normalized = normalize_text(content)
        if not normalized:
            raise ValueError("document content must not be blank")

        words = TOKEN_PATTERN.findall(normalized)
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        child_index = 0

        for parent_index, parent_text in enumerate(
            _sliding_windows(words, self.parent_size, self.parent_overlap)
        ):
            parent = ParentChunk(
                id=_stable_id("parent", source_id, parent_index, parent_text),
                source_id=source_id,
                index=parent_index,
                text=parent_text,
                token_count=token_count(parent_text),
                content_sha256=hashlib.sha256(parent_text.encode("utf-8")).hexdigest(),
            )
            parents.append(parent)
            for child_text in _sliding_windows(
                TOKEN_PATTERN.findall(parent_text), self.child_size, self.child_overlap
            ):
                children.append(
                    ChildChunk(
                        id=_stable_id("child", source_id, child_index, child_text),
                        parent_id=parent.id,
                        source_id=source_id,
                        parent_index=parent.index,
                        index=child_index,
                        text=child_text,
                        token_count=token_count(child_text),
                        content_sha256=hashlib.sha256(child_text.encode("utf-8")).hexdigest(),
                    )
                )
                child_index += 1

        return HierarchicalDocument(source_id=source_id, parents=parents, children=children)

