"""Structure-aware small-to-big document chunking with stable provenance IDs."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\S+")
SPACE_PATTERN = re.compile(r"\s+")
BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
BLOCK_SEPARATOR = "\n\n"
ID_NAMESPACE = uuid.UUID("7f5554b6-73d9-48d3-a3ee-c3b7cb12fb03")

# Whitespace words undercount subword LLM tokens. Budgeting in words alone made
# a "1000 token" parent closer to 1300 real tokens, so budgets are declared in
# tokens and converted with this ratio before any windowing happens.
TOKENS_PER_WORD = 1.3


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
    """Flatten a run of text to one line so chunk containment stays literal."""
    return SPACE_PATTERN.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def token_count(text: str) -> int:
    """Estimate LLM tokens for chunk budgeting and reporting."""
    return math.ceil(word_count(text) * TOKENS_PER_WORD)


def words_for_tokens(tokens: int) -> int:
    return max(1, int(tokens / TOKENS_PER_WORD))


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


def _atomic_blocks(text: str, size: int, overlap: int) -> list[str]:
    """Split text on blank lines, windowing only blocks larger than the budget."""
    units: list[str] = []
    for block in BLOCK_SPLIT_PATTERN.split(text):
        normalized = normalize_text(block)
        if not normalized:
            continue
        words = TOKEN_PATTERN.findall(normalized)
        if len(words) <= size:
            units.append(normalized)
        else:
            units.extend(_sliding_windows(words, size, overlap))
    return units


def _pack_blocks(units: list[str], size: int, overlap: int) -> list[str]:
    """Greedily fill chunks with whole blocks, carrying an overlap tail forward."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        unit_words = word_count(unit)
        if current and current_words + unit_words > size:
            chunks.append(BLOCK_SEPARATOR.join(current))
            carry: list[str] = []
            carry_words = 0
            for previous in reversed(current):
                previous_words = word_count(previous)
                if carry_words + previous_words > overlap:
                    break
                carry.insert(0, previous)
                carry_words += previous_words
            current, current_words = carry, carry_words
        current.append(unit)
        current_words += unit_words

    if current:
        chunks.append(BLOCK_SEPARATOR.join(current))
    return chunks


class HierarchicalChunker:
    """Creates searchable child chunks that preserve a larger parent context.

    Budgets are token estimates; paragraph boundaries are respected so a chunk
    rarely starts or ends mid-sentence, which keeps citation excerpts readable.
    """

    def __init__(
        self,
        parent_size: int = 1_000,
        parent_overlap: int = 100,
        child_size: int = 200,
        child_overlap: int = 40,
    ) -> None:
        self.parent_words = words_for_tokens(parent_size)
        self.parent_overlap_words = min(words_for_tokens(parent_overlap), self.parent_words - 1)
        self.child_words = words_for_tokens(child_size)
        self.child_overlap_words = min(words_for_tokens(child_overlap), self.child_words - 1)

    def chunk(self, source_id: str, content: str) -> HierarchicalDocument:
        if not normalize_text(content):
            raise ValueError("document content must not be blank")

        parent_units = _atomic_blocks(content, self.parent_words, self.parent_overlap_words)
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        child_index = 0

        parent_texts = _pack_blocks(parent_units, self.parent_words, self.parent_overlap_words)
        for parent_index, parent_text in enumerate(parent_texts):
            parent = ParentChunk(
                id=_stable_id("parent", source_id, parent_index, parent_text),
                source_id=source_id,
                index=parent_index,
                text=parent_text,
                token_count=token_count(parent_text),
                content_sha256=hashlib.sha256(parent_text.encode("utf-8")).hexdigest(),
            )
            parents.append(parent)

            child_units = _atomic_blocks(parent_text, self.child_words, self.child_overlap_words)
            for child_text in _pack_blocks(child_units, self.child_words, self.child_overlap_words):
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
