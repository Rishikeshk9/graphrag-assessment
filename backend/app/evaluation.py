"""Deterministic, source-aware evaluation for GraphRAG retrieval quality."""

from __future__ import annotations

import asyncio
import re

from app.retrieval import RetrievalService
from app.schemas import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRequest,
    EvaluationResponse,
    RetrievalRequest,
)


def _canonical(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _recall(expected: set[object], actual: set[object]) -> float:
    return 1.0 if not expected else len(expected & actual) / len(expected)


class EvaluationService:
    """Scores retrieval independent of generative-model phrasing.

    It tracks expected-source recall, expected-graph-fact recall, and whether
    each child citation is contained in its returned parent context.
    """

    def __init__(self, retrieval: RetrievalService) -> None:
        self.retrieval = retrieval

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        results = await asyncio.gather(*(self._evaluate_case(case, request) for case in request.cases))
        total = len(results)
        return EvaluationResponse(
            cases=results,
            mean_source_recall=sum(item.source_recall for item in results) / total,
            mean_graph_recall=sum(item.graph_recall for item in results) / total,
            mean_citation_grounding_rate=sum(item.citation_grounding_rate for item in results) / total,
            pass_rate=sum(item.passed for item in results) / total,
        )

    async def _evaluate_case(self, case: EvaluationCase, request: EvaluationRequest) -> EvaluationCaseResult:
        retrieved = await self.retrieval.retrieve(
            RetrievalRequest(
                query=case.query,
                child_limit=request.child_limit,
                graph_hops=request.graph_hops,
                graph_limit=request.graph_limit,
            )
        )
        source_ids = {item.source_id for item in retrieved.child_citations}
        expected_sources = set(case.expected_source_ids)
        actual_triples = {
            (_canonical(item.subject), _canonical(item.predicate), _canonical(item.object))
            for item in retrieved.graph_triples
        }
        expected_triples = {
            (_canonical(subject), _canonical(predicate), _canonical(object))
            for subject, predicate, object in case.expected_graph_triples
        }
        parents = {item.parent_chunk_id: _canonical(item.text) for item in retrieved.parent_contexts}
        grounded = [
            citation for citation in retrieved.child_citations
            if _canonical(citation.excerpt) in parents.get(citation.parent_chunk_id, "")
        ]
        grounding_rate = len(grounded) / len(retrieved.child_citations) if retrieved.child_citations else 0.0
        source_recall = _recall(expected_sources, source_ids)
        graph_recall = _recall(expected_triples, actual_triples)
        return EvaluationCaseResult(
            id=case.id,
            query=case.query,
            retrieved_source_ids=sorted(source_ids),
            source_recall=source_recall,
            graph_recall=graph_recall,
            citation_grounding_rate=grounding_rate,
            passed=source_recall == 1.0 and graph_recall == 1.0 and grounding_rate == 1.0,
        )
