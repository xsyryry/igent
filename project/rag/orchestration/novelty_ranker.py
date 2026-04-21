"""Local relevance + novelty ranker."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from project.config import get_config


@dataclass(slots=True)
class RankedCandidate:
    doc: dict[str, Any]
    relevance_score: float
    slot_coverage_score: float
    novelty_score: float
    history_penalty: float
    final_score: float


class NoveltyRanker:
    """Rank candidates by relevance and novelty using local MMR-style scoring."""

    def __init__(self) -> None:
        config = get_config()
        self.novelty_weight = float(config.retrieval_novelty_weight)
        self.history_penalty_weight = float(config.retrieval_history_penalty)
        self.backend = "local_mmr"

    def rank(
        self,
        *,
        query: str,
        gap_text: str,
        candidates: list[dict[str, Any]],
        seen_texts: list[str],
        top_k: int,
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        ranked_pool: list[RankedCandidate] = []
        history_vectors = [_token_vector(text) for text in seen_texts if text.strip()]
        query_vector = _token_vector(f"{query}\n{gap_text}")
        for candidate in candidates:
            candidate_text = _candidate_text(candidate)
            candidate_vector = _token_vector(candidate_text)
            relevance = _cosine(query_vector, candidate_vector)
            slot_coverage = _slot_coverage(gap_text, candidate)
            history_penalty = 0.0
            if history_vectors:
                history_penalty = max(_cosine(candidate_vector, history_vector) for history_vector in history_vectors)
            ranked_pool.append(
                RankedCandidate(
                    doc=candidate,
                    relevance_score=relevance,
                    slot_coverage_score=slot_coverage,
                    novelty_score=1.0,
                    history_penalty=history_penalty,
                    final_score=relevance + slot_coverage - self.history_penalty_weight * history_penalty,
                )
            )

        selected: list[RankedCandidate] = []
        remaining = list(ranked_pool)
        while remaining and len(selected) < top_k:
            best_index = 0
            best_score = -math.inf
            for index, item in enumerate(remaining):
                novelty = 1.0
                if selected:
                    novelty = 1.0 - max(
                        _cosine(
                            _token_vector(_candidate_text(item.doc)),
                            _token_vector(_candidate_text(selected_item.doc)),
                        )
                        for selected_item in selected
                    )
                final_score = (
                    (1.0 - self.novelty_weight) * item.relevance_score
                    + 0.25 * item.slot_coverage_score
                    + self.novelty_weight * novelty
                    - self.history_penalty_weight * item.history_penalty
                )
                if final_score > best_score:
                    best_score = final_score
                    best_index = index
            chosen = remaining.pop(best_index)
            chosen.novelty_score = (
                1.0
                if not selected
                else 1.0
                - max(
                    _cosine(
                        _token_vector(_candidate_text(chosen.doc)),
                        _token_vector(_candidate_text(selected_item.doc)),
                    )
                    for selected_item in selected
                )
            )
            chosen.final_score = best_score
            selected.append(chosen)

        return selected


def _candidate_text(candidate: dict[str, Any]) -> str:
    chunks = candidate.get("chunks", [])
    content = "\n".join(chunk for chunk in chunks if isinstance(chunk, str))
    return f"{candidate.get('source', '')}\n{content}".strip()


def _slot_coverage(gap_text: str, candidate: dict[str, Any]) -> float:
    gap_tokens = set(_token_vector(gap_text))
    if not gap_tokens:
        return 0.0
    candidate_tokens = set(_token_vector(_candidate_text(candidate)))
    metadata = candidate.get("metadata", {}) if isinstance(candidate.get("metadata"), dict) else {}
    for value in metadata.values():
        if isinstance(value, str):
            candidate_tokens.update(_token_vector(value))
        elif isinstance(value, list):
            for item in value:
                candidate_tokens.update(_token_vector(str(item)))
    return len(gap_tokens & candidate_tokens) / max(len(gap_tokens), 1)


def _token_vector(text: str) -> dict[str, float]:
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
