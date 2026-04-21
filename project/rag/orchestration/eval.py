"""Evaluation helpers for gap-driven retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from project.rag.orchestration.gap_retrieval import (
    build_writing_review_retrieval_state,
    run_gap_retrieval_round,
    summarize_gap_retrieval_state,
)


def load_retrieval_eval_cases() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "data" / "retrieval_eval_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gap_retrieval_cases(limit: int | None = None) -> dict[str, Any]:
    cases = load_retrieval_eval_cases()
    if limit is not None:
        cases = cases[:limit]

    case_results: list[dict[str, Any]] = []
    for case in cases:
        state = build_writing_review_retrieval_state(
            prompt_text=str(case["prompt_text"]),
            essay_text=str(case["essay_text"]),
            dataset_scope="writing",
        )
        while not state.get("complete"):
            state = run_gap_retrieval_round(state)
        summary = summarize_gap_retrieval_state(state)
        trace = list(summary.get("retrieval_trace", []))
        duplicate_rates = [float(item.get("duplicate_rate", 0.0) or 0.0) for item in trace]
        new_facts = [int(item.get("new_facts", 0) or 0) for item in trace]
        support_text = (
            str(summary.get("answer") or "").lower()
            + " "
            + " ".join(str(item).lower() for item in summary.get("known_facts", []))
        )
        expected_keywords = [str(item).lower() for item in case.get("expected_keywords", [])]
        matched_keywords = sum(1 for keyword in expected_keywords if keyword in support_text)
        case_results.append(
            {
                "case_id": case["case_id"],
                "rounds": len(trace),
                "avg_duplicate_rate": round(mean(duplicate_rates), 3) if duplicate_rates else 0.0,
                "new_facts_per_round": new_facts,
                "total_new_facts": sum(new_facts),
                "gap_fill_rate": float(summary.get("gap_fill_rate", 0.0) or 0.0),
                "final_answer_accuracy": round(matched_keywords / max(len(expected_keywords), 1), 3),
                "latency_ms": int(summary.get("metrics", {}).get("latency_ms", 0) or 0),
                "token_estimate": int(summary.get("metrics", {}).get("token_estimate", 0) or 0),
                "stop_reason": summary.get("stop_reason", ""),
            }
        )

    summary = {
        "case_count": len(case_results),
        "avg_cross_round_duplicate_rate": round(mean([item["avg_duplicate_rate"] for item in case_results]), 3)
        if case_results
        else 0.0,
        "avg_new_facts": round(mean([item["total_new_facts"] for item in case_results]), 3) if case_results else 0.0,
        "avg_gap_fill_rate": round(mean([item["gap_fill_rate"] for item in case_results]), 3) if case_results else 0.0,
        "avg_final_answer_accuracy": round(mean([item["final_answer_accuracy"] for item in case_results]), 3)
        if case_results
        else 0.0,
        "avg_rounds": round(mean([item["rounds"] for item in case_results]), 3) if case_results else 0.0,
        "avg_latency_ms": round(mean([item["latency_ms"] for item in case_results]), 3) if case_results else 0.0,
        "avg_token_estimate": round(mean([item["token_estimate"] for item in case_results]), 3)
        if case_results
        else 0.0,
    }
    return {"summary": summary, "cases": case_results}
