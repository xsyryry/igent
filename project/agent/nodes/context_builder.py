"""Context builder node.

This node turns structured state into a compact text context that a future LLM
generator can consume directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from project.agent.state import AgentState
from project.agent.nodes.tracing import trace_node
from project.config import get_config
from project.memory.extractor import pop_completed_memory_extraction
from project.memory.retriever import retrieve_relevant_memories
from project.memory.snapshot import build_memory_snapshot

logger = logging.getLogger(__name__)


def _format_section(title: str, data: Any) -> str:
    if not data:
        return f"{title}:\n- none"

    if isinstance(data, str):
        return f"{title}:\n{data}"

    return f"{title}:\n{json.dumps(data, ensure_ascii=False, indent=2)}"


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _prepare_retrieved_references(rag_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate retrieved chunks and truncate to top-k source entries."""

    config = get_config()
    grouped_docs: list[dict[str, Any]] = []
    seen_doc_keys: set[str] = set()
    seen_chunk_keys: set[str] = set()

    for doc in rag_result.get("retrieved_docs", []):
        source = str(doc.get("source", "")).strip() or "Reference"
        doc_id = str(doc.get("id", "")).strip() or source
        doc_key = f"{doc_id}::{source}"
        if doc_key in seen_doc_keys:
            continue

        unique_chunks: list[str] = []
        for chunk in doc.get("chunks", []):
            if not isinstance(chunk, str):
                continue
            chunk_text = chunk.strip()
            if not chunk_text:
                continue
            chunk_key = f"{_normalize_text(source)}::{_normalize_text(chunk_text)}"
            if chunk_key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(chunk_key)
            unique_chunks.append(chunk_text)

        if not unique_chunks:
            continue

        seen_doc_keys.add(doc_key)
        grouped_docs.append(
            {
                "id": doc_id,
                "source": source,
                "chunks": unique_chunks,
            }
        )

    return grouped_docs[: config.default_top_k]


def _format_retrieved_docs(rag_result: dict[str, Any]) -> str:
    answer = str(rag_result.get("answer", "")).strip()
    docs = _prepare_retrieved_references(rag_result)
    if not answer and not docs:
        return "Retrieved Knowledge:\n- none"

    lines = [
        "Retrieved Knowledge:",
        f"- query_mode: {rag_result.get('query_mode', 'mix')}",
    ]
    if answer:
        lines.append(f"- answer: {answer}")

    for index, doc in enumerate(docs, start=1):
        joined_chunks = "\n".join(f"  - {chunk}" for chunk in doc["chunks"])
        lines.append(f"{index}. source={doc['source']}\n{joined_chunks}")
    return "\n".join(lines)


def _get_user_id(state: AgentState) -> str:
    profile = state.get("user_profile", {})
    return str(profile.get("id") or profile.get("user_id") or "demo_user")


def _merge_completed_background_memory(state: AgentState) -> tuple[AgentState, bool]:
    extraction = pop_completed_memory_extraction(_get_user_id(state))
    if not extraction:
        return state, False

    merged_state: AgentState = dict(state)
    user_profile = dict(state.get("user_profile", {}))
    study_context = dict(state.get("study_context", {}))
    if extraction.get("short_term_memory"):
        study_context["short_term_memory"] = extraction["short_term_memory"]
    if extraction.get("memory_watermark") is not None:
        study_context["memory_watermark"] = extraction["memory_watermark"]
    if isinstance(extraction.get("profile_updates"), dict):
        user_profile.update(extraction["profile_updates"])
    merged_state["user_profile"] = user_profile
    merged_state["study_context"] = study_context
    return merged_state, True


def _memory_snapshot(state: AgentState) -> dict[str, Any]:
    try:
        return build_memory_snapshot(
            user_id=_get_user_id(state),
            working_memory=state.get("study_context", {}),
        )
    except Exception as exc:  # pragma: no cover - memory should not block answers
        logger.warning("Failed to build memory snapshot: %s", exc)
        return {"error": "memory_snapshot_unavailable"}


def _relevant_memory(state: AgentState) -> dict[str, Any]:
    try:
        result = retrieve_relevant_memories(
            state["user_input"],
            user_id=_get_user_id(state),
            working_memory=state.get("study_context", {}),
            k=get_config().default_top_k,
        )
    except Exception as exc:  # pragma: no cover - memory should not block answers
        logger.warning("Failed to retrieve relevant memory: %s", exc)
        return {"available": False, "items": [], "error": "unavailable"}

    route = result.get("route", {})
    items = result.get("items", [])
    if not route.get("should_search") or not items:
        return {"available": True, "route": route, "items": []}
    return {"available": True, "route": route, "items": items[: get_config().default_top_k]}


@trace_node("context_builder")
def build_context_node(state: AgentState) -> dict[str, Any]:
    """Assemble structured answer context without storing prompt text."""

    state, memory_updated = _merge_completed_background_memory(state)
    rag_result = state.get("tool_results", {}).get("rag", {})
    answer_context = {
        "user_input": state["user_input"],
        "intent": state["intent"],
        "plan": state.get("plan", []),
        "react": {
            "finish_reason": state.get("react_finish_reason"),
            "observations": state.get("observations", []),
        },
        "agent_events": state.get("agent_events", []),
        "agent_outputs": state.get("agent_outputs", {}),
        "relevant_memory": _relevant_memory(state),
        "memory_snapshot": _memory_snapshot(state),
        "user_profile": state.get("user_profile", {}),
        "study_context": state.get("study_context", {}),
        "tool_results": state.get("tool_results", {}),
        "tool_health": state.get("tool_health", {}),
        "retrieved_knowledge": {
            "answer": rag_result.get("answer", ""),
            "query_mode": rag_result.get("query_mode", "mix"),
            "references": _prepare_retrieved_references(rag_result),
        },
    }
    logger.info("Structured answer context built with %s top-level keys", len(answer_context))
    updates: dict[str, Any] = {
        "answer_context": answer_context,
        "retrieved_docs": state.get("retrieved_docs", []),
    }
    if memory_updated:
        updates["user_profile"] = state.get("user_profile", {})
        updates["study_context"] = state.get("study_context", {})
    return updates
