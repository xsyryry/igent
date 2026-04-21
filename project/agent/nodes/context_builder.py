"""Context builder node.

This node turns structured state into a compact text context that a future LLM
generator can consume directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from project.agent.state import AgentState
from project.config import get_config
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


def _format_memory_snapshot(state: AgentState) -> str:
    try:
        snapshot = build_memory_snapshot(
            user_id=_get_user_id(state),
            working_memory=state.get("study_context", {}),
        )
    except Exception as exc:  # pragma: no cover - memory should not block answers
        logger.warning("Failed to build memory snapshot: %s", exc)
        snapshot = {"error": "memory_snapshot_unavailable"}
    return _format_section("Memory Snapshot", snapshot)


def build_context_node(state: AgentState) -> dict[str, str]:
    """Assemble a context summary string from tool outputs and memory."""

    rag_result = state.get("tool_results", {}).get("rag", {})
    sections = [
        _format_section("User Input", state["user_input"]),
        _format_section("Intent", state["intent"]),
        _format_section("Plan", state.get("plan", [])),
        _format_memory_snapshot(state),
        _format_section("User Profile", state.get("user_profile", {})),
        _format_section("Study Context", state.get("study_context", {})),
        _format_section("Tool Results", state.get("tool_results", {})),
        _format_retrieved_docs(rag_result),
    ]
    context_summary = "\n\n".join(sections)
    logger.info("Context summary built with %s sections", len(sections))
    return {
        "context_summary": context_summary,
        "retrieved_docs": state.get("retrieved_docs", []),
    }
