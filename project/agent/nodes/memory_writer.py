"""Memory writer node.

Phase two keeps the existing short-term memory flow and adds lightweight
long-term profile updates into SQLite.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from project.agent.state import AgentState, Message, StudyContext
from project.memory.extractor import request_memory_extraction
from project.memory.profile_service import ensure_user_profile, update_profile_fields

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "demo_user"
DATA_COLLECTION_MAX_ATTEMPTS = 5


def _extract_target_score(text: str) -> str | None:
    patterns = (
        r"(?:目标分|目标成绩|目标是|想考到|考到|冲到)\s*(?:是|为)?\s*[:：]?\s*(\d(?:\.\d)?)\s*分?",
        r"ielts\s*(\d(?:\.\d)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_exam_date(text: str) -> str | None:
    iso_match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    zh_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if zh_match:
        year, month, day = zh_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return None


def _persist_long_term_updates(state: AgentState) -> dict[str, str]:
    user_id = str(state.get("user_profile", {}).get("user_id", DEFAULT_USER_ID))
    user_input = state["user_input"]

    target_score = _extract_target_score(user_input)
    exam_date = _extract_exam_date(user_input)
    if target_score is None and exam_date is None:
        return {}

    ensure_user_profile(user_id)
    updated_profile = update_profile_fields(
        user_id,
        target_score=target_score,
        exam_date=exam_date,
    )

    updates: dict[str, str] = {}
    if target_score is not None:
        updates["target_score"] = str(updated_profile.get("target_score", target_score))
    if exam_date is not None:
        updates["exam_date"] = str(updated_profile.get("exam_date", exam_date))
    return updates


def _summarize_user_input(text: str, profile_updates: dict[str, str]) -> str:
    if profile_updates:
        pairs = ", ".join(f"{key}={value}" for key, value in profile_updates.items())
        return f"[Profile updated: {pairs}]"
    normalized = " ".join(text.split()).strip()
    if len(normalized) <= 180:
        return normalized
    return normalized[:177] + "..."


def _summarize_retrieval_state(state: AgentState) -> dict[str, Any]:
    review_state = state.get("writing_review_state", {})
    if not isinstance(review_state, dict):
        return {}
    trace = review_state.get("retrieval_trace", [])
    return {
        "rounds": review_state.get("retrieval_round"),
        "stop_reason": review_state.get("retrieval_state", {}).get("stop_reason")
        if isinstance(review_state.get("retrieval_state"), dict)
        else "",
        "gap_fill_rate": review_state.get("rag_result", {}).get("gap_fill_rate"),
        "last_query": trace[-1].get("query") if isinstance(trace, list) and trace else "",
    }


def _extract_data_request_details(text: str) -> dict[str, Any]:
    normalized = text.lower()
    module = None
    if "阅读" in text or "reading" in normalized:
        module = "reading"
    elif "听力" in text or "listening" in normalized:
        module = "listening"
    elif "写作" in text or "writing" in normalized:
        module = "writing"
    elif "口语" in text or "speaking" in normalized:
        module = "speaking"

    task_type = None
    for label, markers in {
        "true_false_not_given": ("true false not given", "tfng", "判断题"),
        "matching_headings": ("matching headings", "标题匹配"),
        "task1": ("task 1", "小作文"),
        "task2": ("task 2", "大作文"),
        "part1": ("part 1", "part1"),
        "part2": ("part 2", "part2", "cue card"),
        "part3": ("part 3", "part3"),
    }.items():
        if any(marker in normalized or marker in text for marker in markers):
            task_type = label
            break

    year_match = re.search(r"(20\d{2}|(?<!\d)\d{2}(?=年))", text)
    month_match = re.search(r"(?:20\d{2}\s*年)?\s*(1[0-2]|0?[1-9])\s*月", text)
    year = int(year_match.group(1)) if year_match else None
    if year is not None and year < 100:
        year += 2000
    count_match = re.search(r"(?<!\d)(\d{1,2})\s*(?:份|道|个|套|篇)", text)
    count = int(count_match.group(1)) if count_match else None
    requested_format = "pdf" if "pdf" in normalized else "txt" if "txt" in normalized or "文本" in text else "json" if "json" in normalized else None
    return {
        "module": module,
        "task_type": task_type,
        "year": year,
        "month": int(month_match.group(1)) if month_match else None,
        "format": requested_format,
        "count": count,
    }


def _merge_data_details(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {key: current.get(key) or previous.get(key) for key in ("module", "task_type", "year", "month", "format", "count")}


def _update_data_collection_memory(state: AgentState, context: dict[str, Any]) -> None:
    if state.get("intent") != "data_collection":
        return

    previous = context.get("data_collection_request", {})
    previous = previous if isinstance(previous, dict) else {}
    original_request = str(previous.get("original_request") or state["user_input"])
    previous_details = previous.get("details", {}) if isinstance(previous.get("details"), dict) else {}
    details = _merge_data_details(previous_details, _extract_data_request_details(state["user_input"]))
    attempts = int(previous.get("attempts", 0) or 0)
    plan = state.get("plan", [])
    result = state.get("tool_results", {}).get("collect_data")

    if "data_collection_error_report" in plan:
        context["data_collection_request"] = {
            **previous,
            "active": False,
            "status": "data_collection_failed",
            "attempts": attempts,
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        }
        return

    if "clarify_data_collection_request" in plan:
        context["data_collection_request"] = {
            "active": True,
            "status": "needs_more_details",
            "original_request": original_request,
            "details": details,
            "attempts": min(attempts + 1, DATA_COLLECTION_MAX_ATTEMPTS),
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        }
        return

    if isinstance(result, dict) and int(result.get("success_count", 0) or 0) > 0:
        context["data_collection_request"] = {
            "active": False,
            "status": "satisfied",
            "original_request": original_request,
            "details": details,
            "attempts": attempts,
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        }
        return

    if isinstance(result, dict) and int(result.get("success_count", 0) or 0) == 0 and int(result.get("fail_count", 0) or 0) > 0:
        context["data_collection_request"] = {
            "active": True,
            "status": "failed_retryable",
            "original_request": original_request,
            "details": details,
            "last_failures": result.get("failures", []),
            "attempts": min(attempts + 1, DATA_COLLECTION_MAX_ATTEMPTS),
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        }


def write_memory_node(state: AgentState) -> dict[str, list[Message] | StudyContext | dict[str, Any]]:
    """Append this turn to messages, refresh short-term memory, and persist profile changes."""

    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": state["user_input"]})
    messages.append({"role": "assistant", "content": state["final_answer"]})

    previous_context = dict(state.get("study_context", {}))
    total_turns = int(previous_context.get("total_turns", 0)) + 1
    profile_updates: dict[str, str] = {}
    user_profile = dict(state.get("user_profile", {}))
    try:
        profile_updates = _persist_long_term_updates(state)
        user_profile.update(profile_updates)
    except Exception as exc:  # pragma: no cover - defensive logging for runtime issues
        logger.warning("Failed to persist long-term profile updates: %s", exc)

    updated_context: StudyContext = {
        **previous_context,
        "total_turns": total_turns,
        "last_intent": state["intent"],
        "last_user_input": _summarize_user_input(state["user_input"], profile_updates),
        "last_user_input_summary": _summarize_user_input(state["user_input"], profile_updates),
        "last_answer": state["final_answer"],
        "updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
    }
    retrieval_summary = _summarize_retrieval_state(state)
    if retrieval_summary:
        updated_context["retrieval_state_summary"] = retrieval_summary
    _update_data_collection_memory(state, updated_context)
    prompt_result = state.get("tool_results", {}).get("get_random_task2_prompt")
    if isinstance(prompt_result, dict) and prompt_result.get("success"):
        topic = prompt_result.get("topic", {})
        updated_context["active_writing_topic_id"] = str(topic.get("id", ""))
        updated_context["active_writing_prompt"] = str(topic.get("prompt_text", ""))

    review_result = state.get("tool_results", {}).get("review_task2_submission")
    if isinstance(review_result, dict) and review_result.get("success"):
        topic = review_result.get("topic", {})
        updated_context["active_writing_topic_id"] = str(topic.get("id", previous_context.get("active_writing_topic_id", "")))
        updated_context["active_writing_prompt"] = str(
            topic.get("prompt_text", previous_context.get("active_writing_prompt", ""))
        )

    try:
        extraction = request_memory_extraction(
            user_id=str(user_profile.get("user_id") or user_profile.get("id") or DEFAULT_USER_ID),
            messages=messages,
            study_context=updated_context,
        )
        if extraction.get("short_term_memory"):
            updated_context["short_term_memory"] = extraction["short_term_memory"]
        if extraction.get("memory_watermark") is not None:
            updated_context["memory_watermark"] = extraction["memory_watermark"]
        if isinstance(extraction.get("profile_updates"), dict):
            user_profile.update(extraction["profile_updates"])
    except Exception as exc:  # pragma: no cover - memory extraction should not block answers
        logger.warning("Failed to extract coalesced memory: %s", exc)

    logger.info("Memory updated, total_turns=%s", total_turns)
    return {
        "messages": messages,
        "study_context": updated_context,
        "user_profile": user_profile,
    }
