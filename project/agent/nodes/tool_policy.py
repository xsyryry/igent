"""Tool action selection, retry policy, and failure classification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from project.agent.state import AgentState, ToolCall


@dataclass(frozen=True)
class ToolPolicy:
    max_attempts: int
    timeout_seconds: float
    retry_backoff_seconds: float = 0.25
    circuit_failure_threshold: int = 3


DEFAULT_TOOL_POLICY = ToolPolicy(max_attempts=2, timeout_seconds=20.0)
TOOL_POLICIES: dict[str, ToolPolicy] = {
    "db": ToolPolicy(max_attempts=1, timeout_seconds=5.0),
    "rag": ToolPolicy(max_attempts=2, timeout_seconds=15.0),
    "calendar": ToolPolicy(max_attempts=2, timeout_seconds=10.0),
    "data": ToolPolicy(max_attempts=2, timeout_seconds=45.0, retry_backoff_seconds=0.5),
    "cambridge_crawler": ToolPolicy(max_attempts=2, timeout_seconds=120.0, retry_backoff_seconds=0.5),
    "question_pdf": ToolPolicy(max_attempts=1, timeout_seconds=60.0),
    "export_question_pdf": ToolPolicy(max_attempts=1, timeout_seconds=60.0),
    "mistake": ToolPolicy(max_attempts=2, timeout_seconds=20.0),
    "writing": ToolPolicy(max_attempts=2, timeout_seconds=30.0),
    "web_search": ToolPolicy(max_attempts=2, timeout_seconds=20.0, retry_backoff_seconds=0.5),
    "web": ToolPolicy(max_attempts=2, timeout_seconds=20.0, retry_backoff_seconds=0.5),
}

CACHEABLE_TOOLS = {
    "db.get_user_profile",
    "db.get_study_plan",
    "db.get_mistake_records",
    "rag.retrieve_knowledge",
}


def policy_for(tool_call: ToolCall) -> ToolPolicy:
    return TOOL_POLICIES.get(str(tool_call.get("tool_name") or ""), DEFAULT_TOOL_POLICY)


def tool_id(tool_call: ToolCall) -> str:
    return f"{tool_call.get('tool_name', 'unknown')}.{tool_call.get('action', 'unknown')}"


def result_key_for(tool_call: ToolCall) -> str:
    tool_name = str(tool_call.get("tool_name") or "")
    action = str(tool_call.get("action") or "")
    if tool_name == "rag":
        return "rag"
    if tool_name in {"web_search", "web"} and action in {"search", "search_web"}:
        return "search_web"
    return action


def cache_key_for(tool_call: ToolCall) -> str:
    return json.dumps(
        {
            "tool": tool_id(tool_call),
            "args": tool_call.get("args", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def is_cacheable(tool_call: ToolCall) -> bool:
    return tool_id(tool_call) in CACHEABLE_TOOLS


def classify_error_text(message: str, *, default_retryable: bool) -> tuple[str, bool]:
    text = message.lower()
    if any(token in text for token in ("timed out", "timeout", "deadline")):
        return "timeout", True
    if any(token in text for token in ("rate limit", "429", "too many requests")):
        return "rate_limit", True
    if any(token in text for token in ("not configured", "api_key", "api key", "credential", "unauthorized", "401", "403")):
        return "configuration", False
    if any(token in text for token in ("unsupported", "invalid", "missing", "keyerror", "bad request", "400")):
        return "invalid_request", False
    if any(token in text for token in ("connection", "network", "temporar", "502", "503", "504", "500", "service unavailable")):
        return "transient", True
    return "execution_error", default_retryable


def build_failure(
    *,
    tool_call: ToolCall,
    category: str,
    message: str,
    retryable: bool,
    attempt: int,
    attempts: int,
    elapsed_ms: int,
    result_key: str | None = None,
) -> dict[str, Any]:
    return {
        "tool_name": tool_call.get("tool_name"),
        "action": tool_call.get("action"),
        "result_key": result_key or result_key_for(tool_call),
        "category": category,
        "retryable": retryable,
        "attempt": attempt,
        "attempts": attempts,
        "elapsed_ms": elapsed_ms,
        "error": message,
    }


def fallback_for(tool_call: ToolCall, failure: dict[str, Any], state: AgentState) -> ToolCall | None:
    tool_name = str(tool_call.get("tool_name") or "")
    action = str(tool_call.get("action") or "")
    user_input = state.get("user_input", "")
    category = str(failure.get("category") or "")

    if category == "invalid_request":
        return None
    if _fallback_already_attempted(state, tool_call):
        return None

    if tool_name == "rag":
        return {"tool_name": "web_search", "action": "search_web", "args": {"query": user_input, "max_results": 5}}
    if tool_name in {"web_search", "web"} and state.get("intent") == "knowledge_qa":
        return {
            "tool_name": "rag",
            "action": "retrieve_knowledge",
            "args": {"question": user_input, "query_mode": "mix", "top_k": 5},
        }
    if tool_name == "calendar" and action == "create_study_event":
        return {"tool_name": "calendar", "action": "get_schedule", "args": {"date": None}}
    if tool_name == "writing" and action == "review_task2_submission":
        return {"tool_name": "writing", "action": "get_random_task2_prompt", "args": {"essay_type": None}}
    if tool_name == "data" and action == "collect_data":
        return {"tool_name": "web_search", "action": "search_web", "args": {"query": user_input, "max_results": 5}}
    if tool_name == "cambridge_crawler":
        return None
    return None


def select_fallback_after_failure(state: AgentState) -> ToolCall | None:
    failure = state.get("tool_results", {}).get("_last_error")
    action = state.get("selected_tool_call")
    if not isinstance(failure, dict) or not isinstance(action, dict):
        return None
    return fallback_for(action, failure, state)


def is_circuit_open(tool_health: dict[str, Any], tool_call: ToolCall) -> bool:
    health = tool_health.get(tool_id(tool_call))
    return isinstance(health, dict) and health.get("status") == "open"


def should_retry(failure: dict[str, Any], attempt: int, policy: ToolPolicy) -> bool:
    return bool(failure.get("retryable")) and attempt < policy.max_attempts


def is_duplicate_tool_call(state: AgentState, tool_call: ToolCall) -> bool:
    key = cache_key_for(tool_call)
    history = state.get("tool_call_history", [])
    return key in history


def append_tool_history(history: list[str], tool_call: ToolCall) -> list[str]:
    key = cache_key_for(tool_call)
    if key in history:
        return history
    return [*history, key][-12:]


def _fallback_already_attempted(state: AgentState, original_call: ToolCall) -> bool:
    original_id = tool_id(original_call)
    fallbacks = state.get("tool_results", {}).get("_tool_fallbacks", [])
    if not isinstance(fallbacks, list):
        return False
    return any(isinstance(item, dict) and item.get("from_tool") == original_id for item in fallbacks)
