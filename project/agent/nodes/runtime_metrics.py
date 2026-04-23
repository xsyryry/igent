"""Runtime metrics for ReAct tool-chain quality and cost."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from project.agent.state import AgentState
from project.agent.nodes.tracing import trace_node


SUCCESS_FINISH_REASONS = {"enough_information"}
NO_GAIN_FINISH_REASONS = {"repeated_no_gain"}
FAILURE_FINISH_REASONS = {"tool_failed", "unsupported_action", "recursion_limit"}

_LLM_USAGE_EVENTS: list[dict[str, Any]] = []


@dataclass
class _AggregateMetrics:
    tasks: int = 0
    successful_tasks: int = 0
    selected_tool_calls: int = 0
    accurate_tool_selections: int = 0
    steps_successful_total: int = 0
    duplicate_tool_calls: int = 0
    repeated_no_gain_tasks: int = 0
    fallback_attempts: int = 0
    fallback_successes: int = 0
    cacheable_tool_executions: int = 0
    cache_hits: int = 0
    tool_failure_tasks: int = 0
    recovered_failure_tasks: int = 0
    latency_ms_total: int = 0
    token_cost_total: int = 0


_AGGREGATE = _AggregateMetrics()


def reset_llm_usage_events() -> None:
    _LLM_USAGE_EVENTS.clear()


def record_llm_usage(event: dict[str, Any]) -> None:
    _LLM_USAGE_EVENTS.append(event)


def _llm_usage_summary() -> dict[str, Any]:
    prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in _LLM_USAGE_EVENTS)
    completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in _LLM_USAGE_EVENTS)
    total_tokens = sum(int(item.get("total_tokens", 0) or 0) for item in _LLM_USAGE_EVENTS)
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "call_count": len(_LLM_USAGE_EVENTS),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "events": list(_LLM_USAGE_EVENTS[-8:]),
    }


@trace_node("runtime_metrics_init")
def metrics_init_node(state: AgentState) -> dict[str, Any]:
    """Start per-task runtime measurement."""

    reset_llm_usage_events()
    return {
        "runtime_metrics": {
            "started_at_ms": int(time.time() * 1000),
            "task": {
                "intent": state.get("intent", "unknown"),
                "user_input_length": len(str(state.get("user_input", ""))),
            },
        }
    }


@trace_node("runtime_metrics_finalize")
def metrics_finalize_node(state: AgentState) -> dict[str, Any]:
    """Compute task-level and cumulative runtime metrics."""

    started_at_ms = int(state.get("runtime_metrics", {}).get("started_at_ms") or int(time.time() * 1000))
    latency_ms = max(int(time.time() * 1000) - started_at_ms, 0)
    finish_reason = str(state.get("react_finish_reason") or "no_react")
    tool_results = state.get("tool_results", {})
    observations = list(state.get("observations", []))
    traces = tool_results.get("_tool_policy_trace", [])
    traces = traces if isinstance(traces, list) else []
    failures = tool_results.get("_tool_failures", [])
    failures = failures if isinstance(failures, list) else []
    fallbacks = tool_results.get("_tool_fallbacks", [])
    fallbacks = fallbacks if isinstance(fallbacks, list) else []

    selected_tool_calls = len([item for item in observations if isinstance(item.get("action"), dict)])
    accurate_tool_selections = _count_accurate_tool_selections(observations)
    duplicate_tool_calls = 1 if finish_reason == "duplicate_tool_call" else 0
    repeated_no_gain = 1 if finish_reason in NO_GAIN_FINISH_REASONS else 0
    fallback_attempts = len(fallbacks)
    fallback_successes = sum(1 for item in fallbacks if isinstance(item, dict) and item.get("success"))
    cache_hits = sum(1 for item in traces if isinstance(item, dict) and item.get("status") == "cache_hit")
    cacheable_tool_executions = sum(1 for item in traces if isinstance(item, dict) and item.get("status") in {"success", "cache_hit"})
    had_tool_failure = bool(failures)
    recovered_from_failure = bool(had_tool_failure and not tool_results.get("_last_error"))
    task_success = _is_task_success(state, finish_reason)
    llm_usage = _llm_usage_summary()
    token_cost = int(llm_usage.get("total_tokens", 0) or 0)

    current = {
        "task_success": task_success,
        "finish_reason": finish_reason,
        "task_success_rate": 1.0 if task_success else 0.0,
        "tool_selection_accuracy": _safe_ratio(accurate_tool_selections, selected_tool_calls),
        "average_steps_per_successful_task": len(observations) if task_success else 0.0,
        "duplicate_tool_call_rate": _safe_ratio(duplicate_tool_calls, max(selected_tool_calls, 1)),
        "repeated_no_gain_rate": float(repeated_no_gain),
        "fallback_success_rate": _safe_ratio(fallback_successes, fallback_attempts),
        "cache_hit_rate": _safe_ratio(cache_hits, cacheable_tool_executions),
        "tool_failure_recovery_rate": _safe_ratio(1 if recovered_from_failure else 0, 1 if had_tool_failure else 0),
        "average_latency_ms": float(latency_ms),
        "average_token_cost": float(token_cost),
        "counts": {
            "steps": len(observations),
            "selected_tool_calls": selected_tool_calls,
            "accurate_tool_selections": accurate_tool_selections,
            "duplicate_tool_calls": duplicate_tool_calls,
            "repeated_no_gain_tasks": repeated_no_gain,
            "fallback_attempts": fallback_attempts,
            "fallback_successes": fallback_successes,
            "cache_hits": cache_hits,
            "cacheable_tool_executions": cacheable_tool_executions,
            "tool_failures": len(failures),
            "tool_failure_recovered": recovered_from_failure,
            "latency_ms": latency_ms,
            "token_cost": token_cost,
        },
        "llm_usage": llm_usage,
    }
    aggregate = _update_aggregate(current)
    return {
        "runtime_metrics": {
            **dict(state.get("runtime_metrics", {})),
            "current": current,
            "aggregate": aggregate,
        }
    }


def _is_task_success(state: AgentState, finish_reason: str) -> bool:
    if finish_reason in SUCCESS_FINISH_REASONS:
        return bool(str(state.get("final_answer", "")).strip())
    if state.get("intent") == "general_chat" and str(state.get("final_answer", "")).strip():
        return True
    return finish_reason not in FAILURE_FINISH_REASONS and bool(str(state.get("final_answer", "")).strip())


def _count_accurate_tool_selections(observations: list[dict[str, Any]]) -> int:
    count = 0
    for item in observations:
        action = item.get("action")
        observation = item.get("observation", {})
        if not isinstance(action, dict) or not isinstance(observation, dict):
            continue
        if observation.get("success") or observation.get("fallback_used"):
            count += 1
    return count


def _update_aggregate(current: dict[str, Any]) -> dict[str, float | int]:
    counts = current.get("counts", {})
    _AGGREGATE.tasks += 1
    _AGGREGATE.successful_tasks += 1 if current.get("task_success") else 0
    _AGGREGATE.selected_tool_calls += int(counts.get("selected_tool_calls", 0) or 0)
    _AGGREGATE.accurate_tool_selections += int(counts.get("accurate_tool_selections", 0) or 0)
    if current.get("task_success"):
        _AGGREGATE.steps_successful_total += int(counts.get("steps", 0) or 0)
    _AGGREGATE.duplicate_tool_calls += int(counts.get("duplicate_tool_calls", 0) or 0)
    _AGGREGATE.repeated_no_gain_tasks += int(counts.get("repeated_no_gain_tasks", 0) or 0)
    _AGGREGATE.fallback_attempts += int(counts.get("fallback_attempts", 0) or 0)
    _AGGREGATE.fallback_successes += int(counts.get("fallback_successes", 0) or 0)
    _AGGREGATE.cacheable_tool_executions += int(counts.get("cacheable_tool_executions", 0) or 0)
    _AGGREGATE.cache_hits += int(counts.get("cache_hits", 0) or 0)
    had_tool_failure = int(counts.get("tool_failures", 0) or 0) > 0
    _AGGREGATE.tool_failure_tasks += 1 if had_tool_failure else 0
    _AGGREGATE.recovered_failure_tasks += 1 if counts.get("tool_failure_recovered") else 0
    _AGGREGATE.latency_ms_total += int(counts.get("latency_ms", 0) or 0)
    _AGGREGATE.token_cost_total += int(counts.get("token_cost", 0) or 0)

    return {
        "task_count": _AGGREGATE.tasks,
        "task_success_rate": _safe_ratio(_AGGREGATE.successful_tasks, _AGGREGATE.tasks),
        "tool_selection_accuracy": _safe_ratio(_AGGREGATE.accurate_tool_selections, _AGGREGATE.selected_tool_calls),
        "average_steps_per_successful_task": _safe_ratio(_AGGREGATE.steps_successful_total, _AGGREGATE.successful_tasks),
        "duplicate_tool_call_rate": _safe_ratio(_AGGREGATE.duplicate_tool_calls, max(_AGGREGATE.selected_tool_calls, 1)),
        "repeated_no_gain_rate": _safe_ratio(_AGGREGATE.repeated_no_gain_tasks, _AGGREGATE.tasks),
        "fallback_success_rate": _safe_ratio(_AGGREGATE.fallback_successes, _AGGREGATE.fallback_attempts),
        "cache_hit_rate": _safe_ratio(_AGGREGATE.cache_hits, _AGGREGATE.cacheable_tool_executions),
        "tool_failure_recovery_rate": _safe_ratio(_AGGREGATE.recovered_failure_tasks, _AGGREGATE.tool_failure_tasks),
        "average_latency_ms": _safe_ratio(_AGGREGATE.latency_ms_total, _AGGREGATE.tasks),
        "average_token_cost": _safe_ratio(_AGGREGATE.token_cost_total, _AGGREGATE.tasks),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)
