"""Tool execution node.

This node only executes the selected tool action. Action choice, fallback
selection, and observation compression live in separate graph nodes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import logging
import time
from typing import Any

from project.agent.nodes.tool_policy import (
    build_failure,
    cache_key_for,
    classify_error_text,
    is_cacheable,
    is_circuit_open,
    policy_for,
    result_key_for,
    should_retry,
    tool_id,
)
from project.agent.nodes.tracing import trace_node
from project.agent.state import AgentState, RetrievedDoc, ToolCall, UserProfile
from project.tools.calendar_tool import create_study_event, get_schedule
from project.tools.cambridge_crawler_tool import ENTRY_URL as CAMBRIDGE_CRAWLER_ENTRY_URL
from project.tools.cambridge_crawler_tool import crawl_writing_questions
from project.tools.data_tool import collect_cambridge_writing_questions, collect_data
from project.tools.db_tool import get_mistake_records, get_study_plan, get_user_profile
from project.tools.mistake_tool import grade_submission
from project.tools.question_pdf_export_tool import export_question_pdf
from project.tools.rag_tool import retrieve_knowledge
from project.tools.web_search_tool import search
from project.tools.writing_tool import (
    get_random_task2_prompt,
    prepare_task2_review_context,
    review_task2_submission,
)

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 300
_TOOL_RESULT_CACHE: dict[str, tuple[float, "ToolOutcome"]] = {}


class ToolTimeoutError(TimeoutError):
    """Raised when a tool call exceeds its execution budget."""


@dataclass(frozen=True)
class ToolOutcome:
    tool_name: str
    action: str
    result_key: str
    result: Any
    cache_hit: bool = False


def _execute_db_action(action: str) -> Any:
    if action == "get_user_profile":
        return get_user_profile()
    if action == "get_study_plan":
        return get_study_plan()
    if action == "get_mistake_records":
        return get_mistake_records()
    raise ValueError(f"Unsupported db action: {action}")


def _execute_rag_action(action: str, args: dict[str, Any]) -> dict[str, Any]:
    if action == "retrieve_knowledge":
        return retrieve_knowledge(
            question=args["question"],
            dataset_scope=args.get("dataset_scope"),
            top_k=args.get("top_k", 5),
            mode=str(args.get("query_mode") or "mix"),
            filters=args.get("filters"),
            user_id=args.get("user_id"),
            banned_doc_ids=args.get("banned_doc_ids"),
            banned_chunk_ids=args.get("banned_chunk_ids"),
        )
    raise ValueError(f"Unsupported rag action: {action}")


def _execute_calendar_action(action: str, args: dict[str, Any]) -> Any:
    if action == "create_study_event":
        return create_study_event(
            title=args.get("title", "IELTS Study Session"),
            start_time=args.get("start_time") or args.get("scheduled_at") or "TBD",
            end_time=args.get("end_time") or "TBD",
            description=args.get("description") or args.get("note"),
        )
    if action == "get_schedule":
        return get_schedule(date=args.get("date") or "2026-04-20")
    raise ValueError(f"Unsupported calendar action: {action}")


def _execute_data_action(action: str, args: dict[str, Any]) -> Any:
    if action == "collect_data":
        return collect_data(
            user_input=args["user_input"],
            category=args.get("category"),
            limit=int(args.get("limit", 4)),
        )
    if action == "collect_cambridge_writing_questions":
        task_no = _optional_int(args.get("task_no"))
        cambridge_book = _optional_int(args.get("cambridge_book"))
        part_no = _optional_int(args.get("part_no"))
        return collect_cambridge_writing_questions(
            entry_url=args.get("entry_url") or args.get("url") or CAMBRIDGE_CRAWLER_ENTRY_URL,
            max_pages=int(args.get("max_pages", args.get("count", 80)) or 80),
            task_no=task_no,
            cambridge_book=cambridge_book,
            part_no=part_no,
            save_json=bool(args.get("save_json", True)),
            download_images=bool(args.get("download_images", True)),
            verify_ssl=bool(args.get("verify_ssl", True)),
            use_local_entry=bool(args.get("use_local_entry", False)),
            use_env_proxy=bool(args.get("use_env_proxy", True)),
        )
    raise ValueError(f"Unsupported data action: {action}")


def _execute_cambridge_crawler_action(action: str, args: dict[str, Any]) -> Any:
    if action == "crawl_writing_questions":
        task_no = _optional_int(args.get("task_no"))
        return crawl_writing_questions(
            entry_url=args.get("entry_url") or args.get("url") or args.get("source_url") or CAMBRIDGE_CRAWLER_ENTRY_URL,
            max_pages=int(args.get("max_pages", 80) or 80),
            task_no=task_no,
            cambridge_book=_optional_int(args.get("cambridge_book")),
            part_no=_optional_int(args.get("part_no")),
            save_json=bool(args.get("save_json", True)),
            download_images=bool(args.get("download_images", True)),
            verify_ssl=bool(args.get("verify_ssl", True)),
            use_local_entry=bool(args.get("use_local_entry", False)),
            use_env_proxy=bool(args.get("use_env_proxy", True)),
        )
    raise ValueError(f"Unsupported cambridge_crawler action: {action}")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _execute_question_pdf_action(action: str, args: dict[str, Any]) -> Any:
    if action in {"export_question_pdf", "export_pdf"}:
        return export_question_pdf(
            user_input=args.get("user_input", ""),
            count=args.get("count"),
            cambridge_book=args.get("cambridge_book"),
            task_no=args.get("task_no"),
            part_no=args.get("part_no"),
            include_images=args.get("include_images"),
            output_filename=args.get("output_filename"),
        )
    raise ValueError(f"Unsupported question_pdf action: {action}")


def _execute_mistake_action(action: str, args: dict[str, Any]) -> Any:
    if action == "grade_submission":
        return grade_submission(user_input=args["user_input"])
    raise ValueError(f"Unsupported mistake action: {action}")


def _execute_writing_action(action: str, args: dict[str, Any]) -> Any:
    if action == "get_random_task2_prompt":
        return get_random_task2_prompt(essay_type=args.get("essay_type"))
    if action == "prepare_task2_review_context":
        return prepare_task2_review_context(user_input=args["user_input"], topic_id=args["topic_id"])
    if action == "review_task2_submission":
        return review_task2_submission(user_input=args["user_input"], topic_id=args["topic_id"])
    raise ValueError(f"Unsupported writing action: {action}")


@trace_node("tool_executor")
def execute_tools_node(state: AgentState) -> dict[str, Any]:
    """Execute the selected action and write only raw results into state."""

    tool_call = state.get("selected_tool_call")
    if not isinstance(tool_call, dict):
        return {"tool_calls": [], "selected_tool_call": None}

    tool_results: dict[str, Any] = dict(state.get("tool_results", {}))
    previous_error = tool_results.pop("_last_error", None)
    tool_health: dict[str, Any] = dict(state.get("tool_health", {}))
    retrieved_docs: list[RetrievedDoc] = list(state.get("retrieved_docs", []))
    user_profile: UserProfile = dict(state.get("user_profile", {}))
    writing_review_state: dict[str, Any] = dict(state.get("writing_review_state", {}))
    policy = policy_for(tool_call)

    if is_circuit_open(tool_health, tool_call):
        failure = build_failure(
            tool_call=tool_call,
            category="circuit_open",
            message=f"Circuit breaker is open for {tool_id(tool_call)}",
            retryable=False,
            attempt=0,
            attempts=0,
            elapsed_ms=0,
        )
        _append_failure(tool_results, failure)
        tool_results["_last_error"] = failure
        return {"tool_results": tool_results, "tool_health": tool_health}

    outcome: ToolOutcome | None = None
    last_failure: dict[str, Any] | None = None
    for attempt in range(1, policy.max_attempts + 1):
        started_at = time.monotonic()
        try:
            outcome = _execute_with_cache_and_timeout(tool_call, state, policy)
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            last_failure = _classify_result_failure(tool_call, outcome, attempt, policy.max_attempts, elapsed_ms)
            if last_failure is None:
                _apply_tool_outcome(outcome, tool_results, retrieved_docs, user_profile, writing_review_state)
                _record_tool_success(tool_health, tool_id(tool_call))
                _append_trace(tool_results, tool_call, attempt, "cache_hit" if outcome.cache_hit else "success", elapsed_ms, outcome.result_key)
                _append_fallback_result_if_needed(tool_results, previous_error, tool_call, success=True)
                return {
                    "tool_results": tool_results,
                    "tool_health": tool_health,
                    "retrieved_docs": retrieved_docs,
                    "user_profile": user_profile,
                    "writing_review_state": writing_review_state,
                }

            _append_failure(tool_results, last_failure)
            _append_trace(tool_results, tool_call, attempt, last_failure["category"], elapsed_ms, outcome.result_key)
            if should_retry(last_failure, attempt, policy):
                time.sleep(policy.retry_backoff_seconds * attempt)
                continue
            break
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            last_failure = _classify_exception_failure(tool_call, exc, attempt, policy.max_attempts, elapsed_ms)
            _append_failure(tool_results, last_failure)
            _append_trace(tool_results, tool_call, attempt, last_failure["category"], elapsed_ms, result_key_for(tool_call))
            logger.warning("Tool %s failed attempt %s/%s: %s", tool_id(tool_call), attempt, policy.max_attempts, exc)
            if should_retry(last_failure, attempt, policy):
                time.sleep(policy.retry_backoff_seconds * attempt)
                continue
            break

    if last_failure is not None:
        _record_tool_failure(tool_health, tool_id(tool_call), last_failure, policy.circuit_failure_threshold)
        tool_results["_last_error"] = last_failure
        _append_fallback_result_if_needed(tool_results, previous_error, tool_call, success=False)

    return {
        "tool_results": tool_results,
        "tool_health": tool_health,
        "retrieved_docs": retrieved_docs,
        "user_profile": user_profile,
        "writing_review_state": writing_review_state,
    }


def _execute_with_cache_and_timeout(tool_call: ToolCall, state: AgentState, policy: Any) -> ToolOutcome:
    cache_key = cache_key_for(tool_call)
    if is_cacheable(tool_call):
        cached = _TOOL_RESULT_CACHE.get(cache_key)
        if cached and time.time() - cached[0] <= CACHE_TTL_SECONDS:
            cached_outcome = cached[1]
            return ToolOutcome(
                tool_name=cached_outcome.tool_name,
                action=cached_outcome.action,
                result_key=cached_outcome.result_key,
                result=cached_outcome.result,
                cache_hit=True,
            )

    outcome = _invoke_with_timeout(tool_call, state, policy.timeout_seconds)
    if is_cacheable(tool_call):
        _TOOL_RESULT_CACHE[cache_key] = (time.time(), outcome)
    return outcome


def _invoke_with_timeout(tool_call: ToolCall, state: AgentState, timeout_seconds: float) -> ToolOutcome:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-tool")
    future = executor.submit(_invoke_tool_action, tool_call, dict(state.get("user_profile", {})), state)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise ToolTimeoutError(f"Tool timed out after {timeout_seconds:.0f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _invoke_tool_action(tool_call: ToolCall, user_profile: UserProfile, state: AgentState) -> ToolOutcome:
    tool_name = tool_call["tool_name"]
    action = tool_call["action"]
    args = tool_call["args"]

    if tool_name == "rag":
        args = dict(args)
        args.setdefault("user_id", user_profile.get("id") or user_profile.get("user_id") or "demo_user")
        result = _execute_rag_action(action, args)
        return ToolOutcome(tool_name=tool_name, action=action, result_key="rag", result=result)
    if tool_name == "db":
        result = _execute_db_action(action)
        return ToolOutcome(tool_name=tool_name, action=action, result_key=action, result=result)
    if tool_name == "calendar":
        result = _execute_calendar_action(action, args)
        return ToolOutcome(tool_name=tool_name, action=action, result_key=action, result=result)
    if tool_name == "data":
        result = _execute_data_action(action, args)
        return ToolOutcome(tool_name=tool_name, action=action, result_key=action, result=result)
    if tool_name == "cambridge_crawler":
        result = _execute_cambridge_crawler_action(action, args)
        return ToolOutcome(tool_name=tool_name, action=action, result_key=action, result=result)
    if tool_name in {"question_pdf", "export_question_pdf"}:
        result = _execute_question_pdf_action(action, args)
        return ToolOutcome(tool_name=tool_name, action=action, result_key=action, result=result)
    if tool_name == "mistake":
        result = _execute_mistake_action(action, args)
        return ToolOutcome(tool_name=tool_name, action=action, result_key=action, result=result)
    if tool_name == "writing":
        result = _execute_writing_action(action, args)
        return ToolOutcome(tool_name=tool_name, action=action, result_key=action, result=result)
    if tool_name in {"web_search", "web"}:
        result = search(
            query=args["query"],
            search_type=args.get("search_type", "web"),
            max_results=int(args.get("max_results", 5) or 5),
            domains=args.get("domains"),
            recency_days=args.get("recency_days"),
            need_extract=bool(args.get("need_extract", False)),
        )
        return ToolOutcome(tool_name=tool_name, action=action, result_key=result_key_for(tool_call), result=result)
    raise ValueError(f"Unsupported tool name: {tool_name}")


def _apply_tool_outcome(
    outcome: ToolOutcome,
    tool_results: dict[str, Any],
    retrieved_docs: list[RetrievedDoc],
    user_profile: UserProfile,
    writing_review_state: dict[str, Any],
) -> None:
    tool_results[outcome.result_key] = outcome.result
    if outcome.tool_name == "rag" and isinstance(outcome.result, dict):
        retrieved_docs.extend(outcome.result.get("documents", []))
    if outcome.tool_name == "db" and outcome.action == "get_user_profile" and isinstance(outcome.result, dict):
        user_profile.update(outcome.result)
    if outcome.tool_name in {"mistake", "writing"} and isinstance(outcome.result, dict):
        updated_profile = outcome.result.get("updated_profile", {})
        if isinstance(updated_profile, dict):
            user_profile.update(updated_profile)
        if outcome.action == "prepare_task2_review_context" and outcome.result.get("success"):
            prepared_state = outcome.result.get("review_state", {})
            if isinstance(prepared_state, dict):
                writing_review_state.clear()
                writing_review_state.update(prepared_state)


def _classify_result_failure(
    tool_call: ToolCall,
    outcome: ToolOutcome,
    attempt: int,
    attempts: int,
    elapsed_ms: int,
) -> dict[str, Any] | None:
    result = outcome.result
    if not isinstance(result, dict):
        return None
    if result.get("success") is not False and not (result.get("error") and result.get("success") is not True):
        return None
    message = str(result.get("error") or result.get("message") or "Tool returned an unsuccessful result")
    category, retryable = classify_error_text(message, default_retryable=False)
    return build_failure(
        tool_call=tool_call,
        category=category,
        message=message,
        retryable=retryable,
        attempt=attempt,
        attempts=attempts,
        elapsed_ms=elapsed_ms,
        result_key=outcome.result_key,
    )


def _classify_exception_failure(
    tool_call: ToolCall,
    exc: Exception,
    attempt: int,
    attempts: int,
    elapsed_ms: int,
) -> dict[str, Any]:
    if isinstance(exc, ToolTimeoutError):
        category, retryable = "timeout", True
    elif isinstance(exc, (KeyError, ValueError)):
        category, retryable = classify_error_text(str(exc), default_retryable=False)
    else:
        category, retryable = classify_error_text(str(exc), default_retryable=True)
    return build_failure(
        tool_call=tool_call,
        category=category,
        message=str(exc),
        retryable=retryable,
        attempt=attempt,
        attempts=attempts,
        elapsed_ms=elapsed_ms,
    )


def _record_tool_success(tool_health: dict[str, Any], current_tool_id: str) -> None:
    tool_health[current_tool_id] = {
        "status": "healthy",
        "consecutive_failures": 0,
        "last_error_category": "",
        "updated_at_ms": int(time.time() * 1000),
    }


def _record_tool_failure(
    tool_health: dict[str, Any],
    current_tool_id: str,
    failure: dict[str, Any],
    threshold: int,
) -> None:
    previous = tool_health.get(current_tool_id, {})
    previous_count = int(previous.get("consecutive_failures", 0) or 0) if isinstance(previous, dict) else 0
    failure_count = previous_count + 1
    tool_health[current_tool_id] = {
        "status": "open" if failure_count >= threshold else "degraded",
        "consecutive_failures": failure_count,
        "last_error_category": failure.get("category", "unknown"),
        "last_error": failure.get("error", ""),
        "updated_at_ms": int(time.time() * 1000),
    }


def _append_failure(tool_results: dict[str, Any], failure: dict[str, Any]) -> None:
    failures = list(tool_results.get("_tool_failures", []))
    failures.append(failure)
    tool_results["_tool_failures"] = failures[-8:]


def _append_trace(
    tool_results: dict[str, Any],
    tool_call: ToolCall,
    attempt: int,
    status: str,
    elapsed_ms: int,
    result_key: str,
) -> None:
    trace = list(tool_results.get("_tool_policy_trace", []))
    trace.append(
        {
            "tool": tool_id(tool_call),
            "attempt": attempt,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "result_key": result_key,
        }
    )
    tool_results["_tool_policy_trace"] = trace[-12:]


def _append_fallback_result_if_needed(
    tool_results: dict[str, Any],
    previous_error: Any,
    fallback_call: ToolCall,
    *,
    success: bool,
) -> None:
    if not isinstance(previous_error, dict):
        return
    from_tool = f"{previous_error.get('tool_name')}.{previous_error.get('action')}"
    fallbacks = list(tool_results.get("_tool_fallbacks", []))
    fallbacks.append(
        {
            "from_tool": from_tool,
            "to_tool": tool_id(fallback_call),
            "result_key": result_key_for(fallback_call),
            "reason": previous_error.get("category", "unknown"),
            "success": success,
        }
    )
    tool_results["_tool_fallbacks"] = fallbacks[-6:]
