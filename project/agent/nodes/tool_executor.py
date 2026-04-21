"""Tool execution node."""

from __future__ import annotations

import logging
from typing import Any

from project.agent.state import AgentState, RetrievedDoc, ToolCall, UserProfile
from project.tools.calendar_tool import create_study_event, get_schedule
from project.tools.cambridge_crawler_tool import ENTRY_URL as CAMBRIDGE_CRAWLER_ENTRY_URL
from project.tools.cambridge_crawler_tool import crawl_writing_questions
from project.tools.data_tool import collect_data
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


def _to_legacy_documents(retrieved_docs: list[dict[str, Any]]) -> list[RetrievedDoc]:
    """Convert retrieved references to the legacy chunk format used by generator fallback."""

    legacy_docs: list[RetrievedDoc] = []
    for item in retrieved_docs:
        chunks = item.get("chunks", [])
        content = "\n".join(
            chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()
        ).strip()
        if not content:
            continue
        legacy_docs.append(
            {
                "source": str(item.get("source", "")),
                "title": str(item.get("source") or item.get("id") or "Reference"),
                "content": content,
                "score": 1.0,
            }
        )
    return legacy_docs


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
        start_time = args.get("start_time") or args.get("scheduled_at") or "TBD"
        end_time = args.get("end_time") or "TBD"
        return create_study_event(
            title=args.get("title", "IELTS Study Session"),
            start_time=start_time,
            end_time=end_time,
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
    raise ValueError(f"Unsupported data action: {action}")


def _execute_cambridge_crawler_action(action: str, args: dict[str, Any]) -> Any:
    if action == "crawl_writing_questions":
        return crawl_writing_questions(
            entry_url=args.get("entry_url") or args.get("url") or args.get("source_url") or CAMBRIDGE_CRAWLER_ENTRY_URL,
            max_pages=int(args.get("max_pages", 80) or 80),
            save_json=bool(args.get("save_json", True)),
            download_images=bool(args.get("download_images", True)),
            verify_ssl=bool(args.get("verify_ssl", True)),
            use_local_entry=bool(args.get("use_local_entry", False)),
            use_env_proxy=bool(args.get("use_env_proxy", True)),
        )
    raise ValueError(f"Unsupported cambridge_crawler action: {action}")


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
        return prepare_task2_review_context(
            user_input=args["user_input"],
            topic_id=args["topic_id"],
        )
    if action == "review_task2_submission":
        return review_task2_submission(
            user_input=args["user_input"],
            topic_id=args["topic_id"],
        )
    raise ValueError(f"Unsupported writing action: {action}")


def execute_tools_node(
    state: AgentState,
) -> dict[str, dict[str, Any] | list[RetrievedDoc] | UserProfile]:
    """Execute planned tool calls and write results back into state."""

    tool_results: dict[str, Any] = dict(state.get("tool_results", {}))
    retrieved_docs: list[RetrievedDoc] = list(state.get("retrieved_docs", []))
    user_profile: UserProfile = dict(state.get("user_profile", {}))
    writing_review_state: dict[str, Any] = dict(state.get("writing_review_state", {}))

    for tool_call in state.get("tool_calls", []):
        _execute_single_tool(
            tool_call,
            tool_results,
            retrieved_docs,
            user_profile,
            writing_review_state,
            state,
        )

    logger.info("Tool execution finished with %s result bucket(s)", len(tool_results))
    return {
        "tool_results": tool_results,
        "retrieved_docs": retrieved_docs,
        "user_profile": user_profile,
        "writing_review_state": writing_review_state,
    }


def _execute_single_tool(
    tool_call: ToolCall,
    tool_results: dict[str, Any],
    retrieved_docs: list[RetrievedDoc],
    user_profile: UserProfile,
    writing_review_state: dict[str, Any],
    state: AgentState,
) -> None:
    tool_name = tool_call["tool_name"]
    action = tool_call["action"]
    args = tool_call["args"]
    logger.info("Executing tool %s.%s", tool_name, action)

    if tool_name == "rag":
        args = dict(args)
        args.setdefault("user_id", user_profile.get("id") or user_profile.get("user_id") or "demo_user")
        result = _execute_rag_action(action, args)
        tool_results["rag"] = result
        retrieved_docs.extend(result.get("documents", []))
        return

    if tool_name == "db":
        result = _execute_db_action(action)
        tool_results[action] = result
        if action == "get_user_profile":
            user_profile.update(result)
        return

    if tool_name == "calendar":
        tool_results[action] = _execute_calendar_action(action, args)
        return

    if tool_name == "data":
        tool_results[action] = _execute_data_action(action, args)
        return

    if tool_name == "cambridge_crawler":
        tool_results[action] = _execute_cambridge_crawler_action(action, args)
        return

    if tool_name in {"question_pdf", "export_question_pdf"}:
        tool_results[action] = _execute_question_pdf_action(action, args)
        return

    if tool_name == "mistake":
        result = _execute_mistake_action(action, args)
        tool_results[action] = result
        updated_profile = result.get("updated_profile", {})
        if isinstance(updated_profile, dict):
            user_profile.update(updated_profile)
        return

    if tool_name == "writing":
        result = _execute_writing_action(action, args)
        tool_results[action] = result
        updated_profile = result.get("updated_profile", {})
        if isinstance(updated_profile, dict):
            user_profile.update(updated_profile)
        if action == "prepare_task2_review_context" and result.get("success"):
            prepared_state = result.get("review_state", {})
            if isinstance(prepared_state, dict):
                writing_review_state.clear()
                writing_review_state.update(prepared_state)
        return

    if tool_name in {"web_search", "web"}:
        result_key = "search_web" if action in {"search", "search_web"} else action
        tool_results[result_key] = search(
            query=args["query"],
            search_type=args.get("search_type", "web"),
            max_results=int(args.get("max_results", 5) or 5),
            domains=args.get("domains"),
            recency_days=args.get("recency_days"),
            need_extract=bool(args.get("need_extract", False)),
        )
        return

    raise ValueError(f"Unsupported tool name: {tool_name}")
