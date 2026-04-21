"""Planner node with LLM-first and rule-based fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from project.agent.state import AgentState, ToolCall
from project.llm.client import LLMClient
from project.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from project.tools.mistake_tool import looks_like_submission_request
from project.tools.writing_tool import extract_requested_essay_type, looks_like_writing_submission

logger = logging.getLogger(__name__)


REALTIME_WEB_SEARCH_KEYWORDS = (
    "最新",
    "recent",
    "news",
    "趋势",
    "官网",
    "policy",
    "政策",
    "报名时间",
    "考试时间",
)

RAG_HINT_KEYWORDS = (
    "题库",
    "讲义",
    "评分标准",
    "评分",
    "用户上传",
    "上传资料",
    "资料",
    "错题总结",
    "总结",
    "范文",
    "sample answer",
    "band descriptor",
)

MISTAKE_RAG_KEYWORDS = (
    "错题总结",
    "归纳",
    "总结",
    "讲义",
    "资料",
    "pattern",
    "review notes",
)

ALLOWED_SCOPES = {"writing", "speaking", "reading", "mistakes", None}
ALLOWED_QUERY_MODES = {"mix", "local", "global", None}
DATA_COLLECTION_MAX_ATTEMPTS = 5


def _is_question_data_request(user_input: str) -> bool:
    normalized = user_input.lower()
    return any(token in user_input for token in ("真题", "样题", "练习题", "题目")) or any(
        token in normalized for token in ("practice test", "sample question", "past paper")
    )


def _is_question_pdf_export_request(user_input: str) -> bool:
    normalized = user_input.lower()
    has_export_action = any(token in normalized for token in ("导出", "生成", "整理成", "export"))
    has_pdf = "pdf" in normalized
    has_question_object = any(token in normalized for token in ("剑雅", "剑桥", "writing", "写作题", "作文题", "真题", "本地题库"))
    return has_export_action and has_pdf and has_question_object


def _extract_data_request_details(user_input: str) -> dict[str, Any]:
    normalized = user_input.lower()
    module = None
    if "阅读" in user_input or "reading" in normalized:
        module = "reading"
    elif "听力" in user_input or "listening" in normalized:
        module = "listening"
    elif "写作" in user_input or "writing" in normalized:
        module = "writing"
    elif "口语" in user_input or "speaking" in normalized:
        module = "speaking"

    task_type = None
    task_markers = {
        "true_false_not_given": ("true false not given", "tfng", "判断题"),
        "matching_headings": ("matching headings", "标题匹配"),
        "task1": ("task 1", "小作文"),
        "task2": ("task 2", "大作文"),
        "part1": ("part 1", "part1"),
        "part2": ("part 2", "part2", "cue card"),
        "part3": ("part 3", "part3"),
    }
    for label, markers in task_markers.items():
        if any(marker in normalized or marker in user_input for marker in markers):
            task_type = label
            break

    year_match = re.search(r"(20\d{2}|(?<!\d)\d{2}(?=年))", user_input)
    month_match = re.search(r"(?:20\d{2}\s*年)?\s*(1[0-2]|0?[1-9])\s*月", user_input)
    year = int(year_match.group(1)) if year_match else None
    if year is not None and year < 100:
        year += 2000
    count_match = re.search(r"(?<!\d)(\d{1,2})\s*(?:份|道|个|套|篇)", user_input)
    count = int(count_match.group(1)) if count_match else None
    requested_format = "pdf" if "pdf" in normalized else "txt" if "txt" in normalized or "文本" in user_input else "json" if "json" in normalized else None
    return {
        "module": module,
        "task_type": task_type,
        "year": year,
        "month": int(month_match.group(1)) if month_match else None,
        "format": requested_format,
        "count": count,
    }


def _merge_data_request_details(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {key: current.get(key) or previous.get(key) for key in ("module", "task_type", "year", "month", "format", "count")}


def _missing_question_details(details: dict[str, Any]) -> list[str]:
    missing = []
    if not details.get("module"):
        missing.append("module")
    if not details.get("task_type"):
        missing.append("question_type")
    if not details.get("format"):
        missing.append("format")
    return missing


def _compose_data_collection_query(original_request: str, details: dict[str, Any]) -> str:
    parts = [original_request.strip()]
    if details.get("module"):
        parts.append(f"模块: {details['module']}")
    if details.get("task_type"):
        parts.append(f"题型: {details['task_type']}")
    if details.get("year") and details.get("month"):
        parts.append(f"日期: {details['year']}年{int(details['month']):02d}月")
    elif details.get("year"):
        parts.append(f"日期: {details['year']}年")
    if details.get("count"):
        parts.append(f"数量: {int(details['count'])}份")
    if details.get("format"):
        parts.append(f"格式: {details['format']}")
    return "\n".join(part for part in parts if part)


def _needs_web_search(user_input: str) -> bool:
    normalized = user_input.lower()
    return any(keyword in normalized for keyword in REALTIME_WEB_SEARCH_KEYWORDS)


def _should_use_rag(intent: str, user_input: str) -> bool:
    if _needs_web_search(user_input):
        return False

    normalized = user_input.lower()
    if intent == "knowledge_qa":
        return True

    if intent == "mistake_review":
        return any(keyword in normalized for keyword in MISTAKE_RAG_KEYWORDS)

    return any(keyword in normalized for keyword in RAG_HINT_KEYWORDS)


def _infer_dataset_scope(intent: str, user_input: str) -> str | None:
    normalized = user_input.lower()
    if intent == "mistake_review":
        return "mistakes"
    if "写作" in user_input or "write" in normalized or "评分标准" in user_input:
        return "writing"
    if "口语" in user_input or "speak" in normalized:
        return "speaking"
    if "阅读" in user_input or "read" in normalized or "题库" in user_input:
        return "reading"
    return None


def _infer_query_mode(intent: str, user_input: str) -> str:
    """Choose a minimal retrieval mode for the current request."""

    normalized = user_input.lower()
    summary_keywords = ("总结", "归纳", "概览", "整体", "总览", "pattern", "overview", "summary")
    focused_qa_keywords = ("是什么", "怎么", "如何", "解释", "评分标准", "讲义", "题库", "why", "how", "what")

    if intent == "mistake_review":
        return "global" if any(keyword in normalized or keyword in user_input for keyword in summary_keywords) else "mix"

    if intent == "knowledge_qa":
        if any(keyword in normalized or keyword in user_input for keyword in summary_keywords):
            return "global"
        if any(keyword in normalized or keyword in user_input for keyword in focused_qa_keywords):
            return "local"

    return "mix"


def _build_calendar_call(user_input: str) -> ToolCall:
    create_keywords = ("创建", "安排", "加入", "提醒", "add", "create", "schedule")
    should_create = any(keyword in user_input.lower() for keyword in create_keywords)

    if should_create:
        return {
            "tool_name": "calendar",
            "action": "create_study_event",
            "args": {
                "title": "IELTS Study Session",
                "start_time": "2026-04-20T09:00:00",
                "end_time": "2026-04-20T10:30:00",
                "description": user_input,
            },
        }

    return {
        "tool_name": "calendar",
        "action": "get_schedule",
        "args": {"date": None},
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from raw model output."""

    stripped = text.strip()
    candidates = [stripped]

    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1))

    brace_match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _build_plan_and_tool_calls(
    *,
    intent: str,
    user_input: str,
    use_rag: bool,
    use_db: bool,
    use_calendar: bool,
    use_web_search: bool,
    use_writing: bool,
    use_data_collection: bool,
    dataset_scope: str | None,
    query_mode: str,
    writing_mode: str | None = None,
    plan_override: list[str] | None = None,
) -> tuple[list[str], list[ToolCall]]:
    """Build tool calls from structured planner decisions."""

    plan: list[str] = []
    tool_calls: list[ToolCall] = []

    if use_db:
        plan.append("use_db")
        tool_calls.append({"tool_name": "db", "action": "get_user_profile", "args": {}})
        if intent == "study_plan":
            tool_calls.append({"tool_name": "db", "action": "get_study_plan", "args": {}})
        elif intent == "mistake_review":
            tool_calls.append({"tool_name": "db", "action": "get_mistake_records", "args": {}})

    if use_rag:
        plan.append("use_rag")
        tool_calls.append(
            {
                "tool_name": "rag",
                "action": "retrieve_knowledge",
                "args": {
                    "question": user_input,
                    "dataset_scope": dataset_scope,
                    "query_mode": query_mode,
                    "top_k": 5,
                },
            }
        )

    if use_calendar:
        plan.append("use_calendar")
        tool_calls.append(_build_calendar_call(user_input))

    if use_writing:
        plan.append("use_writing")
        if writing_mode == "review":
            tool_calls.append(
                {
                    "tool_name": "writing",
                    "action": "prepare_task2_review_context",
                    "args": {
                        "user_input": user_input,
                    },
                }
            )
        else:
            requested_essay_type = extract_requested_essay_type(user_input)
            tool_calls.append(
                {
                    "tool_name": "writing",
                    "action": "get_random_task2_prompt",
                    "args": {"essay_type": requested_essay_type},
                }
            )

    if use_web_search:
        plan.append("use_web_search")
        tool_calls.append(
            {
                "tool_name": "web_search",
                "action": "search_web",
                "args": {"query": user_input},
            }
        )

    if use_data_collection:
        plan.append("use_get_data_skill")
        tool_calls.append(
            {
                "tool_name": "data",
                "action": "collect_data",
                "args": {"user_input": user_input},
            }
        )

    default_generation_step = {
        "study_plan": "generate_study_plan_response",
        "knowledge_qa": "generate_knowledge_answer",
        "mistake_review": "generate_review_feedback",
        "writing_practice": "generate_writing_feedback",
        "data_collection": "generate_data_collection_feedback",
        "calendar_action": "generate_calendar_feedback",
        "general_chat": "generate_general_chat_response",
    }[intent]

    if plan_override:
        normalized_override = [step for step in plan_override if isinstance(step, str) and step.strip()]
        if default_generation_step not in normalized_override:
            normalized_override.append(default_generation_step)
        return normalized_override, tool_calls

    if intent == "mistake_review" and "analyze_mistakes" not in plan:
        plan.append("analyze_mistakes")
    plan.append(default_generation_step)
    return plan, tool_calls


def _fallback_plan(intent: str, user_input: str) -> tuple[list[str], list[ToolCall]]:
    """Rule-based planner fallback."""

    if intent == "mistake_review" and looks_like_submission_request(user_input):
        return (
            ["grade_submission", "generate_review_feedback"],
            [
                {
                    "tool_name": "mistake",
                    "action": "grade_submission",
                    "args": {"user_input": user_input},
                }
            ],
        )

    active_topic_id = ""
    if intent == "writing_practice":
        # This branch is handled in plan_node with access to study_context.
        return _build_plan_and_tool_calls(
            intent=intent,
            user_input=user_input,
            use_rag=False,
            use_db=True,
            use_calendar=False,
            use_web_search=False,
            use_writing=True,
            use_data_collection=False,
            dataset_scope=None,
            query_mode="mix",
            writing_mode="prompt" if not active_topic_id else "review",
        )

    use_web_search = _needs_web_search(user_input)
    use_rag = _should_use_rag(intent, user_input)
    use_db = intent in {"study_plan", "knowledge_qa", "mistake_review"}
    use_calendar = intent == "calendar_action"
    use_data_collection = intent == "data_collection"
    dataset_scope = _infer_dataset_scope(intent, user_input) if use_rag else None
    query_mode = _infer_query_mode(intent, user_input) if use_rag else "mix"

    if intent == "general_chat":
        use_db = False
        use_rag = False
        use_calendar = False
        use_data_collection = False

    if intent == "data_collection":
        use_db = False
        use_rag = False
        use_calendar = False
        use_web_search = False
        use_data_collection = True

    if intent == "study_plan":
        use_rag = False
        use_calendar = False

    return _build_plan_and_tool_calls(
        intent=intent,
        user_input=user_input,
        use_rag=use_rag,
        use_db=use_db,
        use_calendar=use_calendar,
        use_web_search=use_web_search,
        use_writing=False,
        use_data_collection=use_data_collection,
        dataset_scope=dataset_scope,
        query_mode=query_mode,
    )


def _sanitize_llm_decision(intent: str, user_input: str, data: dict[str, Any]) -> tuple[list[str], list[ToolCall]]:
    """Normalize LLM planner output before turning it into tool calls."""

    use_rag = bool(data.get("use_rag", False))
    use_db = bool(data.get("use_db", False))
    use_calendar = bool(data.get("use_calendar", False))
    use_web_search = bool(data.get("use_web_search", False))
    use_writing = bool(data.get("use_writing", False))
    use_data_collection = bool(data.get("use_data_collection", False))
    dataset_scope = data.get("dataset_scope")
    if dataset_scope not in ALLOWED_SCOPES:
        dataset_scope = None
    query_mode = data.get("query_mode")
    if query_mode not in ALLOWED_QUERY_MODES:
        query_mode = None

    if intent == "general_chat":
        use_rag = False
        use_db = False
        use_calendar = False
    elif intent == "calendar_action":
        use_calendar = True
        use_rag = False
    elif intent == "study_plan":
        use_db = True
        use_calendar = False
        use_rag = False
    elif intent == "knowledge_qa":
        use_db = bool(data.get("use_db", True))
        if _needs_web_search(user_input):
            use_rag = False
            use_web_search = True
    elif intent == "mistake_review":
        use_db = True
    elif intent == "writing_practice":
        use_db = True
        use_rag = False
        use_calendar = False
        use_web_search = False
        use_writing = True
    elif intent == "data_collection":
        use_rag = False
        use_db = False
        use_calendar = False
        use_web_search = False
        use_writing = False
        use_data_collection = True

    if intent != "data_collection" and _needs_web_search(user_input):
        use_web_search = True
        use_rag = False

    if use_rag and dataset_scope is None:
        dataset_scope = _infer_dataset_scope(intent, user_input)
    if use_rag and query_mode is None:
        query_mode = _infer_query_mode(intent, user_input)
    if not use_rag:
        query_mode = "mix"

    writing_mode = data.get("writing_mode")
    if writing_mode not in {"prompt", "review"}:
        writing_mode = "prompt"

    plan_override = data.get("plan") if isinstance(data.get("plan"), list) else None
    if intent == "data_collection":
        plan_override = None
    return _build_plan_and_tool_calls(
        intent=intent,
        user_input=user_input,
        use_rag=use_rag,
        use_db=use_db,
        use_calendar=use_calendar,
        use_web_search=use_web_search,
        use_writing=use_writing,
        use_data_collection=use_data_collection,
        dataset_scope=dataset_scope,
        query_mode=query_mode,
        writing_mode=writing_mode,
        plan_override=plan_override,
    )


def _try_llm_plan(intent: str, user_input: str) -> tuple[list[str], list[ToolCall]] | None:
    """Ask the configured LLM for tool planning."""

    client = LLMClient.from_config()
    response_text = client.generate_text(
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=build_planner_user_prompt(intent, user_input),
        temperature=0.0,
        max_tokens=300,
    )
    if not response_text:
        return None

    data = _extract_json_object(response_text)
    if data is None:
        logger.warning("LLM planner returned invalid JSON: %s", response_text)
        return None

    return _sanitize_llm_decision(intent, user_input, data)


def plan_node(state: AgentState) -> dict[str, Any]:
    """Create a simple execution plan from the detected intent."""

    intent = state["intent"]
    user_input = state["user_input"]
    if intent == "data_collection":
        plan, tool_calls, details, missing = _plan_data_collection(state)
        logger.info("Plan created with %s tool call(s)", len(tool_calls))
        return {
            "plan": plan,
            "tool_calls": tool_calls,
            "data_collection_details": details,
            "data_collection_missing": missing,
        }

    active_topic_id = str(state.get("study_context", {}).get("active_writing_topic_id", "")).strip()
    if intent == "writing_practice":
        writing_mode = "review" if looks_like_writing_submission(user_input, active_topic_id) else "prompt"
        plan, tool_calls = _build_plan_and_tool_calls(
            intent=intent,
            user_input=user_input,
            use_rag=False,
            use_db=True,
            use_calendar=False,
            use_web_search=False,
            use_writing=True,
            use_data_collection=False,
            dataset_scope=None,
            query_mode="mix",
            writing_mode=writing_mode,
        )
        if writing_mode == "review" and active_topic_id:
            tool_calls[1]["args"]["topic_id"] = active_topic_id
        logger.info("Plan created with %s tool call(s)", len(tool_calls))
        return {"plan": plan, "tool_calls": tool_calls}
    if intent == "mistake_review" and looks_like_submission_request(user_input):
        plan, tool_calls = _fallback_plan(intent, user_input)
        logger.info("Plan created with %s tool call(s)", len(tool_calls))
        return {"plan": plan, "tool_calls": tool_calls}
    llm_result = _try_llm_plan(intent, user_input)
    if llm_result is None:
        plan, tool_calls = _fallback_plan(intent, user_input)
    else:
        plan, tool_calls = llm_result

    logger.info("Plan created with %s tool call(s)", len(tool_calls))
    return {"plan": plan, "tool_calls": tool_calls}


def _plan_data_collection(state: AgentState) -> tuple[list[str], list[ToolCall], dict[str, Any], list[str]]:
    user_input = state["user_input"]
    if _is_question_pdf_export_request(user_input):
        return (
            ["use_export_question_pdf_skill", "generate_data_collection_feedback"],
            [
                {
                    "tool_name": "question_pdf",
                    "action": "export_question_pdf",
                    "args": {"user_input": user_input},
                }
            ],
            {},
            [],
        )

    pending = state.get("study_context", {}).get("data_collection_request")
    pending = pending if isinstance(pending, dict) and pending.get("active") else {}
    attempts = int(pending.get("attempts", 0) or 0)
    original_request = str(pending.get("original_request") or user_input)
    previous_details = pending.get("details", {}) if isinstance(pending.get("details"), dict) else {}
    current_details = _extract_data_request_details(user_input)
    details = _merge_data_request_details(previous_details, current_details)
    query = _compose_data_collection_query(original_request, details)

    if attempts >= DATA_COLLECTION_MAX_ATTEMPTS:
        return ["data_collection_error_report", "generate_data_collection_feedback"], [], details, []

    missing: list[str] = []
    if _is_question_data_request(original_request):
        missing = _missing_question_details(details)
        if missing:
            return ["clarify_data_collection_request", "generate_data_collection_feedback"], [], details, missing

    return (
        ["use_get_data_skill", "generate_data_collection_feedback"],
        [
            {
                "tool_name": "data",
                "action": "collect_data",
                "args": {"user_input": query},
            }
        ],
        details,
        missing,
    )
