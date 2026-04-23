"""Generic single-step ReAct loop nodes."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from project.agent.state import AgentState, ToolCall
from project.agent.nodes.tool_policy import (
    append_tool_history,
    cache_key_for,
    fallback_for,
    is_duplicate_tool_call,
    result_key_for,
    select_fallback_after_failure,
    tool_id,
)
from project.agent.nodes.tracing import trace_node
from project.llm.client import LLMClient
from project.prompts.react_prompt import REACT_CONTROLLER_SYSTEM_PROMPT, build_react_reason_prompt
from project.tools.mistake_tool import looks_like_submission_request
from project.tools.writing_tool import extract_requested_essay_type, looks_like_writing_submission

logger = logging.getLogger(__name__)

DEFAULT_REACT_MAX_STEPS = 4
ABSOLUTE_REACT_RECURSION_LIMIT = 8
SIMPLE_INTENTS = {"general_chat"}
SPECIALIST_TOOL_NAMES = {"writing", "data", "cambridge_crawler", "question_pdf", "export_question_pdf"}
TIME_SENSITIVE_MARKERS = ("latest", "recent", "news", "official", "policy", "最新", "官网", "政策", "报名时间", "考试时间")


@trace_node("react_init")
def react_init_node(state: AgentState) -> dict[str, Any]:
    """Initialize ReAct state for this user turn."""

    intent = state.get("intent", "general_chat")
    react_active = intent not in SIMPLE_INTENTS
    max_steps = _infer_max_steps(str(intent))
    plan = ["react_loop"] if react_active else ["direct_response"]
    return {
        "plan": plan,
        "tool_calls": [],
        "selected_tool_call": None,
        "tool_call_history": list(state.get("tool_call_history", [])),
        "observations": [],
        "last_observation": {},
        "react_active": react_active,
        "react_step": 0,
        "react_max_steps": max_steps if react_active else 0,
        "react_last_thought": "",
        "react_should_continue": react_active,
        "react_finish_reason": None,
        "react_decision": {},
    }


@trace_node("reason")
def reason_node(state: AgentState) -> dict[str, Any]:
    """Make one ReAct reasoning decision."""

    if not state.get("react_active"):
        return {"react_should_continue": False}

    step = int(state.get("react_step", 0) or 0) + 1
    max_steps = int(state.get("react_max_steps", DEFAULT_REACT_MAX_STEPS) or DEFAULT_REACT_MAX_STEPS)
    if step > min(max_steps, ABSOLUTE_REACT_RECURSION_LIMIT):
        return {
            "react_should_continue": False,
            "react_finish_reason": "recursion_limit",
        }
    deterministic_decision = _deterministic_ready_decision(state)
    if deterministic_decision is None:
        deterministic_decision = _deterministic_next_action_decision(state)
    if deterministic_decision is not None:
        decision = deterministic_decision
    elif str(state.get("intent", "")) == "data_collection" and _looks_like_cambridge_writing_request(state["user_input"]):
        decision = _fallback_reason(state)
    else:
        decision = _try_llm_reason(state, step, max_steps) or _fallback_reason(state)
    decision = _sanitize_decision(decision)
    decision = _force_local_pdf_for_cambridge_request(state, decision)
    logger.info("ReAct step %s decision: %s", step, decision.get("action_name") or decision.get("finish_reason"))
    return {
        "react_step": step,
        "react_decision": decision,
        "react_last_thought": str(decision.get("thought") or ""),
        "react_finish_reason": decision.get("finish_reason") if decision.get("finish") else None,
    }


@trace_node("action_selector")
def action_dispatch_node(state: AgentState) -> dict[str, Any]:
    """Select exactly one tool action, including fallback after failures."""

    fallback_action = select_fallback_after_failure(state)
    if fallback_action is not None:
        return _selected_action_update(state, fallback_action, selection_reason="fallback_after_failure")

    decision = state.get("react_decision", {})
    if not isinstance(decision, dict) or decision.get("finish") or not decision.get("need_action"):
        return {"tool_calls": [], "selected_tool_call": None}

    action = _decision_to_tool_call(state, decision)
    if action is None:
        return {
            "tool_calls": [],
            "selected_tool_call": None,
            "react_finish_reason": "unsupported_action",
        }
    if is_duplicate_tool_call(state, action):
        return {
            "tool_calls": [],
            "selected_tool_call": None,
            "react_should_continue": False,
            "react_finish_reason": "duplicate_tool_call",
        }
    return _selected_action_update(state, action, selection_reason="reason_decision")


def _selected_action_update(state: AgentState, action: ToolCall, *, selection_reason: str) -> dict[str, Any]:
    history = append_tool_history(list(state.get("tool_call_history", [])), action)
    return {
        "tool_calls": [action],
        "selected_tool_call": action,
        "tool_call_history": history,
        "action_selection": {
            "tool": tool_id(action),
            "reason": selection_reason,
            "cache_key": cache_key_for(action),
        },
    }


@trace_node("observation_compressor")
def observation_node(state: AgentState) -> dict[str, Any]:
    """Compress the latest raw tool result into a structured observation."""

    action = state.get("selected_tool_call")
    if not isinstance(action, dict):
        observation = {
            "success": True,
            "tool": "",
            "result_key": "",
            "summary": "no action was executed; answer can be finalized",
            "missing": [],
        }
    else:
        key = result_key_for(action)
        tool_results = state.get("tool_results", {})
        result = _fallback_result_for_action(action, tool_results)
        if result is None:
            result = tool_results.get(key)
        if result is None:
            result = tool_results.get("_last_error")
        observation = _summarize_result(action, result)

    observations = list(state.get("observations", []))
    observations.append(
        {
            "step": state.get("react_step", 0),
            "thought": state.get("react_last_thought", ""),
            "action": action,
            "observation": observation,
        }
    )
    return {
        "last_observation": observation,
        "observations": observations[-8:],
    }


@trace_node("react_control")
def react_control_node(state: AgentState) -> dict[str, Any]:
    """Decide whether ReAct should continue or finalize."""

    if not state.get("react_active"):
        return {"react_should_continue": False}

    if state.get("react_should_continue") is False:
        return _finish(str(state.get("react_finish_reason") or "stopped_by_guard"))

    decision = state.get("react_decision", {})
    step = int(state.get("react_step", 0) or 0)
    max_steps = int(state.get("react_max_steps", DEFAULT_REACT_MAX_STEPS) or DEFAULT_REACT_MAX_STEPS)
    finish_reason = str(state.get("react_finish_reason") or "")

    if isinstance(decision, dict) and (decision.get("finish") or decision.get("answer_ready")):
        return _finish("enough_information" if not finish_reason else finish_reason)

    if finish_reason == "unsupported_action":
        return _finish("unsupported_action")

    if step >= max_steps:
        return _finish("max_steps")

    if state.get("tool_results", {}).get("_last_error"):
        fallback_action = select_fallback_after_failure(state)
        if fallback_action is not None:
            return {
                "react_should_continue": True,
                "react_finish_reason": "continue_with_fallback",
            }
        return _finish("tool_failed")

    if _is_repeated_no_gain(state):
        return _finish("repeated_no_gain")

    return {
        "react_should_continue": True,
        "react_finish_reason": "continue_reasoning",
    }


def should_run_react(state: AgentState) -> str:
    """Route after ReAct initialization."""

    return "reason" if state.get("react_active") else "context_builder"


def route_after_react_control(state: AgentState) -> str:
    """Route after one ReAct step."""

    return "reason" if state.get("react_should_continue") else "context_builder"


def route_after_action_selection(state: AgentState) -> str | list[str]:
    """Skip tool execution when the selector decided to stop."""

    tool_call = state.get("selected_tool_call")
    if not isinstance(tool_call, dict):
        return "react_control"
    tool_name = str(tool_call.get("tool_name") or "")
    if tool_name in SPECIALIST_TOOL_NAMES:
        return ["writing_agent", "data_agent"]
    return "tool_executor"


def _finish(reason: str) -> dict[str, Any]:
    return {
        "react_active": False,
        "react_should_continue": False,
        "react_finish_reason": reason,
    }


def _infer_max_steps(intent: str) -> int:
    if intent in {"writing_practice", "data_collection"}:
        return 3
    if intent in {"study_plan", "mistake_review"}:
        return 4
    return DEFAULT_REACT_MAX_STEPS


def _try_llm_reason(state: AgentState, step: int, max_steps: int) -> dict[str, Any] | None:
    client = LLMClient.from_config()
    if not client.is_configured:
        return None

    response = client.generate_text(
        system_prompt=REACT_CONTROLLER_SYSTEM_PROMPT,
        user_prompt=build_react_reason_prompt(
            user_query=state["user_input"],
            intent=str(state.get("intent", "general_chat")),
            known_facts=_known_facts(state),
            last_observation=_observation_to_text(state.get("last_observation", {})),
            action_history=list(state.get("observations", [])),
            step_idx=step,
            max_steps=max_steps,
        ),
        temperature=0.0,
        max_tokens=360,
    )
    return _parse_json_object(response)


def _fallback_reason(state: AgentState) -> dict[str, Any]:
    intent = str(state.get("intent", "general_chat"))
    user_input = state["user_input"]
    results = state.get("tool_results", {})

    if intent == "study_plan":
        if "get_user_profile" not in results:
            return _action("先读取用户画像，确认目标分、弱项和可用时间。", "tool", "db.get_user_profile")
        if "get_study_plan" not in results:
            return _action("用户画像已具备，再读取最近学习计划。", "tool", "db.get_study_plan")
        return _ready("用户画像和学习计划都已具备，可以生成回答。")

    if intent == "knowledge_qa":
        if _needs_web_search(user_input):
            if _fallback_succeeded(results, "web_search.search_web", "rag.retrieve_knowledge"):
                return _ready("外部检索不可用，但本地检索 fallback 已返回可用证据，可以带限制回答。")
            if "search_web" not in results:
                return _action("问题包含时效信息，先查外部最新结果。", "skill", "web_search_skill", {"query": user_input})
            return _ready("外部检索结果已具备，可以回答。")
        if "rag" not in results:
            return _action("需要本地资料支撑，先做一次 RAG 检索。", "skill", "knowledge_retrieval_skill", {"query": user_input})
        return _ready("本地检索结果已具备，可以回答。")

    if intent == "mistake_review":
        if looks_like_submission_request(user_input) and "grade_submission" not in results:
            return _action("用户提交了答案，需要先批改并记录错题。", "skill", "mistake_review_skill", {"mode": "grade"})
        if "get_user_profile" not in results:
            return _action("先读取用户画像，辅助错题复盘。", "tool", "db.get_user_profile")
        if "get_mistake_records" not in results:
            return _action("再读取最近错题记录。", "tool", "db.get_mistake_records")
        return _ready("错题记录已具备，可以复盘。")

    if intent == "writing_practice":
        active_topic_id = str(state.get("study_context", {}).get("active_writing_topic_id", "")).strip()
        if looks_like_writing_submission(user_input, active_topic_id):
            if "review_task2_submission" not in results:
                return _action("检测到作文提交，调用写作批改 skill。", "skill", "writing_review_skill")
            return _ready("写作批改结果已生成，可以反馈。")
        if "get_random_task2_prompt" not in results:
            essay_type = extract_requested_essay_type(user_input)
            return _action("用户需要写作练习题，先从本地题库抽题。", "skill", "writing_prompt_skill", {"essay_type": essay_type})
        return _ready("写作题目已返回，可以反馈。")

    if intent == "data_collection":
        if "export_question_pdf" not in results and (
            _looks_like_question_pdf_export(user_input) or _looks_like_cambridge_writing_request(user_input)
        ):
            return _action("用户要导出题库 PDF，调用 PDF 导出工具。", "tool", "question_pdf.export_question_pdf")
        if "collect_data" not in results and "export_question_pdf" not in results:
            return _action("用户要收集资料，调用资料采集 skill。", "skill", "data_collection_skill")
        return _ready("资料处理结果已具备，可以反馈。")

    if intent == "calendar_action":
        if "create_study_event" not in results and "get_schedule" not in results:
            return _action("用户涉及日程，调用日历工具。", "skill", "calendar_skill")
        return _ready("日历工具结果已具备，可以反馈。")

    return _ready("无需工具，直接回答即可。")


def _deterministic_ready_decision(state: AgentState) -> dict[str, Any] | None:
    """Stop without another LLM call once required evidence is present."""

    intent = str(state.get("intent", "general_chat"))
    results = state.get("tool_results", {})
    user_input = state["user_input"]

    if intent == "study_plan" and "get_user_profile" in results and "get_study_plan" in results:
        return _ready("study profile and plan are already available")

    if intent == "knowledge_qa":
        if _needs_web_search(user_input):
            if _fallback_succeeded(results, "web_search.search_web", "rag.retrieve_knowledge"):
                return _ready("web search failed but local RAG fallback returned evidence")
            if "search_web" in results and not results.get("_last_error"):
                return _ready("web search result is available")
        elif "rag" in results:
            return _ready("local RAG evidence is available")

    if intent == "mistake_review":
        if "grade_submission" in results or ("get_user_profile" in results and "get_mistake_records" in results):
            return _ready("mistake review evidence is available")

    if intent == "writing_practice":
        if "get_random_task2_prompt" in results or "review_task2_submission" in results:
            return _ready("writing result is available")

    if intent == "data_collection":
        if any(key in results for key in ("export_question_pdf", "collect_data", "collect_cambridge_writing_questions")):
            return _ready("data collection result is available")

    return None


def _deterministic_next_action_decision(state: AgentState) -> dict[str, Any] | None:
    """Prefer hard runtime routing for cases where tool order is contractual."""

    intent = str(state.get("intent", "general_chat"))
    results = state.get("tool_results", {})
    user_input = state["user_input"]

    if intent == "knowledge_qa" and _needs_web_search(user_input):
        if "search_web" not in results:
            return _action(
                "time-sensitive question requires external search before local evidence",
                "skill",
                "web_search_skill",
                {"query": user_input},
            )

    if intent == "study_plan":
        if "get_user_profile" not in results:
            return _action("load learner profile before study planning", "tool", "db.get_user_profile")
        if "get_study_plan" not in results:
            return _action("load current study plan after profile", "tool", "db.get_study_plan")

    return None


def _action(
    thought: str,
    action_type: str,
    action_name: str,
    action_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "thought": thought,
        "need_action": True,
        "action_type": action_type,
        "action_name": action_name,
        "action_input": action_input or {},
        "finish": False,
        "finish_reason": "",
        "answer_ready": False,
        "missing_info": [],
    }


def _ready(thought: str) -> dict[str, Any]:
    return {
        "thought": thought,
        "need_action": False,
        "action_type": "",
        "action_name": "",
        "action_input": {},
        "finish": True,
        "finish_reason": "enough_information",
        "answer_ready": True,
        "missing_info": [],
    }


def _fallback_succeeded(results: dict[str, Any], from_tool: str, to_tool: str) -> bool:
    fallbacks = results.get("_tool_fallbacks", [])
    if not isinstance(fallbacks, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("from_tool") == from_tool
        and item.get("to_tool") == to_tool
        and item.get("success")
        for item in fallbacks
    )


def _sanitize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    action_input = decision.get("action_input")
    if not isinstance(action_input, dict):
        action_input = {}
    missing_info = decision.get("missing_info")
    if not isinstance(missing_info, list):
        missing_info = []
    need_action = bool(decision.get("need_action", False))
    finish = bool(decision.get("finish", False))
    if finish:
        need_action = False
    return {
        "thought": str(decision.get("thought") or "本轮决策。").strip()[:240],
        "need_action": need_action,
        "action_type": str(decision.get("action_type") or "").strip(),
        "action_name": str(decision.get("action_name") or "").strip(),
        "action_input": action_input,
        "finish": finish,
        "finish_reason": str(decision.get("finish_reason") or ("enough_information" if finish else "")).strip(),
        "answer_ready": bool(decision.get("answer_ready", finish)),
        "missing_info": [str(item) for item in missing_info[:5]],
    }


def _force_local_pdf_for_cambridge_request(state: AgentState, decision: dict[str, Any]) -> dict[str, Any]:
    if state.get("intent") != "data_collection":
        return decision
    if not _looks_like_cambridge_writing_request(state["user_input"]):
        return decision
    if not decision.get("need_action") or decision.get("finish"):
        return decision
    action_name = str(decision.get("action_name") or "")
    if action_name in {
        "data_collection_skill",
        "data.collect_data",
        "collect_data",
        "cambridge_crawler.crawl_writing_questions",
    }:
        return _action(
            "User wants Cambridge IELTS writing questions; export from local collected bank.",
            "tool",
            "question_pdf.export_question_pdf",
        )
    return decision


def _decision_to_tool_call(state: AgentState, decision: dict[str, Any]) -> ToolCall | None:
    action_name = str(decision.get("action_name") or "").strip()
    action_input = decision.get("action_input", {})
    action_input = action_input if isinstance(action_input, dict) else {}
    if "." in action_name:
        tool_name, action = action_name.split(".", 1)
        return _tool_call_for_direct_action(state, tool_name, action, action_input)
    return _tool_call_for_skill(state, action_name, action_input)


def _tool_call_for_skill(state: AgentState, skill_name: str, action_input: dict[str, Any]) -> ToolCall | None:
    user_input = state["user_input"]
    intent = str(state.get("intent", "general_chat"))
    if skill_name in {"knowledge_retrieval_skill", "rag_retrieval_skill", "retrieve_policy_skill"}:
        return _rag_call(user_input, action_input.get("dataset_scope") or _infer_dataset_scope(intent, user_input))
    if skill_name == "web_search_skill":
        return {"tool_name": "web_search", "action": "search_web", "args": {"query": str(action_input.get("query") or user_input)}}
    if skill_name == "writing_prompt_skill":
        return {
            "tool_name": "writing",
            "action": "get_random_task2_prompt",
            "args": {"essay_type": action_input.get("essay_type") or extract_requested_essay_type(user_input)},
        }
    if skill_name == "writing_review_skill":
        topic_id = str(action_input.get("topic_id") or state.get("study_context", {}).get("active_writing_topic_id", "")).strip()
        return {"tool_name": "writing", "action": "review_task2_submission", "args": {"user_input": user_input, "topic_id": topic_id}}
    if skill_name == "mistake_review_skill":
        if not looks_like_submission_request(user_input):
            if "get_user_profile" not in state.get("tool_results", {}):
                return {"tool_name": "db", "action": "get_user_profile", "args": {}}
            return {"tool_name": "db", "action": "get_mistake_records", "args": {}}
        return {"tool_name": "mistake", "action": "grade_submission", "args": {"user_input": user_input}}
    if skill_name == "data_collection_skill":
        if _looks_like_cambridge_writing_request(user_input):
            return {"tool_name": "question_pdf", "action": "export_question_pdf", "args": {"user_input": user_input}}
        return {"tool_name": "data", "action": "collect_data", "args": {"user_input": user_input}}
    if skill_name == "calendar_skill":
        return _calendar_call(user_input)
    if skill_name == "study_plan_skill":
        if "get_user_profile" in state.get("tool_results", {}):
            return {"tool_name": "db", "action": "get_study_plan", "args": {}}
        return {"tool_name": "db", "action": "get_user_profile", "args": {}}
    return None


def _tool_call_for_direct_action(
    state: AgentState,
    tool_name: str,
    action: str,
    action_input: dict[str, Any],
) -> ToolCall | None:
    user_input = state["user_input"]
    if tool_name == "db":
        return {"tool_name": "db", "action": action, "args": {}}
    if tool_name == "rag":
        return _rag_call(str(action_input.get("question") or action_input.get("query") or user_input), action_input.get("dataset_scope"))
    if tool_name in {"web", "web_search"}:
        return {"tool_name": "web_search", "action": "search_web", "args": {"query": str(action_input.get("query") or user_input)}}
    if tool_name == "calendar":
        return _calendar_call(user_input)
    if tool_name == "data":
        if action == "collect_cambridge_writing_questions":
            return {"tool_name": "data", "action": action, "args": action_input}
        if _looks_like_cambridge_writing_request(user_input):
            return {"tool_name": "question_pdf", "action": "export_question_pdf", "args": {"user_input": user_input}}
        return {"tool_name": "data", "action": "collect_data", "args": {"user_input": user_input}}
    if tool_name == "cambridge_crawler":
        args = _cambridge_crawler_args(user_input)
        args.update(action_input)
        return {"tool_name": "cambridge_crawler", "action": action, "args": args}
    if tool_name in {"question_pdf", "export_question_pdf"}:
        return {"tool_name": "question_pdf", "action": "export_question_pdf", "args": {"user_input": user_input}}
    if tool_name == "mistake":
        return {"tool_name": "mistake", "action": "grade_submission", "args": {"user_input": user_input}}
    if tool_name == "writing":
        if action == "get_random_task2_prompt":
            return {"tool_name": "writing", "action": action, "args": {"essay_type": action_input.get("essay_type")}}
        topic_id = str(action_input.get("topic_id") or state.get("study_context", {}).get("active_writing_topic_id", "")).strip()
        return {"tool_name": "writing", "action": action, "args": {"user_input": user_input, "topic_id": topic_id}}
    return None


def _rag_call(question: str, dataset_scope: Any) -> ToolCall:
    return {
        "tool_name": "rag",
        "action": "retrieve_knowledge",
        "args": {
            "question": question,
            "dataset_scope": dataset_scope,
            "query_mode": "mix",
            "top_k": 5,
        },
    }


def _calendar_call(user_input: str) -> ToolCall:
    if any(token in user_input.lower() for token in ("创建", "安排", "加入", "提醒", "add", "create", "schedule")):
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
    return {"tool_name": "calendar", "action": "get_schedule", "args": {"date": None}}


def _known_facts(state: AgentState) -> dict[str, Any]:
    return {
        "user_profile_keys": sorted(list(state.get("user_profile", {}).keys())),
        "study_context": state.get("study_context", {}),
        "tool_result_keys": sorted(list(state.get("tool_results", {}).keys())),
        "last_observation": state.get("last_observation", {}),
        "react_finish_reason": state.get("react_finish_reason"),
    }


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _needs_web_search(user_input: str) -> bool:
    normalized = user_input.lower()
    return any(marker in normalized for marker in TIME_SENSITIVE_MARKERS)


def _infer_dataset_scope(intent: str, user_input: str) -> str | None:
    normalized = user_input.lower()
    if intent == "mistake_review":
        return "mistakes"
    if "写作" in user_input or "writing" in normalized or "task 2" in normalized:
        return "writing"
    if "口语" in user_input or "speaking" in normalized:
        return "speaking"
    if "阅读" in user_input or "reading" in normalized:
        return "reading"
    return None


def _looks_like_question_pdf_export(user_input: str) -> bool:
    normalized = user_input.lower()
    return "pdf" in normalized and any(token in normalized for token in ("导出", "生成", "export"))


def _looks_like_cambridge_writing_crawl(user_input: str) -> bool:
    normalized = user_input.lower()
    has_cambridge = any(token in normalized for token in ("cambridge", "剑桥", "剑雅"))
    has_writing = any(token in normalized for token in ("writing", "作文", "写作", "task 1", "task 2"))
    has_collection = any(token in normalized for token in ("爬取", "抓取", "收集", "下载", "真题", "题目", "题"))
    return has_cambridge and has_writing and has_collection and "pdf" not in normalized


def _cambridge_crawler_args(user_input: str) -> dict[str, Any]:
    args: dict[str, Any] = {
        "max_pages": _extract_requested_count(user_input) or 80,
        "save_json": True,
        "download_images": True,
    }
    task_no = _extract_requested_task_no(user_input)
    if task_no:
        args["task_no"] = task_no
    return args


def _extract_requested_count(user_input: str) -> int | None:
    match = re.search(r"(?<!\d)(\d{1,3})\s*(?:道|条|篇|个|items?|questions?)?", user_input, flags=re.IGNORECASE)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 200))


def _extract_requested_task_no(user_input: str) -> int | None:
    normalized = user_input.lower()
    if re.search(r"task\s*2", normalized) or "大作文" in user_input:
        return 2
    if re.search(r"task\s*1", normalized) or "小作文" in user_input:
        return 1
    return None


def _looks_like_cambridge_writing_request(user_input: str) -> bool:
    normalized = user_input.lower()
    has_cambridge = any(token in normalized for token in ("cambridge", "剑桥", "剑雅"))
    has_writing = any(token in normalized for token in ("writing", "作文", "写作", "task 1", "task 2"))
    wants_questions = any(token in normalized for token in ("题", "真题", "题目", "pdf", "导出", "生成", "爬取", "抓取", "收集", "下载"))
    return has_cambridge and has_writing and wants_questions


def _fallback_result_for_action(action: dict[str, Any], tool_results: dict[str, Any]) -> dict[str, Any] | None:
    source_tool = f"{action.get('tool_name')}.{action.get('action')}"
    fallbacks = tool_results.get("_tool_fallbacks", [])
    if not isinstance(fallbacks, list):
        return None
    for item in reversed(fallbacks):
        if not isinstance(item, dict):
            continue
        if item.get("from_tool") == source_tool and item.get("success"):
            return {
                "success": True,
                "fallback_used": True,
                "message": (
                    f"primary failed with {item.get('reason', 'unknown')}; "
                    f"fallback {item.get('to_tool')} succeeded"
                ),
                "fallback_result_key": item.get("result_key"),
            }
    return None


def _summarize_result(action: dict[str, Any], result: Any) -> dict[str, Any]:
    label = f"{action.get('tool_name')}.{action.get('action')}"
    if result is None:
        return {"tool": label, "success": False, "summary": "no result returned", "missing": ["useful tool output"]}
    if isinstance(result, dict) and result.get("error"):
        return {
            "tool": label,
            "success": False,
            "category": result.get("category", "execution_error"),
            "summary": str(result.get("error", ""))[:360],
            "missing": ["alternative action or final caveat"],
        }
    if action.get("tool_name") == "rag" and isinstance(result, dict):
        docs = result.get("documents") or result.get("retrieved_docs") or []
        answer = str(result.get("answer") or result.get("message") or "").strip()
        return {
            "tool": label,
            "success": True,
            "result_key": "rag",
            "summary": f"retrieved {len(docs)} reference(s); {answer[:220] or 'no direct answer'}",
            "missing": [] if docs else ["relevant references"],
        }
    if isinstance(result, dict):
        success = result.get("success")
        facts = _compact_dict_facts(result)
        missing = _missing_hint(result)
        return {
            "tool": label,
            "success": success is not False,
            "summary": facts,
            "missing": [] if missing == "none" else [missing],
            "fallback_used": bool(result.get("fallback_used")),
        }
    if isinstance(result, list):
        preview = "; ".join(str(item)[:120] for item in result[:2])
        return {"tool": label, "success": True, "summary": f"returned {len(result)} item(s); {preview}", "missing": []}
    return {"tool": label, "success": True, "summary": str(result)[:260], "missing": []}


def _observation_to_text(observation: dict[str, Any]) -> str:
    return (
        f"Observation: action={observation.get('tool', '')}; "
        f"success={observation.get('success')}; "
        f"summary={observation.get('summary', '')}; "
        f"missing={observation.get('missing', [])}"
    )


def _compact_dict_facts(result: dict[str, Any]) -> str:
    preferred = (
        "message",
        "status",
        "collection_status",
        "matched_count",
        "success_count",
        "saved_count",
        "requested_count",
        "completion_status",
        "overall_band",
        "phase",
    )
    parts: list[str] = []
    for key in preferred:
        if key in result and result.get(key) not in (None, "", []):
            parts.append(f"{key}={str(result.get(key))[:120]}")
    if "topic" in result and isinstance(result["topic"], dict):
        parts.append(f"topic={str(result['topic'].get('prompt_text') or result['topic'].get('id'))[:160]}")
    if "evaluation" in result and isinstance(result["evaluation"], dict):
        parts.append(f"score={result['evaluation'].get('overall_band')}")
        parts.append(f"issue={str(result['evaluation'].get('priority_issue') or '')[:120]}")
    if not parts:
        parts = [f"keys={','.join(list(result.keys())[:8])}"]
    return "; ".join(parts)[:360]


def _missing_hint(result: dict[str, Any]) -> str:
    if result.get("success") is False:
        return str(result.get("message") or result.get("error") or "tool failed")[:180]
    failures = result.get("failures")
    if failures:
        return f"{len(failures)} failure(s) need review"
    return "none"


def _is_repeated_no_gain(state: AgentState) -> bool:
    observations = list(state.get("observations", []))
    if len(observations) < 2:
        return False
    last = observations[-1]
    prev = observations[-2]
    if last.get("action") != prev.get("action"):
        return False
    observation = str(last.get("observation", {}).get("summary", "")).lower()
    return any(marker in observation for marker in ("no result", "no direct answer", "success: false", "missing"))


action_selector_node = action_dispatch_node
observation_compressor_node = observation_node
