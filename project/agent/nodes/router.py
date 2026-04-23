"""Intent router node with LLM-first and rule-based fallback."""

from __future__ import annotations

import json
import logging
import re

from project.agent.state import AgentState, IntentType
from project.agent.nodes.tracing import trace_node
from project.llm.client import LLMClient
from project.prompts.router_prompt import ALLOWED_INTENTS, ROUTER_SYSTEM_PROMPT, build_router_user_prompt
from project.tools.mistake_tool import looks_like_submission_request
from project.tools.writing_tool import looks_like_writing_practice_request, looks_like_writing_submission

logger = logging.getLogger(__name__)

INTENT_ALIASES = {
    "study": "study_plan",
    "plan": "study_plan",
    "qa": "knowledge_qa",
    "knowledge": "knowledge_qa",
    "mistake": "mistake_review",
    "mistakes": "mistake_review",
    "error_review": "mistake_review",
    "writing": "writing_practice",
    "essay": "writing_practice",
    "calendar": "calendar_action",
    "schedule": "calendar_action",
    "data": "data_collection",
    "collection": "data_collection",
    "collect": "data_collection",
    "download": "data_collection",
    "corpus": "data_collection",
}

DATA_COLLECTION_VERBS = (
    "收集",
    "下载",
    "导出",
    "生成",
    "整理",
    "找一些",
    "找点",
    "补充",
    "准备",
    "爬取",
    "归档",
    "collect",
    "download",
    "gather",
    "prepare",
    "archive",
)

DATA_COLLECTION_OBJECTS = (
    "资料",
    "数据",
    "知识库",
    "资料库",
    "rag",
    "官方样题",
    "样题",
    "真题",
    "剑雅",
    "剑桥雅思",
    "练习题",
    "作文题",
    "写作题",
    "讲义",
    "外刊",
    "语料",
    "corpus",
    "materials",
    "sample questions",
    "practice tests",
)


INTENT_KEYWORDS: dict[IntentType, tuple[str, ...]] = {
    "study_plan": (
        "学习计划",
        "备考计划",
        "学习安排",
        "复习计划",
        "怎么学",
        "计划",
        "schedule",
        "study plan",
        "roadmap",
    ),
    "knowledge_qa": (
        "什么是",
        "怎么做",
        "怎么写",
        "技巧",
        "模板",
        "例句",
        "解释",
        "阅读",
        "听力",
        "口语",
        "写作",
        "ielts",
        "task 1",
        "task 2",
        "评分标准",
        "政策",
        "报名",
        "考试",
    ),
    "mistake_review": (
        "错题",
        "错误",
        "复盘",
        "薄弱",
        "弱项",
        "我总是错",
        "mistake",
        "error",
        "review my mistakes",
    ),
    "writing_practice": (
        "作文练习",
        "写作文",
        "大作文",
        "作文题",
        "一题作文",
        "一道作文",
        "task 2 题目",
        "task 2 练习",
        "writing task 2 prompt",
        "ielts writing task 2 prompt",
        "writing prompt",
        "give me one ielts writing",
        "give me a writing task",
        "give me one writing task",
        "帮我批改作文",
        "批改作文",
        "essay feedback",
        "review my essay",
    ),
    "data_collection": (
        "收集资料",
        "下载资料",
        "补充资料库",
        "准备数据",
        "准备 rag",
        "rag 知识库准备数据",
        "知识库准备数据",
        "为 rag",
        "找一些外刊",
        "外刊文章",
        "雅思讲义资料",
        "官方样题",
        "collect data",
        "download materials",
        "prepare rag data",
        "build corpus",
    ),
    "calendar_action": (
        "日程",
        "安排到日历",
        "提醒",
        "预约",
        "创建日程",
        "calendar",
        "schedule me",
        "event",
    ),
    "general_chat": (),
}


def detect_intent(user_input: str) -> IntentType:
    """Infer the user's intent with simple keyword rules."""

    normalized = user_input.strip().lower()
    if not normalized:
        return "general_chat"

    if _looks_like_data_collection(normalized):
        return "data_collection"

    for intent in ("mistake_review", "writing_practice", "data_collection", "calendar_action", "study_plan", "knowledge_qa"):
        keywords = INTENT_KEYWORDS[intent]
        if any(keyword in normalized for keyword in keywords):
            return intent

    return "general_chat"


def _looks_like_data_collection(normalized_input: str) -> bool:
    """Strict rule for obvious material collection requests."""

    if _looks_like_question_pdf_export(normalized_input):
        return True

    has_action = any(token in normalized_input for token in DATA_COLLECTION_VERBS)
    has_object = any(token in normalized_input for token in DATA_COLLECTION_OBJECTS)
    return has_action and has_object


def _looks_like_question_pdf_export(normalized_input: str) -> bool:
    has_export_action = any(token in normalized_input for token in ("导出", "生成", "整理成", "export"))
    has_pdf = "pdf" in normalized_input
    has_question_object = any(token in normalized_input for token in ("剑雅", "剑桥", "writing", "写作题", "作文题", "真题", "本地题库"))
    return has_export_action and has_pdf and has_question_object


def _try_llm_detect_intent(user_input: str) -> IntentType | None:
    """Ask the configured LLM to classify intent."""

    client = LLMClient.from_config()
    response_text = client.generate_text(
        system_prompt=ROUTER_SYSTEM_PROMPT,
        user_prompt=build_router_user_prompt(user_input),
        temperature=0.0,
        max_tokens=120,
    )
    if not response_text:
        return None

    parsed_intent = _extract_intent(response_text)
    if parsed_intent is None:
        logger.warning("LLM router returned unrecognized content: %s", response_text)
    return parsed_intent


def _extract_intent(text: str) -> IntentType | None:
    """Extract a valid intent label from plain text or JSON."""

    normalized_text = _normalize_llm_router_output(text)
    try:
        data = json.loads(normalized_text)
        if isinstance(data, dict):
            return _normalize_intent_label(data.get("intent"))
    except json.JSONDecodeError:
        pass

    label = _normalize_intent_label(normalized_text)
    if label:
        return label

    for intent in sorted(ALLOWED_INTENTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(intent)}\b", normalized_text):
            return intent
    return None


def _normalize_llm_router_output(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    json_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if json_match:
        return json_match.group(0).strip()
    return stripped.strip().strip("`'\" ,.，。:：;；").lower()


def _normalize_intent_label(value: object) -> IntentType | None:
    if not isinstance(value, str):
        return None
    label = value.strip().lower().strip("`'\" ,.，。:：;；")
    label = re.sub(r"[^a-z_]+", "_", label).strip("_")
    if label in ALLOWED_INTENTS:
        return label  # type: ignore[return-value]
    alias = INTENT_ALIASES.get(label)
    if alias in ALLOWED_INTENTS:
        return alias  # type: ignore[return-value]
    return None


@trace_node("router")
def route_node(state: AgentState) -> dict[str, IntentType]:
    """LangGraph node entry for routing."""

    user_input = state["user_input"]
    active_topic_id = str(state.get("study_context", {}).get("active_writing_topic_id", "")).strip()
    pending_data_request = state.get("study_context", {}).get("data_collection_request")
    if isinstance(pending_data_request, dict) and pending_data_request.get("active"):
        logger.info("Continuing pending data collection request")
        return {"intent": "data_collection"}
    if looks_like_writing_submission(user_input, active_topic_id):
        logger.info("Detected essay-style submission, routing to writing_practice")
        return {"intent": "writing_practice"}
    if looks_like_writing_practice_request(user_input):
        logger.info("Detected writing practice request, routing to writing_practice")
        return {"intent": "writing_practice"}
    if looks_like_submission_request(user_input):
        logger.info("Detected submission-style request, routing to mistake_review")
        return {"intent": "mistake_review"}
    heuristic_intent = detect_intent(user_input)
    if heuristic_intent == "data_collection":
        logger.info("Detected data collection request, routing to data_collection")
        return {"intent": "data_collection"}
    llm_intent = _try_llm_detect_intent(user_input)
    if heuristic_intent != "general_chat" and llm_intent in {None, "general_chat"}:
        intent = heuristic_intent
    else:
        intent = llm_intent or heuristic_intent
    logger.info("Detected intent: %s", intent)
    return {"intent": intent}
