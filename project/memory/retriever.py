"""Memory retrieval with query routing, pool weighting, and conflict filtering."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import logging
import math
import re
from typing import Any

from project.db.models import get_connection, init_db
from project.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryDoc:
    id: str
    source: str
    pool: str
    text: str
    is_long_term: bool
    field: str = ""
    value: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueryRoute:
    should_search: bool
    query_type: str
    reason: str = ""


QUERY_TYPES = {"profile_query", "update_query", "recent_query", "mixed_query"}
POOL_WEIGHTS = {
    "profile_query": {"user_profile": 1.0, "memory_events": 0.6, "short_term_memory": 0.2},
    "update_query": {"memory_events": 1.0, "user_profile": 0.5, "short_term_memory": 0.2},
    "recent_query": {"short_term_memory": 1.0, "user_profile": 0.4, "memory_events": 0.3},
    "mixed_query": {"user_profile": 0.8, "short_term_memory": 0.8, "memory_events": 0.5},
}

FIELD_ALIASES = {
    "target_school": ("学校", "院校", "志愿", "港中文", "cuhk", "大学"),
    "major": ("专业", "本科方向"),
    "target_program_type": ("授课型", "研究型", "项目类型", "硕士类型"),
    "career_goal": ("工作", "职业", "研发", "算法", "大厂", "发展"),
    "target_score": ("目标分", "雅思目标", "总分", "分数", "target score"),
    "exam_date": ("考试时间", "考试日期", "几月", "日期"),
    "exam_type": ("雅思", "托福", "考试类型"),
    "subscore_requirement": ("单项", "小分", "不要低于"),
    "weak_skills": ("弱项", "薄弱", "拖后腿"),
    "budget_constraint": ("预算", "费用", "经济", "性价比", "付费"),
    "response_preference": ("喜欢", "风格", "建议", "解释", "输出"),
    "language_preference": ("中文", "英文", "交流", "语言"),
    "preferred_focus": ("重点", "优先", "主攻", "先补", "focus"),
    "available_hours_per_week": ("每周", "时间", "小时", "学习几天"),
    "available_days_per_week": ("每周", "几天", "学习天数"),
    "current_stage_goal": ("当前阶段", "核心任务", "这阶段", "申请港中文", "雅思备考计划"),
    "current_task_mode": ("每天", "小任务", "当前任务", "这轮"),
    "retrieval_source_preference": ("题库", "本地题库", "本地库", "本地 JSON", "公开资料"),
    "progress_review_cycle": ("按周", "按天", "进展", "评估"),
    "export_format_requirements": ("序号", "明确序号", "题目之间", "间隔"),
    "export_content_constraints": ("图片", "网页导航", "评论区", "网页噪音", "保留图片"),
    "export_layout_preferences": ("分页", "单独分页", "同页分节", "布局"),
    "study_strategy": ("弱项优先", "四项并行", "先补口语", "同步推进写作", "策略"),
    "weak_skill_details": ("最担心", "没信心", "阅读基础", "口语", "写作"),
    "writing_difficulty": ("写作困难", "展开论证", "限时写作", "超时"),
    "listening_difficulty": ("听力", "抓细节", "细节"),
    "learning_style": ("模板", "死记硬背", "高频反馈", "小步快跑", "备考节奏"),
    "system_workflow_preference": ("系统命令", "普通用户", "改成系统", "工作流"),
    "data_source_policy": ("本地数据库", "本地库", "本地 JSON", "直接查"),
    "tool_usage_policy": ("普通用户", "联网抓取", "临时调用", "系统命令"),
}

FIELD_INTENT_RULES = {
    "target_school": ("学校", "院校", "志愿", "大学", "cuhk", "港中文"),
    "major": ("专业", "本科", "方向"),
    "target_program_type": ("授课型", "研究型", "项目类型", "硕士类型"),
    "career_goal": ("工作", "职业", "研发", "算法", "大厂"),
    "target_score": ("目标分", "总分", "分数", "雅思目标"),
    "exam_date": ("考试时间", "考试日期", "几月"),
    "exam_type": ("雅思", "托福", "考试类型"),
    "subscore_requirement": ("单项", "小分", "不要低于"),
    "weak_skills": ("弱项", "薄弱", "拖后腿", "担心"),
    "budget_constraint": ("预算", "费用", "经济", "性价比", "付费", "免费", "课程"),
    "response_preference": ("偏好", "风格", "喜欢", "建议", "解释", "输出", "表格", "清单"),
    "language_preference": ("中文", "英文", "语言", "交流"),
    "preferred_focus": ("优先", "主攻", "先补", "重点"),
    "available_hours_per_week": ("每周", "几天", "小时", "学习时间"),
    "available_days_per_week": ("每周", "几天", "学习天数"),
    "current_stage_goal": ("核心任务", "当前阶段", "这阶段", "申请港中文", "备考计划"),
    "current_task_mode": ("每天", "小任务", "当前任务", "这轮"),
    "retrieval_source_preference": ("从哪里", "题库", "本地题库", "本地库", "本地 JSON", "公开资料", "读取"),
    "progress_review_cycle": ("按天", "按周", "进展", "评估"),
    "export_format_requirements": ("序号", "明显间隔", "明确序号"),
    "export_content_constraints": ("图片", "网页导航", "评论区", "混进", "保留图片"),
    "export_layout_preferences": ("分页", "单独分页", "同页分节", "导出格式"),
    "study_strategy": ("备考策略", "弱项优先", "四项并行", "先补口语", "同步推进写作"),
    "weak_skill_details": ("最担心", "没信心", "阅读基础", "口语相比", "哪个基础"),
    "writing_difficulty": ("写作", "展开论证", "限时写作", "超时"),
    "listening_difficulty": ("听力", "抓细节"),
    "learning_style": ("模板", "死记硬背", "高频反馈", "小步快跑", "备考节奏"),
    "system_workflow_preference": ("系统命令", "普通用户", "改成系统"),
    "data_source_policy": ("本地数据库", "本地库", "本地 JSON", "直接查"),
    "tool_usage_policy": ("普通用户", "联网", "临时调用", "爬取"),
}

CURRENT_MARKERS = ("现在", "当前", "最新", "后来", "改成", "最终")
HISTORY_MARKERS = ("原来", "最开始", "最初", "之前", "一开始", "以前")
RECENT_MARKERS = ("这轮", "这次", "刚刚", "当前阶段", "当前任务", "本周", "现在导出", "当前在")
MIXED_MARKERS = ("结合", "根据", "按我的", "同时", "综合")


def collect_memory_docs(user_id: str, working_memory: dict[str, Any] | None = None) -> list[MemoryDoc]:
    """Collect searchable docs from user_profile, memory_events, and short_term_memory."""

    init_db()
    docs: list[MemoryDoc] = []
    with get_connection() as connection:
        user_rows = connection.execute(
            "SELECT id, target_score, exam_date, weak_skills, preferences, updated_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchall()
        event_rows = connection.execute(
            """
            SELECT id, memory_type, field_name, old_value, new_value, source_message, confidence, created_at
            FROM memory_events
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()

    for row in user_rows:
        updated_at = row["updated_at"] or ""
        _add_profile_doc(docs, "target_score", row["target_score"], updated_at)
        _add_profile_doc(docs, "exam_date", row["exam_date"], updated_at)
        _add_profile_doc(docs, "weak_skills", _deserialize_value(row["weak_skills"], []), updated_at)
        for field, value in _flatten_preferences(_deserialize_value(row["preferences"], {})):
            _add_profile_doc(docs, field, value, updated_at)

    latest_event_by_field: dict[str, int] = {}
    for row in event_rows:
        if row["memory_type"] != "core_memory":
            continue
        field = str(row["field_name"] or "")
        latest_event_by_field[field] = max(latest_event_by_field.get(field, 0), int(row["id"]))

    for row in event_rows:
        if row["memory_type"] != "core_memory":
            continue
        old_value = _deserialize_value(row["old_value"])
        new_value = _deserialize_value(row["new_value"])
        field = str(row["field_name"] or "")
        is_latest = int(row["id"]) == latest_event_by_field.get(field)
        text = "\n".join(
            [
                "pool=memory_events",
                f"type: {row['memory_type']}",
                f"field: {field}",
                f"old_value: {_serialize_value(old_value)}",
                f"new_value: {_serialize_value(new_value)}",
                f"is_latest: {is_latest}",
                f"created_at: {row['created_at']}",
                f"source: {row['source_message']}",
            ]
        )
        docs.append(
            MemoryDoc(
                id=f"event:{row['id']}",
                source="memory_events",
                pool="memory_events",
                field=field,
                value=_serialize_value(new_value),
                text=text,
                is_long_term=True,
                metadata={
                    "old_value": old_value,
                    "new_value": new_value,
                    "is_latest": is_latest,
                    "created_at": row["created_at"],
                    "memory_type": row["memory_type"],
                },
            )
        )

    memory = (working_memory or {}).get("short_term_memory", {})
    items = memory.get("items", []) if isinstance(memory, dict) else []
    if isinstance(items, list):
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict) or item.get("retrievable") is False:
                continue
            field = str(item.get("type") or item.get("item_type") or "")
            text = "\n".join(f"{key}: {_serialize_value(value)}" for key, value in item.items())
            docs.append(
                MemoryDoc(
                    id=f"short:{index}",
                    source="short_term_memory",
                    pool="short_term_memory",
                    field=field,
                    value=_serialize_value(item.get("value")),
                    text=text,
                    is_long_term=False,
                    metadata=dict(item),
                )
            )
    return [doc for doc in docs if doc.text.strip()]


def retrieve_relevant_memories(
    query: str,
    *,
    user_id: str,
    working_memory: dict[str, Any] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    docs = collect_memory_docs(user_id, working_memory)
    route, ranked = weighted_memory_retrieve(query, docs, k=k)
    return {
        "route": {"should_search": route.should_search, "query_type": route.query_type, "reason": route.reason},
        "items": [
            {
                "id": doc.id,
                "source": doc.source,
                "pool": doc.pool,
                "field": doc.field,
                "value": doc.value,
                "score": round(score, 4),
                "text": doc.text,
                "is_long_term": doc.is_long_term,
                "metadata": doc.metadata or {},
            }
            for doc, score in ranked
        ],
    }


def weighted_memory_retrieve(query: str, docs: list[MemoryDoc], k: int = 5) -> tuple[QueryRoute, list[tuple[MemoryDoc, float]]]:
    route = route_memory_query(query)
    if not route.should_search:
        return route, []

    weights = POOL_WEIGHTS.get(route.query_type, POOL_WEIGHTS["mixed_query"])
    intended_fields = query_field_intents(query)
    scored_all: list[tuple[MemoryDoc, float]] = []
    for pool, weight in weights.items():
        pool_docs = [doc for doc in docs if doc.pool == pool]
        if intended_fields and pool in {"user_profile", "memory_events"}:
            matching_docs = [doc for doc in pool_docs if doc.field in intended_fields]
            if matching_docs:
                pool_docs = matching_docs
        if not pool_docs or weight <= 0:
            continue
        for doc, score in bm25_rank(query, pool_docs, k=max(k * 2, 8)):
            final_score = (score + metadata_boost(query, doc, route)) * weight
            if final_score > 0:
                scored_all.append((doc, final_score))

    scored_all.sort(key=lambda item: item[1], reverse=True)
    return route, resolve_conflicts(query, scored_all, k=k)


def route_memory_query(query: str) -> QueryRoute:
    llm_route = _route_memory_query_with_llm(query)
    if llm_route is not None:
        return llm_route
    return _route_memory_query_with_rules(query)


def query_field_intents(query: str) -> set[str]:
    fields = {
        field
        for field, markers in FIELD_INTENT_RULES.items()
        if any(marker.lower() in query.lower() or marker in query for marker in markers)
    }
    if "target_school" in fields and "target_score" in fields and "分" not in query and "总分" not in query:
        fields.discard("target_score")
    if "target_score" in fields and "学校" not in query and "院校" not in query and "志愿" not in query:
        fields.discard("target_school")
    if (
        "exam_type" in fields
        and ("雅思还是托福" not in query and "考试类型" not in query)
        and any(marker in query for marker in ("目标分", "总分", "分数", "单项", "几月", "考试时间", "考试日期"))
    ):
        fields.discard("exam_type")
    if "writing_difficulty" in fields:
        fields.discard("weak_skills")
        fields.discard("response_preference")
    if "listening_difficulty" in fields:
        fields.discard("weak_skills")
    if "export_layout_preferences" in fields or "export_format_requirements" in fields or "export_content_constraints" in fields:
        fields.discard("response_preference")
    if "retrieval_source_preference" in fields or "data_source_policy" in fields:
        fields.discard("budget_constraint")
    return fields


def bm25_rank(query: str, docs: list[MemoryDoc], k: int = 5) -> list[tuple[MemoryDoc, float]]:
    tokenized_docs = [_char_tokens(doc.text) for doc in docs]
    query_tokens = _char_tokens(query)
    if not query_tokens or not docs:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] += 1

    avgdl = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    total_docs = len(docs)
    scored: list[tuple[MemoryDoc, float]] = []
    for doc, tokens in zip(docs, tokenized_docs):
        token_counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for token, query_weight in Counter(query_tokens).items():
            freq = token_counts.get(token, 0)
            if not freq:
                continue
            idf = math.log(1 + (total_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(avgdl, 1))
            score += idf * (freq * 2.5 / denom) * min(query_weight, 3)
        scored.append((doc, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def metadata_boost(query: str, doc: MemoryDoc, route: QueryRoute) -> float:
    boost = 0.0
    intended_fields = query_field_intents(query)
    query_norm = _normalize_text(query)
    field_norm = _normalize_text(doc.field)
    value_norm = _normalize_text(doc.value)
    if intended_fields:
        if doc.field in intended_fields:
            boost += 3.0
        elif doc.pool in {"user_profile", "memory_events"}:
            boost -= 20.0
    elif field_norm and field_norm in query_norm:
        boost += 0.25
    if value_norm and value_norm in query_norm:
        boost += 0.25
    if doc.field in intended_fields:
        aliases = FIELD_ALIASES.get(doc.field, ())
        if any(alias.lower() in query.lower() or alias in query for alias in aliases):
            boost += 0.4

    metadata = doc.metadata or {}
    if doc.pool == "memory_events":
        if metadata.get("memory_type") != "core_memory":
            boost -= 1.0
        if any(marker in query for marker in CURRENT_MARKERS) and metadata.get("is_latest"):
            boost += 0.45
        if any(marker in query for marker in HISTORY_MARKERS) and metadata.get("old_value") not in (None, "", []):
            boost += 0.45
        if "改" in query or "后来" in query:
            boost += 0.25
    elif doc.pool == "short_term_memory":
        if any(marker in query for marker in RECENT_MARKERS):
            boost += 0.45
        if metadata.get("importance") == "focus":
            boost += 0.15
    elif doc.pool == "user_profile":
        if route.query_type == "profile_query":
            boost += 0.2
        if any(marker in query for marker in CURRENT_MARKERS):
            boost += 0.2
    return boost


def resolve_conflicts(query: str, ranked: list[tuple[MemoryDoc, float]], *, k: int) -> list[tuple[MemoryDoc, float]]:
    selected: list[tuple[MemoryDoc, float]] = []
    seen_text: set[str] = set()
    selected_fields: set[str] = set()
    wants_current = any(marker in query for marker in CURRENT_MARKERS)
    wants_history = any(marker in query for marker in HISTORY_MARKERS)

    for doc, score in ranked:
        text_key = _normalize_text(doc.text)
        if text_key in seen_text:
            continue
        metadata = doc.metadata or {}
        if wants_current and doc.pool == "memory_events" and not metadata.get("is_latest") and doc.field in selected_fields:
            continue
        if wants_history and doc.pool == "user_profile" and doc.field in selected_fields:
            continue
        selected.append((doc, score))
        seen_text.add(text_key)
        if doc.field:
            selected_fields.add(doc.field)
        if len(selected) >= k:
            break
    return selected


def _route_memory_query_with_llm(query: str) -> QueryRoute | None:
    try:
        client = LLMClient.from_memory_config()
        if not client.is_configured:
            return None
        raw = client.generate_text(
            system_prompt=(
                "You are a memory query router. Decide if the user question needs memory search. "
                "If yes, classify it as profile_query, update_query, recent_query, or mixed_query. "
                "Return JSON only: {\"should_search\":true,\"query_type\":\"profile_query\",\"reason\":\"...\"}."
            ),
            user_prompt=f"User question:\n{query}",
            temperature=0.0,
            max_tokens=120,
        )
        parsed = _parse_json_object(raw)
        if not isinstance(parsed, dict):
            return None
        query_type = str(parsed.get("query_type") or "mixed_query")
        if query_type not in QUERY_TYPES:
            query_type = "mixed_query"
        return QueryRoute(
            should_search=bool(parsed.get("should_search", True)),
            query_type=query_type,
            reason=str(parsed.get("reason") or "llm_router"),
        )
    except Exception as exc:
        logger.debug("Memory query LLM router failed, fallback enabled: %s", exc)
        return None


def _route_memory_query_with_rules(query: str) -> QueryRoute:
    normalized = query.lower()
    if not any(marker in query for marker in ("我", "我的", "记得", "现在", "当前", "之前", "这轮", "这次")):
        return QueryRoute(False, "mixed_query", "no_memory_marker")
    if any(marker in query for marker in MIXED_MARKERS):
        return QueryRoute(True, "mixed_query", "mixed_marker")
    if any(marker in query for marker in HISTORY_MARKERS) or "最新" in query or "改" in query or "后来" in query:
        return QueryRoute(True, "update_query", "update_marker")
    if any(marker in query for marker in RECENT_MARKERS):
        return QueryRoute(True, "recent_query", "recent_marker")
    if any(marker in normalized for marker in ("目标", "学校", "专业", "弱项", "预算", "喜欢", "风格", "本科", "雅思")):
        return QueryRoute(True, "profile_query", "profile_marker")
    return QueryRoute(True, "mixed_query", "default")


def _add_profile_doc(docs: list[MemoryDoc], field: str, value: Any, updated_at: str = "") -> None:
    if value in (None, "", [], {}):
        return
    serialized = _serialize_value(value)
    text = f"pool=user_profile\nfield={field}\nvalue={serialized}\nsummary={field}: {serialized}\nupdated_at={updated_at}"
    docs.append(
        MemoryDoc(
            id=f"profile:{field}",
            source="user_profile",
            pool="user_profile",
            field=field,
            value=serialized,
            text=text,
            is_long_term=True,
            metadata={"updated_at": updated_at},
        )
    )


def _flatten_preferences(preferences: Any) -> list[tuple[str, Any]]:
    if not isinstance(preferences, dict):
        return []
    return [(str(key), value) for key, value in preferences.items() if key != "memory_highlights"]


def _serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _deserialize_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_text(text: Any) -> str:
    raw = json.dumps(text, ensure_ascii=False) if not isinstance(text, str) else text
    return re.sub(r"[\s\W_]+", "", raw.lower(), flags=re.UNICODE)


def _char_tokens(text: str) -> list[str]:
    return [ch for ch in _normalize_text(text) if ch]
