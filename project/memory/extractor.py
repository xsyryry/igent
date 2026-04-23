"""Coalesced short-term memory extraction."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import re
import threading
from typing import Any

from project.agent.state import Message
from project.db.repository import save_memory_event
from project.llm.client import LLMClient
from project.memory.profile_service import ensure_user_profile, update_profile_fields

MIN_MESSAGE_DELTA = 4
MAX_SHORT_TERM_ITEMS = 24
logger = logging.getLogger(__name__)
PROFILE_PREFERENCE_FIELDS = {
    "target_school",
    "major",
    "target_program_type",
    "career_goal",
    "budget_constraint",
    "response_preference",
    "language_preference",
    "exam_type",
    "subscore_requirement",
    "available_hours_per_week",
    "available_days_per_week",
    "preferred_focus",
    "current_stage_goal",
    "current_task_mode",
    "retrieval_source_preference",
    "progress_review_cycle",
    "export_format_requirements",
    "export_content_constraints",
    "export_layout_preferences",
    "study_strategy",
    "weak_skill_details",
    "writing_difficulty",
    "listening_difficulty",
    "learning_style",
    "system_workflow_preference",
    "data_source_policy",
    "tool_usage_policy",
}

_lock = threading.Lock()
_running = False
_dirty = False
_watermark = 0
_pending: tuple[str, list[Message], dict[str, Any]] | None = None
_async_results: dict[str, dict[str, Any]] = {}


def request_memory_extraction(
    *,
    user_id: str,
    messages: list[Message],
    study_context: dict[str, Any],
) -> dict[str, Any]:
    """Run one coalesced extraction when enough new messages accumulated."""

    global _running, _dirty, _pending
    with _lock:
        _pending = (user_id, list(messages), dict(study_context))
        if _running:
            _dirty = True
            return {}
        _running = True

    return _run_extraction_loop()


def request_memory_extraction_async(
    *,
    user_id: str,
    messages: list[Message],
    study_context: dict[str, Any],
) -> None:
    """Schedule coalesced extraction in a daemon thread and return immediately."""

    global _running, _dirty, _pending
    with _lock:
        _pending = (user_id, list(messages), dict(study_context))
        if _running:
            _dirty = True
            return
        _running = True

    thread = threading.Thread(
        target=_run_async_worker,
        args=(user_id,),
        name=f"memory-extractor-{user_id}",
        daemon=True,
    )
    thread.start()


def pop_completed_memory_extraction(user_id: str) -> dict[str, Any]:
    """Return the latest background extraction result for this user, if any."""

    with _lock:
        return _async_results.pop(user_id, {})


def _run_async_worker(user_id: str) -> None:
    try:
        result = _run_extraction_loop()
    except Exception as exc:  # pragma: no cover - background best effort
        logger.warning("Background memory extraction failed: %s", exc)
        return
    if not result:
        return
    with _lock:
        existing = _async_results.get(user_id, {})
        _async_results[user_id] = _merge_results(existing, result)


def _run_extraction_loop() -> dict[str, Any]:
    global _running, _dirty
    merged: dict[str, Any] = {}
    try:
        while True:
            current = _pending
            if current is not None:
                result = _extract_if_ready(*current)
                merged = _merge_results(merged, result)

            with _lock:
                if not _dirty:
                    _running = False
                    return merged
                _dirty = False
    except Exception:
        with _lock:
            _running = False
            _dirty = False
        raise


def _extract_if_ready(user_id: str, messages: list[Message], study_context: dict[str, Any]) -> dict[str, Any]:
    global _watermark
    with _lock:
        start = _watermark
    if len(messages) - start < MIN_MESSAGE_DELTA:
        return {}

    new_messages = messages[start:]
    items = _extract_short_term_items(new_messages)
    with _lock:
        _watermark = len(messages)
    if not items:
        return {"memory_watermark": _watermark}

    promoted = _promote_focus_items(user_id, items)
    memory = dict(study_context.get("short_term_memory", {})) if isinstance(study_context.get("short_term_memory"), dict) else {}
    existing = list(memory.get("items", [])) if isinstance(memory.get("items"), list) else []
    memory.update(
        {
            "items": (existing + items)[-MAX_SHORT_TERM_ITEMS:],
            "last_extracted_at": _utc_now(),
            "last_watermark": _watermark,
        }
    )
    return {
        "memory_watermark": _watermark,
        "short_term_memory": memory,
        "profile_updates": promoted,
    }


def _extract_short_term_items(messages: list[Message]) -> list[dict[str, Any]]:
    llm_items = _extract_short_term_items_with_llm(messages)
    if llm_items is not None:
        return _dedupe_items(llm_items)
    return _extract_short_term_items_with_rules(messages)


def _extract_short_term_items_with_rules(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        items.extend(_extract_profile_items(text))
        if not items or items[-1].get("source_text") != text:
            items.append(_item("recent_request", _summarize(text), "normal", text))
    return _dedupe_items(items)


def _extract_short_term_items_with_llm(messages: list[Message]) -> list[dict[str, Any]] | None:
    client = LLMClient.from_memory_config()
    if not client.is_configured:
        return None

    raw = client.generate_text(
        system_prompt=_memory_system_prompt(),
        user_prompt=_memory_user_prompt(messages),
        temperature=0.0,
        max_tokens=700,
    )
    parsed = _parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    raw_items = parsed.get("items", [])
    if not isinstance(raw_items, list):
        return None
    items = [_normalize_llm_item(item) for item in raw_items if isinstance(item, dict)]
    return [item for item in items if item is not None]


def _memory_system_prompt() -> str:
    return (
        "You are a memory extraction module for an IELTS study agent. "
        "Extract only durable, useful user memory from the new conversation messages. "
        "Return JSON only. Do not explain. "
        "Use items with fields: type, value, importance, summary, source_text. "
        "importance must be 'focus' for long-term-worthy facts, otherwise 'normal'. "
        "Allowed focus types: target_school, major, target_program_type, career_goal, "
        "budget_constraint, response_preference, language_preference, exam_type, target_score, "
        "subscore_requirement, exam_date, weak_skill, available_hours_per_week, preferred_focus, "
        "current_stage_goal, current_task_mode, retrieval_source_preference, progress_review_cycle, "
        "export_format_requirements, export_content_constraints, export_layout_preferences, "
        "study_strategy, weak_skill_details, writing_difficulty, listening_difficulty, learning_style, "
        "system_workflow_preference, data_source_policy, tool_usage_policy. "
        "Allowed normal types: preference, learning_goal, constraint, task_state, other. "
        "Use retrievable=false for recent_request shells, internal controls, planner/policy notes, and transient chatter. "
        "Ignore assistant promises, transient small talk, and facts not about the user."
    )


def _memory_user_prompt(messages: list[Message]) -> str:
    compact_messages = [
        {"role": message.get("role", ""), "content": str(message.get("content", ""))[:1000]}
        for message in messages
    ]
    return (
        "New messages since the previous memory watermark:\n"
        f"{json.dumps(compact_messages, ensure_ascii=False)}\n\n"
        "Return exactly this JSON shape:\n"
        '{"items":[{"type":"target_school","value":"CUHK","importance":"focus",'
        '"summary":"User target school is CUHK.","source_text":"...","retrievable":true}]}'
    )


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


def _normalize_llm_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = str(item.get("type") or "other").strip()
    value = item.get("value")
    if value in (None, "", []):
        return None
    importance = str(item.get("importance") or "normal").strip().lower()
    importance = "focus" if importance == "focus" else "normal"
    source_text = str(item.get("source_text") or item.get("evidence") or "").strip()
    summary = str(item.get("summary") or f"{item_type}: {value}").strip()
    cleaned = _clean_focus_item(item_type, value, source_text)
    if cleaned is None and importance == "focus":
        return None
    if cleaned is not None:
        item_type, value = cleaned
    normalized = _item(item_type, value, importance, source_text)
    normalized["summary"] = summary[:240]
    normalized["extractor"] = "llm"
    normalized["retrievable"] = bool(item.get("retrievable", _default_retrievable(item_type, importance)))
    normalized["item_type"] = str(item.get("item_type") or item_type)
    return normalized


def _clean_focus_item(item_type: str, value: Any, source_text: str) -> tuple[str, Any] | None:
    text = f"{value} {source_text}"
    if item_type == "language_preference":
        if "港中文" in text or "香港中文大学" in text:
            return ("target_school", "香港中文大学（CUHK）")
        if not any(token in text for token in ("中文交流", "中文解释", "中文为主", "英文材料", "英文训练", "语言")):
            return None
    if item_type == "budget_constraint":
        if not any(token in text for token in ("预算", "经济", "费用", "性价比", "高价", "低价", "付费", "免费", "低成本", "课程")):
            return None
    if item_type == "response_preference":
        if not any(token in text for token in ("建议", "解释", "风格", "清晰", "具体", "执行", "分步骤", "表格", "清单", "空泛", "模板")):
            return None
    if item_type == "target_program_type" and not any(token in text for token in ("授课型", "研究型")):
        if "计算机" in text and ("项目" in text or "硕士" in text):
            return ("major", "计算机科学相关硕士项目")
        return None
    if item_type in {"current_stage_goal", "current_task_mode", "retrieval_source_preference", "progress_review_cycle"}:
        if not any(token in text for token in ("当前", "这阶段", "这轮", "现在", "计划", "题库", "本地", "按周", "按天", "进展")):
            return None
    if item_type in {"export_format_requirements", "export_content_constraints", "export_layout_preferences"}:
        if not any(token in text for token in ("导出", "PDF", "分页", "序号", "图片", "网页", "导航", "评论区", "格式")):
            return None
    if item_type in {"study_strategy", "weak_skill_details", "writing_difficulty", "listening_difficulty", "learning_style"}:
        if not any(token in text for token in ("策略", "弱项", "口语", "写作", "听力", "阅读", "超时", "细节", "模板", "高频反馈", "小步快跑")):
            return None
    if item_type in {"system_workflow_preference", "data_source_policy", "tool_usage_policy"}:
        if not any(token in text for token in ("系统命令", "本地数据库", "本地库", "联网", "爬取", "普通用户", "调用", "工具")):
            return None
    if item_type == "target_score":
        match = re.search(r"(\d(?:\.\d)?)", str(value))
        return ("target_score", match.group(1)) if match else None
    if item_type == "subscore_requirement":
        match = re.search(r"(\d(?:\.\d)?)", str(value))
        return ("subscore_requirement", f"单项尽量不要低于 {match.group(1)}") if match else None
    return (item_type, value)


def _extract_profile_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    items.extend(_extract_structured_profile_items(text))
    target = _extract_target_score(text)
    if target:
        items.append(_item("target_score", target, "focus", text))
    exam_date = _extract_exam_date(text)
    if exam_date:
        items.append(_item("exam_date", exam_date, "focus", text))
    weak_skills = _extract_weak_skills(text)
    for skill in weak_skills:
        items.append(_item("weak_skill", skill, "focus", text))
    hours = _extract_weekly_hours(text)
    if hours is not None:
        items.append(_item("available_hours_per_week", hours, "focus", text))
    focus = _extract_preferred_focus(text)
    if focus:
        items.append(_item("preferred_focus", focus, "focus", text))
    return items


def _extract_structured_profile_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    lowered = text.lower()
    if "香港中文大学" in text or "港中文" in text or "cuhk" in lowered:
        items.append(_item("target_school", "香港中文大学（CUHK）", "focus", text))
    if "计算机" in text and ("专业" in text or "项目" in text or "本科" in text):
        items.append(_item("major", "计算机相关专业", "focus", text))
    if any(token in text for token in ("人工智能", "数据科学", "软件工程")):
        items.append(_item("major", "人工智能、数据科学、软件工程等相近项目", "focus", text))
    if "授课型" in text:
        items.append(_item("target_program_type", "授课型硕士", "focus", text))
    if any(token in text for token in ("技术研发", "算法工程", "大厂", "科技公司")):
        items.append(_item("career_goal", text, "focus", text))
    if "经济条件" in text or "预算" in text:
        items.append(_item("budget_constraint", "家庭经济条件中等，预算需要控制", "focus", text))
    elif "性价比" in text or "高价课程" in text or "一对一" in text:
        items.append(_item("budget_constraint", "倾向高性价比备考，不接受高价课程", "focus", text))
    elif "免费" in text or "低成本" in text or "低价付费" in text:
        items.append(_item("budget_constraint", text, "focus", text))
    if any(token in text for token in ("清晰", "具体", "直接执行", "分步骤", "不要太空泛")):
        items.append(_item("response_preference", "喜欢清晰、具体、可执行、分步骤的建议", "focus", text))
    elif any(token in text for token in ("表格", "清单")):
        items.append(_item("response_preference", "喜欢表格或清单式输出", "focus", text))
    if ("中文交流" in text or "中文解释" in text or "中文为主" in text or "英文材料" in text) and "港中文" not in text:
        items.append(_item("language_preference", "更习惯中文交流，也接受英文材料训练", "focus", text))
    if "雅思" in text and "托福" not in text:
        items.append(_item("exam_type", "雅思", "focus", text))
    if "托福" in text and "不考虑" not in text:
        items.append(_item("exam_type", "托福", "focus", text))
    subscore_match = re.search(r"单项.*?(\d(?:\.\d)?)", text)
    if subscore_match:
        items.append(_item("subscore_requirement", f"单项尽量不要低于 {subscore_match.group(1)}", "focus", text))
    if "口语" in text and any(token in text for token in ("担心", "弱", "拖后腿", "提上去", "主攻", "先补")):
        items.append(_item("weak_skill", "speaking", "focus", text))
        items.append(_item("weak_skill_details", text, "focus", text))
    if "写作" in text and any(token in text for token in ("担心", "弱", "拖后腿", "展开", "超时", "主攻", "兼顾")):
        items.append(_item("weak_skill", "writing", "focus", text))
        items.append(_item("weak_skill_details", text, "focus", text))
    if "阅读" in text and any(token in text for token in ("弱", "训练", "保持")):
        items.append(_item("weak_skill", "reading", "focus", text))
        items.append(_item("weak_skill_details", text, "focus", text))
    if "听力" in text and any(token in text for token in ("弱", "训练", "保持")):
        items.append(_item("weak_skill", "listening", "focus", text))
        items.append(_item("weak_skill_details", text, "focus", text))
    if any(token in text for token in ("先补口语", "主攻口语", "口语提上去")):
        items.append(_item("preferred_focus", "speaking", "focus", text))
    if "主攻写作" in text:
        items.append(_item("preferred_focus", "writing", "focus", text))
    days_match = re.search(r"每周.*?(\d{1,2})\s*天", text)
    if days_match:
        items.append(_item("available_days_per_week", int(days_match.group(1)), "focus", text))
    if any(token in text for token in ("核心任务", "这阶段", "当前阶段")) and any(token in text for token in ("雅思备考计划", "申请港中文", "英语成绩")):
        items.append(_item("current_stage_goal", text, "focus", text))
    if any(token in text for token in ("先补口语", "同步推进写作", "弱项优先", "四项并行")):
        items.append(_item("study_strategy", text, "focus", text))
    if "每天" in text and "小任务" in text:
        items.append(_item("current_task_mode", "每天都有可直接执行的小任务", "focus", text))
    if any(token in text for token in ("本地题库", "本地库", "本地 JSON", "公开资料", "题库")):
        items.append(_item("retrieval_source_preference", text, "focus", text))
    if "按周" in text and "进展" in text:
        items.append(_item("progress_review_cycle", "按周评估学习进展", "focus", text))
    if "每道题" in text and "序号" in text:
        items.append(_item("export_format_requirements", "每道题都要有明确序号", "focus", text))
    if "每道题单独分页" in text:
        items.append(_item("export_layout_preferences", "每道题单独分页", "focus", text))
    if any(token in text for token in ("明显间隔", "分页", "单独分页", "同页分节")) and "导出" in text:
        items.append(_item("export_layout_preferences", text, "focus", text))
    if any(token in text for token in ("网页导航", "评论区", "保留图片", "默认优先保留图片")):
        items.append(_item("export_content_constraints", text, "focus", text))
    if "限时写作" in text or "超时" in text or "展开论证" in text:
        items.append(_item("writing_difficulty", text, "focus", text))
    if "听力" in text and "细节" in text:
        items.append(_item("listening_difficulty", text, "focus", text))
    if "死记硬背" in text or "模板" in text or "高频反馈" in text or "小步快跑" in text:
        items.append(_item("learning_style", text, "focus", text))
    if "系统命令" in text or "普通用户" in text:
        items.append(_item("system_workflow_preference", text, "focus", text))
    if "本地数据库" in text or "本地库" in text or "本地 JSON" in text:
        items.append(_item("data_source_policy", text, "focus", text))
    if "普通用户" in text and any(token in text for token in ("联网", "临时调用", "爬取", "系统命令")):
        items.append(_item("tool_usage_policy", text, "focus", text))
    return items


def _promote_focus_items(user_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    focus_items = [item for item in items if item.get("importance") == "focus"]
    if not focus_items:
        return {}

    profile = ensure_user_profile(user_id)
    weak_skills = list(profile.get("weak_skills", [])) if isinstance(profile.get("weak_skills"), list) else []
    preferences = dict(profile.get("preferences", {})) if isinstance(profile.get("preferences"), dict) else {}
    target_score = None
    exam_date = None

    for item in focus_items:
        key = str(item.get("type"))
        value = item.get("value")
        if key == "target_score":
            target_score = str(value)
        elif key == "exam_date":
            exam_date = str(value)
        elif key == "weak_skill" and str(value) not in weak_skills:
            weak_skills.append(str(value))
        elif key in PROFILE_PREFERENCE_FIELDS:
            preferences[key] = value

    highlights = list(preferences.get("memory_highlights", [])) if isinstance(preferences.get("memory_highlights"), list) else []
    highlights.extend(str(item.get("summary") or item.get("value")) for item in focus_items)
    preferences["memory_highlights"] = highlights[-12:]

    updated = update_profile_fields(
        user_id,
        target_score=target_score,
        exam_date=exam_date,
        weak_skills=weak_skills if weak_skills != profile.get("weak_skills", []) else None,
        preferences=preferences,
    )
    return {
        "target_score": updated.get("target_score"),
        "exam_date": updated.get("exam_date"),
        "weak_skills": updated.get("weak_skills", []),
        "preferences": updated.get("preferences", {}),
    }


def _item(item_type: str, value: Any, importance: str, source_text: str) -> dict[str, Any]:
    return {
        "type": item_type,
        "item_type": item_type,
        "value": value,
        "importance": importance,
        "summary": f"{item_type}: {value}",
        "source_text": source_text,
        "retrievable": _default_retrievable(item_type, importance),
        "created_at": _utc_now(),
    }


def _default_retrievable(item_type: str, importance: str) -> bool:
    if item_type == "recent_request":
        return False
    if item_type in {"internal_control", "planner_policy", "system_note"}:
        return False
    return importance == "focus" or item_type in {"preference", "learning_goal", "task_state", "constraint"}


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (str(item.get("type")), str(item.get("value")).lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _merge_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if not right:
        return left
    merged = dict(left)
    merged.update(right)
    if left.get("profile_updates") and right.get("profile_updates"):
        profile_updates = dict(left["profile_updates"])
        profile_updates.update(right["profile_updates"])
        merged["profile_updates"] = profile_updates
    return merged


def _extract_target_score(text: str) -> str | None:
    if ("口语" in text or "单项" in text) and "总分" not in text:
        return None
    patterns = (
        r"(?:雅思)?总分目标从\s*\d(?:\.\d)?\s*(?:提高|改|调整)?到\s*(\d(?:\.\d)?)",
        r"(?:雅思)?总分目标\s*(?:至少|是|为|到|能到)?\s*(\d(?:\.\d)?)",
        r"(?:目标分|目标成绩|目标是|想考到|考到|冲到)\s*(?:是|为)?\s*[:：]?\s*(\d(?:\.\d)?)\s*分?",
        r"(?:理想一点|理想状态).{0,12}(?:总分)?.{0,6}(?:到|是)\s*(\d(?:\.\d)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _extract_exam_date(text: str) -> str | None:
    month_change = re.search(r"考试时间从\s*(\d{1,2})\s*月.*?(?:改到|调整到|到)\s*(\d{1,2})\s*月", text)
    if month_change:
        return f"{int(month_change.group(2))} 月"
    initial_month = re.search(r"(?:最初|一开始).*?(\d{1,2})\s*月.*?(?:第一次)?雅思考试", text)
    if initial_month:
        return f"{int(initial_month.group(1))} 月"
    month = re.search(r"(\d{1,2})\s*月.*?(?:参加|考).*?雅思", text)
    if month:
        return f"{int(month.group(1))} 月"
    match = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _extract_weak_skills(text: str) -> list[str]:
    lowered = text.lower()
    skills = []
    for skill, markers in {
        "writing": ("写作", "作文", "writing"),
        "reading": ("阅读", "reading"),
        "listening": ("听力", "listening"),
        "speaking": ("口语", "speaking"),
    }.items():
        if any(marker in lowered or marker in text for marker in markers) and any(token in text for token in ("弱", "差", "提升", "薄弱", "不会")):
            skills.append(skill)
    return skills


def _extract_weekly_hours(text: str) -> int | None:
    match = re.search(r"(?:每周|一周).{0,8}(\d{1,2})\s*(?:小时|h)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_preferred_focus(text: str) -> str | None:
    lowered = text.lower()
    if "重点" not in text and "focus" not in lowered and "优先" not in text:
        return None
    for focus, markers in {
        "writing": ("写作", "作文", "writing"),
        "reading": ("阅读", "reading"),
        "listening": ("听力", "listening"),
        "speaking": ("口语", "speaking"),
    }.items():
        if any(marker in lowered or marker in text for marker in markers):
            return focus
    return None


def _summarize(text: str) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= 120 else normalized[:117] + "..."


def _safe_save_event(user_id: str, item: dict[str, Any]) -> None:
    try:
        save_memory_event(
            user_id=user_id,
            memory_type="short_term_focus",
            field_name=str(item.get("type") or ""),
            new_value=item.get("value"),
            source_message=str(item.get("source_text") or ""),
            confidence=0.75,
        )
    except Exception:
        return


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()
