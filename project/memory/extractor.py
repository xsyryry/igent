"""Coalesced short-term memory extraction."""

from __future__ import annotations

from datetime import datetime
import re
import threading
from typing import Any

from project.agent.state import Message
from project.db.repository import save_memory_event
from project.memory.profile_service import ensure_user_profile, update_profile_fields

MIN_MESSAGE_DELTA = 4
MAX_SHORT_TERM_ITEMS = 24

_lock = threading.Lock()
_running = False
_dirty = False
_watermark = 0
_pending: tuple[str, list[Message], dict[str, Any]] | None = None


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


def _extract_profile_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
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
        elif key in {"available_hours_per_week", "preferred_focus"}:
            preferences[key] = value
        _safe_save_event(user_id, item)

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
        "value": value,
        "importance": importance,
        "summary": f"{item_type}: {value}",
        "source_text": source_text,
        "created_at": _utc_now(),
    }


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
    match = re.search(r"(?:目标分|目标成绩|目标是|想考到|考到|冲到)\s*(?:是|为)?\s*[:：]?\s*(\d(?:\.\d)?)\s*分?", text)
    return match.group(1) if match else None


def _extract_exam_date(text: str) -> str | None:
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
