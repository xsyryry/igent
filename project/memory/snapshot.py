"""Memory snapshot helpers for prompt injection.

The project keeps long-term memory in SQLite and working memory in graph state.
This module builds the compact view that should enter prompts.
"""

from __future__ import annotations

from typing import Any

from project.db.repository import (
    get_latest_study_plan,
    get_recent_mistakes,
    list_memory_events,
    list_writing_submissions,
    summarize_mistake_patterns,
)
from project.memory.profile_service import ensure_user_profile


def build_memory_snapshot(
    *,
    user_id: str = "demo_user",
    working_memory: dict[str, Any] | None = None,
    recent_limit: int = 5,
) -> dict[str, Any]:
    """Return compact Core/Working/Study/Writing/Error/Progress memory."""

    try:
        profile = ensure_user_profile(user_id)
    except Exception:
        return _fallback_snapshot(user_id, working_memory or {})
    plan = _safe_call(lambda: get_latest_study_plan(user_id), None)
    mistake_summary = _safe_call(
        lambda: summarize_mistake_patterns(user_id=user_id, limit=20),
        {"top_error_patterns": [], "focus_recommendations": []},
    )
    recent_mistakes = _safe_call(lambda: get_recent_mistakes(user_id=user_id, limit=recent_limit), [])
    writing_records = _safe_call(lambda: list_writing_submissions(user_id=user_id, limit=recent_limit), [])
    memory_events = _safe_call(
        lambda: list_memory_events(user_id=user_id, memory_type="core_memory", limit=recent_limit),
        [],
    )

    return {
        "core_memory": _core_memory(profile, memory_events),
        "working_memory": _working_memory(working_memory or {}),
        "study_plan_memory": _study_plan_memory(plan),
        "writing_memory": _writing_memory(writing_records),
        "error_pattern_memory": _error_pattern_memory(mistake_summary, recent_mistakes),
        "progress_memory": _progress_memory(writing_records, mistake_summary),
    }


def _safe_call(fn: Any, fallback: Any) -> Any:
    try:
        return fn()
    except Exception:
        return fallback


def _fallback_snapshot(user_id: str, working_memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "core_memory": {"user_id": user_id, "unavailable": True},
        "working_memory": _working_memory(working_memory),
        "study_plan_memory": {"active": False},
        "writing_memory": {"recent_count": 0, "recent_essays": []},
        "error_pattern_memory": {"top_patterns": [], "focus_recommendations": [], "recent_occurrences": []},
        "progress_memory": {"avg_band": None, "current_focus": None},
    }


def _core_memory(profile: dict[str, Any], memory_events: list[dict[str, Any]]) -> dict[str, Any]:
    preferences = profile.get("preferences", {}) if isinstance(profile.get("preferences"), dict) else {}
    return {
        "user_id": profile.get("id") or profile.get("user_id"),
        "target_band": profile.get("target_score"),
        "exam_type": preferences.get("exam_type", "IELTS Academic"),
        "exam_date": profile.get("exam_date"),
        "preferred_language": preferences.get("preferred_language", "Chinese explanation + English examples"),
        "weekly_available_time": preferences.get("available_hours_per_week"),
        "long_term_weaknesses": profile.get("weak_skills", []),
        "preferred_feedback_style": preferences.get("preferred_feedback_style", "concise and actionable"),
        "updated_at": profile.get("updated_at"),
        "confidence": 0.8,
        "version": preferences.get("profile_version", 1),
        "recent_updates": [
            {
                "field_name": item.get("field_name"),
                "old_value": item.get("old_value"),
                "new_value": item.get("new_value"),
                "created_at": item.get("created_at"),
            }
            for item in memory_events
        ],
    }


def _working_memory(working_memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_task_type": working_memory.get("last_intent") or working_memory.get("current_task_type"),
        "current_essay_id": working_memory.get("active_writing_topic_id"),
        "current_instruction": working_memory.get("last_user_input_summary") or working_memory.get("last_user_input"),
        "current_focus": working_memory.get("active_goal"),
        "retrieval_state": working_memory.get("retrieval_state_summary"),
        "recent_messages_summary": working_memory.get("recent_messages_summary"),
        "short_term_memory": working_memory.get("short_term_memory"),
        "memory_watermark": working_memory.get("memory_watermark"),
        "updated_at": working_memory.get("updated_at"),
    }


def _study_plan_memory(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {"active": False}
    return {
        "active": True,
        "plan_id": plan.get("id"),
        "title": plan.get("title"),
        "content": plan.get("content"),
        "start_date": plan.get("start_date"),
        "end_date": plan.get("end_date"),
        "updated_at": plan.get("updated_at"),
    }


def _writing_memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "recent_count": len(records),
        "recent_essays": [
            {
                "essay_id": item.get("id"),
                "task_type": item.get("essay_type"),
                "predicted_band": item.get("score"),
                "issue_tags": _issue_tags(item.get("feedback_json", {})),
                "created_at": item.get("created_at"),
            }
            for item in records
        ],
    }


def _error_pattern_memory(
    mistake_summary: dict[str, Any],
    recent_mistakes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "top_patterns": mistake_summary.get("top_error_patterns", []),
        "focus_recommendations": mistake_summary.get("focus_recommendations", []),
        "recent_occurrences": [
            {
                "subject": item.get("subject"),
                "issue_type": item.get("error_type"),
                "summary": item.get("wrong_reason"),
                "created_at": item.get("created_at"),
            }
            for item in recent_mistakes
        ],
    }


def _progress_memory(
    writing_records: list[dict[str, Any]],
    mistake_summary: dict[str, Any],
) -> dict[str, Any]:
    scores = [float(item.get("score")) for item in writing_records if item.get("score") is not None]
    latest = scores[0] if scores else None
    previous = scores[1] if len(scores) > 1 else None
    score_delta = None if latest is None or previous is None else round(latest - previous, 1)
    return {
        "time_window": f"last_{len(writing_records)}_writing_submissions",
        "avg_band": round(sum(scores) / len(scores), 1) if scores else None,
        "latest_band": latest,
        "score_delta_vs_previous": score_delta,
        "persistent_issues": mistake_summary.get("top_error_patterns", [])[:3],
        "current_focus": (mistake_summary.get("focus_recommendations") or [None])[0],
    }


def _issue_tags(feedback: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    priority = feedback.get("priority_issue")
    if priority:
        tags.append(str(priority))
    for item in feedback.get("issues", []) if isinstance(feedback.get("issues"), list) else []:
        if item and len(tags) < 4:
            tags.append(str(item)[:80])
    return tags
