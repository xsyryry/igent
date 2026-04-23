"""Long-term learner profile service."""

from __future__ import annotations

from typing import Any

from project.db.repository import (
    get_user_by_id,
    save_memory_event,
    summarize_mistake_patterns,
    upsert_user_profile,
)


def ensure_user_profile(user_id: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ensure a user profile exists and return it."""

    existing_profile = get_user_by_id(user_id)
    if existing_profile is not None:
        return existing_profile

    defaults = defaults or {}
    return upsert_user_profile(
        user_id=user_id,
        name=defaults.get("name", "Demo User"),
        target_score=defaults.get("target_score", "6.5"),
        exam_date=defaults.get("exam_date", ""),
        weak_skills=defaults.get("weak_skills", ["writing", "reading"]),
        preferences=defaults.get(
            "preferences",
            {"preferred_focus": "writing", "available_hours_per_week": 8},
        ),
    )


def get_target_score(user_id: str) -> str:
    """Get the user's target IELTS score."""

    profile = ensure_user_profile(user_id)
    return str(profile.get("target_score", ""))


def get_exam_date(user_id: str) -> str:
    """Get the user's exam date."""

    profile = ensure_user_profile(user_id)
    return str(profile.get("exam_date", ""))


def get_weak_skills(user_id: str) -> list[str]:
    """Get the user's weak skills."""

    profile = ensure_user_profile(user_id)
    weak_skills = profile.get("weak_skills", [])
    return weak_skills if isinstance(weak_skills, list) else []


def update_preferences(user_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
    """Update user preference fields while preserving existing profile data."""

    profile = ensure_user_profile(user_id)
    merged_preferences = dict(profile.get("preferences", {}))
    merged_preferences.update(preferences)
    return upsert_user_profile(
        user_id=user_id,
        name=profile.get("name"),
        target_score=profile.get("target_score"),
        exam_date=profile.get("exam_date"),
        weak_skills=profile.get("weak_skills", []),
        preferences=merged_preferences,
    )


def update_profile_fields(
    user_id: str,
    *,
    target_score: str | None = None,
    exam_date: str | None = None,
    weak_skills: list[str] | None = None,
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update selected profile fields and preserve the rest."""

    profile = ensure_user_profile(user_id)
    next_preferences = dict(profile.get("preferences", {}))
    if preferences:
        next_preferences.update(preferences)

    updated = upsert_user_profile(
        user_id=user_id,
        name=profile.get("name"),
        target_score=target_score if target_score is not None else profile.get("target_score"),
        exam_date=exam_date if exam_date is not None else profile.get("exam_date"),
        weak_skills=weak_skills if weak_skills is not None else profile.get("weak_skills", []),
        preferences=next_preferences,
    )
    _record_core_memory_changes(user_id, profile, updated)
    return updated


def _record_core_memory_changes(user_id: str, before: dict[str, Any], after: dict[str, Any]) -> None:
    for field_name in ("target_score", "exam_date", "weak_skills"):
        if before.get(field_name) == after.get(field_name):
            continue
        _safe_record_change(user_id, field_name, before.get(field_name), after.get(field_name))

    before_preferences = before.get("preferences", {}) if isinstance(before.get("preferences"), dict) else {}
    after_preferences = after.get("preferences", {}) if isinstance(after.get("preferences"), dict) else {}
    for field_name in sorted(set(before_preferences) | set(after_preferences)):
        if field_name == "memory_highlights":
            continue
        old_value = before_preferences.get(field_name)
        new_value = after_preferences.get(field_name)
        if old_value == new_value:
            continue
        _safe_record_change(user_id, field_name, old_value, new_value)


def _safe_record_change(user_id: str, field_name: str, old_value: Any, new_value: Any) -> None:
    try:
        save_memory_event(
            user_id=user_id,
            memory_type="core_memory",
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            confidence=0.85,
        )
    except Exception:
        return


def refresh_profile_from_mistakes(user_id: str, limit: int = 20) -> dict[str, Any]:
    """Update long-term profile fields from recent mistake patterns."""

    profile = ensure_user_profile(user_id)
    summary = summarize_mistake_patterns(user_id=user_id, limit=limit)
    merged_preferences = dict(profile.get("preferences", {}))
    merged_preferences["mistake_patterns"] = summary.get("top_error_patterns", [])
    merged_preferences["focus_recommendations"] = summary.get("focus_recommendations", [])
    merged_preferences["recent_mistake_count"] = summary.get("recent_mistake_count", 0)

    return upsert_user_profile(
        user_id=user_id,
        name=profile.get("name"),
        target_score=profile.get("target_score"),
        exam_date=profile.get("exam_date"),
        weak_skills=summary.get("weak_skills", profile.get("weak_skills", [])),
        preferences=merged_preferences,
    )
