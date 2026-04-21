"""SQLite-backed database tool interface."""

from __future__ import annotations

import logging
from typing import Any

from project.agent.state import UserProfile
from project.db.models import init_db
from project.db.repository import (
    get_latest_study_plan,
    get_recent_mistakes,
    list_writing_task2_topics,
    save_mistake_record,
    save_study_plan,
    summarize_mistake_patterns,
)
from project.memory.profile_service import ensure_user_profile, get_weak_skills

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "demo_user"
_DB_AVAILABLE = True


def _mark_db_unavailable(exc: Exception) -> None:
    global _DB_AVAILABLE
    _DB_AVAILABLE = False
    logger.warning("SQLite unavailable, using fallback data: %s", exc)


def _db_unavailable() -> bool:
    return not _DB_AVAILABLE


def _seed_demo_data(user_id: str) -> None:
    """Create minimal demo records on first use."""

    profile = ensure_user_profile(
        user_id=user_id,
        defaults={
            "name": "Demo User",
            "target_score": "6.5",
            "exam_date": "",
            "weak_skills": ["writing", "speaking"],
            "preferences": {
                "preferred_focus": "writing",
                "available_hours_per_week": 8,
            },
        },
    )

    if get_latest_study_plan(user_id) is None:
        save_study_plan(
            user_id=user_id,
            title="Foundation Building Plan",
            content={
                "phase": "foundation_building",
                "weekly_goals": [
                    "完成 2 次写作 Task 2 结构训练",
                    "完成 2 次写作评分标准对照复盘",
                    "完成 2 次口语 Part 2 录音复盘",
                ],
                "daily_tasks": [
                    "背诵并改写 5 个写作常用观点句",
                    "完成 1 段写作主体段展开训练",
                    "整理 3 个近期错题原因",
                ],
            },
            start_date="2026-04-16",
            end_date="2026-04-30",
        )

    if not get_recent_mistakes(user_id, limit=1):
        demo_mistakes = [
            {
                "subject": "writing",
                "question_source": "Task 2 practice",
                "question_text": "Opinion essay paragraph development",
                "wrong_reason": "观点展开不充分",
                "correction_note": "每个主体段补上解释句和具体例子。",
            },
            {
                "subject": "listening",
                "question_source": "Section 1 practice",
                "question_text": "Form completion singular/plural",
                "wrong_reason": "单复数漏听",
                "correction_note": "审题时圈出名词形式，并在听后复核拼写和单复数。",
            },
        ]
        for record in demo_mistakes:
            save_mistake_record(user_id=user_id, **record)

    logger.info("SQLite demo data ready for user %s", profile["id"])


def _profile_to_agent_profile(profile: dict[str, Any]) -> UserProfile:
    preferences = profile.get("preferences", {})
    if _db_unavailable():
        weak_skills = profile.get("weak_skills", ["writing", "reading"])
    else:
        try:
            weak_skills = get_weak_skills(str(profile["id"]))
        except Exception:
            weak_skills = profile.get("weak_skills", ["writing", "reading"])
    preferred_focus = preferences.get("preferred_focus") or (weak_skills[0] if weak_skills else "")

    return {
        "user_id": str(profile["id"]),
        "current_level": str(preferences.get("current_level", "IELTS 5.5-6.0")),
        "target_score": str(profile.get("target_score", "")),
        "preferred_focus": str(preferred_focus),
        "available_hours_per_week": int(preferences.get("available_hours_per_week", 8)),
        "exam_date": str(profile.get("exam_date", "")),
        "weak_skills": weak_skills,
        "preferences": preferences,
        "name": str(profile.get("name", "")),
    }


def get_user_profile(user_id: str = DEFAULT_USER_ID) -> UserProfile:
    """Query the user's learning profile from SQLite."""

    if _db_unavailable():
        return _fallback_profile(user_id)
    try:
        init_db()
        _seed_demo_data(user_id)
        profile = ensure_user_profile(user_id)
        return _profile_to_agent_profile(profile)
    except Exception as exc:
        _mark_db_unavailable(exc)
        return _fallback_profile(user_id)


def get_study_plan(user_id: str = DEFAULT_USER_ID) -> dict[str, Any]:
    """Query the latest study plan from SQLite."""

    try:
        if _db_unavailable():
            latest_plan = None
        else:
            init_db()
            _seed_demo_data(user_id)
            latest_plan = get_latest_study_plan(user_id)
    except Exception as exc:
        _mark_db_unavailable(exc)
        latest_plan = None
    if latest_plan is None:
        return {
            "phase": "not_started",
            "weekly_goals": [],
            "daily_tasks": [],
        }

    content = latest_plan["content"]
    if isinstance(content, dict):
        return content

    return {
        "phase": "not_started",
        "weekly_goals": [],
        "daily_tasks": [],
        "raw_content": content,
    }


def get_mistake_records(
    user_id: str = DEFAULT_USER_ID,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Query recent mistake records from SQLite."""

    try:
        if _db_unavailable():
            records = []
        else:
            init_db()
            _seed_demo_data(user_id)
            records = get_recent_mistakes(user_id=user_id, limit=limit)
    except Exception as exc:
        _mark_db_unavailable(exc)
        records = []
    return [
        {
            "skill": record["subject"],
            "error_type": record.get("error_type") or record["wrong_reason"],
            "count": 1,
            "question_type": record.get("question_type"),
            "question_source": record["question_source"],
            "question_text": record["question_text"],
            "user_answer": record.get("user_answer"),
            "reference_answer": record.get("reference_answer"),
            "is_correct": record.get("is_correct"),
            "score": record.get("score"),
            "correction_note": record["correction_note"],
            "source_of_truth": record.get("source_of_truth"),
            "metadata": record.get("metadata", {}),
            "created_at": record["created_at"],
        }
        for record in records
    ]


def get_mistake_patterns(user_id: str = DEFAULT_USER_ID, limit: int = 20) -> dict[str, Any]:
    """Aggregate recent mistakes into user-level learning patterns."""

    if _db_unavailable():
        return {"weak_skills": [], "top_error_patterns": [], "focus_recommendations": [], "recent_mistake_count": 0}
    try:
        init_db()
        _seed_demo_data(user_id)
        return summarize_mistake_patterns(user_id=user_id, limit=limit)
    except Exception as exc:
        _mark_db_unavailable(exc)
        return {"weak_skills": [], "top_error_patterns": [], "focus_recommendations": [], "recent_mistake_count": 0}


def get_writing_task2_bank(limit: int = 20, essay_type: str | None = None) -> list[dict[str, Any]]:
    """Query stored IELTS Writing Task 2 topics from SQLite."""

    if _db_unavailable():
        return []
    try:
        init_db()
        return list_writing_task2_topics(limit=limit, essay_type=essay_type)
    except Exception as exc:
        _mark_db_unavailable(exc)
        return []


def _fallback_profile(user_id: str) -> UserProfile:
    return _profile_to_agent_profile(
        {
            "id": user_id,
            "name": "Demo User",
            "target_score": "6.5",
            "exam_date": "",
            "weak_skills": ["writing", "reading"],
            "preferences": {"preferred_focus": "writing", "available_hours_per_week": 8},
        }
    )
