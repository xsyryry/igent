"""Minimal repository layer for SQLite persistence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
import random
from typing import Any

from project.db.models import get_connection, init_db

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _deserialize_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON field, fallback used.")
        return default


def _row_to_user_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "target_score": row["target_score"],
        "exam_date": row["exam_date"],
        "weak_skills": _deserialize_json(row["weak_skills"], []),
        "preferences": _deserialize_json(row["preferences"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    """Fetch a user profile by ID."""

    init_db()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, name, target_score, exam_date, weak_skills, preferences, created_at, updated_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    return _row_to_user_dict(row)


def upsert_user_profile(
    user_id: str,
    name: str | None = None,
    target_score: str | None = None,
    exam_date: str | None = None,
    weak_skills: list[str] | None = None,
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert or update a user profile."""

    init_db()
    existing_user = get_user_by_id(user_id)
    timestamp = _utc_now()

    merged_user = {
        "id": user_id,
        "name": name if name is not None else (existing_user or {}).get("name", "Demo User"),
        "target_score": (
            target_score
            if target_score is not None
            else (existing_user or {}).get("target_score", "")
        ),
        "exam_date": exam_date if exam_date is not None else (existing_user or {}).get("exam_date", ""),
        "weak_skills": (
            weak_skills if weak_skills is not None else (existing_user or {}).get("weak_skills", [])
        ),
        "preferences": (
            preferences if preferences is not None else (existing_user or {}).get("preferences", {})
        ),
        "created_at": (existing_user or {}).get("created_at", timestamp),
        "updated_at": timestamp,
    }

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO users (
                id, name, target_score, exam_date, weak_skills, preferences, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                target_score = excluded.target_score,
                exam_date = excluded.exam_date,
                weak_skills = excluded.weak_skills,
                preferences = excluded.preferences,
                updated_at = excluded.updated_at
            """,
            (
                merged_user["id"],
                merged_user["name"],
                merged_user["target_score"],
                merged_user["exam_date"],
                _serialize_json(merged_user["weak_skills"]),
                _serialize_json(merged_user["preferences"]),
                merged_user["created_at"],
                merged_user["updated_at"],
            ),
        )
        connection.commit()

    saved_user = get_user_by_id(user_id)
    if saved_user is None:
        raise RuntimeError("Failed to save user profile.")
    return saved_user


def save_memory_event(
    *,
    user_id: str,
    memory_type: str,
    field_name: str | None = None,
    old_value: Any = None,
    new_value: Any = None,
    source_message: str | None = None,
    confidence: float = 0.8,
) -> dict[str, Any]:
    """Persist a lightweight memory change event."""

    init_db()
    timestamp = _utc_now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO memory_events (
                user_id, memory_type, field_name, old_value, new_value, source_message, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                memory_type,
                field_name,
                _serialize_json(old_value),
                _serialize_json(new_value),
                source_message,
                confidence,
                timestamp,
            ),
        )
        connection.commit()
        event_id = cursor.lastrowid

    return {
        "id": event_id,
        "user_id": user_id,
        "memory_type": memory_type,
        "field_name": field_name,
        "old_value": old_value,
        "new_value": new_value,
        "source_message": source_message,
        "confidence": confidence,
        "created_at": timestamp,
    }


def list_memory_events(
    *,
    user_id: str,
    memory_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """List recent memory change events."""

    init_db()
    query = """
        SELECT id, user_id, memory_type, field_name, old_value, new_value, source_message, confidence, created_at
        FROM memory_events
        WHERE user_id = ?
    """
    params: list[Any] = [user_id]
    if memory_type:
        query += " AND memory_type = ?"
        params.append(memory_type)
    query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()

    return [
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "memory_type": row["memory_type"],
            "field_name": row["field_name"],
            "old_value": _deserialize_json(row["old_value"], None),
            "new_value": _deserialize_json(row["new_value"], None),
            "source_message": row["source_message"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_latest_study_plan(user_id: str) -> dict[str, Any] | None:
    """Fetch the most recent study plan for a user."""

    init_db()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, user_id, title, content, start_date, end_date, created_at, updated_at
            FROM study_plans
            WHERE user_id = ?
            ORDER BY datetime(updated_at) DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "title": row["title"],
        "content": _deserialize_json(row["content"], row["content"]),
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def save_study_plan(
    user_id: str,
    title: str,
    content: dict[str, Any] | str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Persist a study plan."""

    init_db()
    timestamp = _utc_now()
    serialized_content = content if isinstance(content, str) else _serialize_json(content)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO study_plans (user_id, title, content, start_date, end_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, title, serialized_content, start_date, end_date, timestamp, timestamp),
        )
        connection.commit()
        plan_id = cursor.lastrowid

    saved_plan = get_latest_study_plan(user_id)
    if saved_plan is None or saved_plan["id"] != plan_id:
        raise RuntimeError("Failed to save study plan.")
    return saved_plan


def get_recent_mistakes(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent mistake records for a user."""

    init_db()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                user_id,
                subject,
                question_type,
                question_source,
                question_text,
                user_answer,
                reference_answer,
                is_correct,
                score,
                error_type,
                wrong_reason,
                correction_note,
                source_of_truth,
                metadata,
                created_at
            FROM mistake_records
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "subject": row["subject"],
            "question_type": row["question_type"],
            "question_source": row["question_source"],
            "question_text": row["question_text"],
            "user_answer": row["user_answer"],
            "reference_answer": row["reference_answer"],
            "is_correct": None if row["is_correct"] is None else bool(row["is_correct"]),
            "score": row["score"],
            "error_type": row["error_type"],
            "wrong_reason": row["wrong_reason"],
            "correction_note": row["correction_note"],
            "source_of_truth": row["source_of_truth"],
            "metadata": _deserialize_json(row["metadata"], {}),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def save_mistake_record(
    user_id: str,
    subject: str,
    question_text: str,
    wrong_reason: str,
    question_source: str | None = None,
    question_type: str | None = None,
    user_answer: str | None = None,
    reference_answer: str | None = None,
    is_correct: bool | None = None,
    score: float | None = None,
    error_type: str | None = None,
    correction_note: str | None = None,
    source_of_truth: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a mistake record."""

    init_db()
    timestamp = _utc_now()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO mistake_records (
                user_id,
                subject,
                question_type,
                question_source,
                question_text,
                user_answer,
                reference_answer,
                is_correct,
                score,
                error_type,
                wrong_reason,
                correction_note,
                source_of_truth,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                subject,
                question_type,
                question_source,
                question_text,
                user_answer,
                reference_answer,
                None if is_correct is None else int(is_correct),
                score,
                error_type,
                wrong_reason,
                correction_note,
                source_of_truth,
                _serialize_json(metadata or {}),
                timestamp,
            ),
        )
        connection.commit()
        record_id = cursor.lastrowid

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                user_id,
                subject,
                question_type,
                question_source,
                question_text,
                user_answer,
                reference_answer,
                is_correct,
                score,
                error_type,
                wrong_reason,
                correction_note,
                source_of_truth,
                metadata,
                created_at
            FROM mistake_records
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    if row is None:
        raise RuntimeError("Failed to save mistake record.")

    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "subject": row["subject"],
        "question_type": row["question_type"],
        "question_source": row["question_source"],
        "question_text": row["question_text"],
        "user_answer": row["user_answer"],
        "reference_answer": row["reference_answer"],
        "is_correct": None if row["is_correct"] is None else bool(row["is_correct"]),
        "score": row["score"],
        "error_type": row["error_type"],
        "wrong_reason": row["wrong_reason"],
        "correction_note": row["correction_note"],
        "source_of_truth": row["source_of_truth"],
        "metadata": _deserialize_json(row["metadata"], {}),
        "created_at": row["created_at"],
    }


def summarize_mistake_patterns(user_id: str, limit: int = 20) -> dict[str, Any]:
    """Aggregate recent mistake records into lightweight user patterns."""

    recent_records = get_recent_mistakes(user_id=user_id, limit=limit)
    subject_counts: dict[str, int] = {}
    error_type_counts: dict[str, int] = {}

    for record in recent_records:
        subject = str(record.get("subject") or "").strip()
        if subject:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        error_type = str(record.get("error_type") or record.get("wrong_reason") or "").strip()
        if error_type:
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

    weak_skills = [
        item[0]
        for item in sorted(subject_counts.items(), key=lambda item: (-item[1], item[0]))
        if item[1] > 0
    ]
    top_error_patterns = [
        {"error_type": item[0], "count": item[1]}
        for item in sorted(error_type_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    focus_recommendations = [
        _build_focus_recommendation(skill_name)
        for skill_name in weak_skills[:3]
    ]

    return {
        "weak_skills": weak_skills,
        "top_error_patterns": top_error_patterns,
        "focus_recommendations": focus_recommendations,
        "recent_mistake_count": len(recent_records),
    }


def build_writing_task2_topic_id(exam_date: str, prompt_text: str, essay_type: str) -> str:
    """Create a stable ID for a Task 2 topic."""

    normalized_prompt = " ".join(str(prompt_text).split()).strip().lower()
    seed = f"{exam_date.strip()}|{essay_type.strip()}|{normalized_prompt}"
    return f"task2_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"


def upsert_writing_task2_topic(
    *,
    exam_date: str,
    prompt_text: str,
    essay_type: str,
    topic_category: str | None = None,
    prompt_translation: str | None = None,
    source_title: str | None = None,
    source_file: str | None = None,
    source_excerpt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert or update an IELTS Writing Task 2 topic."""

    init_db()
    topic_id = build_writing_task2_topic_id(exam_date, prompt_text, essay_type)
    existing = get_writing_task2_topic_by_id(topic_id)
    timestamp = _utc_now()

    merged = {
        "id": topic_id,
        "exam_date": exam_date.strip(),
        "prompt_text": " ".join(prompt_text.split()).strip(),
        "essay_type": essay_type.strip(),
        "topic_category": (
            topic_category.strip()
            if isinstance(topic_category, str) and topic_category.strip()
            else (existing or {}).get("topic_category", "")
        ),
        "prompt_translation": (
            prompt_translation.strip()
            if isinstance(prompt_translation, str) and prompt_translation.strip()
            else (existing or {}).get("prompt_translation", "")
        ),
        "source_title": (
            source_title.strip()
            if isinstance(source_title, str) and source_title.strip()
            else (existing or {}).get("source_title", "")
        ),
        "source_file": (
            source_file.strip()
            if isinstance(source_file, str) and source_file.strip()
            else (existing or {}).get("source_file", "")
        ),
        "source_excerpt": (
            source_excerpt.strip()
            if isinstance(source_excerpt, str) and source_excerpt.strip()
            else (existing or {}).get("source_excerpt", "")
        ),
        "metadata": (
            metadata
            if metadata is not None
            else (existing or {}).get("metadata", {})
        ),
        "created_at": (existing or {}).get("created_at", timestamp),
        "updated_at": timestamp,
    }

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO writing_task2_topics (
                id,
                exam_date,
                prompt_text,
                essay_type,
                topic_category,
                prompt_translation,
                source_title,
                source_file,
                source_excerpt,
                metadata,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                exam_date = excluded.exam_date,
                prompt_text = excluded.prompt_text,
                essay_type = excluded.essay_type,
                topic_category = excluded.topic_category,
                prompt_translation = excluded.prompt_translation,
                source_title = excluded.source_title,
                source_file = excluded.source_file,
                source_excerpt = excluded.source_excerpt,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                merged["id"],
                merged["exam_date"],
                merged["prompt_text"],
                merged["essay_type"],
                merged["topic_category"],
                merged["prompt_translation"],
                merged["source_title"],
                merged["source_file"],
                merged["source_excerpt"],
                _serialize_json(merged["metadata"]),
                merged["created_at"],
                merged["updated_at"],
            ),
        )
        connection.commit()

    saved = get_writing_task2_topic_by_id(topic_id)
    if saved is None:
        raise RuntimeError("Failed to save writing Task 2 topic.")
    return saved


def get_writing_task2_topic_by_id(topic_id: str) -> dict[str, Any] | None:
    """Fetch a Writing Task 2 topic by its stable ID."""

    init_db()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                exam_date,
                prompt_text,
                essay_type,
                topic_category,
                prompt_translation,
                source_title,
                source_file,
                source_excerpt,
                metadata,
                created_at,
                updated_at
            FROM writing_task2_topics
            WHERE id = ?
            """,
            (topic_id,),
        ).fetchone()
    return _row_to_task2_topic(row)


def list_writing_task2_topics(
    *,
    limit: int = 20,
    essay_type: str | None = None,
) -> list[dict[str, Any]]:
    """List stored Writing Task 2 topics."""

    init_db()
    query = """
        SELECT
            id,
            exam_date,
            prompt_text,
            essay_type,
            topic_category,
            prompt_translation,
            source_title,
            source_file,
            source_excerpt,
            metadata,
            created_at,
            updated_at
        FROM writing_task2_topics
    """
    params: list[Any] = []
    if essay_type:
        query += " WHERE essay_type = ?"
        params.append(essay_type)
    query += " ORDER BY datetime(updated_at) DESC, exam_date DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return [_row_to_task2_topic(row) for row in rows if row is not None]


def save_writing_sample(
    *,
    content: str,
    sample_type: str,
    task2_topic_id: str | None = None,
    title: str | None = None,
    source_label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a writing sample such as a reference essay."""

    init_db()
    timestamp = _utc_now()
    normalized_content = " ".join(content.split()).strip()
    sample_id = f"sample_{hashlib.sha1((sample_type + normalized_content).encode('utf-8')).hexdigest()[:16]}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO writing_samples (
                id,
                task2_topic_id,
                sample_type,
                title,
                content,
                source_label,
                metadata,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                task2_topic_id = excluded.task2_topic_id,
                sample_type = excluded.sample_type,
                title = excluded.title,
                content = excluded.content,
                source_label = excluded.source_label,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                sample_id,
                task2_topic_id,
                sample_type,
                title,
                content.strip(),
                source_label,
                _serialize_json(metadata or {}),
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, task2_topic_id, sample_type, title, content, source_label, metadata, created_at, updated_at
            FROM writing_samples
            WHERE id = ?
            """,
            (sample_id,),
        ).fetchone()

    if row is None:
        raise RuntimeError("Failed to save writing sample.")
    return _row_to_writing_sample(row)


def save_writing_scoring_descriptor(
    *,
    writing_type: str,
    criterion_name: str,
    descriptor_text: str,
    band_level: str | None = None,
    source_label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a writing scoring descriptor."""

    init_db()
    timestamp = _utc_now()
    seed = "|".join(
        [
            writing_type.strip(),
            criterion_name.strip(),
            (band_level or "").strip(),
            " ".join(descriptor_text.split()).strip(),
        ]
    )
    descriptor_id = f"descriptor_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO writing_scoring_descriptors (
                id,
                writing_type,
                criterion_name,
                band_level,
                descriptor_text,
                source_label,
                metadata,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                writing_type = excluded.writing_type,
                criterion_name = excluded.criterion_name,
                band_level = excluded.band_level,
                descriptor_text = excluded.descriptor_text,
                source_label = excluded.source_label,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                descriptor_id,
                writing_type.strip(),
                criterion_name.strip(),
                (band_level or "").strip(),
                descriptor_text.strip(),
                source_label,
                _serialize_json(metadata or {}),
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                writing_type,
                criterion_name,
                band_level,
                descriptor_text,
                source_label,
                metadata,
                created_at,
                updated_at
            FROM writing_scoring_descriptors
            WHERE id = ?
            """,
            (descriptor_id,),
        ).fetchone()

    if row is None:
        raise RuntimeError("Failed to save writing scoring descriptor.")
    return _row_to_writing_descriptor(row)


def _row_to_task2_topic(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "exam_date": row["exam_date"],
        "prompt_text": row["prompt_text"],
        "essay_type": row["essay_type"],
        "topic_category": row["topic_category"],
        "prompt_translation": row["prompt_translation"],
        "source_title": row["source_title"],
        "source_file": row["source_file"],
        "source_excerpt": row["source_excerpt"],
        "metadata": _deserialize_json(row["metadata"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_writing_sample(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task2_topic_id": row["task2_topic_id"],
        "sample_type": row["sample_type"],
        "title": row["title"],
        "content": row["content"],
        "source_label": row["source_label"],
        "metadata": _deserialize_json(row["metadata"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_writing_descriptor(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "writing_type": row["writing_type"],
        "criterion_name": row["criterion_name"],
        "band_level": row["band_level"],
        "descriptor_text": row["descriptor_text"],
        "source_label": row["source_label"],
        "metadata": _deserialize_json(row["metadata"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_random_writing_task2_topic(essay_type: str | None = None) -> dict[str, Any] | None:
    """Fetch one random Writing Task 2 topic."""

    init_db()
    query = """
        SELECT
            id,
            exam_date,
            prompt_text,
            essay_type,
            topic_category,
            prompt_translation,
            source_title,
            source_file,
            source_excerpt,
            metadata,
            created_at,
            updated_at
        FROM writing_task2_topics
    """
    params: list[Any] = []
    if essay_type:
        query += " WHERE essay_type = ?"
        params.append(essay_type)
    query += " ORDER BY exam_date DESC, updated_at DESC"

    with get_connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    if not rows:
        return None
    row = random.choice(rows)
    return _row_to_task2_topic(row)


def list_writing_samples(
    *,
    task2_topic_id: str | None = None,
    sample_type: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """List writing samples for a given topic or sample type."""

    init_db()
    query = """
        SELECT id, task2_topic_id, sample_type, title, content, source_label, metadata, created_at, updated_at
        FROM writing_samples
    """
    clauses: list[str] = []
    params: list[Any] = []
    if task2_topic_id:
        clauses.append("task2_topic_id = ?")
        params.append(task2_topic_id)
    if sample_type:
        clauses.append("sample_type = ?")
        params.append(sample_type)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY datetime(updated_at) DESC, id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return [_row_to_writing_sample(row) for row in rows]


def list_writing_scoring_descriptors(
    *,
    writing_type: str = "task2",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List writing scoring descriptors for a writing type."""

    init_db()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                writing_type,
                criterion_name,
                band_level,
                descriptor_text,
                source_label,
                metadata,
                created_at,
                updated_at
            FROM writing_scoring_descriptors
            WHERE writing_type = ?
            ORDER BY criterion_name ASC, band_level ASC, datetime(updated_at) DESC
            LIMIT ?
            """,
            (writing_type, limit),
        ).fetchall()
    return [_row_to_writing_descriptor(row) for row in rows]


def save_writing_submission(
    *,
    user_id: str,
    task2_topic_id: str,
    essay_text: str,
    word_count: int,
    score: float | None,
    feedback_json: dict[str, Any] | None = None,
    source_of_truth: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a reviewed writing submission."""

    init_db()
    timestamp = _utc_now()
    seed = f"{user_id}|{task2_topic_id}|{essay_text.strip()}|{timestamp}"
    submission_id = f"submission_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO writing_submissions (
                id,
                user_id,
                task2_topic_id,
                essay_text,
                word_count,
                score,
                feedback_json,
                source_of_truth,
                metadata,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission_id,
                user_id,
                task2_topic_id,
                essay_text.strip(),
                word_count,
                score,
                _serialize_json(feedback_json or {}),
                source_of_truth,
                _serialize_json(metadata or {}),
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                user_id,
                task2_topic_id,
                essay_text,
                word_count,
                score,
                feedback_json,
                source_of_truth,
                metadata,
                created_at,
                updated_at
            FROM writing_submissions
            WHERE id = ?
            """,
            (submission_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Failed to save writing submission.")
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "task2_topic_id": row["task2_topic_id"],
        "essay_text": row["essay_text"],
        "word_count": row["word_count"],
        "score": row["score"],
        "feedback_json": _deserialize_json(row["feedback_json"], {}),
        "source_of_truth": row["source_of_truth"],
        "metadata": _deserialize_json(row["metadata"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_writing_submissions(
    *,
    user_id: str,
    limit: int = 10,
    essay_type: str | None = None,
    exclude_submission_id: str | None = None,
) -> list[dict[str, Any]]:
    """List recent writing submissions, optionally filtered by essay type."""

    init_db()
    query = """
        SELECT
            submissions.id,
            submissions.user_id,
            submissions.task2_topic_id,
            submissions.essay_text,
            submissions.word_count,
            submissions.score,
            submissions.feedback_json,
            submissions.source_of_truth,
            submissions.metadata,
            submissions.created_at,
            submissions.updated_at,
            topics.essay_type,
            topics.prompt_text,
            topics.exam_date
        FROM writing_submissions AS submissions
        LEFT JOIN writing_task2_topics AS topics
            ON topics.id = submissions.task2_topic_id
        WHERE submissions.user_id = ?
    """
    params: list[Any] = [user_id]
    if essay_type:
        query += " AND topics.essay_type = ?"
        params.append(essay_type)
    if exclude_submission_id:
        query += " AND submissions.id != ?"
        params.append(exclude_submission_id)
    query += " ORDER BY datetime(submissions.created_at) DESC, submissions.id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()

    return [
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "task2_topic_id": row["task2_topic_id"],
            "essay_text": row["essay_text"],
            "word_count": row["word_count"],
            "score": row["score"],
            "feedback_json": _deserialize_json(row["feedback_json"], {}),
            "source_of_truth": row["source_of_truth"],
            "metadata": _deserialize_json(row["metadata"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "essay_type": row["essay_type"],
            "prompt_text": row["prompt_text"],
            "exam_date": row["exam_date"],
        }
        for row in rows
    ]


def _build_focus_recommendation(skill_name: str) -> str:
    skill_name = skill_name.lower()
    if skill_name == "reading":
        return "优先强化阅读定位与逻辑比对，尤其是判断题和同义替换识别。"
    if skill_name == "writing":
        return "优先强化写作任务回应与段落展开，减少观点有但论证不足的问题。"
    if skill_name == "listening":
        return "优先强化听力审题与细节复核，特别是单复数、数字和拼写。"
    if skill_name == "speaking":
        return "优先强化口语延展与流利度，避免回答过短和词汇重复。"
    return f"优先复盘 {skill_name} 相关错题，并针对高频错误做小步练习。"
