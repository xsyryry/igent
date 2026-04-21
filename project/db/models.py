"""SQLite database bootstrap and connection helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from datetime import datetime


DEFAULT_DB_FILENAME = "ielts_assistant.db"


def get_database_path() -> Path:
    """Resolve the SQLite database path and ensure its parent directory exists."""

    configured_path = os.getenv("IELTS_DB_PATH")
    if configured_path:
        db_path = Path(configured_path).expanduser()
    else:
        db_path = (
            Path(__file__).resolve().parents[1]
            / "storage"
            / "sqlite"
            / DEFAULT_DB_FILENAME
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection with Row access enabled."""

    connection = sqlite3.connect(get_database_path(), timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Initialize the SQLite schema if it does not exist yet."""

    schema_statements = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            target_score TEXT,
            exam_date TEXT,
            weak_skills TEXT,
            preferences TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            start_date TEXT,
            end_date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS mistake_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            question_type TEXT,
            question_source TEXT,
            question_text TEXT NOT NULL,
            user_answer TEXT,
            reference_answer TEXT,
            is_correct INTEGER,
            score REAL,
            error_type TEXT,
            wrong_reason TEXT NOT NULL,
            correction_note TEXT,
            source_of_truth TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS writing_task2_topics (
            id TEXT PRIMARY KEY,
            exam_date TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            essay_type TEXT NOT NULL,
            topic_category TEXT,
            prompt_translation TEXT,
            source_title TEXT,
            source_file TEXT,
            source_excerpt TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (exam_date, prompt_text, essay_type)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS writing_samples (
            id TEXT PRIMARY KEY,
            task2_topic_id TEXT,
            sample_type TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            source_label TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (task2_topic_id) REFERENCES writing_task2_topics(id) ON DELETE SET NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS writing_scoring_descriptors (
            id TEXT PRIMARY KEY,
            writing_type TEXT NOT NULL,
            criterion_name TEXT NOT NULL,
            band_level TEXT,
            descriptor_text TEXT NOT NULL,
            source_label TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (writing_type, criterion_name, band_level, descriptor_text)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS writing_submissions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            task2_topic_id TEXT NOT NULL,
            essay_text TEXT NOT NULL,
            word_count INTEGER,
            score REAL,
            feedback_json TEXT,
            source_of_truth TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (task2_topic_id) REFERENCES writing_task2_topics(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS writing_questions (
            id TEXT PRIMARY KEY,
            source_site TEXT NOT NULL,
            source_url TEXT NOT NULL,
            cambridge_book INTEGER,
            part_no INTEGER,
            task_no INTEGER,
            prompt_text TEXT NOT NULL,
            image_url TEXT,
            image_local_path TEXT,
            module TEXT NOT NULL,
            question_type TEXT NOT NULL,
            crawl_time TEXT NOT NULL,
            parse_status TEXT NOT NULL,
            raw_snapshot_path TEXT,
            UNIQUE (source_site, cambridge_book, part_no, task_no, prompt_text)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            field_name TEXT,
            old_value TEXT,
            new_value TEXT,
            source_message TEXT,
            confidence REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
    ]
    migration_columns = {
        "mistake_records": {
            "question_type": "TEXT",
            "user_answer": "TEXT",
            "reference_answer": "TEXT",
            "is_correct": "INTEGER",
            "score": "REAL",
            "error_type": "TEXT",
            "source_of_truth": "TEXT",
            "metadata": "TEXT",
        },
        "writing_task2_topics": {
            "topic_category": "TEXT",
            "prompt_translation": "TEXT",
            "source_title": "TEXT",
            "source_file": "TEXT",
            "source_excerpt": "TEXT",
            "metadata": "TEXT",
            "updated_at": "TEXT",
        },
        "writing_samples": {
            "task2_topic_id": "TEXT",
            "sample_type": "TEXT",
            "title": "TEXT",
            "source_label": "TEXT",
            "metadata": "TEXT",
            "updated_at": "TEXT",
        },
        "writing_scoring_descriptors": {
            "writing_type": "TEXT",
            "criterion_name": "TEXT",
            "band_level": "TEXT",
            "source_label": "TEXT",
            "metadata": "TEXT",
            "updated_at": "TEXT",
        },
        "writing_submissions": {
            "word_count": "INTEGER",
            "score": "REAL",
            "feedback_json": "TEXT",
            "source_of_truth": "TEXT",
            "metadata": "TEXT",
            "updated_at": "TEXT",
        },
    }

    db_path = get_database_path()
    should_apply_column_migrations = os.getenv("IELTS_ENABLE_DB_MIGRATIONS", "0") == "1"
    try:
        with get_connection() as connection:
            for statement in schema_statements:
                connection.execute(statement)
            if should_apply_column_migrations:
                _apply_schema_migrations(connection, migration_columns)
            connection.commit()
    except sqlite3.OperationalError as exc:
        if "malformed" in str(exc).lower() and db_path.exists():
            broken_path = db_path.with_suffix(
                f"{db_path.suffix}.broken.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            )
            db_path.rename(broken_path)

            with get_connection() as connection:
                for statement in schema_statements:
                    connection.execute(statement)
                if should_apply_column_migrations:
                    _apply_schema_migrations(connection, migration_columns)
                connection.commit()
            return
        raise


def _apply_schema_migrations(
    connection: sqlite3.Connection,
    migration_columns: dict[str, dict[str, str]],
) -> None:
    """Add newly introduced columns to existing SQLite tables."""

    for table_name, columns in migration_columns.items():
        existing_columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_type in columns.items():
            if column_name in existing_columns:
                continue
            try:
                connection.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
            existing_columns.add(column_name)
