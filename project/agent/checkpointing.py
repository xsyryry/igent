"""Checkpoint helpers for LangGraph runtime state history."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from project.config import get_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREAD_ID = "demo_user"

_MEMORY_CHECKPOINTER: Any | None = None
_SQLITE_CONNECTIONS: list[sqlite3.Connection] = []
_SQLITE_CONTEXTS: list[Any] = []


def build_checkpointer() -> Any | None:
    """Create a LangGraph checkpointer from runtime config.

    Supported backends:
    - memory: process-local state history, useful for tests/dev
    - sqlite: persistent local checkpoint DB, preferred for local production
    - none/off/disabled: no checkpointing
    """

    backend = get_config().checkpoint_backend.strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return None
    if backend == "memory":
        return _build_memory_checkpointer()
    if backend == "sqlite":
        return _build_sqlite_checkpointer()

    logger.warning("Unknown checkpoint backend %r; checkpointing disabled", backend)
    return None


def checkpoint_config(thread_id: str | None = None) -> dict[str, dict[str, str]]:
    """Build the required LangGraph invoke config for checkpointed runs."""

    normalized_thread_id = str(thread_id or DEFAULT_THREAD_ID).strip() or DEFAULT_THREAD_ID
    return {"configurable": {"thread_id": normalized_thread_id}}


def _build_memory_checkpointer() -> Any | None:
    global _MEMORY_CHECKPOINTER
    if _MEMORY_CHECKPOINTER is not None:
        return _MEMORY_CHECKPOINTER

    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except Exception:
        try:
            from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
        except Exception as exc:  # pragma: no cover - optional dependency boundary
            logger.warning("LangGraph memory checkpoint saver unavailable: %s", exc)
            return None

    _MEMORY_CHECKPOINTER = InMemorySaver()
    return _MEMORY_CHECKPOINTER


def _build_sqlite_checkpointer() -> Any | None:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        logger.warning("SQLite checkpoint saver unavailable; install langgraph-checkpoint-sqlite: %s", exc)
        return None

    db_path = _resolve_checkpoint_path(get_config().checkpoint_sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _SQLITE_CONNECTIONS.append(conn)

    try:
        saver = SqliteSaver(conn)
    except TypeError:
        _SQLITE_CONNECTIONS.pop()
        conn.close()
        context = SqliteSaver.from_conn_string(str(db_path))
        saver = context.__enter__()
        _SQLITE_CONTEXTS.append(context)

    setup = getattr(saver, "setup", None)
    if callable(setup):
        setup()
    return saver


def _resolve_checkpoint_path(raw_path: str) -> Path:
    path = Path(raw_path or "data/langgraph_checkpoints.sqlite")
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
