"""LangSmith-friendly node tracing helpers."""

from __future__ import annotations

import os
from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable[..., object])


def _enable_langsmith_env() -> None:
    """Enable LangSmith tracing when a project or API key is configured."""

    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "ielts-agent"))


def trace_node(name: str) -> Callable[[F], F]:
    """Decorate a graph node with LangSmith tracing when langsmith is installed."""

    _enable_langsmith_env()
    try:
        from langsmith import traceable
    except Exception:  # pragma: no cover - optional dependency
        def decorator(fn: F) -> F:
            return fn

        return decorator

    def decorator(fn: F) -> F:
        return traceable(name=name, run_type="chain")(fn)  # type: ignore[return-value]

    return decorator
