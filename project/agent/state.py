"""Shared state definitions for the LangGraph pipeline."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


IntentType = Literal[
    "study_plan",
    "knowledge_qa",
    "mistake_review",
    "writing_practice",
    "data_collection",
    "calendar_action",
    "general_chat",
]


class Message(TypedDict):
    """A minimal chat message structure."""

    role: str
    content: str


class ToolCall(TypedDict):
    """A normalized tool invocation request."""

    tool_name: str
    action: str
    args: dict[str, Any]


class RetrievedDoc(TypedDict):
    """A retrieved document chunk placeholder."""

    source: str
    title: str
    content: str
    score: float


class UserProfile(TypedDict, total=False):
    """Learner profile loaded from the database layer."""

    user_id: str
    current_level: str
    target_score: str
    preferred_focus: str
    available_hours_per_week: int


class StudyContext(TypedDict, total=False):
    """Lightweight session and study memory."""

    total_turns: int
    last_intent: str
    last_user_input: str
    last_answer: str
    active_goal: str
    active_writing_topic_id: str
    active_writing_prompt: str


class AgentState(TypedDict, total=False):
    """State carried across the graph."""

    messages: list[Message]
    user_input: str
    intent: IntentType
    plan: list[str]
    tool_calls: list[ToolCall]
    tool_results: dict[str, Any]
    retrieved_docs: list[RetrievedDoc]
    user_profile: UserProfile
    study_context: StudyContext
    writing_review_state: dict[str, Any]
    context_summary: str
    final_answer: str


def build_initial_state(
    user_input: str,
    messages: list[Message] | None = None,
    user_profile: UserProfile | None = None,
    study_context: StudyContext | None = None,
) -> AgentState:
    """Create a fresh state object for one graph invocation."""

    return AgentState(
        messages=list(messages or []),
        user_input=user_input,
        intent="general_chat",
        plan=[],
        tool_calls=[],
        tool_results={},
        retrieved_docs=[],
        user_profile=user_profile or {},
        study_context=study_context or {"total_turns": 0},
        writing_review_state={},
        context_summary="",
        final_answer="",
    )
