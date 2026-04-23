"""Shared state definitions for the LangGraph pipeline."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict


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


def merge_dict_state(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Merge concurrent branch dict updates with right-hand keys taking precedence."""

    merged: dict[str, Any] = {}
    if isinstance(left, dict):
        merged.update(left)
    if isinstance(right, dict):
        merged.update(right)
    return merged


def merge_tool_results(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Merge tool results produced by parallel agent branches."""

    merged = merge_dict_state(left, right)
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    if right and "_last_error" not in right:
        merged.pop("_last_error", None)
    recent_limits = {
        "_tool_failures": 8,
        "_tool_policy_trace": 12,
        "_tool_fallbacks": 6,
    }
    for key, limit in recent_limits.items():
        left_items = left.get(key, [])
        right_items = right.get(key, [])
        combined = []
        if isinstance(left_items, list):
            combined.extend(left_items)
        if isinstance(right_items, list):
            combined.extend(right_items)
        if combined:
            merged[key] = combined[-limit:]
    return merged


def append_recent_events(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Append branch-local agent events while bounding state growth."""

    events: list[dict[str, Any]] = []
    if isinstance(left, list):
        events.extend(item for item in left if isinstance(item, dict))
    if isinstance(right, list):
        events.extend(item for item in right if isinstance(item, dict))
    elif isinstance(right, dict):
        events.append(right)
    return events[-12:]


class AgentState(TypedDict, total=False):
    """State carried across the graph."""

    messages: list[Message]
    user_input: str
    intent: IntentType
    plan: list[str]
    tool_calls: list[ToolCall]
    selected_tool_call: ToolCall | None
    tool_call_history: list[str]
    agent_events: Annotated[list[dict[str, Any]], append_recent_events]
    agent_outputs: Annotated[dict[str, Any], merge_dict_state]
    tool_results: Annotated[dict[str, Any], merge_tool_results]
    tool_health: Annotated[dict[str, Any], merge_dict_state]
    observations: list[dict[str, Any]]
    last_observation: dict[str, Any]
    retrieved_docs: list[RetrievedDoc]
    user_profile: Annotated[UserProfile, merge_dict_state]
    study_context: Annotated[StudyContext, merge_dict_state]
    writing_review_state: Annotated[dict[str, Any], merge_dict_state]
    react_active: bool
    react_step: int
    react_max_steps: int
    react_last_thought: str
    react_should_continue: bool
    react_finish_reason: str | None
    react_decision: dict[str, Any]
    answer_context: dict[str, Any]
    runtime_metrics: dict[str, Any]
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
        selected_tool_call=None,
        tool_call_history=[],
        agent_events=[],
        agent_outputs={},
        tool_results={},
        tool_health={},
        observations=[],
        last_observation={},
        retrieved_docs=[],
        user_profile=user_profile or {},
        study_context=study_context or {"total_turns": 0},
        writing_review_state={},
        react_active=False,
        react_step=0,
        react_max_steps=0,
        react_last_thought="",
        react_should_continue=False,
        react_finish_reason=None,
        react_decision={},
        answer_context={},
        runtime_metrics={},
        final_answer="",
    )
