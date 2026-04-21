"""Graph-level ReAct loop helpers for writing review."""

from __future__ import annotations

import logging
from typing import Any

from project.agent.state import AgentState, UserProfile
from project.rag.orchestration.gap_retrieval import (
    apply_retrieval_state_to_review_state,
    is_gap_retrieval_complete,
    run_gap_retrieval_round,
)
from project.tools.writing_tool import finalize_task2_review

logger = logging.getLogger(__name__)


def is_writing_review_active(state: AgentState) -> bool:
    """Whether the current state is inside a writing-review loop."""

    review_state = state.get("writing_review_state", {})
    return bool(isinstance(review_state, dict) and review_state.get("active"))


def should_continue_writing_retrieval(state: AgentState) -> bool:
    """Whether another retrieval round should run."""

    review_state = state.get("writing_review_state", {})
    if not isinstance(review_state, dict) or not review_state.get("active"):
        return False

    retrieval_state = review_state.get("retrieval_state", {})
    if isinstance(retrieval_state, dict) and retrieval_state:
        return not is_gap_retrieval_complete(retrieval_state)

    enough_context = bool(review_state.get("enough_context", False))
    retrieval_round = int(review_state.get("retrieval_round", 0) or 0)
    max_rounds = int(review_state.get("max_rounds", 3) or 3)
    return not enough_context and retrieval_round < max_rounds


def writing_retrieval_node(state: AgentState) -> dict[str, dict[str, Any]]:
    """Run one retrieve-judge round for writing review."""

    review_state = state.get("writing_review_state", {})
    if not isinstance(review_state, dict) or not review_state.get("active"):
        return {}

    retrieval_state = review_state.get("retrieval_state", {})
    if not isinstance(retrieval_state, dict) or not retrieval_state:
        return {}

    updated_retrieval_state = run_gap_retrieval_round(retrieval_state)
    updated_review_state = apply_retrieval_state_to_review_state(review_state, updated_retrieval_state)
    logger.info(
        "Writing retrieval round %s finished (enough=%s)",
        updated_review_state.get("retrieval_round", 0),
        updated_retrieval_state.get("complete", False),
    )
    return {"writing_review_state": updated_review_state}


def finalize_writing_review_node(
    state: AgentState,
) -> dict[str, dict[str, Any] | UserProfile]:
    """Finalize writing review after retrieval rounds complete."""

    review_state = state.get("writing_review_state", {})
    if not isinstance(review_state, dict) or not review_state.get("active"):
        return {}

    result = finalize_task2_review(review_state)
    tool_results = dict(state.get("tool_results", {}))
    tool_results["review_task2_submission"] = result

    user_profile: UserProfile = dict(state.get("user_profile", {}))
    updated_profile = result.get("updated_profile", {})
    if isinstance(updated_profile, dict):
        user_profile.update(updated_profile)

    finalized_state = dict(review_state)
    finalized_state["active"] = False
    finalized_state["finalized"] = True

    logger.info("Writing review finalized after %s retrieval round(s)", review_state.get("retrieval_round", 0))
    return {
        "tool_results": tool_results,
        "user_profile": user_profile,
        "writing_review_state": finalized_state,
    }
