"""LangGraph pipeline assembly.

The demo prefers real LangGraph. If the dependency is not installed yet, we use
an in-process sequential fallback so the phase-one CLI remains runnable.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    END = "__end__"
    START = "__start__"
    StateGraph = None

from project.agent.nodes.context_builder import build_context_node
from project.agent.nodes.generator import generate_node
from project.agent.nodes.memory_writer import write_memory_node
from project.agent.nodes.planner import plan_node
from project.agent.nodes.router import route_node
from project.agent.nodes.tool_executor import execute_tools_node
from project.agent.nodes.writing_react import (
    finalize_writing_review_node,
    is_writing_review_active,
    should_continue_writing_retrieval,
    writing_retrieval_node,
)
from project.agent.state import AgentState


NodeFn = Callable[[AgentState], dict[str, Any]]


class _SequentialGraph:
    """Minimal fallback executor with a LangGraph-like invoke interface."""

    def __init__(self, nodes: list[NodeFn]) -> None:
        self._nodes = nodes

    def invoke(self, state: AgentState) -> AgentState:
        current_state: AgentState = dict(state)
        for index, node in enumerate(self._nodes):
            updates = node(current_state)
            current_state.update(updates)
            if index == 2 and is_writing_review_active(current_state):
                while should_continue_writing_retrieval(current_state):
                    current_state.update(writing_retrieval_node(current_state))
                current_state.update(finalize_writing_review_node(current_state))
        return current_state


def _route_after_tool_executor(state: AgentState) -> str:
    if is_writing_review_active(state):
        return "writing_retrieval"
    return "context_builder"


def _route_after_writing_retrieval(state: AgentState) -> str:
    if should_continue_writing_retrieval(state):
        return "writing_retrieval"
    return "writing_finalize"


def build_graph():
    """Build and compile the phase-one LangGraph workflow."""

    if StateGraph is None:
        return _SequentialGraph(
            [
                route_node,
                plan_node,
                execute_tools_node,
                build_context_node,
                generate_node,
                write_memory_node,
            ]
        )

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("router", route_node)
    graph_builder.add_node("planner", plan_node)
    graph_builder.add_node("tool_executor", execute_tools_node)
    graph_builder.add_node("writing_retrieval", writing_retrieval_node)
    graph_builder.add_node("writing_finalize", finalize_writing_review_node)
    graph_builder.add_node("context_builder", build_context_node)
    graph_builder.add_node("generator", generate_node)
    graph_builder.add_node("memory_writer", write_memory_node)

    graph_builder.add_edge(START, "router")
    graph_builder.add_edge("router", "planner")
    graph_builder.add_edge("planner", "tool_executor")
    graph_builder.add_conditional_edges(
        "tool_executor",
        _route_after_tool_executor,
        {
            "writing_retrieval": "writing_retrieval",
            "context_builder": "context_builder",
        },
    )
    graph_builder.add_conditional_edges(
        "writing_retrieval",
        _route_after_writing_retrieval,
        {
            "writing_retrieval": "writing_retrieval",
            "writing_finalize": "writing_finalize",
        },
    )
    graph_builder.add_edge("writing_finalize", "context_builder")
    graph_builder.add_edge("context_builder", "generator")
    graph_builder.add_edge("generator", "memory_writer")
    graph_builder.add_edge("memory_writer", END)

    return graph_builder.compile()
