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
from project.agent.nodes.data_agent import data_agent_node
from project.agent.nodes.generator import generate_node
from project.agent.nodes.memory_writer import write_memory_node
from project.agent.nodes.runtime_metrics import metrics_finalize_node, metrics_init_node
from project.agent.nodes.react_loop import (
    action_selector_node,
    observation_compressor_node,
    react_control_node,
    react_init_node,
    reason_node,
    route_after_action_selection,
    route_after_react_control,
    should_run_react,
)
from project.agent.nodes.router import route_node
from project.agent.nodes.tool_executor import execute_tools_node
from project.agent.nodes.writing_agent import writing_agent_node
from project.agent.checkpointing import build_checkpointer
from project.agent.state import AgentState


NodeFn = Callable[[AgentState], dict[str, Any]]


class _SequentialGraph:
    """Minimal fallback executor with a LangGraph-like invoke interface."""

    def __init__(self, nodes: list[NodeFn]) -> None:
        self._nodes = nodes

    def invoke(self, state: AgentState, config: dict[str, Any] | None = None) -> AgentState:
        del config
        current_state: AgentState = dict(state)
        current_state.update(metrics_init_node(current_state))
        current_state.update(route_node(current_state))
        current_state.update(react_init_node(current_state))
        while current_state.get("react_should_continue"):
            current_state.update(reason_node(current_state))
            current_state.update(action_selector_node(current_state))
            if current_state.get("selected_tool_call"):
                route = route_after_action_selection(current_state)
                routes = route if isinstance(route, list) else [route]
                if "writing_agent" in routes:
                    current_state.update(writing_agent_node(current_state))
                if "data_agent" in routes:
                    current_state.update(data_agent_node(current_state))
                if "tool_executor" in routes:
                    current_state.update(execute_tools_node(current_state))
                current_state.update(observation_compressor_node(current_state))
            current_state.update(react_control_node(current_state))
        for node in self._nodes:
            current_state.update(node(current_state))
        current_state.update(metrics_finalize_node(current_state))
        current_state.update(write_memory_node(current_state))
        return current_state


_AUTO_CHECKPOINTER = object()


def build_graph(*, checkpointer: Any = _AUTO_CHECKPOINTER):
    """Build and compile the ReAct workflow."""

    if StateGraph is None:
        return _SequentialGraph(
            [
                build_context_node,
                generate_node,
            ]
        )

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("metrics_init", metrics_init_node)
    graph_builder.add_node("router", route_node)
    graph_builder.add_node("react_init", react_init_node)
    graph_builder.add_node("reason", reason_node)
    graph_builder.add_node("action_selector", action_selector_node)
    graph_builder.add_node("writing_agent", writing_agent_node)
    graph_builder.add_node("data_agent", data_agent_node)
    graph_builder.add_node("tool_executor", execute_tools_node)
    graph_builder.add_node("observation_compressor", observation_compressor_node)
    graph_builder.add_node("react_control", react_control_node)
    graph_builder.add_node("context_builder", build_context_node)
    graph_builder.add_node("generator", generate_node)
    graph_builder.add_node("metrics_finalize", metrics_finalize_node)
    graph_builder.add_node("memory_writer", write_memory_node)

    graph_builder.add_edge(START, "metrics_init")
    graph_builder.add_edge("metrics_init", "router")
    graph_builder.add_edge("router", "react_init")
    graph_builder.add_conditional_edges(
        "react_init",
        should_run_react,
        {
            "reason": "reason",
            "context_builder": "context_builder",
        },
    )
    graph_builder.add_edge("reason", "action_selector")
    graph_builder.add_conditional_edges(
        "action_selector",
        route_after_action_selection,
        {
            "writing_agent": "writing_agent",
            "data_agent": "data_agent",
            "tool_executor": "tool_executor",
            "react_control": "react_control",
        },
    )
    graph_builder.add_edge("writing_agent", "observation_compressor")
    graph_builder.add_edge("data_agent", "observation_compressor")
    graph_builder.add_edge("tool_executor", "observation_compressor")
    graph_builder.add_edge("observation_compressor", "react_control")
    graph_builder.add_conditional_edges(
        "react_control",
        route_after_react_control,
        {
            "reason": "reason",
            "context_builder": "context_builder",
        },
    )
    graph_builder.add_edge("context_builder", "generator")
    graph_builder.add_edge("generator", "metrics_finalize")
    graph_builder.add_edge("metrics_finalize", "memory_writer")
    graph_builder.add_edge("memory_writer", END)

    effective_checkpointer = build_checkpointer() if checkpointer is _AUTO_CHECKPOINTER else checkpointer
    compile_kwargs: dict[str, Any] = {}
    if effective_checkpointer is not None:
        compile_kwargs["checkpointer"] = effective_checkpointer
    return graph_builder.compile(**compile_kwargs)
