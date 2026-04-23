"""Writing specialist agent node.

The main ReAct controller still decides when writing work is needed. This node
owns execution of writing-domain actions and writes structured handoff events
back into shared state for the controller, observation compressor, and metrics.
"""

from __future__ import annotations

from typing import Any

from project.agent.nodes.tool_executor import execute_tools_node
from project.agent.nodes.tool_policy import result_key_for, tool_id
from project.agent.nodes.tracing import trace_node
from project.agent.state import AgentState


WRITING_AGENT_TOOLS = {"writing"}


def is_writing_agent_tool(tool_name: str) -> bool:
    return tool_name in WRITING_AGENT_TOOLS


@trace_node("writing_agent")
def writing_agent_node(state: AgentState) -> dict[str, Any]:
    """Execute writing-domain actions selected by the supervisor."""

    matching_calls = _matching_tool_calls(state)
    if not matching_calls:
        return _agent_output_update(state, "skipped", [])

    merged_state: AgentState = dict(state)
    updates: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for tool_call in matching_calls:
        call_state: AgentState = dict(merged_state)
        call_state["selected_tool_call"] = tool_call
        call_updates = execute_tools_node(call_state)
        merged_state.update(call_updates)
        updates.update(call_updates)
        events.append(_agent_event(merged_state, tool_call, "handled", call_updates))
    return {
        **updates,
        "agent_events": events,
        **_agent_output_update(merged_state, "handled", matching_calls),
    }


def _matching_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    calls = state.get("tool_calls") or []
    if not calls and isinstance(state.get("selected_tool_call"), dict):
        calls = [state["selected_tool_call"]]
    return [
        call
        for call in calls
        if isinstance(call, dict) and is_writing_agent_tool(str(call.get("tool_name") or ""))
    ]


def _agent_event(
    state: AgentState,
    tool_call: dict[str, Any],
    status: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    result_key = result_key_for(tool_call) if isinstance(tool_call, dict) else ""
    tool_results = updates.get("tool_results") or state.get("tool_results", {})
    result = tool_results.get(result_key) if isinstance(tool_results, dict) else None
    return {
        "agent": "writing_agent",
        "status": status,
        "tool": tool_id(tool_call) if isinstance(tool_call, dict) else "",
        "result_key": result_key,
        "success": bool(not isinstance(result, dict) or result.get("success") is not False),
    }


def _agent_output_update(state: AgentState, status: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    del state
    return {
        "agent_outputs": {
            "writing_agent": {
                "status": status,
                "tools": [tool_id(call) for call in tool_calls],
            }
        }
    }
