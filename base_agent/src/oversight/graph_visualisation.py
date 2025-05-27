# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Module for generating execution visualizations optimized for overseer analysis."""

import textwrap

from typing import Union, Optional, List
from datetime import datetime


from ..callgraph.digraph import CallGraph, FunctionNode
from ..callgraph.manager import CallGraphManager
from ..callgraph.reporting import (
    _format_duration,
    _format_cost,
    _format_relative_time,
    _format_stats_line,
)
from ..types.tool_types import ToolResult
from ..types.event_types import EventType, Event
from ..events.event_bus_utils import get_subagent_events


def _format_overseer_event(event: Event) -> str:
    """Format an overseer event with iteration information."""
    iteration = event.metadata.get("iteration", "N/A")
    return f"[Overseer Iteration {iteration}] {event.content}"


def _format_assistant_event(
    event: Event, start_time: datetime, max_length: int = 500
) -> str:
    """Format an assistant message event with timing and cost details."""
    from ..llm.base import Completion

    completion: Optional[Completion | dict] = event.metadata.get("completion")
    if isinstance(completion, dict):
        try:
            completion = Completion.model_validate(completion)
        except Exception:
            completion = None

    if not completion:
        content = textwrap.shorten(event.content, width=max_length, placeholder="...")
        return f"[Assistant] {content}"

    # Extract metrics from completion
    rel_time = _format_relative_time(completion.timing.start_time, start_time)
    abs_time = completion.timing.start_time.strftime("%H:%M:%S")
    tokens = completion.usage.total_tokens
    cached = completion.usage.cached_prompt_tokens
    cached_percent = cached / tokens * 100 if tokens > 0 else 0.0
    cost = completion.calculate_cost()

    content = textwrap.shorten(event.content, width=max_length, placeholder="...")

    return (
        f"[Assistant] {abs_time} ({rel_time}) | {tokens} tokens (cached {cached_percent:.2f}%)| "
        f'{_format_cost(cost)} | "{content}"'
    )


def _format_tool_result_event(event: Event) -> str:
    """Format a tool event concisely with absolute timestamp."""
    result = event.metadata.get("tool_result")
    try:
        result = ToolResult.model_validate(result)
    except Exception:
        pass
    if not isinstance(result, ToolResult):
        return f"[Tool] [Badly formatted tool response]"
    tool_name = result.tool_name
    duration = result.duration
    duration_str = f" | {duration:.1f}s" if duration is not None else ""

    # NOTE: this is important to allow the overseer to see the next step
    # instructions in reasoning structures, and offer good guidance along those
    # steps.
    if "reasoning_structure" in tool_name or tool_name.endswith("_complete"):
        output_str = f" | {str(result.output)}" if result.output else ""
    else:
        output_str = f" | {str(result.output)[:200]}..." if result.output else ""

    abs_time = event.timestamp.strftime("%H:%M:%S")
    return f"[Tool] {abs_time} | {tool_name}{duration_str} → {'Success' if result.success else 'Failed'}{output_str}"

def _format_timeout_event(event: Event) -> str:
    """Format a tool event concisely."""
    duration = event.metadata.get("duration")
    duration_str = f"{duration:.1f}s" if duration is not None else ""
    return f"[Timeout] Timed out after {duration_str}"

async def generate_overseer_execution_tree(
    graph_or_manager: Union[CallGraph, CallGraphManager],
    max_content_length: int = 500,
) -> str:
    """
    Generate an execution tree visualization optimized for overseer analysis.

    Args:
        graph_or_manager: Either a CallGraph instance or CallGraphManager instance
        max_content_length: Maximum length of event content before truncation

    Returns:
        A formatted string containing the execution tree
    """
    # Get the graph instance
    graph = (
        graph_or_manager.graph
        if isinstance(graph_or_manager, CallGraphManager)
        else graph_or_manager
    )

    if not graph.root:
        return "No executions recorded."

    lines = ["EXECUTION TREE (OVERSEER VIEW)", "==========================="]
    start_time = None
    for node in graph.nodes.values():
        if node.started_at and (not start_time or node.started_at < start_time):
            start_time = node.started_at
    start_time = start_time or datetime.now()

    async def format_node(node: FunctionNode, index: str, level: int = 0) -> List[str]:
        indent = "   " * level
        node_lines = []

        # Node header with metrics and absolute start time
        abs_time = node.started_at.strftime("%H:%M:%S") if node.started_at else "N/A"
        rel_time = (
            _format_relative_time(node.started_at, start_time)
            if node.started_at
            else "N/A"
        )
        percent_cached = (
            node.num_cached_tokens / node.token_count if node.token_count > 0 else 0.0
        )

        node_lines.append(
            f"{indent}{index} {node.name} [{node.id}] "
            f"(Started: {abs_time} {rel_time} | {_format_duration(node.duration_seconds)} | "
            f"{node.token_count} tokens (cached {percent_cached:.2f}%) | "
            f"{_format_cost(node.cost)} | "
            f"{'Running...' if node.success is None else 'Success' if node.success else 'Failed'})"
        )

        events = await get_subagent_events(node.id, set(EventType))

        # Add stats line
        node_lines.append(f"{indent}   {await _format_stats_line(events)}")

        # Create timeline of events and child functions
        timeline = []

        # Add events
        for event in events:
            timeline.append((event.timestamp, "event", event))

        # Add child functions
        for i, child_id in enumerate(sorted(node.children), 1):
            child = graph.get_node(child_id)
            if child and child.started_at:
                child_index = f"{index}.{i}"
                timeline.append(
                    (child.started_at, "function_start", (child, child_index))
                )
                if child.completed_at:
                    timeline.append((child.completed_at, "function_end", child))

        # Sort timeline
        timeline.sort(key=lambda x: x[0])

        # Process timeline
        function_level = 0
        for _, item_type, item in timeline:
            if item_type == "event":
                event_indent = f"{indent}   {'   ' * function_level}"
                if item.type == EventType.ASSISTANT_MESSAGE:
                    node_lines.append(
                        f"{event_indent}{_format_assistant_event(item, start_time, max_content_length)}"
                    )
                elif item.type == EventType.TOOL_RESULT:
                    node_lines.append(
                        f"{event_indent}{_format_tool_result_event(item)}"
                    )
                elif item.type == EventType.OVERSEER_UPDATE:
                    node_lines.append(f"{event_indent}{_format_overseer_event(item)}")
                elif item.type == EventType.OVERSEER_NOTIFICATION:
                    abs_time = item.timestamp.strftime("%H:%M:%S")
                    content = textwrap.shorten(
                        item.content, width=max_content_length, placeholder="..."
                    )
                    node_lines.append(
                        f"{event_indent}[Overseer Notification] {abs_time} | {content}"
                    )
                elif item.type == EventType.TIMEOUT:
                    node_lines.append(
                        f"{event_indent}{_format_timeout_event(item)}"
                    )
                elif item.type not in (
                    EventType.CORE_PROMPT_UPDATE,
                    EventType.SYSTEM_PROMPT_UPDATE,
                ):
                    # Include other events except prompt updates
                    abs_time = item.timestamp.strftime("%H:%M:%S")
                    content = textwrap.shorten(
                        item.content, width=max_content_length, placeholder="..."
                    )
                    node_lines.append(
                        f"{event_indent}[{item.type.value}] {abs_time} | {content}"
                    )
            elif item_type == "function_start":
                child, child_index = item
                node_lines.extend(await format_node(child, child_index, level + 1))
                function_level += 1
            elif item_type == "function_end":
                function_level -= 1

        return node_lines

    # Generate tree starting from root
    tree_lines = await format_node(graph.root, "1")
    lines.extend(tree_lines)

    # Add summary
    metrics = graph.get_execution_metrics()
    from ..llm.metering import get_total_usage

    usage = get_total_usage()
    lines.extend(
        [
            "",
            f"Total Duration: {_format_duration(metrics['total_duration'])}",
            f"Total Tokens: {usage.total_tokens} (of which cached {usage.cached_prompt_tokens})",
            f"Total Cost: {_format_cost(metrics['total_cost'])}",
        ]
    )

    full_content = "\n".join(lines)
    chars_per_token = 4
    max_tokens = 100000
    max_chars = max_tokens * chars_per_token
    if len(full_content) > max_chars:
        # Get the LAST max_chars characters
        full_content = full_content[-max_chars:]
        full_content = (
            f"\n[Content was truncated to {max_tokens} tokens at this point...]\n\n"
            + full_content
        )
    return full_content
