# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Module for generating execution reports from call graphs."""

import difflib
import textwrap

from typing import Union, Optional, List
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

from .digraph import CallGraph, FunctionNode
from .manager import CallGraphManager
from ..events.event_bus_utils import get_subagent_events
from ..llm.base import Completion
from ..llm.metering import get_total_cost, get_total_usage
from ..types.tool_types import ToolResult
from ..types.event_types import EventType, Event


def _format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds to a readable string."""
    if seconds is None:
        return "N/A"
    return f"{seconds:.1f}s"


def _format_cost(cost: float) -> str:
    """Format cost in dollars."""
    return f"${cost:.3f}"


def _format_relative_time(time: datetime, start_time: datetime) -> str:
    """Format time relative to start time."""
    delta = time - start_time
    return f"t+{delta.total_seconds():.1f}s"


async def _format_assistant_event(
    event: Event, start_time: datetime, truncate: Union[int, bool] = 50
) -> tuple[str, str]:
    """Format an assistant message event with timing and cost details.

    Args:
        event: The event to format
        start_time: The start time of the execution
        truncate: If an integer, truncate to that many characters. If True, truncate to 50 characters.
                 If False, don't truncate.

    Returns:
        A tuple of (info_line, content)
    """
    completion: Optional[Completion | dict] = event.metadata.get("completion")
    if isinstance(completion, dict):
        try:
            completion = Completion.model_validate(completion)
        except Exception:
            completion = None
    if not completion:
        # Default truncation length if truncate is True
        truncate_length = 50 if truncate is True else truncate if isinstance(truncate, int) else None

        if truncate_length is not None:
            content = event.content[:truncate_length].replace("\n", "").strip() + "..."
        else:
            content = event.content.strip()

        return f"[Assistant] {content}", ""

    # Extract metrics from completion
    rel_time = _format_relative_time(completion.timing.start_time, start_time)
    tokens = completion.usage.total_tokens
    cached = completion.usage.cached_prompt_tokens
    cached_percent = cached / tokens * 100 if tokens > 0 else 0.0
    cost = completion.calculate_cost()
    ttft = (
        completion.timing.time_to_first_token
        if completion.timing.time_to_first_token
        else 0
    )

    # Determine how to format content based on truncate parameter
    if not truncate:
        content = textwrap.fill(event.content.strip(), width=80)
        return (
            f"[Assistant] {rel_time} | {tokens} tokens (cached {cached_percent:.2f}%)| {_format_cost(cost)} | ",
            content,
        )
    else:
        # Set truncation length based on the truncate parameter
        truncate_length = 50 if truncate is True else truncate
        content = event.content[:truncate_length].replace("\n", "").strip() + "..."

        return (
            f"[Assistant] {rel_time} | {tokens} tokens (cached {cached_percent:.2f}%)| {_format_cost(cost)} | "
            f'"{content}"',
            "",
        )


def _format_tool_result_event(event: Event) -> str:
    """Format a tool event concisely."""
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
    # TODO: make 'Success' here more nuanced *e.g. execute command with exit code 1"
    return f"[Tool] {tool_name}{duration_str} → {'Success' if result.success else 'Failed'}"


def _format_timeout_event(event: Event) -> str:
    """Format a tool event concisely."""
    duration = event.metadata.get("duration")
    duration_str = f"{duration:.1f}s" if duration is not None else ""
    return f"[Timeout] Timed out after {duration_str}"


def _format_generic_event(event: Event) -> str:
    """Format a tool event concisely."""
    event_type = event.type.value
    content = event.content[:50].replace("\n", "").strip()
    return f"[{event_type}] {content}..."


async def _format_stats_line(events: List[Event]) -> str:
    """Format a statistics summary line for a set of events."""
    event_counts = defaultdict(int)
    prompt_lines = defaultdict(int)

    for event in events:
        event_counts[event.type] += 1
        if event.type in (EventType.SYSTEM_PROMPT_UPDATE, EventType.CORE_PROMPT_UPDATE):
            prompt_lines[event.type] += len(event.content.splitlines())

    parts = []
    if event_counts[EventType.SYSTEM_PROMPT_UPDATE]:
        parts.append(
            f"{event_counts[EventType.SYSTEM_PROMPT_UPDATE]} system updates "
            f"({prompt_lines[EventType.SYSTEM_PROMPT_UPDATE]} lines)"
        )
    if event_counts[EventType.CORE_PROMPT_UPDATE]:
        parts.append(
            f"{event_counts[EventType.CORE_PROMPT_UPDATE]} core updates "
            f"({prompt_lines[EventType.CORE_PROMPT_UPDATE]} lines)"
        )
    if event_counts[EventType.TOOL_CALL]:
        parts.append(f"{event_counts[EventType.TOOL_CALL]} tool calls")
    if event_counts[EventType.ASSISTANT_MESSAGE]:
        parts.append(f"{event_counts[EventType.ASSISTANT_MESSAGE]} messages")

    return f"[Stats] Events: {', '.join(parts)}"


def _find_execution_start_time(graph: CallGraph) -> datetime:
    """Find the earliest start time in the execution."""
    start_time = None
    for node in graph.nodes.values():
        if node.started_at and (not start_time or node.started_at < start_time):
            start_time = node.started_at
    return start_time or datetime.now()


async def generate_execution_tree(
    graph_or_manager: Union[CallGraph, CallGraphManager],
    truncate_assistant_events: bool | int = 50,
    include_all_events: bool = False,  # Just includes assistant and tool results by default
) -> str:
    """
    Generate a standalone execution tree visualization.

    Args:
        graph_or_manager: Either a CallGraph instance or CallGraphManager instance
        event_bus: EventBus instance for event retrieval
        truncate_assistant_events: Make the representation concise by truncating assistant messages

    Returns:
        A formatted string containing just the execution tree
    """
    # Get the graph instance
    graph = (
        graph_or_manager.graph
        if isinstance(graph_or_manager, CallGraphManager)
        else graph_or_manager
    )

    if not graph.root:
        return "No executions recorded."

    lines = ["EXECUTION TREE", "=============="]
    start_time = _find_execution_start_time(graph)

    async def format_node(node: FunctionNode, index: str, level: int = 0) -> List[str]:
        indent = "   " * level
        node_lines = []

        percent_cached = (
            node.num_cached_tokens / node.token_count if node.token_count > 0 else 0.0
        ) * 100

        # Node header with metrics
        node_lines.append(
            f"{indent}{index} {node.name} [{node.id}] "
            f"({_format_duration(node.duration_seconds)} | "
            f"{node.token_count} tokens (cached {percent_cached:.2f}%) | "
            f"{_format_cost(node.cost)} | "
            f"{'Running...' if node.success is None else 'Success' if node.success else 'Failed'})"
        )

        event_types = {
            EventType.ASSISTANT_MESSAGE,
            EventType.TOOL_RESULT,
            EventType.TOOL_CALL,
            EventType.TIMEOUT
        }
        events = await get_subagent_events(node.id, event_types)

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
                if item.type == EventType.ASSISTANT_MESSAGE:
                    if truncate_assistant_events:
                        # Use the truncate parameter as an integer (default 50)
                        line_content, _ = await _format_assistant_event(
                            item, start_time, truncate_assistant_events
                        )
                        node_lines.append(
                            f"{indent}   {'   ' * function_level}{line_content}"
                        )
                    else:
                        # Pass False to indicate no truncation
                        info_line, content = await _format_assistant_event(
                            item, start_time, False
                        )
                        node_lines.append(
                            f"{indent}   {'   ' * function_level}{info_line}"
                        )
                        for l in content.splitlines():
                            node_lines.append(
                                f"{indent}   {'   ' * function_level}  {l}"
                            )

                elif item.type == EventType.TOOL_RESULT:
                    node_lines.append(
                        f"{indent}   {'   ' * function_level}{_format_tool_result_event(item)}"
                    )
                elif item.type == EventType.TIMEOUT:
                    node_lines.append(
                        f"{indent}   {'   ' * function_level}{_format_timeout_event(item)}"
                    )
                elif include_all_events and item.type != EventType.OVERSEER_UPDATE:
                    node_lines.append(
                        f"{indent}   {'   ' * function_level}{_format_generic_event(item)}"
                    )
            elif item_type == "function_start":
                child, child_index = item
                child_lines = await format_node(child, child_index, level + 1)
                node_lines.extend(child_lines)
                function_level += 1
            elif item_type == "function_end":
                function_level -= 1

        return node_lines

    # Generate tree starting from root
    tree_lines = await format_node(graph.root, "1")
    lines.extend(tree_lines)

    # Add summary
    metrics = graph.get_execution_metrics()
    usage = get_total_usage()
    lines.extend(
        [
            "",
            f"Total Duration: {_format_duration(metrics['total_duration'])}",
            f"Total Tokens: {usage.total_tokens} (of which cached {usage.cached_prompt_tokens})",
            f"Total Cost: {_format_cost(get_total_cost())}",
        ]
    )

    return "\n".join(lines)


@dataclass
class PromptState:
    """Tracks the state of system and core prompts."""

    system_prompt: Optional[str] = None
    core_prompt: Optional[str] = None


async def _format_linear_trace(
    node: FunctionNode, exclude_system_prompt: bool = True
) -> str:
    """Format a complete linear trace of all events for a function node."""
    lines = []

    # Start trace marker
    lines.append(f"\n{node.id} trace start {'=' * 50}")

    percent_cached = (
        node.num_cached_tokens / node.token_count if node.token_count > 0 else 0.0
    )

    # Add function header
    lines.append(
        f"Function: {node.name}\n"
        f"Duration: {_format_duration(node.duration_seconds)}\n"
        f"Tokens: {node.token_count} (cached: {percent_cached:.2f}%)\n"
        f"Cost: {_format_cost(node.cost)}\n"
        f"Status: {'Success' if node.success else 'Failed'}\n"
    )

    # Add all events in chronological order
    prompt_state = PromptState()
    events = await get_subagent_events(node.id)
    for event in events:
        if event.type == EventType.SYSTEM_PROMPT_UPDATE:
            if prompt_state.system_prompt is None:
                if not exclude_system_prompt:
                    lines.append("\nSystem Prompt:")
                    lines.append(event.content)
                prompt_state.system_prompt = event.content
            else:
                previous_lines = prompt_state.system_prompt.splitlines()
                updated_lines = event.content.splitlines()
                diff = difflib.ndiff(previous_lines, updated_lines)
                num_diff_lines = sum(1 for line in diff if line.startswith(("+", "-")))

                if num_diff_lines > 0:
                    lines.append("\nSystem Prompt Update:")
                    prompt_len = len(updated_lines)
                    lines.append(
                        f"\nSystem prompt was updated with {num_diff_lines} differing lines. "
                        f"Total length is now {prompt_len}.\n"
                    )
                prompt_state.system_prompt = event.content

        elif event.type == EventType.CORE_PROMPT_UPDATE:
            if prompt_state.core_prompt is None:
                lines.append("\nCore Prompt:")
                lines.append(event.content)
                prompt_state.core_prompt = event.content
            else:
                previous_lines = prompt_state.core_prompt.splitlines()
                updated_lines = event.content.splitlines()
                diff = difflib.ndiff(previous_lines, updated_lines)
                num_diff_lines = sum(1 for line in diff if line.startswith(("+", "-")))

                if num_diff_lines > 0:
                    lines.append("\nCore Prompt Update:")
                    prompt_len = len(updated_lines)
                    lines.append(
                        f"\nCore prompt was updated with {num_diff_lines} differing lines. "
                        f"Total length is now {prompt_len}.\n"
                    )
                prompt_state.core_prompt = event.content

        elif event.type == EventType.ASSISTANT_MESSAGE:
            lines.append(f"\nAssistant Message:")
            lines.append(event.content)

        elif event.type == EventType.TOOL_CALL:
            lines.append("\nTool Call:")
            tool_name = event.metadata.get("name", "unknown_tool")
            args = event.metadata.get("args", {})
            lines.append(f"Tool: {tool_name}")
            lines.append(f"Args: {args}")

        elif event.type == EventType.TOOL_RESULT:
            lines.append("\nTool Result:")
            lines.append(event.content)

        else:
            lines.append(f"\n{event.type.value}:")
            lines.append(event.content)

    # End trace marker
    lines.append(f"\n{node.id} trace end {'=' * 52}\n")

    return "\n".join(lines)


async def generate_execution_report(
    graph_or_manager: Union[CallGraph, CallGraphManager]
) -> str:
    """
    Generate a detailed execution report from a CallGraph or CallGraphManager.

    Args:
        graph_or_manager: Either a CallGraph instance or CallGraphManager instance
        event_bus: EventBus instance for event retrieval

    Returns:
        A formatted string containing the execution report
    """
    # Get the graph instance
    graph = (
        graph_or_manager.graph
        if isinstance(graph_or_manager, CallGraphManager)
        else graph_or_manager
    )

    if not graph.root:
        return "No executions recorded."

    sections = []

    # Add execution summary
    metrics = graph.get_execution_metrics()
    summary = [
        "EXECUTION SUMMARY",
        "================",
        f"Total Duration: {_format_duration(metrics['total_duration'])}",
        f"Total Tokens: {metrics['total_tokens']} (of which cached: {metrics['num_cached_tokens']})",
        f"Total Cost: {_format_cost(metrics['total_cost'])}",
        f"Status: {'Completed Successfully' if metrics['failed_calls'] == 0 else 'Failed'}",
    ]
    sections.append("\n".join(summary))

    # Add linear traces for all nodes
    sections.append("\nCOMPLETE EVENT TRACES\n====================")
    for node in graph.iter_dfs():
        sections.append(await _format_linear_trace(node))

    # Add tree visualization
    sections.append(await generate_execution_tree(graph))

    return "\n".join(sections)
