# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utility functions for working with the event bus.

Note: these are entirely un-optimised, and many require inefficient iterations
over the event lists to reconstruct 'views' on the event bus. For SWE bench
style tasks, with only up to hundreds of messages, this is not important.
"""

from typing import Optional, Set, List

from .event_bus import EventBus
from ..types.tool_types import ToolResult
from ..types.agent_types import AgentResult
from ..types.event_types import EventType, Event, FileOperation, FileEvent


async def log_to_stdout(event: Event | FileEvent):
    """Print important events to stdout with clear formatting.

    This function is important to format since it is the visual feedback
    that the meta-agent will get back when test-running itself
    """

    # Common formatting constants
    max_content_len = 50  # Reduced to allow for richer metadata
    prefix_width = 10

    def truncate(text: str, length: int = max_content_len) -> str:
        """Helper to truncate text and handle newlines"""
        text = text.replace("\n", " ")
        return f"{text[:length]}..." if len(text) > length else text

    def format_output(prefix: str, content: str, metadata: str = "") -> None:
        """Helper to format and print consistent output"""
        print(
            f"{prefix:<{prefix_width}s} => {content}{' | ' + metadata if metadata else ''}"
        )

    event_content = truncate(str(event.content))

    if event.type in (EventType.CORE_PROMPT_UPDATE, EventType.SYSTEM_PROMPT_UPDATE):
        return
    elif event.type == EventType.ASSISTANT_MESSAGE:
        format_output(event.type.value, event_content)
    elif event.type == EventType.TOOL_CALL:
        name = event.metadata.get("name", "unknown tool")
        args = truncate(str(event.metadata.get("args", {})))
        format_output(event.type.value, f"{name}, {args}")
    elif event.type == EventType.TOOL_RESULT:
        result = event.metadata.get("tool_result")
        if not isinstance(result, ToolResult):
            return
        content = f"{result.tool_name}, success: {result.success}, "
        content += f"duration: {result.duration:.1f}, {event_content} "
        format_output(event.type.value, content)
    elif event.type == EventType.AGENT_CALL:
        name = event.metadata.get("name", "unknown agent")
        args = truncate(str(event.metadata.get("args", {})))
        format_output(event.type.value, f"{name}, {args}")
    elif event.type == EventType.AGENT_RESULT:
        result = event.metadata.get("agent_result")
        if not isinstance(result, AgentResult):
            return
        name = result.agent_name
        status = result.status.value
        duration = result.metrics.duration_seconds or 0.0
        cost = result.metrics.cost
        res = truncate(result.result, 20)
        content = f"{name}, status: {status}, duration: {duration:.1f}, cost: ${cost:.4f}, {res}"
        format_output(event.type.value, content)
    else:
        format_output(event.type.value, event_content)


async def get_problem_statement() -> str:
    """Get the initial problem statement."""
    event_bus = await EventBus.get_instance()
    # There should only be one, but we handle the case when it was updated somehow
    ps_events = event_bus.get_events_by_type(EventType.PROBLEM_STATEMENT)
    return "\n".join(ps.content for ps in ps_events) if len(ps_events) else ""


async def get_budget_info() -> dict[str, int | float | None]:
    """Get the initial problem statement."""
    event_bus = await EventBus.get_instance()
    # There should only be one, but we handle the case when it was updated somehow
    ps_events = event_bus.get_events_by_type(EventType.BUDGET_INFO)
    if ps_events:
        return ps_events[-1].metadata
    else:
        return dict()

async def get_latest_sys_prompt_event(agent_id: str | None = None) -> Optional[Event]:
    """Get the latest system prompt update event."""
    event_bus = await EventBus.get_instance()
    events = (
        event_bus.get_events_by_type(EventType.SYSTEM_PROMPT_UPDATE)
        if not agent_id
        else event_bus.get_events(agent_id)
    )
    system_prompts = [e for e in events if e.type == EventType.SYSTEM_PROMPT_UPDATE]
    return system_prompts[-1] if system_prompts else None


async def get_latest_core_prompt_event(agent_id: str | None = None) -> Optional[Event]:
    """Get the latest core prompt update event."""
    event_bus = await EventBus.get_instance()
    events = (
        event_bus.get_events_by_type(EventType.CORE_PROMPT_UPDATE)
        if not agent_id
        else event_bus.get_events(agent_id)
    )
    core_prompts = [e for e in events if e.type == EventType.CORE_PROMPT_UPDATE]
    return core_prompts[-1] if core_prompts else None


async def get_open_file_set(agent_id: str | None = None) -> Set[FileEvent]:
    """Get the set of currently open files."""
    event_bus = await EventBus.get_instance()
    open_files: dict[str, FileEvent] = {}
    events = (
        event_bus.get_events_by_type(EventType.FILE_EVENT)
        if not agent_id
        else [
            e for e in event_bus.get_events(agent_id) if e.type == EventType.FILE_EVENT
        ]
    )

    for event in events:
        if isinstance(event, FileEvent):
            if event.operation == FileOperation.CLOSE and event.path in open_files:
                open_files.pop(event.path)
            elif event.operation == FileOperation.OPEN:
                open_files[event.path] = event
    return set(open_files.values())


async def is_file_open(file_path: str, agent_id: str | None = None) -> bool:
    """Check if a specific file is open."""
    open_files = await get_open_file_set(agent_id)
    return any(file_event.path == file_path for file_event in open_files)


async def get_latest_file_event(
    file_path: str,
    agent_id: str | None = None,
    exclude_close: bool = False,
) -> Optional[FileEvent]:
    """Get the most recent file event for a given path."""
    event_bus = await EventBus.get_instance()
    events = (
        event_bus.get_events_by_type(EventType.FILE_EVENT)
        if not agent_id
        else [
            e for e in event_bus.get_events(agent_id) if e.type == EventType.FILE_EVENT
        ]
    )

    file_events = [
        e
        for e in events
        if isinstance(e, FileEvent)
        and e.path == file_path
        and (e.operation != FileOperation.CLOSE if exclude_close else True)
    ]
    return file_events[-1] if file_events else None


async def get_file_content_size(agent_id: str | None = None) -> int:
    """Calculate total size of content from file events."""
    event_bus = await EventBus.get_instance()
    total_size = 0
    events = (
        event_bus.get_events_by_type(EventType.FILE_EVENT)
        if not agent_id
        else [
            e for e in event_bus.get_events(agent_id) if e.type == EventType.FILE_EVENT
        ]
    )

    for event in events:
        if isinstance(event, FileEvent):
            total_size += len(event.content.encode("utf-8"))
    return total_size


async def get_subagent_events(
    agent_id: str,
    event_types: Set[EventType] = set(EventType),
    # event_types: Set[EventType] = {
    #     EventType.ASSISTANT_MESSAGE,
    #     EventType.TOOL_RESULT,
    #     EventType.AGENT_RESULT,
    #     EventType.FILE_EVENT,
    #     EventType.EXTERNAL_MESSAGE,
    # },
) -> List[Event]:
    """Get events for prefilling assistant messages."""
    event_bus = await EventBus.get_instance()
    all_events = event_bus.get_events_in_chain(agent_id)
    return [e for e in all_events if e.type in event_types]
