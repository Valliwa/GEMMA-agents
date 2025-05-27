# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import difflib

from pathlib import Path
from datetime import datetime, timedelta

from ...types.tool_types import ToolResult
from ...types.agent_types import AgentInterface, InheritanceFlags
from ...events.event_bus_utils import is_file_open, get_latest_file_event


async def edit_preflight_check(
    path: Path, tool_name: str, calling_agent: AgentInterface
) -> ToolResult | None:
    inherits_parent_files = (
        InheritanceFlags.OPEN_FILES in calling_agent.INHERITANCE.flags
    )

    file_open: bool = await is_file_open(str(path), calling_agent._id)
    if inherits_parent_files and not file_open:
        file_open = await is_file_open(str(path), calling_agent._parent_id)

    # Verify file is open
    if not file_open:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            errors=f"File {path} must be opened first using the open_file tool",
        )

    eps = timedelta(seconds=0.5)
    latest_file_event = await get_latest_file_event(
        str(path), calling_agent._id, exclude_close=True
    )
    # Assumes agent runs are blocking (i.e. all agent file events will be newer
    # than parent file events)
    if inherits_parent_files and not latest_file_event:
        latest_file_event = await get_latest_file_event(
            str(path),
            calling_agent._parent_id,
            exclude_close=True,
        )

    last_mod = datetime.fromtimestamp(path.stat().st_mtime)
    if not latest_file_event or last_mod > latest_file_event.timestamp + eps:
        last_viewed = (
            latest_file_event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if latest_file_event
            else "Never"
        )
        return ToolResult(
            tool_name=tool_name,
            success=False,
            errors=(
                f"File {path} was changed at {last_mod.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"which is after you last viewed or edited it at {last_viewed}."
                "Please view it again to get its latest contents before making your edit."
            ),
        )


def generate_edit_event_content(
    old_content: str, new_content: str, path: str
) -> tuple[str, str]:
    """Generate a diff between old and new content for file events.

    Returns:
        tuple[str, str]: A tuple containing (content_for_event, content_hash)
        where content_for_event contains the diff and content_hash is the hash of new_content
    """
    if not old_content and new_content:
        # For new files, return the full content
        content_hash = hashlib.sha256(new_content.encode()).hexdigest()
        return new_content, content_hash

    # Generate unified diff
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )

    if diff:
        diff_content = "\n".join(diff)
    else:
        diff_content = "No changes"

    content_hash = hashlib.sha256(new_content.encode()).hexdigest()
    return diff_content, content_hash
