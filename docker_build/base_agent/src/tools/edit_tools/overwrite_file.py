# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import logging

from pathlib import Path
from pydantic import Field

from .utils import edit_preflight_check, generate_edit_event_content
from ...schemas.json_parsing import json_str_to_dict
from ..base_tool import BaseTool, extract_between_patterns
from ...events import EventBus
from ...types.tool_types import ToolResult
from ...types.event_types import EventType, FileOperation, FileEvent
from ...types.agent_types import AgentInterface
from ...types.common import ArgFormat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OverwriteFile(BaseTool):
    """Tool to overwrite an existing file or create a new one with content."""

    TOOL_NAME = "overwrite_file"
    TOOL_DESCRIPTION = f"""Use this tool when you want to write content verbatim to a file, either overwriting an existing file or creating a new one.

For existing files:
- You MUST have called the `open_file` tool to view the file before over-writing it
- This is to make sure we're not over-writing anything of value that needs to be kept
- The entire content will be replaced verbatim with the new content provided

For new files:
- 'Overwriting' a not-yet-existing file will create it
- The file will be automatically opened in the context window after creation

Very important notes:
- The content you provide to this tool will be that file's new content. You must make sure to include absolutely everything you still need
- Do NOT "fold" any code sections because this will cause errors. Instead, write out everything verbatim.

- DO NOT, under any circumstances, call this tool for a file edit that exceeds about 500 lines. It will be slow, inefficient, costly and error-prone. For these types of large-file edits, you should seek to use more efficient editing tools.
- You do not need to write out the file ahed of time before invoking this tool.
"""

    filepath: str = Field(
        ...,
        description="The full absolute filepath of the file to write. For existing files, must be already open in context window.",
    )
    full_unabridged_new_content: str = Field(
        ...,
        description="The full content to write to the file, which will entirely replace any existing content.",
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    @classmethod
    async def args_str_to_dict(
        cls, args_str: str, arg_format: ArgFormat = ArgFormat.XML
    ) -> tuple[dict | None, str | None]:
        if arg_format == ArgFormat.XML:
            # Carefully extract the content, with the assumption that there _will_
            # be conflicting tags.
            # First, manually get the content between <filepath>
            filepath_pattern = f"<filepath>(.*?)</filepath>"
            filepath_match = re.search(filepath_pattern, args_str)
            filepath = filepath_match.group(1) if filepath_match else None
            if not filepath:
                return None, "Could not parse filepath"

            # Find the first <content> opening tag
            content = extract_between_patterns(
                args_str, "<full_unabridged_new_content>", "</full_unabridged_new_content>"
            )
            if not content:
                return None, "Could not parse file content"

            return dict(filepath=filepath, full_unabridged_new_content=content), None
        else:
            return await json_str_to_dict(args_str, guide_obj=cls)

    async def run(self) -> ToolResult:
        try:
            path = Path(self.filepath)
            event_bus = await EventBus.get_instance()

            # Check if file exists
            file_exists = path.exists()

            if not file_exists:
                # Create directory structure if needed
                path.parent.mkdir(parents=True, exist_ok=True)

                # For new files, write content first
                try:
                    path.write_text(self.full_unabridged_new_content)

                    event = FileEvent(
                        type=EventType.FILE_EVENT,
                        content=self.full_unabridged_new_content,
                        path=str(path),
                        operation=FileOperation.OPEN,
                    )

                    await event_bus.publish(event, self._calling_agent._id)

                    return ToolResult(
                        tool_name=self.TOOL_NAME,
                        success=True,
                        output=f"Successfully created new file {path}",
                    )
                except Exception as e:
                    return ToolResult(
                        tool_name=self.TOOL_NAME,
                        success=False,
                        errors=f"Failed to create new file {path}: {str(e)}",
                    )
            else:
                # For existing files, verify it's open first
                result = await edit_preflight_check(
                    path, self.TOOL_NAME, self._calling_agent
                )
                if result:
                    return result

                prev_content = path.read_text()

                # Now write new content
                try:
                    path.write_text(self.full_unabridged_new_content)

                    diff_content, content_hash = generate_edit_event_content(
                        prev_content, self.full_unabridged_new_content, str(path)
                    )

                    event = FileEvent(
                        type=EventType.FILE_EVENT,
                        content=diff_content,
                        path=str(path),
                        operation=FileOperation.EDIT,
                        content_hash=content_hash,
                        mtime=path.stat().st_mtime,
                    )

                    await event_bus.publish(event, self._calling_agent._id)

                    return ToolResult(
                        tool_name=self.TOOL_NAME,
                        success=True,
                        output=f"Successfully overwrote content of {path}",
                    )
                except Exception as e:
                    return ToolResult(
                        tool_name=self.TOOL_NAME,
                        success=False,
                        errors=f"Failed to write to file {path}: {str(e)}",
                    )

        except Exception as e:
            return ToolResult(tool_name=self.TOOL_NAME, success=False, errors=str(e))

    @classmethod
    def generate_examples(cls) -> list[tuple["OverwriteFile", ToolResult]]:
        from ...agents.implementations import DemoAgent

        return [
            # Example 1: Create new file
            (
                cls(
                    calling_agent=DemoAgent(),
                    filepath="/home/agent/workdir/new_file.txt",
                    full_unabridged_new_content="Content for the new file",
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output="Successfully created new file /home/agent/workdir/new_file.txt",
                ),
            ),
            # Example 2: Overwrite existing file
            (
                cls(
                    calling_agent=DemoAgent(),
                    filepath="/home/agent/workdir/example.txt",
                    full_unabridged_new_content="New content for existing file",
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output="Successfully overwrote content of /home/agent/workdir/example.txt",
                ),
            ),
        ]
