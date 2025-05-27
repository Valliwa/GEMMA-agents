# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from pathlib import Path
from pydantic import Field

from .base_tool import BaseTool
from ..events import EventBus
from ..events.event_bus_utils import get_open_file_set
from ..types.tool_types import ToolResult
from ..types.event_types import EventType, FileOperation, FileEvent
from ..types.agent_types import AgentInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenFile(BaseTool):
    TOOL_NAME = "open_files"
    TOOL_DESCRIPTION = """A file viewer tool that allows you to see the contents of one or more files in your context window.

Note, you should use /home/agent/workdir as your working directory if possible, althogh specifying file paths outside this directory will work.

Features:
- View multiple files at once
- Optional line number display for easier reference
- Automatic warning for non-text files

VERY IMPORTANT NOTE: you should only open files that are in a plain text format (e.g. something that you might open in a code editor). Opening media files, binary formats or any other non-text format will lead to unpredictable results.
"""

    file_paths: list[str] = Field(
        ...,
        description="A list of one or more absolute filepaths to add to open in your context window",
    )
    show_line_numbers: bool = Field(
        False,
        description="When True, displays line numbers in the left margin of the file for easier reference.",
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        try:
            output_strings = []
            warnings = []
            total_lines = 0
            for fpath in self.file_paths:
                path = Path(fpath)
                if not path.exists():
                    warnings.append(f"File path: {path} does not exist")
                    continue

                output_strings.append(f"The file at {path} was opened successfully.")

                file_content = path.read_text()
                event = FileEvent(
                    type=EventType.FILE_EVENT,
                    content=file_content,
                    path=str(fpath),
                    operation=FileOperation.OPEN,
                    metadata={"show_line_numbers": self.show_line_numbers},
                )
                total_lines += len(file_content.splitlines())

                event_bus = await EventBus.get_instance()
                await event_bus.publish(event, self._calling_agent._id)

            if total_lines > 750:
                warnings.append(f"You have added {total_lines} of content to the context, which is quite high. If this file is not immediately relevant to the task at hand, you should make sure to close it (and any other long files) with the close_file tool.")

            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=True,
                output="\n".join(output_strings) if output_strings else None,
                warnings="\n".join(warnings) if warnings else None,
            )
        except Exception as e:
            return ToolResult(tool_name=self.TOOL_NAME, success=False, errors=str(e))

    @classmethod
    def generate_examples(cls) -> list[tuple["OpenFile", ToolResult]]:
        from ..agents.implementations import DemoAgent

        return [
            (
                cls(
                    calling_agent=DemoAgent(),
                    file_paths=["/home/agent/workdir/example.txt"],
                    show_line_numbers=False,
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
            (
                cls(
                    calling_agent=DemoAgent(),
                    file_paths=["/tmp/example.txt", "/home/agent/workdir/new.txt"],
                    show_line_numbers=True,
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
        ]


class CloseFile(BaseTool):
    TOOL_NAME = "close_files"
    TOOL_DESCRIPTION = """Close one or more open files to clear up space in the context window.

Note that you can call this tool with the empty list [] as the file_paths to close all open files.
"""

    file_paths: list[str] = Field(
        ...,
        description="A list of one or more absolute file paths to close. If this is the empty list, then all files will be closed",
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        try:
            event_bus = await EventBus.get_instance()

            if len(self.file_paths) == 0:
                open_files = await get_open_file_set(self._calling_agent._id)
                for open_event in open_files:
                    close_event = FileEvent(
                        type=EventType.FILE_EVENT,
                        content="",
                        path=open_event.path,
                        operation=FileOperation.CLOSE,
                    )
                    await event_bus.publish(close_event, self._calling_agent._id)
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=True,
                )

            for fpath in self.file_paths:
                close_event = FileEvent(
                    type=EventType.FILE_EVENT,
                    content="",
                    path=fpath,
                    operation=FileOperation.CLOSE,
                )
                await event_bus.publish(close_event, self._calling_agent._id)

            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=True,
            )
        except Exception as e:
            return ToolResult(tool_name=self.TOOL_NAME, success=False, errors=str(e))

    @classmethod
    def generate_examples(cls) -> list[tuple["CloseFile", ToolResult]]:
        from ..agents.implementations import DemoAgent

        return [
            (
                cls(
                    calling_agent=DemoAgent(),
                    file_paths=["/home/agent/workdir/example.txt"],
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
            (
                cls(
                    calling_agent=DemoAgent(),
                    file_paths=[],
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
        ]
