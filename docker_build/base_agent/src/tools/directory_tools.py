# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from pathlib import Path
from pydantic import Field

from .base_tool import BaseTool
from ..utils.file_views import create_filetree, FileTreeOptions
from ..types.tool_types import ToolResult
from ..types.agent_types import AgentInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ViewDirectory(BaseTool):
    """Tool to generate a detailed view of directory contents."""

    TOOL_NAME = "view_directory"
    TOOL_DESCRIPTION = """View the contents of a directory with configurable depth and detail options.

The tool provides a formatted tree view of the directory structure, including:
- File and directory sizes
- Permissions
- Modification times
- Smart collapsing of large directories
- Configurable depth and detail level"""

    directory: str = Field(
        ...,
        description="The directory path to view",
    )
    max_depth: int = Field(
        default=2,
        description="Maximum depth to traverse (None for unlimited)",
    )
    show_hidden: bool = Field(
        default=False,
        description="Whether to show hidden files and directories",
    )
    collapse_threshold: int = Field(
        default=15,
        description="Number of items before a directory is collapsed (None for no collapsing)",
    )
    show_timestamps: bool = Field(
        default=False,
        description="Whether to show file modification timestamps",
    )
    exclude_patterns: list[str] = Field(
        default=[],
        description="List of glob patterns to exclude (e.g. '.git' or '*.pyc')",
    )
    show_full_filepaths: bool = Field(
        default=False,
        description="Whether to show the full filepaths from the root directory",
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        try:
            path = Path(self.directory)
            if not path.exists():
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors=f"Directory does not exist: {path}",
                )
            if not path.is_dir():
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors=f"Path is not a directory: {path}",
                )

            # Create options for the tree generation
            options = FileTreeOptions(
                collapse_threshold=self.collapse_threshold,
                show_hidden=self.show_hidden,
                exclude_patterns=(
                    self.exclude_patterns
                    if len(self.exclude_patterns) > 0 or self.show_hidden
                    else None
                ),
                show_mtime=self.show_timestamps,
                min_dir_level=(
                    0 if self.max_depth is None else max(0, self.max_depth - 1)
                ),
                show_full_path=self.show_full_filepaths,
            )

            # Generate the tree
            tree_output = create_filetree(path, options)

            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=True,
                output=f"Directory contents of {path}:\n{tree_output}",
            )

        except Exception as e:
            return ToolResult(tool_name=self.TOOL_NAME, success=False, errors=str(e))

    @classmethod
    def generate_examples(cls) -> list[tuple["ViewDirectory", ToolResult]]:
        from ..agents.implementations import DemoAgent

        return [
            # Basic directory view
            (
                cls(
                    calling_agent=DemoAgent(),
                    directory="/home/agent/workdir",
                    max_depth=2,
                    show_hidden=False,
                    show_timestamps=False,
                    exclude_patterns=[],
                    collapse_threshold=20,
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output="Directory contents of /home/agent/workdir:\n"
                    "workdir/ [0755] (1.2MB, 25 files, 5 dirs)\n"
                    "  src/ [0755] (800KB, 15 files, 3 dirs)\n"
                    "    main.py [0644] 50KB\n"
                    "    utils.py [0644] 30KB\n"
                    "  tests/ [0755] (400KB, 10 files, 2 dirs) [collapsed]\n",
                ),
            ),
            # Detailed view with timestamps
            (
                cls(
                    calling_agent=DemoAgent(),
                    directory="/home/agent/project",
                    max_depth=1,
                    show_hidden=True,
                    show_timestamps=True,
                    exclude_patterns=[".git", "*.pyc"],
                    collapse_threshold=15,
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output="Directory contents of /home/agent/project:\n"
                    "project/ [0755] (2.5MB, 40 files, 8 dirs) 2024-01-14 10:00\n"
                    "  .env [0644] 2KB 2024-01-14 09:55\n"
                    "  README.md [0644] 15KB 2024-01-14 09:50\n"
                    "  src/ [0755] (1.5MB, 25 files, 5 dirs) 2024-01-14 10:00\n"
                    "  tests/ [0755] (1MB, 15 files, 3 dirs) 2024-01-14 09:45\n",
                ),
            ),
        ]
