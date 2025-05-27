# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A module of Agent tools
"""

from .base_tool import BaseTool, tool_registry
from .file_tools import CloseFile, OpenFile
from .edit_tools import OverwriteFile
from .execute_command import ExecuteCommand
from .directory_tools import ViewDirectory
from .ripgrep_tool import RipGrepTool

# TODO: expand the concept of toolkits and use throughout the agent implementations
toolkits: dict[str, list[BaseTool]] = dict(
    coding=[
        ViewDirectory,
        ExecuteCommand,
        OpenFile,
        CloseFile,
        OverwriteFile,
        RipGrepTool,
    ]
)
