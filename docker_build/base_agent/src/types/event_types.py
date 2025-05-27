# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hashlib

from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import field, dataclass


class EventType(Enum):
    ASSISTANT_MESSAGE = "assistant_message"
    ASSISTANT_REASONING = "assistant_reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_CALL = "agent_call"
    AGENT_RESULT = "agent_result"
    CORE_PROMPT_UPDATE = "core_prompt_update"
    SYSTEM_PROMPT_UPDATE = "system_prompt_update"
    FILE_EVENT = "file_event"
    APPLICATION_ERROR = "application_error"
    APPLICATION_WARNING = "application_warning"
    PROBLEM_STATEMENT = "problem_statement"  # initial problem statement
    EXTERNAL_MESSAGE = "external_message"  # subsequent update messages
    OVERSEER_NOTIFICATION = "overseer_notification"
    OVERSEER_UPDATE = "overseer_update"  # for debugging
    BUDGET_INFO = "budget_info"
    TIMEOUT = "timeout"
    COST_LIMIT = "cost_limit"


@dataclass
class Event:
    """Base class for all events in the stream"""

    type: EventType
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class FileOperation(Enum):
    OPEN = "open"
    CLOSE = "close"
    EDIT = "edit"


@dataclass
class FileEvent:
    """Special event for file operations"""

    type: EventType
    content: str  # NOTE: this is the formatted content, not just the raw file content (e.g. with line numbers, content hash, lsp diagnostics, etc)
    operation: FileOperation
    path: str

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)  # NOTE: unused

    mtime: float = field(default=0.0)
    content_hash: str = field(default="")
    diff: str | None = None
    # lsp_diagnosdics: list = field(default_factory=list)

    def __post_init__(self):
        """Compute hash on initialization if not provided"""
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()

        if self.mtime == 0.0:
            try:
                self.mtime = Path(self.path).stat().st_mtime
            except Exception:
                pass

    def __hash__(self):
        return hash((self.type, self.operation, self.path, self.content_hash))

    def __eq__(self, other):
        if not isinstance(other, FileEvent):
            return False
        return (
            self.type == other.type
            and self.operation == other.operation
            and self.path == other.path
            and self.content_hash == other.content_hash
        )
