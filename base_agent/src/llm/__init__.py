# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LLM integration module for the Self-Referential Agent System.

This module provides a unified interface for interacting with various LLM providers
including Anthropic, OpenAI, and DeepSeek.
"""

import logging

from .base import (
    Message,
    Completion,
    CompletionChunk,
    TimingInfo,
    TextContent,
    ToolResultContent,
)
from .api import create_completion, create_streaming_completion
from .metering import token_meter, get_total_cost

# Quieten LLM API call logs to make stdout more useful
logging.getLogger("httpx").setLevel(logging.WARNING)

__all__ = [
    "Message",
    "Completion",
    "CompletionChunk",
    "TimingInfo",
    "create_completion",
    "create_streaming_completion",
    "token_meter",
    "get_total_cost",
    "TextContent",
    "ToolResultContent",
]
