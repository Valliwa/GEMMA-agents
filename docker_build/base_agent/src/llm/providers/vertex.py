# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Vertex AI specific implementation for Anthropic Claude models."""

import time
import logging
import asyncio

from typing import List, Optional, Union, AsyncGenerator, Type
from datetime import datetime

from pydantic import BaseModel
from anthropic.types import Message as AntMessage

from .base_provider import (
    Message,
    Completion,
    CompletionChunk,
    TimingInfo,
    BaseProvider,
)
from ...types.llm_types import TokenUsage, FCI, Model, StopReason, ReasoningEffort, TextContent, ReasoningContent, ToolCallContent, ToolResultContent, ContentTypes
from ...types.agent_types import AgentInterface
from ...types.tool_types import ToolInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Initialize tokenizer for token estimation if available
try:
    import tiktoken
    TOKENIZER = tiktoken.encoding_for_model("cl100k_base")  # Claude uses cl100k_base encoding
except Exception:
    TOKENIZER = None


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in the text."""
    if TOKENIZER:
        return len(TOKENIZER.encode(text))
    else:
        # Rough approximation: 4 characters per token on average
        return len(text) // 4


def get_suffix_difference(shorter: str, longer: str) -> str:
    """For debugging cache breaks"""
    # Find the index where the prefix ends
    common_length = len(shorter)

    if not longer.startswith(shorter):
        # Find actual common prefix length if shorter is not fully a prefix
        common_length = next((i for i, (a, b) in enumerate(zip(shorter, longer)) if a != b), 0)

    return longer[common_length:]


class VertexProvider(BaseProvider):
    """Provider implementation for Anthropic's Claude models hosted on Google Vertex AI."""

    def __init__(self, client, max_retries: int = 3, retry_base_delay: float = 1.0):
        self.client = client
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._last_sys = ""
        self._last_msg = ""

    def map_stop_reason(self, response: AntMessage) -> tuple[StopReason, Optional[str]]:
        """Map Anthropic-specific stop reasons to our standard format.

        Returns the standardised stop reason, and optionally the stop sequence
        that caused it.
        """
        # Handle streaming chunks
        if hasattr(response, "type") and response.type == "message_delta":
            if hasattr(response.delta, "stop_reason"):
                raw_stop = response.delta.stop_reason
            else:
                return StopReason.COMPLETE, None

        raw_stop_reason = response.stop_reason
        if raw_stop_reason == "end_turn":
            return StopReason.COMPLETE, None  # Natural termination
        elif raw_stop_reason == "max_tokens":
            return StopReason.LENGTH, None
        elif raw_stop_reason == "stop_sequence":
            return StopReason.STOP_TOKEN, response.stop_sequence
        elif raw_stop_reason == "tool_use":
            return StopReason.TOOL_CALL, None
        else:
            return StopReason.ERROR, None

    def _content_mapping(self, block: ContentTypes) -> dict:
        """Maps our message content types, into provider-specific message formats"""
        if isinstance(block, TextContent):
            return {"type": "text", "text": block.text,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
        elif isinstance(block, ReasoningContent):
            return {"type": "thinking", "thinking": block.text,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
        elif isinstance(block, ToolCallContent):
            return {"type": "tool_use", "id": block.call_id, "name": block.tool_name, "input": block.tool_args,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
        elif isinstance(block, ToolResultContent):
            return {"type": "tool_result", "tool_use_id": block.call_id, "content": block.content,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
        else:
            raise ValueError(f"Unhandled content type in provider Vertex: {block}")

    def _prepare_messages(
        self, messages: list[Message]
    ) -> tuple[Optional[list[dict]], list[dict]]:
        """Process and prepare messages for the Anthropic API.

        Anthropic has specific requirements for message formatting:
        1. System messages must be separated and structured
        2. Assistant messages need content trimming
        3. Names and cache control flags must be properly formatted

        Returns:
            Tuple of (system_content, anthropic_messages)
        """
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = [self._content_mapping(block) for block in msg.content]
            else:
                msg_content = [self._content_mapping(block) for block in msg.content]
                anthropic_messages.append(
                    {
                        "role": msg.role,
                        "content": msg_content,
                        **({"name": msg.name} if msg.name else {}),
                    }
                )

        for i in range(len(anthropic_messages)-1, 0, -1):
            # Remove potential empty message blocks from the end
            while (len(anthropic_messages[i]["content"]) > 0 and
                   anthropic_messages[i]["content"][-1]["type"] == "text" and
                   anthropic_messages[i]["content"][-1]["text"] == ""):
                anthropic_messages[i]["content"].pop()

            if len(anthropic_messages[i]["content"]) == 0:
                anthropic_messages.pop(i)

            elif anthropic_messages[i]["content"][-1]["type"] == "text" and anthropic_messages[i]["content"][-1]["text"] != "":
                break

        # Ensure the last assistant message in the prefill doesn't have
        # trailing whitespace
        if len(anthropic_messages) > 0 and anthropic_messages[-1]["role"] == "assistant":
            for block in reversed(anthropic_messages[-1]["content"]):
                if block["type"] == "text":
                    block["text"] = block["text"].rstrip()
                    break

        # Default to setting a cache break point at the last message
        if len(anthropic_messages) > 0 and len(anthropic_messages[-1]["content"]) > 0:
            anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        if isinstance(system_content, list) and len(system_content) > 0:
            system_content[-1]["cache_control"] = {"type": "ephemeral"}

        return system_content, anthropic_messages

    def _create_token_usage(self, usage_data) -> TokenUsage:
        """Create TokenUsage object from response usage data."""

        empty_usage = TokenUsage()
        if not usage_data:
            return empty_usage

        try:
            usage_dict = (
                usage_data.model_dump()
                if hasattr(usage_data, "model_dump")
                else vars(usage_data)
            )
            return TokenUsage(
                uncached_prompt_tokens=usage_dict.get("input_tokens", 0),
                cache_write_prompt_tokens=usage_dict.get(
                    "cache_creation_input_tokens", 0
                ),
                cached_prompt_tokens=usage_dict.get("cache_read_input_tokens", 0),
                completion_tokens=usage_dict.get("output_tokens", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to create token usage: {e}")
            return empty_usage

    def pydantic_to_native_tool(self, tool: ToolInterface | AgentInterface) -> dict:
        if hasattr(tool, 'TOOL_NAME') and hasattr(tool, 'TOOL_DESCRIPTION'):
            return {
                "name": tool.TOOL_NAME,
                "description": tool.TOOL_DESCRIPTION,
                "input_schema": tool.model_json_schema()
            }
        elif hasattr(tool, 'AGENT_NAME') and hasattr(tool, 'AGENT_DESCRIPTION'):
            return {
                "name": tool.AGENT_NAME,
                "description": tool.AGENT_DESCRIPTION,
                "input_schema": tool.model_json_schema()
            }
        else:
            # Fall back to guess
            return {
                "name": tool.__class__.__name__,
                "description": "No description provided.",
                "input_schema": tool.model_json_schema()
            }

        raise ValueError("tool is not BaseTool or BaseAgent")

    def _reasoning_budgets_to_tokens(self, reasoning_effort: ReasoningEffort) -> int:
        if reasoning_effort == ReasoningEffort.LOW:
            return 16000
        elif reasoning_effort == ReasoningEffort.MEDIUM:
            return 32000
        elif reasoning_effort == ReasoningEffort.HIGH:
            return 60000

    def _estimate_request_tokens(self, messages: List[Message]) -> int:
        """
        Estimate the number of tokens in the messages.

        Args:
            messages: The list of messages to estimate tokens for

        Returns:
            int: Estimated token count
        """
        total_tokens = 0

        for msg in messages:
            # Add tokens for role
            total_tokens += 4  # Average tokens used for role

            # Add tokens for each content block
            for block in msg.content:
                if hasattr(block, "text"):
                    total_tokens += estimate_token_count(block.text)
                elif hasattr(block, "tool_name"):
                    # Tool calls have a JSON structure
                    tool_tokens = (
                        estimate_token_count(block.tool_name) +
                        estimate_token_count(str(block.tool_args))
                    )
                    total_tokens += tool_tokens + 10  # Extra tokens for JSON structure

        # Add a buffer for any metadata or other tokens
        return int(total_tokens * 1.1)  # 10% buffer

    async def create_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[Type[BaseModel]] | None = None,
        num_completions: int = 1,
    ) -> Completion:
        """Create a completion using Vertex's Anthropic API."""
        for attempt in range(self.max_retries + 1):
            try:
                start_time = datetime.now()

                logger.debug(f"Creating completion with stop tokens: {stop}")

                # Process messages
                system_content, anthropic_messages = self._prepare_messages(messages)

                # Prepare arguments
                args = {
                    "messages": anthropic_messages,
                    "model": model.id,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens or model.max_output_tokens,
                }

                if system_content:
                    args["system"] = system_content

                if stop:
                    stop_sequences = [stop] if isinstance(stop, str) else stop
                    args["stop_sequences"] = stop_sequences

                if available_tools and model.fci == FCI.CONSTRAINED:
                    args["tools"] = [self.pydantic_to_native_tool(t) for t in available_tools]

                if model.is_reasoner and reasoning_effort is not None:
                    args["temperature"] = 1.0
                    del args["top_p"]
                    args["extra_body"] = {
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": self._reasoning_budgets_to_tokens(reasoning_effort),
                        }
                    }

                # Execute the request
                response = await self.client.messages.create(**args)

                end_time = datetime.now()

                # Extract usage information - note that Vertex may have different usage reporting
                usage_info = self._create_token_usage(response.usage)
                logger.debug(usage_info)

                duration = end_time - start_time
                tokens_per_second = (
                    usage_info.completion_tokens / duration.total_seconds()
                    if usage_info and duration.total_seconds() > 0
                    else None
                )

                timing_info = TimingInfo(
                    start_time=start_time,
                    end_time=end_time,
                    total_duration=duration,
                    first_token_time=None,
                    time_to_first_token=None,
                    tokens_per_second=tokens_per_second,
                )

                # Create cache metrics if available
                cache_metrics = None
                if usage_info:
                    cache_metrics = {
                        "cache_hits": usage_info.cached_prompt_tokens,
                        "cache_misses": usage_info.uncached_prompt_tokens,
                        "cache_writes": usage_info.cache_write_prompt_tokens,
                    }

                logger.debug(f"\n\nUsage: {response.usage}\n{cache_metrics}\n\n")

                # Map stop reason
                stop_reason, stop_sequence = self.map_stop_reason(response)

                response_content = []
                for block in response.content:
                    match block.type:
                        case 'text':
                            response_content.append(TextContent(text=block.text))
                        case 'tool_use':
                            response_content.append(
                                ToolCallContent(
                                    call_id=block.id, tool_name=block.name, tool_args=block.input,
                                    call_type=FCI.CONSTRAINED,
                                )
                            )
                        case 'thinking':
                            response_content.append(
                                ReasoningContent(text=block.thinking)
                            )
                        case _:
                            logger.warning(
                                f"Unhandled response block type {block.type} in Vertex completion"
                            )

                logger.debug(f"\n\nVERTEX CACHE METRICS:\n{cache_metrics}\n\n\n")

                completion = Completion(
                    id=response.id,
                    content=response_content,
                    model=model,
                    usage=usage_info,
                    timing=timing_info,
                    cache_metrics=cache_metrics,
                    stop_reason=stop_reason,
                    stop_sequence=stop_sequence,
                )

                # Store raw response data
                completion.raw_response = {
                    "stop_reason": getattr(response, "stop_reason", None),
                    "stop_sequence": getattr(response, "stop_sequence", None),
                }

                return completion

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit_error = "rate_limit" in error_msg or "rate limit" in error_msg

                if is_rate_limit_error and attempt < self.max_retries:
                    # Calculate backoff delay with exponential backoff and jitter
                    delay = self.retry_base_delay * (2 ** attempt) * (0.5 + 0.5 * time.time() % 1)
                    logger.warning(f"Rate limit exceeded, retrying in {delay:.2f} seconds (attempt {attempt+1}/{self.max_retries})")
                    print(f"\n\n\nRate limit exceeded, retrying in {delay:.2f} seconds (attempt {attempt+1}/{self.max_retries})\n\n\n")
                    await asyncio.sleep(delay)
                    continue

                # Re-raise other errors or if we've exhausted retries
                logger.error(f"Error during completion: {str(e)}")
                raise

    async def create_streaming_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[Type[BaseModel]] | None = None,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Create a streaming completion using Vertex's Anthropic API."""
        start_time = datetime.now()
        first_token_time = None
        message_id = None
        final_usage = None
        current_stop_reason = None
        current_stop_sequence = None
        usage_info = None
        initial_prompt_tokens = 0  # Track initial prompt tokens

        # Process messages
        system_content, anthropic_messages = self._prepare_messages(messages)

        # Prepare arguments
        args = {
            "messages": anthropic_messages,
            "model": model.id,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens or model.max_output_tokens,
            "stream": True,
        }

        if system_content:
            args["system"] = system_content

        if stop:
            stop_sequences = [stop] if isinstance(stop, str) else stop
            args["stop_sequences"] = stop_sequences

        # Track cache metrics
        cache_metrics = None

        async for stream_resp in await self.client.messages.create(**args):
            current_time = datetime.now()

            # Handle message start
            if stream_resp.type == "message_start":
                message_id = stream_resp.message.id
                # Get initial usage info and store prompt tokens
                if stream_resp.message.usage:
                    usage_info = stream_resp.message.usage
                    initial_prompt_tokens = usage_info.input_tokens
                    cache_metrics = {
                        "cache_hits": usage_info.cache_read_input_tokens,
                        "cache_misses": usage_info.input_tokens - usage_info.cache_read_input_tokens,
                        "cache_writes": usage_info.cache_creation_input_tokens,
                    }
                continue

            # Handle content updates
            elif stream_resp.type == "content_block_delta":
                if stream_resp.delta.type == "text_delta" and stream_resp.delta.text:
                    content = stream_resp.delta.text
                    if first_token_time is None:
                        first_token_time = current_time

                    timing_info = TimingInfo(
                        start_time=start_time,
                        end_time=current_time,
                        total_duration=current_time - start_time,
                        first_token_time=first_token_time,
                        time_to_first_token=(
                            (first_token_time - start_time).total_seconds()
                            if first_token_time
                            else None
                        ),
                        tokens_per_second=None,  # Calculate in final chunk
                    )

                    yield CompletionChunk(
                        id=message_id,
                        content=content,
                        model=model,
                        is_finished=False,
                        timing=timing_info,
                        usage=None,  # Usage only in final chunk
                    )

            # Handle message updates
            elif stream_resp.type == "message_delta":
                # Update usage info
                if hasattr(stream_resp, "usage"):
                    usage_info = stream_resp.usage
                current_stop_reason, current_stop_sequence = self.map_stop_reason(
                    stream_resp
                )

            # Handle final chunk
            elif stream_resp.type == "message_stop":
                # Process final usage info with initial prompt tokens
                usage_dict = (
                    usage_info.model_dump()
                    if hasattr(usage_info, "model_dump")
                    else vars(usage_info)
                )
                final_usage = TokenUsage(
                    uncached_prompt_tokens=initial_prompt_tokens,  # Use stored prompt tokens
                    cache_write_prompt_tokens=usage_dict.get(
                        "cache_creation_input_tokens", 0
                    ),
                    cached_prompt_tokens=usage_dict.get("cache_read_input_tokens", 0),
                    completion_tokens=usage_dict.get("output_tokens", 0),
                )

                # Calculate timing
                duration = current_time - start_time
                tokens_per_second = (
                    final_usage.completion_tokens / duration.total_seconds()
                    if final_usage and duration.total_seconds() > 0
                    else None
                )

                timing_info = TimingInfo(
                    start_time=start_time,
                    end_time=current_time,
                    total_duration=duration,
                    first_token_time=first_token_time,
                    time_to_first_token=(
                        (first_token_time - start_time).total_seconds()
                        if first_token_time
                        else None
                    ),
                    tokens_per_second=tokens_per_second,
                )

                # Send final chunk
                completion = CompletionChunk(
                    id=message_id,
                    content="",  # Empty content in final chunk
                    model=model,
                    is_finished=True,
                    timing=timing_info,
                    usage=final_usage,
                    cache_metrics=cache_metrics,
                    stop_reason=current_stop_reason,
                    raw_response={
                        "stop_reason": (
                            current_stop_reason.value if current_stop_reason else None
                        ),
                        "stop_sequence": current_stop_sequence,
                    },
                )

                yield completion
