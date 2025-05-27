# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Base models and shared functionality for LLM interactions."""

from typing import Dict, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from ..types.llm_types import TokenUsage, Model, StopReason, TextContent, ReasoningContent, ToolCallContent, ToolResultContent, ContentTypes

# NOTE: perhaps move the rest of these classes to the llm_types for consistency


class Message(BaseModel):
    """A message in a conversation with an LLM."""

    role: str
    content: list[ContentTypes]
    name: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"Message from role={self.role}"]
        for c in self.content:
            if isinstance(c, TextContent):
                parts.append(f"Text {'-'*10}\n{c.text}")
            elif isinstance(c, ReasoningContent):
                parts.append(f"Reasoning {'-'*10}\n{c.text}")
            elif isinstance(c, ToolCallContent):
                parts.append(f"{'-'*10}\nTool call {c.tool_name} (id: {c.call_id}) {c.call_type}: {str(c.tool_args)}\n{'-'*10}")
            elif isinstance(c, ToolResultContent):
                parts.append(f"{'-'*10}\nTool result {c.tool_name} (id: {c.call_id}): {c.content}\n{'-'*10}")
        # return "\n".join([p.replace("\n", "").strip() for p in parts])
        return "\n".join(parts)


class TimingInfo(BaseModel):
    """Timing information for LLM interactions."""

    start_time: datetime = Field(description="When the request started")
    end_time: datetime = Field(description="When the response completed")
    total_duration: timedelta = Field(description="Total duration of the request")
    first_token_time: Optional[datetime] = Field(
        None, description="When the first token was received"
    )
    time_to_first_token: Optional[float] = Field(
        None, description="Duration until first token received"
    )
    tokens_per_second: Optional[float] = Field(
        None, description="Average tokens per second for completion"
    )

    def __str__(self) -> str:
        # Format datetime fields to a readable format
        fmt = "%Y-%m-%d %H:%M:%S"
        parts = [
            f"- Start {self.start_time.strftime(fmt)}, End {self.end_time.strftime(fmt)}",
            f"- Duration: {self.total_duration}",
        ]
        if self.time_to_first_token is not None:
            parts.append(f"- TTFT: {self.time_to_first_token:.2f} sec")
        if self.tokens_per_second is not None:
            parts.append(f"- TPS: {self.tokens_per_second:.2f}")
        return "\n".join(parts)

class CacheMetrics(BaseModel):
    """Cache-related metrics."""

    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    cache_writes: int = Field(default=0, description="Number of cache writes")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, int]] = None) -> "CacheMetrics":
        """Create metrics from dictionary, preserving provider values."""
        if data is None:
            data = {"cache_hits": 0, "cache_misses": 0, "cache_writes": 0}
        return cls(
            cache_hits=data.get("cache_hits", 0),
            cache_misses=data.get("cache_misses", 0),
            cache_writes=data.get("cache_writes", 0),
        )

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return self.model_dump()  # Use model_dump instead of dict


# Completion Types ============================================================

class Completion(BaseModel):
    """A completion response from an LLM."""

    id: str
    content: list[ContentTypes] | list[list[ContentTypes]]
    model: Model  # Model identifier string
    usage: TokenUsage
    timing: TimingInfo
    cache_metrics: Optional[Dict[str, int]] = None
    stop_reason: StopReason | list[StopReason] = StopReason.COMPLETE
    stop_sequence: Optional[str] | list[StopReason] = None
    continuation_count: Optional[int] = None
    raw_response: Optional[Dict] = Field(default=None, exclude=True)

    @property
    def finished_early(self) -> bool:
        """Check if completion stopped before finishing normally."""
        return self.stop_reason != StopReason.COMPLETE

    @property
    def hit_token_limit(self) -> bool:
        """Check if completion stopped due to token length."""
        return self.stop_reason == StopReason.LENGTH

    @property
    def errored(self) -> bool:
        """Check if completion encountered an error."""
        return self.stop_reason == StopReason.ERROR

    def get_cache_metric(self, key: str, default: int = 0) -> int:
        """Get a cache metric value safely."""
        if self.cache_metrics is None:
            return default
        return self.cache_metrics.get(key, default)

    def calculate_cost(self) -> float:
        """Calculate the cost for this completion."""
        return self.usage.calculate_cost(self.model.token_cost)

    def __str__(self) -> str:
        comp_str = f"{'='*80}\n"
        if isinstance(self.content[0], list):
            for i, completion in enumerate(self.content):
                comp_str += f"Candidate {i:03d} {70*'-'}\n"
                for block in completion:
                    comp_str += str(block) + "\n"
        else:
            for block in self.content:
                comp_str += str(block) + "\n"
        comp_str += f"\n{'-'*80}\n"
        comp_str += f"Model: {self.model.id}\n"
        comp_str += f"""Tokens used:
- Input {self.usage.input_tokens} (cached: {self.usage.cached_prompt_tokens}, written to cache: {self.usage.cache_write_prompt_tokens})
- Completion {self.usage.completion_tokens}
- Total {self.usage.total_tokens}
"""
        if self.stop_reason != StopReason.COMPLETE:
            comp_str += f"Stop reason: {self.stop_reason}\n"
            if self.stop_sequence:
                comp_str += f"Stop sequence: {self.stop_sequence}\n"

        if self.continuation_count:
            comp_str += f"Continuations: {self.continuation_count}\n"

        if self.timing:
            comp_str += f"Timing:\n{self.timing}\n"

        comp_str += f"Cost: ${self.calculate_cost():.6f}\n"

        comp_str += f"{'='*80}\n"
        return comp_str


class CompletionChunk(BaseModel):
    """A streaming chunk of a completion response."""

    id: str
    content: str  # TODO: make tool call or assistant message string
    model: Model  # Model identifier string
    is_finished: bool = False
    timing: Optional[TimingInfo] = None
    usage: Optional[TokenUsage] = None
    cache_metrics: Optional[Dict[str, int]] = None
    stop_reason: Optional[StopReason] = None
    continuation_count: Optional[int] = None
    raw_response: Optional[Dict] = Field(default=None, exclude=True)

    @property
    def finished_early(self) -> bool:
        """Check if completion stopped before finishing normally."""
        return bool(self.stop_reason and self.stop_reason != StopReason.COMPLETE)

    @property
    def hit_token_limit(self) -> bool:
        """Check if completion stopped due to token length."""
        return bool(self.stop_reason and self.stop_reason == StopReason.LENGTH)

    @property
    def errored(self) -> bool:
        """Check if completion encountered an error."""
        return bool(self.stop_reason and self.stop_reason == StopReason.ERROR)

    def get_cache_metric(self, key: str, default: int = 0) -> int:
        """Get a cache metric value safely."""
        if self.cache_metrics is None:
            return default
        return self.cache_metrics.get(key, default)

    def model_dump(self, **kwargs) -> Dict:
        """Override model_dump to exclude raw_response by default."""
        kwargs.setdefault("exclude", {"raw_response"})
        return super().model_dump(**kwargs)
