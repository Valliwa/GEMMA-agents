"""Gemma provider implementation."""

import logging
from typing import List, Optional, Union, AsyncGenerator, Type, Tuple
from datetime import datetime
from uuid import uuid4

from .base_provider import BaseProvider, Message, Completion, TokenUsage, TimingInfo, StopReason, ReasoningEffort
from ...types.llm_types import Model, TextContent
from ...types.tool_types import ToolInterface
from ...types.agent_types import AgentInterface

logger = logging.getLogger(__name__)

class GemmaProvider(BaseProvider):
    def __init__(self, client):
        # Import here to avoid circular imports
        from ...config import GemmaAPIClient
        self.client = GemmaAPIClient()
        
    async def create_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[Type[AgentInterface] | Type[ToolInterface]] | None = None,
        num_completions: int = 1,
    ) -> Completion:
        """Create a completion using Gemma."""
        start_time = datetime.now()
        
        # Convert messages to prompt
        prompt = self._prepare_messages(messages)
        
        try:
            # Generate response using our Gemma client
            response_text = self.client.generate(
                prompt, 
                max_tokens=max_tokens or 1024,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Gemma generation failed: {e}")
            response_text = f"Error generating response: {e}"
        
        end_time = datetime.now()
        
        # Create completion object
        completion = Completion(
            id=f"gemma_{uuid4().hex[:8]}",
            content=[TextContent(text=response_text)],
            model=model,
            usage=TokenUsage(
                uncached_prompt_tokens=len(prompt.split()),
                completion_tokens=len(response_text.split()),
                cache_write_prompt_tokens=0,
                cached_prompt_tokens=0,
            ),
            timing=TimingInfo(
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
                first_token_time=None,
                time_to_first_token=None,
                tokens_per_second=None,
            ),
            stop_reason=StopReason.COMPLETE,
            stop_sequence=None,
        )
        
        return completion
    
    def _prepare_messages(self, messages: List[Message]) -> str:
        """Prepare messages for Gemma API."""
        return self._format_messages(messages)
    
    def _format_messages(self, messages: List[Message]) -> str:
        """Convert messages to a prompt string."""
        prompt = ""
        for msg in messages:
            role = getattr(msg, 'role', 'user')
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                for content in msg.content:
                    if hasattr(content, 'text'):
                        prompt += f"{role}: {content.text}\n"
            else:
                content_text = getattr(msg, 'content', str(msg))
                prompt += f"{role}: {content_text}\n"
        return prompt
    
    def pydantic_to_native_tool(self, tool) -> dict:
        """Convert Pydantic tool to native format."""
        return {
            "name": getattr(tool, "TOOL_NAME", tool.__class__.__name__),
            "description": getattr(tool, "TOOL_DESCRIPTION", ""),
            "parameters": {}
        }
    
    def map_stop_reason(self, response) -> Tuple[StopReason, Optional[str]]:
        """Map stop reason from response."""
        return (StopReason.COMPLETE, None)
        
    async def create_streaming_completion(self, *args, **kwargs):
        """Streaming not implemented for Gemma."""
        raise NotImplementedError("Streaming not implemented for Gemma")
