"""
Google provider - Disabled for Gemma-only setup
This prevents import errors while keeping the framework intact
"""
import logging
from typing import Any, List, Optional, Union, AsyncGenerator, Type
from datetime import datetime

# Import base classes to maintain compatibility
from .base_provider import (
    Message,
    Completion,
    CompletionChunk,
    TokenUsage,
    TimingInfo,
    StopReason,
    BaseProvider,
    ReasoningEffort,
)

logger = logging.getLogger(__name__)

class GoogleProvider(BaseProvider):
    """Disabled Google provider - using Gemma instead"""
    
    def __init__(self, *args, **kwargs):
        logger.info("Google provider disabled - using Gemma instead")
        self._disabled = True
    
    def map_stop_reason(self, *args, **kwargs):
        """Disabled method"""
        return StopReason.COMPLETE, None
    
    async def create_completion(self, *args, **kwargs):
        """Disabled method"""
        raise NotImplementedError("Google provider disabled - using Gemma instead")
    
    async def create_streaming_completion(self, *args, **kwargs):
        """Disabled method"""
        raise NotImplementedError("Google provider disabled - using Gemma instead")
    
    def pydantic_to_native_tool(self, *args, **kwargs):
        """Disabled method"""
        return None
    
    def tool_name(self, *args, **kwargs):
        """Disabled method"""
        return "disabled_tool"

# Maintain API compatibility
def create_google_provider(*args, **kwargs):
    return GoogleProvider(*args, **kwargs)
