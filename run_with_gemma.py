#!/usr/bin/env python3
"""
Final Complete Gemma Wrapper Script
This replaces the entire anthropic package with Gemma integration
"""

import os
import sys
import types
import requests
import logging
import argparse
import asyncio
import inspect

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_fake_anthropic_module():
    """Create a complete fake anthropic package that routes to Gemma"""
    
    # Create fake anthropic module
    fake_anthropic = types.ModuleType('anthropic')
    
    class MessageHandler:
        """Handler for client.messages.create() calls"""
        
        def __init__(self, client):
            self.client = client
        
        def create(self, **kwargs):
            """Handle client.messages.create() calls"""
            logger.info("🎯 WRAPPER: messages.create() called -> Gemma")
            return self.client._call_gemma(**kwargs)
        
        def __call__(self, **kwargs):
            """Handle direct client.messages() calls"""
            logger.info("🎯 WRAPPER: messages() called -> Gemma")
            return self.client._call_gemma(**kwargs)
    
    class AsyncMessageHandler:
        """Async handler for client.messages.create() calls"""
        
        def __init__(self, client):
            self.client = client
        
        async def create(self, **kwargs):
            """Handle async client.messages.create() calls"""
            logger.info("🎯 WRAPPER: async messages.create() called -> Gemma")
            loop = asyncio.get_event_loop()
            # Fix: Pass kwargs as a single argument, not as keyword arguments to run_in_executor
            return await loop.run_in_executor(None, lambda: self.client._call_gemma(**kwargs))
        
        async def __call__(self, **kwargs):
            """Handle direct async client.messages() calls"""
            logger.info("🎯 WRAPPER: async messages() called -> Gemma")
            return await self.create(**kwargs)
    
    class GemmaAnthropicClient:
        """Fake Anthropic client that actually calls Gemma"""
        
        def __init__(self, *args, **kwargs):
            self.base_url = "http://localhost:8000"
            self.session = requests.Session()
            logger.info("🎯 WRAPPER: Anthropic client created -> Using Gemma")
            
            # Create message handler object with create method
            self.messages = MessageHandler(self)
        
        def _call_gemma(self, **kwargs):
            """Internal method to call Gemma API"""
            messages = kwargs.get('messages', [])
            max_tokens = kwargs.get('max_tokens', 1024)
            temperature = kwargs.get('temperature', 0.7)
            
            payload = {
                "model": "gemma-2-9b-it",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            logger.info(f"🎯 WRAPPER: Calling Gemma API ({len(messages)} messages)")
            logger.debug(f"🔍 WRAPPER: Payload: {payload}")
            
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    timeout=120
                )
                
                # Log response details for debugging
                logger.debug(f"🔍 WRAPPER: Response status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"🔍 WRAPPER: Response text: {response.text}")
                
                response.raise_for_status()
                result = response.json()
                
                if "content" in result and len(result["content"]) > 0:
                    text = result["content"][0]["text"]
                    
                    # Create response object that matches Anthropic's format with all required fields
                    response_obj = types.SimpleNamespace()
                    response_obj.content = [types.SimpleNamespace(text=text, type="text")]
                    response_obj.usage = types.SimpleNamespace(input_tokens=0, output_tokens=len(text.split()))
                    response_obj.model = kwargs.get('model', 'claude-3-7-sonnet-20250219')
                    response_obj.stop_reason = "end_turn"  # Add missing stop_reason
                    response_obj.stop_sequence = None      # Add missing stop_sequence  
                    response_obj.id = f"msg_{hash(text) % 100000}"  # Add message ID
                    response_obj.type = "message"          # Add message type
                    response_obj.role = "assistant"        # Add role
                    
                    logger.info("✅ WRAPPER: Gemma response received and formatted")
                    return response_obj
                else:
                    logger.error(f"🔍 WRAPPER: Unexpected result format: {result}")
                    
            except requests.exceptions.HTTPError as e:
                logger.error(f"❌ WRAPPER: HTTP error: {e}")
                logger.error(f"🔍 WRAPPER: Response content: {response.text if 'response' in locals() else 'No response'}")
                
                # Test with a simpler request format
                simple_payload = {
                    "messages": messages,
                    "max_tokens": max_tokens
                }
                logger.info(f"🔄 WRAPPER: Retrying with simpler payload: {simple_payload}")
                
                try:
                    response = self.session.post(
                        f"{self.base_url}/v1/messages",
                        json=simple_payload,
                        timeout=120
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if "content" in result and len(result["content"]) > 0:
                        text = result["content"][0]["text"]
                        
                        # Create response with all required fields
                        response_obj = types.SimpleNamespace()
                        response_obj.content = [types.SimpleNamespace(text=text, type="text")]
                        response_obj.usage = types.SimpleNamespace(input_tokens=0, output_tokens=len(text.split()))
                        response_obj.model = kwargs.get('model', 'claude-3-7-sonnet-20250219')
                        response_obj.stop_reason = "end_turn"
                        response_obj.stop_sequence = None
                        response_obj.id = f"msg_{hash(text) % 100000}"
                        response_obj.type = "message"
                        response_obj.role = "assistant"
                        
                        logger.info("✅ WRAPPER: Gemma response received with simpler payload")
                        return response_obj
                        
                except Exception as retry_error:
                    logger.error(f"❌ WRAPPER: Retry also failed: {retry_error}")
                    
            except Exception as e:
                logger.error(f"❌ WRAPPER: Gemma API error: {e}")
            
            # Return error response with all required fields
            error_obj = types.SimpleNamespace()
            error_obj.content = [types.SimpleNamespace(text="Error: Unable to connect to Gemma server", type="text")]
            error_obj.usage = types.SimpleNamespace(input_tokens=0, output_tokens=0)
            error_obj.model = kwargs.get('model', 'claude-3-7-sonnet-20250219')
            error_obj.stop_reason = "error"
            error_obj.stop_sequence = None
            error_obj.id = "msg_error"
            error_obj.type = "message"
            error_obj.role = "assistant"
            return error_obj
    
    class GemmaAsyncAnthropicClient(GemmaAnthropicClient):
        """Fake AsyncAnthropic client that actually calls Gemma"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            logger.info("🎯 WRAPPER: AsyncAnthropic client created -> Using Gemma")
            
            # Override with async message handler
            self.messages = AsyncMessageHandler(self)
    
    # Set up the fake module
    fake_anthropic.Anthropic = GemmaAnthropicClient
    fake_anthropic.AsyncAnthropic = GemmaAsyncAnthropicClient
    fake_anthropic.Client = GemmaAnthropicClient
    fake_anthropic.AsyncClient = GemmaAsyncAnthropicClient
    
    # Create fake types submodule
    fake_types = types.ModuleType('anthropic.types')
    
    # Add common Anthropic types that might be imported
    fake_types.MessageParam = dict
    fake_types.TextBlock = types.SimpleNamespace
    fake_types.Message = types.SimpleNamespace
    fake_types.Usage = types.SimpleNamespace
    fake_types.ContentBlock = types.SimpleNamespace
    fake_types.TextBlockParam = dict
    fake_types.MessageCreateParams = dict
    fake_types.CompletionCreateParams = dict
    
    # Add the types submodule to the fake anthropic module
    fake_anthropic.types = fake_types
    
    # Create other common submodules
    fake_anthropic._types = fake_types
    fake_anthropic.lib = types.ModuleType('anthropic.lib')
    fake_anthropic._client = types.ModuleType('anthropic._client')
    fake_anthropic.resources = types.ModuleType('anthropic.resources')
    
    # Add other attributes
    fake_anthropic.HUMAN_PROMPT = ""
    fake_anthropic.AI_PROMPT = ""
    fake_anthropic.__version__ = "0.0.0-gemma-override"
    
    return fake_anthropic, fake_types

def patch_sys_modules():
    """Patch sys.modules to replace anthropic with our fake module"""
    fake_anthropic, fake_types = create_fake_anthropic_module()
    
    # Register both the main module and all submodules
    sys.modules['anthropic'] = fake_anthropic
    sys.modules['anthropic.types'] = fake_types
    sys.modules['anthropic._types'] = fake_types
    sys.modules['anthropic.lib'] = fake_anthropic.lib
    sys.modules['anthropic._client'] = fake_anthropic._client
    sys.modules['anthropic.resources'] = fake_anthropic.resources
    
    # Also register potential sub-submodules
    sys.modules['anthropic.resources.messages'] = types.ModuleType('anthropic.resources.messages')
    sys.modules['anthropic.resources.completions'] = types.ModuleType('anthropic.resources.completions')
    
    logger.info("🚀 WRAPPER: Replaced 'anthropic' and all submodules in sys.modules with Gemma client")

def set_environment():
    """Set required environment variables"""
    os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-wrapper-override-gemma'
    os.environ['OPENAI_API_KEY'] = 'sk-wrapper-override-gemma'
    os.environ['GEMINI_API_KEY'] = 'wrapper-override-gemma'
    logger.info("✅ WRAPPER: Set fake API keys")

def test_gemma_connection():
    """Test Gemma server before starting agent"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ WRAPPER: Gemma server is healthy")
            return True
        else:
            logger.error(f"❌ WRAPPER: Gemma server returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ WRAPPER: Cannot connect to Gemma server: {e}")
        return False

def main():
    """Main wrapper function"""
    logger.info("🔧 WRAPPER: Starting Final Complete Gemma wrapper for self-improving coding agent")
    
    # Test Gemma connection first
    if not test_gemma_connection():
        logger.error("❌ WRAPPER: Gemma server not accessible. Please start your Gemma server.")
        return 1
    
    # Set environment
    set_environment()
    
    # Patch sys.modules BEFORE importing anything from the agent
    patch_sys_modules()
    
    # Add base_agent to path
    base_agent_path = os.path.join(os.path.dirname(__file__), 'base_agent')
    if base_agent_path not in sys.path:
        sys.path.insert(0, base_agent_path)
    
    logger.info("🚀 WRAPPER: Starting agent with complete Gemma integration active")
    
    try:
        # Import and run the agent AFTER our patches are in place
        from base_agent.agent import main as agent_main
        
        # Check if main is async and handle accordingly
        if inspect.iscoroutinefunction(agent_main):
            logger.info("🔄 WRAPPER: Agent main is async, running with asyncio.run()")
            return asyncio.run(agent_main())
        else:
            logger.info("🔄 WRAPPER: Agent main is sync, calling directly")
            return agent_main()
        
    except ImportError as e:
        logger.error(f"❌ WRAPPER: Failed to import agent: {e}")
        logger.error("Make sure you're running this from the self_improving_coding_agent directory")
        return 1
    except Exception as e:
        logger.error(f"❌ WRAPPER: Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
