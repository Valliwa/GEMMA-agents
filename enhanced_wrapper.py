#!/usr/bin/env python3
"""
Complete Enhanced Gemma Wrapper Script - Fixed Version
This replaces the entire anthropic package with Gemma integration
"""

import sys
import os

# Python 3.10 compatibility fix for asyncio.timeout
if sys.version_info < (3, 11):
    print("🔧 WRAPPER: Applying Python 3.10 compatibility fixes...")
    
    # Install async-timeout if not available
    try:
        import async_timeout
    except ImportError:
        print("📦 Installing async-timeout for Python 3.10 compatibility...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "async-timeout"])
        import async_timeout
    
    # Patch asyncio to add timeout context manager
    import asyncio
    if not hasattr(asyncio, 'timeout'):
        asyncio.timeout = async_timeout.timeout
        print("✅ WRAPPER: asyncio.timeout compatibility added")

import types
import requests
import logging
import argparse
import asyncio
import inspect
import json
import time
from typing import List, Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResponseObject:
    """More robust response object that matches Anthropic's format exactly"""
    
    def __init__(self, text: str, model: str = "claude-3-5-sonnet-20241022", 
                 input_tokens: int = 0, output_tokens: int = 0):
        self.id = f"msg_{int(time.time() * 1000)}_{hash(text) % 10000}"
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.content = [types.SimpleNamespace(type="text", text=text)]
        self.stop_reason = "end_turn"
        self.stop_sequence = None
        self.usage = types.SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def __getattr__(self, name):
        """Handle any missing attributes gracefully"""
        if name == "text":
            return self.content[0].text if self.content else ""
        return None


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
        """Fixed Async handler for client.messages.create() calls"""
        
        def __init__(self, client):
            self.client = client
        
        async def create(self, **kwargs):
            """Handle async client.messages.create() calls"""
            logger.info("🎯 WRAPPER: async messages.create() called -> Gemma")
            
            try:
                # Ensure we await the async call properly
                result = await self.client._call_gemma(**kwargs)
                logger.info("🎯 WRAPPER: async messages.create() completed successfully")
                return result
            except Exception as e:
                logger.error(f"❌ WRAPPER: async messages.create() failed: {e}")
                # Return error response with proper structure
                return ResponseObject(
                    text=f"Error in async message creation: {e}",
                    model=kwargs.get('model', 'claude-3-5-sonnet-20241022'),
                    input_tokens=0,
                    output_tokens=0
                )
        
        async def __call__(self, **kwargs):
            """Handle direct async client.messages() calls"""
            return await self.create(**kwargs)
    
    class GemmaAnthropicClient:
        """Enhanced Fake Anthropic client that actually calls Gemma"""
        
        def __init__(self, *args, **kwargs):
            # Support different server configurations
            self.base_url = kwargs.get('base_url', os.environ.get('GEMMA_BASE_URL', "http://localhost:8000"))
            self.session = requests.Session()
            self.session.headers.update({'Content-Type': 'application/json'})
            
            # Set timeout
            self.timeout = kwargs.get('timeout', 520)
            
            logger.info(f"🎯 WRAPPER: Anthropic client created -> Using Gemma at {self.base_url}")
            
            # Create message handler object with create method
            self.messages = MessageHandler(self)
        
        def _estimate_tokens(self, text: str) -> int:
            """Better token estimation"""
            # Rough approximation: 1 token ≈ 4 characters for English text
            return max(1, len(text) // 4)
        
        def _format_messages_for_logging(self, messages: List[Dict]) -> str:
            """Format messages for logging without exposing sensitive content"""
            summary = []
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if isinstance(content, str):
                    length = len(content)
                    summary.append(f"{role}({length} chars)")
                else:
                    summary.append(f"{role}(complex)")
            return " -> ".join(summary)
        
        def _extract_text_from_content(self, content):
            """Extract plain text from complex content structures"""
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif 'text' in item:
                            text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, 'text'):
                        text_parts.append(item.text)
                return '\n'.join(text_parts)
            else:
                return str(content)
        
        def _call_gemma(self, **kwargs):
            """Internal method to call Gemma API with proper message handling"""
            raw_messages = kwargs.get('messages', [])
            max_tokens = kwargs.get('max_tokens', 1024)
            temperature = kwargs.get('temperature', 0.7)
            model = kwargs.get('model', 'claude-3-5-sonnet-20241022')
            
            # Process messages to extract simple text content
            messages = []
            for msg in raw_messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = self._extract_text_from_content(msg.get('content', ''))
                else:
                    # Handle framework message objects
                    role = getattr(msg, 'role', 'user')
                    raw_content = getattr(msg, 'content', '')
                    content = self._extract_text_from_content(raw_content)
                
                messages.append({
                    'role': role,
                    'content': content
                })
            
            # Log request summary
            msg_summary = " -> ".join([f"{msg['role']}({len(msg['content'])} chars)" for msg in messages])
            logger.info(f"🎯 WRAPPER: Calling Gemma API - {msg_summary}")
            
            payload = {
                "model": "gemma-2-9b-it",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add top_p if provided
            if 'top_p' in kwargs:
                payload['top_p'] = kwargs['top_p']
            
            try:
                # Make request with retry logic
                for attempt in range(3):
                    try:
                        logger.debug(f"🔍 WRAPPER: Attempt {attempt + 1}/3")
                        
                        response = self.session.post(
                            f"{self.base_url}/v1/messages",
                            json=payload,
                            timeout=self.timeout
                        )
                        
                        if response.status_code != 200:
                            logger.error(f"🔍 WRAPPER: HTTP {response.status_code}: {response.text}")
                            if attempt < 2:
                                continue
                        
                        response.raise_for_status()
                        result = response.json()
                        
                        # Extract response text
                        if "content" in result and len(result["content"]) > 0:
                            text = result["content"][0]["text"]
                            
                            # Calculate tokens
                            input_tokens = sum(len(msg['content'].split()) for msg in messages)
                            output_tokens = len(text.split())
                            
                            # Create enhanced response object
                            response_obj = ResponseObject(
                                text=text,
                                model=model,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens
                            )
                            
                            logger.info(f"✅ WRAPPER: Success! Generated {output_tokens} tokens")
                            return response_obj
                        else:
                            logger.error(f"🔍 WRAPPER: Unexpected result format: {result}")
                            
                        break
                        
                    except requests.exceptions.Timeout:
                        logger.warning(f"⏰ WRAPPER: Request timeout on attempt {attempt + 1}")
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)
                        
                    except requests.exceptions.ConnectionError:
                        logger.warning(f"🔌 WRAPPER: Connection error on attempt {attempt + 1}")
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)
                        
                    except requests.exceptions.HTTPError as e:
                        if response.status_code >= 500:
                            logger.warning(f"🔥 WRAPPER: Server error {response.status_code} on attempt {attempt + 1}")
                            if attempt == 2:
                                raise
                            time.sleep(2 ** attempt)
                        else:
                            raise
                            
            except Exception as e:
                logger.error(f"❌ WRAPPER: Gemma API error after all retries: {e}")
                
                # Try health check to diagnose the issue
                try:
                    health_response = self.session.get(f"{self.base_url}/health", timeout=5)
                    if health_response.status_code == 200:
                        logger.error("🏥 WRAPPER: Gemma server is healthy but request failed")
                    else:
                        logger.error(f"🏥 WRAPPER: Gemma server health check failed: {health_response.status_code}")
                except:
                    logger.error("🏥 WRAPPER: Cannot reach Gemma server for health check")
            
            # Return error response
            error_text = f"Error: Unable to connect to Gemma server at {self.base_url}"
            return ResponseObject(
                text=error_text,
                model=model,
                input_tokens=0,
                output_tokens=0
            )
    
    class GemmaAsyncAnthropicClient:
        """Enhanced Fake AsyncAnthropic client that actually calls Gemma"""
        
        def __init__(self, *args, **kwargs):
            # Support different server configurations
            self.base_url = kwargs.get('base_url', os.environ.get('GEMMA_BASE_URL', "http://localhost:8000"))
            self.session = requests.Session()
            self.session.headers.update({'Content-Type': 'application/json'})
            
            # Set timeout
            self.timeout = kwargs.get('timeout', 520)
            
            logger.info(f"🎯 WRAPPER: AsyncAnthropic client created -> Using Gemma at {self.base_url}")
            
            # Create message handler
            self.messages = AsyncMessageHandler(self)
        
        def _extract_text_from_content(self, content):
            """Extract plain text from complex content structures"""
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif 'text' in item:
                            text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, 'text'):
                        text_parts.append(item.text)
                return '\n'.join(text_parts)
            else:
                return str(content)
        
        async def _call_gemma(self, **kwargs):
            """Async method to call Gemma API"""
            logger.info("🎯 WRAPPER: Framework calling async _call_gemma")
            
            try:
                raw_messages = kwargs.get('messages', [])
                max_tokens = kwargs.get('max_tokens', 1024)
                temperature = kwargs.get('temperature', 0.7)
                model = kwargs.get('model', 'claude-3-5-sonnet-20241022')
                
                # Process messages to extract simple text content
                messages = []
                for msg in raw_messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = self._extract_text_from_content(msg.get('content', ''))
                    else:
                        # Handle framework message objects
                        role = getattr(msg, 'role', 'user')
                        raw_content = getattr(msg, 'content', '')
                        content = self._extract_text_from_content(raw_content)
                    
                    messages.append({
                        'role': role,
                        'content': content
                    })
                
                # Log request summary
                msg_summary = " -> ".join([f"{msg['role']}({len(msg['content'])} chars)" for msg in messages])
                logger.info(f"🎯 WRAPPER: Calling Gemma API - {msg_summary}")
                
                payload = {
                    "model": "gemma-2-9b-it",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Add top_p if provided
                if 'top_p' in kwargs:
                    payload['top_p'] = kwargs['top_p']
                
                # Make async HTTP request
                loop = asyncio.get_event_loop()
                
                def make_request():
                    response = self.session.post(
                        f"{self.base_url}/v1/messages",
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()
                
                result = await loop.run_in_executor(None, make_request)
                
                # Extract response text
                if "content" in result and len(result["content"]) > 0:
                    text = result["content"][0]["text"]
                    
                    # Calculate tokens
                    input_tokens = sum(len(msg['content'].split()) for msg in messages)
                    output_tokens = len(text.split())
                    
                    # Create response object
                    response_obj = ResponseObject(
                        text=text,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
                    
                    logger.info(f"✅ WRAPPER: Success! Generated {output_tokens} tokens")
                    return response_obj
                else:
                    logger.error(f"🔍 WRAPPER: Unexpected result format: {result}")
                    
            except Exception as e:
                logger.error(f"❌ WRAPPER: async _call_gemma failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Return error response
            return ResponseObject(
                text=f"Error: Unable to connect to Gemma server at {self.base_url}",
                model=kwargs.get('model', 'claude-3-5-sonnet-20241022'),
                input_tokens=0,
                output_tokens=0
            )
    
    # Set up the fake module with more comprehensive attributes
    fake_anthropic.Anthropic = GemmaAnthropicClient
    fake_anthropic.AsyncAnthropic = GemmaAsyncAnthropicClient
    fake_anthropic.Client = GemmaAnthropicClient
    fake_anthropic.AsyncClient = GemmaAsyncAnthropicClient
    
    # Create comprehensive fake types submodule
    fake_types = types.ModuleType('anthropic.types')
    
    # Add all common Anthropic types
    fake_types.MessageParam = dict
    fake_types.TextBlock = types.SimpleNamespace
    fake_types.Message = ResponseObject
    fake_types.Usage = types.SimpleNamespace
    fake_types.ContentBlock = types.SimpleNamespace
    fake_types.TextBlockParam = dict
    fake_types.MessageCreateParams = dict
    fake_types.CompletionCreateParams = dict
    fake_types.MessageCreateParamsNonStreaming = dict
    fake_types.MessageCreateParamsStreaming = dict
    
    # Add message delta types for streaming (even though we don't support streaming)
    fake_types.MessageDelta = types.SimpleNamespace
    fake_types.MessageDeltaEvent = types.SimpleNamespace
    fake_types.ContentBlockDelta = types.SimpleNamespace
    fake_types.ContentBlockDeltaEvent = types.SimpleNamespace
    
    fake_anthropic.types = fake_types
    fake_anthropic._types = fake_types
    
    # Create other submodules
    fake_anthropic.lib = types.ModuleType('anthropic.lib')
    fake_anthropic._client = types.ModuleType('anthropic._client')
    fake_anthropic.resources = types.ModuleType('anthropic.resources')
    
    # Add version and constants
    fake_anthropic.HUMAN_PROMPT = ""
    fake_anthropic.AI_PROMPT = ""
    fake_anthropic.__version__ = "0.0.0-gemma-override"
    
    # Add common exceptions (create minimal versions)
    class AnthropicError(Exception):
        pass
    
    class APIError(AnthropicError):
        pass
    
    class RateLimitError(AnthropicError):
        pass
    
    fake_anthropic.AnthropicError = AnthropicError
    fake_anthropic.APIError = APIError
    fake_anthropic.RateLimitError = RateLimitError
    
    return fake_anthropic, fake_types


def patch_sys_modules():
    """Enhanced patching of sys.modules"""
    fake_anthropic, fake_types = create_fake_anthropic_module()
    
    # Comprehensive module registration
    modules_to_patch = [
        'anthropic',
        'anthropic.types',
        'anthropic._types',
        'anthropic.lib',
        'anthropic._client',
        'anthropic.resources',
        'anthropic.resources.messages',
        'anthropic.resources.completions',
        'anthropic._exceptions',
        'anthropic._base_client',
        'anthropic._streaming',
    ]
    
    for module_name in modules_to_patch:
        if module_name == 'anthropic':
            sys.modules[module_name] = fake_anthropic
        elif module_name == 'anthropic.types' or module_name == 'anthropic._types':
            sys.modules[module_name] = fake_types
        else:
            # Create minimal module for others
            sys.modules[module_name] = types.ModuleType(module_name)
    
    logger.info("🚀 WRAPPER: Comprehensive anthropic module replacement complete")


def set_environment():
    """Set required environment variables with better defaults"""
    env_vars = {
        'ANTHROPIC_API_KEY': 'sk-ant-wrapper-override-gemma-12345',
        'OPENAI_API_KEY': 'sk-wrapper-override-gemma-12345',
        'GEMINI_API_KEY': 'wrapper-override-gemma-12345',
        'CLAUDE_API_KEY': 'sk-ant-wrapper-override-gemma-12345',
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    logger.info("✅ WRAPPER: Environment variables configured")


def test_gemma_connection():
    """Enhanced Gemma server testing"""
    base_url = os.environ.get('GEMMA_BASE_URL', "http://localhost:8000")
    
    try:
        # Test basic connectivity
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ WRAPPER: Gemma server is healthy")
            logger.info(f"📊 WRAPPER: Server status: {health_data}")
            
            # Test a simple API call
            test_payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            test_response = requests.post(
                f"{base_url}/v1/messages",
                json=test_payload,
                timeout=30
            )
            
            if test_response.status_code == 200:
                logger.info("✅ WRAPPER: Gemma API test call successful")
                return True
            else:
                logger.error(f"❌ WRAPPER: Gemma API test failed: {test_response.status_code}")
                logger.error(f"Response: {test_response.text}")
                return False
        else:
            logger.error(f"❌ WRAPPER: Gemma server returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("❌ WRAPPER: Timeout connecting to Gemma server")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ WRAPPER: Cannot connect to Gemma server at {base_url}")
        return False
    except Exception as e:
        logger.error(f"❌ WRAPPER: Error testing Gemma connection: {e}")
        return False


def main():
    """Enhanced main wrapper function"""
    logger.info("🔧 WRAPPER: Starting Enhanced Gemma wrapper for self-improving coding agent")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gemma wrapper for self-improving coding agent")
    parser.add_argument("--gemma-url", default="http://localhost:8000", 
                       help="Gemma server URL (default: http://localhost:8000)")
    parser.add_argument("--timeout", type=int, default=320,
                       help="Request timeout in seconds (default: 120)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 WRAPPER: Debug logging enabled")
    
    # Set Gemma URL
    os.environ['GEMMA_BASE_URL'] = args.gemma_url
    
    # Test Gemma connection first
    logger.info(f"🔍 WRAPPER: Testing connection to Gemma server at {args.gemma_url}")
    if not test_gemma_connection():
        logger.error("❌ WRAPPER: Gemma server not accessible.")
        logger.error("🔧 WRAPPER: Please ensure your Gemma server is running and accessible.")
        logger.error(f"🔧 WRAPPER: Current URL: {args.gemma_url}")
        return 1
    
    # Set environment
    set_environment()
    
    # Patch sys.modules BEFORE importing anything from the agent
    patch_sys_modules()
    
    # Add base_agent to path
    base_agent_path = os.path.join(os.path.dirname(__file__), 'base_agent')
    if os.path.exists(base_agent_path) and base_agent_path not in sys.path:
        sys.path.insert(0, base_agent_path)
        logger.info(f"📁 WRAPPER: Added {base_agent_path} to Python path")
    
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
        logger.error("🔧 WRAPPER: Make sure you're running this from the self_improving_coding_agent directory")
        logger.error("🔧 WRAPPER: Expected directory structure:")
        logger.error("   self_improving_coding_agent/")
        logger.error("   ├── base_agent/")
        logger.error("   │   └── agent.py")
        logger.error("   └── enhanced_gemma_wrapper.py (this file)")
        return 1
    except Exception as e:
        logger.error(f"❌ WRAPPER: Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
