"""
Complete config.py file for the self-improving coding agent with Gemma integration
Replace the entire base_agent/src/config.py with this file
"""

import os
import logging
from typing import Dict, Any, Optional
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set fake API keys to prevent validation errors
os.environ['ANTHROPIC_API_KEY'] = 'fake-key-for-gemma-integration'
os.environ['OPENAI_API_KEY'] = 'fake-key-for-gemma-integration'
os.environ['GEMINI_API_KEY'] = 'fake-key-for-gemma-integration'

# Import the required types
try:
    from base_agent.src.types.llm_types import Provider, ModelInfo, TokenCost, FCI, ArgFormat
except ImportError:
    # Fallback definitions if types module is not available
    from enum import Enum
    
    class Provider(Enum):
        ANTHROPIC = 1
        OPENAI = 2
        DEEPSEEK = 3
        FIREWORKS = 4
        GOOGLE_REST = 7
        VERTEX = 8
    
    class FCI(Enum):
        CONSTRAINED = 1
        UNCONSTRAINED = 2
    
    class ArgFormat(Enum):
        JSON = "json"
        XML = "xml"
    
    class TokenCost:
        def __init__(self, input_uncached=0.0, input_cached=0.0, cache_write=0.0, output=0.0):
            self.input_uncached = input_uncached
            self.input_cached = input_cached
            self.cache_write = cache_write
            self.output = output
    
    class ModelInfo:
        def __init__(self, api_name, provider, costs, max_tokens, supports_caching, 
                     reasoning_model, context_window, function_calling_interface, preferred_arg_format):
            self.api_name = api_name
            self.provider = provider
            self.costs = costs
            self.max_tokens = max_tokens
            self.supports_caching = supports_caching
            self.reasoning_model = reasoning_model
            self.context_window = context_window
            self.function_calling_interface = function_calling_interface
            self.preferred_arg_format = preferred_arg_format


class GemmaAPIClient:
    """Gemma API client that mimics other LLM providers"""
    
    def __init__(self, base_url="http://localhost:8000", model_name="gemma-2-9b-it"):
        self.base_url = base_url
        self.model_name = model_name
        self.session = requests.Session()
        
        # Set connection pool settings
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'GemmaClient/1.0'
        })
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if the Gemma API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Gemma API connection successful")
            else:
                logger.warning(f"⚠️ Gemma API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Gemma API: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using Gemma API"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
    
    def chat(self, messages: list, **kwargs) -> str:
        """Chat interface for compatibility with agent framework"""
        payload = {
            "model": self.model_name,
            "max_tokens": kwargs.get('max_tokens', 1024),
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.7)
        }
        
        logger.info(f"🔄 Gemma API call: {len(messages)} messages")
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse Gemma API response
            if "content" in result and len(result["content"]) > 0:
                content_item = result["content"][0]
                if isinstance(content_item, dict) and "text" in content_item:
                    logger.info("✅ Gemma API response received")
                    return content_item["text"]
            
            logger.error(f"Unexpected response format: {result}")
            return "Error: Unexpected response format"
            
        except requests.exceptions.Timeout:
            logger.error("⏰ Gemma API timeout")
            return "Error: Request timed out"
        except Exception as e:
            logger.error(f"❌ Gemma API error: {e}")
            return f"Error: {e}"
    
    # Anthropic-style methods for compatibility
    def messages(self, **kwargs):
        """Anthropic-style messages method"""
        messages = kwargs.get('messages', [])
        max_tokens = kwargs.get('max_tokens', 1024)
        response_text = self.chat(messages, max_tokens=max_tokens)
        
        # Return Anthropic-style response
        class MockResponse:
            def __init__(self, text):
                self.content = [MockContent(text)]
                self.usage = MockUsage()
        
        class MockContent:
            def __init__(self, text):
                self.text = text
                self.type = "text"
        
        class MockUsage:
            def __init__(self):
                self.input_tokens = 0
                self.output_tokens = 0
        
        return MockResponse(response_text)
    
    def __call__(self, messages, **kwargs):
        """Make the client callable"""
        response_text = self.chat(messages, **kwargs)
        return {
            "content": [{"type": "text", "text": response_text}],
            "role": "assistant",
            "type": "message"
        }


def create_gemma_llm(**kwargs) -> GemmaAPIClient:
    """Factory function to create Gemma client"""
    return GemmaAPIClient(**kwargs)


def create_standard_model():
    """Use the actual predefined model from the framework"""
    try:
        from base_agent.src.types.llm_types import Model
        # Use the exact predefined SONNET_37 model enum
        return Model.SONNET_37  # This is the actual enum value that passes validation
    except ImportError:
        # Fallback if import fails
        return ModelInfo(
            api_name="claude-3-7-sonnet-20250219",
            provider=Provider.ANTHROPIC,
            costs=TokenCost(
                input_uncached=3.0,
                input_cached=0.3,
                cache_write=3.75,
                output=15.0
            ),
            max_tokens=64000,
            supports_caching=True,
            reasoning_model=True,
            context_window=200000,
            function_calling_interface=FCI.UNCONSTRAINED,
            preferred_arg_format=ArgFormat.XML
        )


class Settings:
    """Settings class for the agent framework"""
    
    def __init__(self):
        # Create standard model that passes validation
        standard_model = create_standard_model()
        
        # Use the same model for all tasks
        self.REASONING_MODEL = standard_model
        self.CODING_MODEL = standard_model
        self.ORCHESTRATOR_MODEL = standard_model
        self.BENCHMARKING_MODEL = standard_model
        self.META_IMPROVEMENT_MODEL = standard_model
        self.MODEL = standard_model
        self.REVIEW_MODEL = standard_model
        self.COMMITTEE_MODEL = standard_model
        self.OVERSIGHT_MODEL = standard_model
        self.SUPERVISOR_MODEL = standard_model
        self.EVALUATOR_MODEL = standard_model
        self.PLANNER_MODEL = standard_model
        self.EXECUTOR_MODEL = standard_model
        self.ANALYZER_MODEL = standard_model
        self.VALIDATOR_MODEL = standard_model
        
        # Other settings
        self.max_iterations = 10
        self.max_context_length = 8192
        self.timeout_seconds = 300
        self.log_level = "INFO"
        self.LOG_LEVEL = "INFO"  # Add uppercase version for compatibility
        self.work_dir = "./work"
        self.WORK_DIR = "./work"  # Add uppercase version
        self.OUTPUT_DIR = "./results"  # Add this too
        self.TEMP_DIR = "/tmp"  # Add this too
        self.server_host = "0.0.0.0"
        self.server_port = 8080
        self.gemma_api_url = os.getenv("GEMMA_API_URL", "http://localhost:8000")
        
        # Ensure directories exist
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"✅ All models set to: {standard_model} (predefined)")
        logger.info(f"✅ Model API name: {getattr(standard_model, 'api_name', getattr(standard_model, 'id', 'unknown'))}")
        logger.info(f"✅ Gemma API URL: {self.gemma_api_url}")
    
    def get_llm(self, llm_name: Optional[str] = None, task_type: Optional[str] = None):
        """Get LLM instance - always returns Gemma"""
        logger.info(f"🔄 Settings.get_llm called: {llm_name}/{task_type} -> Gemma")
        return create_gemma_llm()


# Global settings instance
settings = Settings()


# Core LLM creation functions - all return Gemma
def create_llm(llm_name: str = None):
    """Create LLM instance - always returns Gemma"""
    logger.info(f"🔄 create_llm called: {llm_name} -> Gemma")
    return create_gemma_llm()


def get_llm_for_task(task_type: str = "general"):
    """Get LLM for task - always returns Gemma"""
    logger.info(f"🔄 get_llm_for_task called: {task_type} -> Gemma")
    return create_gemma_llm()


def get_default_llm():
    """Get default LLM - always returns Gemma"""
    logger.info("🔄 get_default_llm called -> Gemma")
    return create_gemma_llm()


def get_llm(name: str = None):
    """Get LLM by name - always returns Gemma"""
    logger.info(f"🔄 get_llm called: {name} -> Gemma")
    return create_gemma_llm()


# Patch the ModelInfo class to add missing attributes (fix recursion)
def patch_modelinfo_class():
    """Add missing attributes to ModelInfo instances without recursion"""
    try:
        from base_agent.src.types.llm_types import ModelInfo, FCI, ArgFormat, Model
        
        # Use object.__getattribute__ to avoid recursion
        def safe_getattr(obj, name, default=None):
            try:
                return object.__getattribute__(obj, name)
            except AttributeError:
                return default
        
        # Patch ModelInfo class
        original_getattribute = ModelInfo.__getattribute__
        
        def patched_getattribute(self, name):
            if name == 'fci':
                return safe_getattr(self, 'function_calling_interface') or FCI.UNCONSTRAINED
            elif name == 'id':
                return safe_getattr(self, 'api_name') or 'unknown'
            elif name == 'arg_format':
                return safe_getattr(self, 'preferred_arg_format') or ArgFormat.XML
            else:
                return original_getattribute(self, name)
        
        ModelInfo.__getattribute__ = patched_getattribute
        
        # Simpler patch for Model enum - just add properties
        def model_fci_property(self):
            return safe_getattr(self.value, 'function_calling_interface') or FCI.UNCONSTRAINED
        
        def model_id_property(self):
            return safe_getattr(self.value, 'api_name') or getattr(self, 'id', 'unknown')
        
        def model_arg_format_property(self):
            return safe_getattr(self.value, 'preferred_arg_format') or ArgFormat.XML
        
        # Add as properties instead of patching __getattribute__
        Model.fci = property(model_fci_property)
        Model.arg_format = property(model_arg_format_property)
        
        logger.info("✅ Patched ModelInfo and Model classes (recursion-safe)")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not patch ModelInfo: {e}")


# Nuclear option: Comprehensive LLM interception
def patch_all_llm_creation():
    """Patch every possible LLM creation method"""
    
    # Patch 1: LLM API functions (recursion-safe)
    try:
        import base_agent.src.llm.api as llm_api
        
        def gemma_create_completion(model, messages, **kwargs):
            logger.info(f"🎯 create_completion intercepted: {model} -> Gemma")
            try:
                client = create_gemma_llm()
                response_text = client.chat(messages, **kwargs)
                
                # Return proper completion object
                try:
                    from base_agent.src.llm.base import Completion
                    return Completion(
                        content=[{"type": "text", "text": response_text}],
                        model=model,
                        usage={"input_tokens": 0, "output_tokens": len(response_text.split())}
                    )
                except ImportError:
                    return {
                        "content": [{"type": "text", "text": response_text}],
                        "model": model,
                        "usage": {"input_tokens": 0, "output_tokens": len(response_text.split())}
                    }
            except Exception as e:
                logger.error(f"❌ Gemma completion failed: {e}")
                return {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "model": model,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
        
        # Only patch if not already patched
        if not hasattr(llm_api.create_completion, '_gemma_patched'):
            llm_api.create_completion = gemma_create_completion
            llm_api.create_completion._gemma_patched = True
            
        if not hasattr(llm_api.create_streaming_completion, '_gemma_patched'):
            llm_api.create_streaming_completion = gemma_create_completion
            llm_api.create_streaming_completion._gemma_patched = True
        
        if hasattr(llm_api, 'get_client') and not hasattr(llm_api.get_client, '_gemma_patched'):
            original_get_client = llm_api.get_client
            def patched_get_client(*args, **kwargs):
                logger.info("🎯 get_client intercepted -> Gemma")
                return create_gemma_llm()
            llm_api.get_client = patched_get_client
            llm_api.get_client._gemma_patched = True
        
        logger.info("✅ Patched llm_api functions (recursion-safe)")
        
    except ImportError:
        logger.debug("llm_api not found")
    except Exception as e:
        logger.warning(f"⚠️ Error patching llm_api: {e}")
    
    # Patch 2: Anthropic provider (fix recursion)
    try:
        import anthropic
        
        # Store the original class
        original_anthropic = anthropic.Anthropic
        
        class GemmaAnthropicClient:
            def __init__(self, *args, **kwargs):
                self.gemma_client = create_gemma_llm()
                logger.info("🎯 Anthropic client creation intercepted -> Gemma")
            
            def messages(self, **kwargs):
                logger.info("🎯 Anthropic messages() called -> Gemma")
                return self.gemma_client.messages(**kwargs)
            
            def create(self, **kwargs):
                logger.info("🎯 Anthropic create() called -> Gemma")
                return self.messages(**kwargs)
            
            def __getattr__(self, name):
                # For any other methods, delegate to gemma client
                logger.info(f"🎯 Anthropic.{name} called -> Gemma")
                return getattr(self.gemma_client, name, lambda *args, **kwargs: self.gemma_client.messages(**kwargs))
        
        # Replace the class only once
        if not hasattr(anthropic.Anthropic, '_gemma_patched'):
            anthropic.Anthropic = GemmaAnthropicClient
            anthropic.Anthropic._gemma_patched = True
            logger.info("✅ Patched anthropic.Anthropic class (recursion-safe)")
        
    except ImportError:
        logger.debug("anthropic module not available")
    except Exception as e:
        logger.warning(f"⚠️ Error patching anthropic: {e}")
    
    # Patch 3: Provider modules (simplified to avoid recursion)
    try:
        import base_agent.src.llm.providers.anthropic_provider as anthropic_provider
        
        # Only patch the key functions, not everything
        key_functions = ['create_anthropic_client', 'get_anthropic_client', 'anthropic_completion']
        
        for func_name in key_functions:
            if hasattr(anthropic_provider, func_name):
                def create_simple_wrapper(original_name):
                    def wrapper(*args, **kwargs):
                        logger.info(f"🎯 anthropic_provider.{original_name} intercepted -> Gemma")
                        return create_gemma_llm()
                    return wrapper
                
                setattr(anthropic_provider, func_name, create_simple_wrapper(func_name))
        
        logger.info("✅ Patched key anthropic_provider functions (recursion-safe)")
        
    except ImportError:
        logger.debug("anthropic_provider not found")
    except Exception as e:
        logger.warning(f"⚠️ Error patching anthropic_provider: {e}")


# Apply all patches
patch_modelinfo_class()
patch_all_llm_creation()

# Test Gemma connection
try:
    test_client = GemmaAPIClient(base_url=settings.gemma_api_url)
    logger.info("✅ Gemma integration test completed")
except Exception as e:
    logger.warning(f"⚠️ Gemma integration test failed: {e}")

# Configuration complete
logger.info("🚀 Gemma configuration complete!")
logger.info("🎯 All LLM calls will be routed to your Gemma server")
