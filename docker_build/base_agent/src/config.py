"""
Complete config.py file for the self-improving coding agent with Gemma integration
Replace the existing base_agent/src/config.py with this file
"""

import os
import logging
from typing import Dict, Any, Optional
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaAPIClient:
    """Simple client to connect to external Gemma API server"""
    
    def __init__(self, base_url="http://host.docker.internal:8000", model_name="gemma-2-9b-it"):
        self.base_url = base_url
        self.model_name = model_name
        self.session = requests.Session()
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if the Gemma API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Connected to Gemma API successfully")
            else:
                logger.warning(f"⚠️ Gemma API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Gemma API at {self.base_url}: {e}")
            logger.error("Make sure your Gemma server is running on the host machine!")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using external Gemma API"""
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                timeout=180  # 3 minutes timeout for long generations
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the text content
            if "content" in result and len(result["content"]) > 0:
                return result["content"][0]["text"]
            else:
                logger.error(f"Unexpected response format: {result}")
                return "Error: Unexpected response format from Gemma API"
                
        except requests.exceptions.Timeout:
            logger.error("⏰ Gemma API request timed out")
            return "Error: Request timed out"
        except requests.exceptions.RequestException as e:
            logger.error(f"🚫 Gemma API request failed: {e}")
            return f"Error: API request failed - {e}"
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return f"Error: {e}"
    
    def chat(self, messages: list, **kwargs) -> str:
        """Chat interface for compatibility with agent framework"""
        # Convert messages to a formatted prompt
        formatted_prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted_prompt += f"System: {content}\n\n"
            elif role == 'user':
                formatted_prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                formatted_prompt += f"Assistant: {content}\n\n"
        
        formatted_prompt += "Assistant: "
        
        return self.generate(
            formatted_prompt, 
            max_tokens=kwargs.get('max_tokens', 1024),
            temperature=kwargs.get('temperature', 0.7)
        )
    
    def __call__(self, messages, **kwargs):
        """Make the client callable"""
        return {"content": self.chat(messages, **kwargs)}

def create_gemma_llm(**kwargs) -> GemmaAPIClient:
    """Factory function to create Gemma client"""
    return GemmaAPIClient(**kwargs)

# LLM Configuration
LLM_CONFIGS = {
    # Gemma (Local API) - Primary choice
    "gemma-2-9b-local": {
        "provider": "gemma",
        "model": "gemma-2-9b-it",
        "max_tokens": 1024,
        "temperature": 0.7,
        "base_url": "http://host.docker.internal:8000",
    },
    
    "gemma-2-9b-creative": {
        "provider": "gemma",
        "model": "gemma-2-9b-it",
        "max_tokens": 1500,
        "temperature": 0.9,  # More creative
        "base_url": "http://host.docker.internal:8000",
    },
    
    "gemma-2-9b-precise": {
        "provider": "gemma",
        "model": "gemma-2-9b-it",
        "max_tokens": 800,
        "temperature": 0.3,  # More focused
        "base_url": "http://host.docker.internal:8000",
    },
    
    # Backup options (if you have API keys)
    "claude-3.7-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    
    "gpt-4o": {
        "provider": "openai", 
        "model": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini", 
        "max_tokens": 2048,
        "temperature": 0.7,
    },
}

# Task-specific LLM assignments
TASK_SPECIFIC_LLMS = {
    "code_generation": "gemma-2-9b-local",
    "code_review": "gemma-2-9b-precise", 
    "debugging": "gemma-2-9b-local",
    "meta_improvement": "gemma-2-9b-creative",  # More creative for self-improvement
    "benchmarking": "gemma-2-9b-local",
    "planning": "gemma-2-9b-local",
    "reasoning": "gemma-2-9b-local",
    "general": "gemma-2-9b-local",
}

# Default LLM
DEFAULT_LLM = "gemma-2-9b-local"

def create_llm(llm_name: str):
    """Create LLM instance based on configuration"""
    if llm_name not in LLM_CONFIGS:
        logger.warning(f"Unknown LLM: {llm_name}. Available: {list(LLM_CONFIGS.keys())}")
        logger.info(f"Falling back to default LLM: {DEFAULT_LLM}")
        llm_name = DEFAULT_LLM
    
    config = LLM_CONFIGS[llm_name]
    provider = config["provider"]
    
    logger.info(f"Creating LLM: {llm_name} (provider: {provider})")
    
    if provider == "gemma":
        return create_gemma_llm(
            base_url=config.get("base_url", "http://host.docker.internal:8000"),
            model_name=config.get("model", "gemma-2-9b-it")
        )
    
    elif provider == "anthropic":
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found, falling back to Gemma")
                return create_llm(DEFAULT_LLM)
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.error("Anthropic package not installed, falling back to Gemma") 
            return create_llm(DEFAULT_LLM)
    
    elif provider == "openai":
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found, falling back to Gemma")
                return create_llm(DEFAULT_LLM)
            return openai.OpenAI(api_key=api_key)
        except ImportError:
            logger.error("OpenAI package not installed, falling back to Gemma")
            return create_llm(DEFAULT_LLM)
    
    else:
        logger.error(f"Unknown provider: {provider}, falling back to Gemma")
        return create_llm(DEFAULT_LLM)

def get_llm_for_task(task_type: str = "general"):
    """Get appropriate LLM for specific task types"""
    llm_name = TASK_SPECIFIC_LLMS.get(task_type, DEFAULT_LLM)
    logger.info(f"Using LLM '{llm_name}' for task: {task_type}")
    return create_llm(llm_name)

class Settings:
    """Settings class for the agent framework"""
    
    def __init__(self):
        # LLM Settings
        self.default_llm = DEFAULT_LLM
        self.llm_configs = LLM_CONFIGS
        
        # Agent Settings
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "300"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Working Directory
        self.work_dir = os.getenv("WORK_DIR", "/tmp/agent_workspace")
        
        # Server Settings
        self.server_host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.server_port = int(os.getenv("SERVER_PORT", "8080"))
        
        # Gemma API Settings
        self.gemma_api_url = os.getenv("GEMMA_API_URL", "http://host.docker.internal:8000")
        self.gemma_max_tokens = int(os.getenv("GEMMA_MAX_TOKENS", "1024"))
        self.gemma_temperature = float(os.getenv("GEMMA_TEMPERATURE", "0.7"))
        
        # Ensure work directory exists
        os.makedirs(self.work_dir, exist_ok=True)
    
    def get_llm(self, llm_name: Optional[str] = None, task_type: Optional[str] = None):
        """Get LLM instance"""
        if task_type:
            return get_llm_for_task(task_type)
        elif llm_name:
            return create_llm(llm_name)
        else:
            return create_llm(self.default_llm)

# Global settings instance
settings = Settings()

# Legacy compatibility functions
def get_default_llm():
    """Get the default LLM instance"""
    return create_llm(DEFAULT_LLM)

def get_llm(name: str = None):
    """Get LLM by name or default"""
    return create_llm(name or DEFAULT_LLM)

# Print configuration info when imported
logger.info("🤖 Agent Configuration Loaded")
logger.info(f"📍 Default LLM: {DEFAULT_LLM}")
logger.info(f"🔧 Available LLMs: {list(LLM_CONFIGS.keys())}")
logger.info(f"🌐 Gemma API URL: {settings.gemma_api_url}")

# Test Gemma connection on import
try:
    test_client = GemmaAPIClient(base_url=settings.gemma_api_url)
    logger.info("✅ Gemma API connection test completed")
except Exception as e:
    logger.warning(f"⚠️ Gemma API connection test failed: {e}")
    logger.info("💡 Make sure your Gemma server is running on the host machine!")

# Environment check
required_dirs = [settings.work_dir]
for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")

logger.info("🚀 Configuration setup complete!")
