"""
Custom LLM provider for integrating local Gemma API with the self-improving coding agent.
Place this file in: base_agent/src/llm/gemma_provider.py
"""

import requests
import json
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class GemmaProvider:
    """Local Gemma API provider for the coding agent framework"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 model_name: str = "gemma-2-9b-it",
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        """
        Initialize Gemma provider
        
        Args:
            base_url: Base URL of the Gemma API server
            model_name: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = requests.Session()
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test if the Gemma API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Connected to Gemma API: {health_data}")
            else:
                logger.warning(f"⚠️ Gemma API health check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Gemma API at {self.base_url}: {e}")
            logger.error("Make sure to start the Gemma API server first!")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send chat completion request to Gemma API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", 1.0)
        }
        
        try:
            # Use Claude-compatible endpoint
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                timeout=120  # 2 minute timeout for generation
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"🤖 Gemma response generated ({result.get('usage', {}).get('output_tokens', 'unknown')} tokens)")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("⏰ Gemma API request timed out")
            raise Exception("Gemma API timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"🚫 Gemma API request failed: {e}")
            raise Exception(f"Gemma API error: {e}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a simple prompt
        
        Args:
            prompt: Input prompt string
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages, **kwargs)
        
        # Extract content from Claude-style response
        if "content" in response and len(response["content"]) > 0:
            return response["content"][0]["text"]
        else:
            logger.error(f"Unexpected response format: {response}")
            return ""
    
    def create_system_message(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """
        Create messages with system prompt
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
            
        Returns:
            List of formatted messages
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

class GemmaLLM:
    """
    Wrapper class to make Gemma compatible with the agent framework's LLM interface
    This mimics the interface expected by the coding agent
    """
    
    def __init__(self, **kwargs):
        """Initialize with Gemma provider"""
        self.provider = GemmaProvider(**kwargs)
        self.model_name = kwargs.get("model_name", "gemma-2-9b-it")
    
    def __call__(self, messages, **kwargs):
        """Make the class callable like other LLM providers"""
        return self.provider.chat_completion(messages, **kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        return self.provider.generate_text(prompt, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat interface - returns just the text content"""
        response = self.provider.chat_completion(messages, **kwargs)
        
        if "content" in response and len(response["content"]) > 0:
            return response["content"][0]["text"]
        else:
            return ""

# Factory function for easy integration
def create_gemma_llm(**kwargs) -> GemmaLLM:
    """Create a Gemma LLM instance"""
    return GemmaLLM(**kwargs)
