#!/usr/bin/env python3
"""
Gemma Pre-Patch Script
Run this BEFORE starting the agent to ensure all patches are applied
"""

import sys
import os
import importlib
import logging

# Add the base_agent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'base_agent', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_anthropic_provider_directly():
    """Directly modify the anthropic provider file"""
    
    provider_file = "base_agent/src/llm/providers/anthropic.py"
    
    if not os.path.exists(provider_file):
        logger.error(f"❌ Anthropic provider file not found: {provider_file}")
        return False
    
    # Read the current file
    with open(provider_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "GEMMA_PATCHED" in content:
        logger.info("✅ Anthropic provider already patched")
        return True
    
    # Create the patch content
    patch_content = '''
# GEMMA_PATCHED - Auto-generated patch for Gemma integration
import requests
import logging

logger = logging.getLogger(__name__)

class GemmaClient:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.session = requests.Session()
    
    def messages(self, **kwargs):
        messages = kwargs.get('messages', [])
        max_tokens = kwargs.get('max_tokens', 1024)
        
        payload = {
            "model": "gemma-2-9b-it",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": 0.7
        }
        
        logger.info(f"🎯 PATCHED: Anthropic call redirected to Gemma ({len(messages)} messages)")
        
        try:
            response = self.session.post(f"{self.base_url}/v1/messages", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if "content" in result and len(result["content"]) > 0:
                text = result["content"][0]["text"]
                
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
                
                logger.info("✅ PATCHED: Gemma response received")
                return MockResponse(text)
            
        except Exception as e:
            logger.error(f"❌ PATCHED: Gemma API error: {e}")
            # Return error response in expected format
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.content = [MockContent(f"Error: {error_msg}")]
                    self.usage = MockUsage()
            
            return ErrorResponse(str(e))

# Global Gemma client
_GLOBAL_GEMMA_CLIENT = GemmaClient()

# Override the completion function
def complete(model, messages, **kwargs):
    """Patched completion function"""
    return _GLOBAL_GEMMA_CLIENT.messages(messages=messages, **kwargs)

# Override any provider client creation
class AnthropicProvider:
    def __init__(self, *args, **kwargs):
        logger.info("🎯 PATCHED: AnthropicProvider created -> Using Gemma")
        self.client = _GLOBAL_GEMMA_CLIENT
    
    def complete(self, model, messages, **kwargs):
        return self.client.messages(messages=messages, **kwargs)
    
    def create_completion(self, model, messages, **kwargs):
        return self.complete(model, messages, **kwargs)

logger.info("🚀 PATCHED: Anthropic provider patched for Gemma integration")

'''
    
    # Add the patch at the beginning of the file
    patched_content = patch_content + "\n\n" + content
    
    # Write the patched file
    with open(provider_file, 'w') as f:
        f.write(patched_content)
    
    logger.info("✅ Successfully patched anthropic provider file")
    return True

def set_environment():
    """Set environment variables"""
    os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-patched-for-gemma-integration'
    os.environ['OPENAI_API_KEY'] = 'sk-patched-for-gemma'
    logger.info("✅ Set environment variables")

def main():
    """Main patching function"""
    logger.info("🔧 Starting Gemma pre-patch process...")
    
    # Set environment
    set_environment()
    
    # Patch the provider file directly
    if patch_anthropic_provider_directly():
        logger.info("🎉 All patches applied successfully!")
        logger.info("🚀 Now run the agent - all LLM calls will go to Gemma")
        return True
    else:
        logger.error("❌ Patching failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
