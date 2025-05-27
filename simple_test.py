#!/usr/bin/env python3
"""
Simple agent test that bypasses complex model validation
"""

import sys
import os
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the base_agent to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'base_agent', 'src'))

class SimpleGemmaClient:
    """Simple Gemma client for testing"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_connection(self):
        """Test if Gemma API is working"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt, max_tokens=100):
        """Generate text using Gemma API"""
        payload = {
            "model": "gemma-2-9b-it",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse the response
            if "content" in result and len(result["content"]) > 0:
                return result["content"][0]["text"]
            else:
                return f"Error: Unexpected response format: {result}"
                
        except Exception as e:
            return f"Error: {e}"

def test_gemma_direct():
    """Test Gemma API directly"""
    print("🧪 Testing Gemma API directly...")
    
    client = SimpleGemmaClient()
    
    # Test connection
    if not client.test_connection():
        print("❌ Gemma API not accessible. Make sure your server is running.")
        return False
    
    print("✅ Gemma API connection successful")
    
    # Test generation
    prompt = "Write a simple Python function that adds two numbers and returns the result."
    print(f"📝 Prompt: {prompt}")
    
    response = client.generate(prompt, max_tokens=200)
    print(f"🤖 Gemma Response:\n{response}")
    
    return True

def test_agent_minimal():
    """Test the agent framework minimally"""
    print("\n🤖 Testing agent framework...")
    
    try:
        # Try to import just what we need
        from config import GemmaAPIClient, settings
        
        print("✅ Config imported successfully")
        print(f"📋 Settings MODEL: {type(settings.MODEL)}")
        
        # Test our Gemma client
        gemma_client = GemmaAPIClient()
        test_messages = [{"role": "user", "content": "Say hello"}]
        
        response = gemma_client.chat(test_messages, max_tokens=50)
        print(f"🤖 Gemma client response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_coding_task():
    """Run a simple coding task using just our Gemma client"""
    print("\n📝 Running simple coding task...")
    
    client = SimpleGemmaClient()
    
    # Test if connection works
    if not client.test_connection():
        print("❌ Gemma server not accessible")
        return
    
    # Define a coding task
    coding_prompt = """
Write a Python function called 'fibonacci' that takes an integer n as input and returns the nth Fibonacci number.
Include docstring and example usage.
"""
    
    print(f"🎯 Task: {coding_prompt}")
    
    # Generate response
    print("🔄 Generating code with Gemma...")
    response = client.generate(coding_prompt, max_tokens=300)
    
    print("🤖 Generated Code:")
    print("=" * 50)
    print(response)
    print("=" * 50)

def main():
    """Main test function"""
    print("🚀 Simple Agent Test with Gemma")
    print("=" * 50)
    
    # Test 1: Direct Gemma API
    gemma_works = test_gemma_direct()
    
    if not gemma_works:
        print("\n❌ Gemma API not working. Please check your server.")
        print("💡 Make sure to run: python3 your_gemma_server.py")
        return
    
    # Test 2: Agent framework (minimal)
    agent_works = test_agent_minimal()
    
    # Test 3: Simple coding task
    run_simple_coding_task()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Gemma API: {'✅' if gemma_works else '❌'}")
    print(f"Agent Framework: {'✅' if agent_works else '❌'}")
    
    if gemma_works:
        print("\n🎉 Your Gemma integration is working!")
        print("💡 You can now work on integrating with the full agent framework.")
    else:
        print("\n❌ Issues found. Please fix Gemma server first.")

if __name__ == "__main__":
    main()
