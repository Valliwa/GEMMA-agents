#!/usr/bin/env python3
"""
Test script to verify native Gemma provider integration works
"""

import sys
import os
import asyncio

# Add path and import config FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'base_agent', 'src'))

def test_config_import():
    """Test that config imports successfully"""
    print("🔧 Testing config import...")
    try:
        from config import GemmaAPIClient, settings
        print("✅ Config imported successfully")
        
        # Test GemmaAPIClient creation
        client = GemmaAPIClient()
        print("✅ GemmaAPIClient created")
        
        # Test if generate method exists
        if hasattr(client, 'generate'):
            print("✅ generate() method found")
        else:
            print("❌ generate() method missing - add it to your GemmaAPIClient class")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

async def test_native_provider():
    """Test the native provider integration"""
    print("\n🎯 Testing native provider integration...")
    
    try:
        # Import framework components
        from base_agent.src.llm.api import create_completion
        from base_agent.src.llm.base import Message, TextContent
        from base_agent.src.types.llm_types import Model
        
        print("✅ Framework imports successful")
        
        # Find the Gemma model enum value
        gemma_models = [model for model in Model if 'GEMMA' in str(model)]
        if not gemma_models:
            print("❌ No GEMMA models found in Model enum")
            print("Available models:")
            for model in Model:
                print(f"   - {model}")
            return False
        
        gemma_model = gemma_models[0]  # Use the first Gemma model found
        print(f"✅ Using model: {gemma_model}")
        
        # Create test message
        messages = [Message(
            role="user", 
            content=[TextContent(text="Say 'Native integration test successful!'")])
        ]
        
        print("🔄 Making native API call...")
        
        # Use native create_completion function
        completion = await create_completion(
            messages=messages,
            model=gemma_model,
            max_tokens=50,
            temperature=0.3
        )
        
        print("✅ Native completion successful!")
        print(f"📝 Response: {completion.content[0].text}")
        print(f"📊 Tokens used: {completion.usage.completion_tokens}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the self_improving_coding_agent directory")
        return False
    except Exception as e:
        print(f"❌ Native provider test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_client():
    """Test direct client usage"""
    print("\n💬 Testing direct client usage...")
    
    try:
        from config import GemmaAPIClient
        
        client = GemmaAPIClient()
        
        # Test the generate method directly
        response = client.generate("Say 'Direct client test successful!'", max_tokens=20)
        print(f"✅ Direct client test: {response}")
        
        return True
    except Exception as e:
        print(f"❌ Direct client test failed: {e}")
        return False

async def run_all_tests():
    """Run all integration tests"""
    print("🧪 Running Native Integration Tests")
    print("=" * 50)
    
    # Test 1: Config import
    if not test_config_import():
        return False
    
    # Test 2: Direct client
    if not test_direct_client():
        return False
    
    # Test 3: Native provider
    if not await test_native_provider():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("Your native Gemma integration is working perfectly!")
    print("🚀 Ready to run the self-improving coding agent!")
    print("=" * 50)
    
    return True

def main():
    """Main test function"""
    try:
        success = asyncio.run(run_all_tests())
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
