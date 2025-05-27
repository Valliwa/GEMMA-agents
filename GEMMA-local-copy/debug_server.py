#!/usr/bin/env python3
"""
Debug script to test Gemma server and identify issues
"""

import requests
import json
import time
import sys

def test_server_connectivity():
    """Test basic server connectivity"""
    print("🔍 Testing server connectivity...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
            return True
        else:
            print(f"❌ Server returned {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server - is it running on localhost:8000?")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_health_endpoint():
    """Test health endpoint"""
    print("\n🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working")
            print(f"   Model loaded: {health_data.get('model_loaded', 'unknown')}")
            print(f"   Tokenizer loaded: {health_data.get('tokenizer_loaded', 'unknown')}")
            print(f"   GPU available: {health_data.get('gpu_available', 'unknown')}")
            print(f"   Device: {health_data.get('device', 'unknown')}")
            return health_data.get('model_loaded', False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_simple_api_call():
    """Test a simple API call"""
    print("\n💬 Testing simple API call...")
    
    payload = {
        "messages": [
            {"role": "user", "content": "Say 'Hello from Gemma test'"}
        ],
        "max_tokens": 20,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/messages",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful")
            if "content" in result and len(result["content"]) > 0:
                text = result["content"][0]["text"]
                print(f"   Response: {text}")
                return True
            else:
                print(f"❌ Unexpected response format: {result}")
                return False
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ API call timed out (this might indicate model loading issues)")
        return False
    except Exception as e:
        print(f"❌ API call error: {e}")
        return False

def test_error_conditions():
    """Test various error conditions"""
    print("\n🧪 Testing error conditions...")
    
    # Test invalid payload
    try:
        response = requests.post(
            "http://localhost:8000/v1/messages",
            json={"invalid": "payload"},
            timeout=10
        )
        print(f"   Invalid payload test: {response.status_code}")
    except Exception as e:
        print(f"   Invalid payload error: {e}")
    
    # Test empty messages
    try:
        response = requests.post(
            "http://localhost:8000/v1/messages",
            json={"messages": [], "max_tokens": 10},
            timeout=10
        )
        print(f"   Empty messages test: {response.status_code}")
    except Exception as e:
        print(f"   Empty messages error: {e}")

def main():
    """Run all tests"""
    print("🚀 Gemma Server Debug Script")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    if not test_server_connectivity():
        print("\n❌ Basic connectivity failed. Server might not be running.")
        print("💡 Start your server with: python gemma_server.py")
        return False
    
    # Test 2: Health check
    model_loaded = test_health_endpoint()
    if not model_loaded:
        print("\n⚠️ Model not loaded properly. Check server logs for loading errors.")
        print("Common issues:")
        print("- Missing or invalid Hugging Face token")
        print("- Insufficient GPU memory")
        print("- Model download interrupted")
        return False
    
    # Test 3: Simple API call
    if not test_simple_api_call():
        print("\n❌ API call failed. Check server logs for generation errors.")
        return False
    
    # Test 4: Error conditions
    test_error_conditions()
    
    print("\n" + "=" * 50)
    print("🎉 All basic tests passed!")
    print("Your Gemma server appears to be working correctly.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
