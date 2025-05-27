#!/usr/bin/env python3
"""
Simple test script for Gemma 3 27B integration
Tests the API server and basic agent functionality step by step
"""

import requests
import json
import time
import sys
from pathlib import Path

def print_header(title):
    """Print a nice header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """Print a test step"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def test_1_server_health():
    """Test 1: Check if Gemma 3 API server is running"""
    print_step("1", "Testing Gemma 3 API Server Health")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Server is healthy!")
            print(f"📊 Status: {health_data.get('status', 'unknown')}")
            print(f"🧠 Model: {health_data.get('model', 'unknown')}")
            
            if 'gpu_memory' in health_data:
                gpu_info = health_data['gpu_memory']
                print(f"💾 GPU Memory: {gpu_info}")
            
            if 'enhanced_features' in health_data:
                features = health_data['enhanced_features']
                print(f"🎯 Features: {features}")
            
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        print("💡 Make sure gemma_api_server.py is running:")
        print("   python3 gemma_api_server.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_2_simple_generation():
    """Test 2: Simple text generation"""
    print_step("2", "Testing Simple Text Generation")
    
    test_payload = {
        "model": "gemma-3-27b-it",
        "messages": [
            {"role": "user", "content": "Hello! Please introduce yourself briefly."}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        print("📤 Sending request to Gemma 3...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/v1/messages",
            json=test_payload,
            timeout=60
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if 'content' in result:
                content = result['content'][0]['text'] if isinstance(result['content'], list) else result['content']
                
                print("✅ Generation successful!")
                print(f"⏱️ Response time: {response_time:.2f} seconds")
                print(f"📝 Response:")
                print("-" * 30)
                print(content)
                print("-" * 30)
                
                if 'usage' in result:
                    usage = result['usage']
                    print(f"📊 Token usage: {usage.get('input_tokens', 0)} input + {usage.get('output_tokens', 0)} output")
                
                return True
            else:
                print("❌ No content in response")
                print(f"Raw response: {result}")
                return False
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation test failed: {str(e)}")
        return False

def test_3_coding_task():
    """Test 3: Coding task to test reasoning capabilities"""
    print_step("3", "Testing Coding Task (Advanced Reasoning)")
    
    coding_prompt = """Write a Python function that:
1. Takes a list of numbers as input
2. Filters out even numbers
3. Squares the odd numbers
4. Returns the sum of the squared odd numbers
5. Include proper error handling and documentation

Example: [1, 2, 3, 4, 5] should return 35 (1² + 3² + 5² = 1 + 9 + 25 = 35)"""

    test_payload = {
        "model": "gemma-3-27b-it", 
        "messages": [
            {"role": "user", "content": coding_prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.3
    }
    
    try:
        print("📤 Sending coding task...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/v1/messages",
            json=test_payload,
            timeout=120
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if 'content' in result:
                content = result['content'][0]['text'] if isinstance(result['content'], list) else result['content']
                
                print("✅ Coding task completed!")
                print(f"⏱️ Response time: {response_time:.2f} seconds")
                print(f"📝 Generated code:")
                print("-" * 50)
                print(content)
                print("-" * 50)
                
                # Check if response contains key elements
                checks = {
                    "Contains 'def'": "def " in content,
                    "Contains 'return'": "return" in content,
                    "Contains error handling": any(word in content.lower() for word in ["try", "except", "error", "raise"]),
                    "Contains documentation": any(word in content for word in ['"""', "'''", "Args:", "Returns:"]),
                    "Mentions filtering": any(word in content.lower() for word in ["filter", "even", "odd"])
                }
                
                print("🔍 Code quality checks:")
                for check, passed in checks.items():
                    status = "✅" if passed else "❌"
                    print(f"   {status} {check}")
                
                passed_checks = sum(checks.values())
                print(f"\n📊 Quality score: {passed_checks}/{len(checks)} checks passed")
                
                if 'usage' in result:
                    usage = result['usage']
                    print(f"📊 Token usage: {usage}")
                
                return passed_checks >= 3  # Consider successful if at least 3/5 checks pass
            else:
                print("❌ No content in response")
                return False
        else:
            print(f"❌ Coding task failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Coding task failed: {str(e)}")
        return False

def test_4_context_window():
    """Test 4: Test large context window capabilities"""
    print_step("4", "Testing Large Context Window (128K)")
    
    # Create a longer context to test Gemma 3's capabilities
    long_context = """
Context: You are analyzing a complex software system. Here are the details:

""" + "\n".join([f"Module {i}: This module handles {'database operations' if i % 3 == 0 else 'user interface' if i % 3 == 1 else 'business logic'} and contains approximately {100 + i*20} lines of code." for i in range(1, 51)])

    context_prompt = f"""{long_context}

Based on all the module information provided above:
1. How many modules handle database operations?
2. What's the total estimated lines of code across all modules?
3. Recommend which modules should be refactored first and why.

Please be specific and reference the module numbers in your analysis."""

    test_payload = {
        "model": "gemma-3-27b-it",
        "messages": [
            {"role": "user", "content": context_prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        print(f"📤 Testing context window with ~{len(context_prompt)} characters...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/v1/messages", 
            json=test_payload,
            timeout=90
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if 'content' in result:
                content = result['content'][0]['text'] if isinstance(result['content'], list) else result['content']
                
                print("✅ Large context processing successful!")
                print(f"⏱️ Response time: {response_time:.2f} seconds")
                print(f"📝 Analysis result:")
                print("-" * 50)
                print(content)
                print("-" * 50)
                
                # Check if the response demonstrates understanding of the context
                context_checks = {
                    "References specific modules": any(f"module {i}" in content.lower() for i in range(1, 51)),
                    "Mentions database operations": "database" in content.lower(),
                    "Provides numerical analysis": any(str(i) in content for i in range(10, 100)),
                    "Makes recommendations": any(word in content.lower() for word in ["recommend", "suggest", "should", "refactor"])
                }
                
                print("🔍 Context understanding checks:")
                for check, passed in context_checks.items():
                    status = "✅" if passed else "❌"
                    print(f"   {status} {check}")
                
                return sum(context_checks.values()) >= 2
            else:
                print("❌ No content in response")
                return False
        else:
            print(f"❌ Context test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Context window test failed: {str(e)}")
        return False

def test_5_server_stats():
    """Test 5: Check server performance statistics"""
    print_step("5", "Testing Server Performance Statistics")
    
    try:
        response = requests.get("http://localhost:8000/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            
            print("✅ Stats retrieved successfully!")
            print("📊 Server Performance:")
            print(f"   Model: {stats.get('model', 'unknown')}")
            
            if 'gpu_memory' in stats:
                gpu_info = stats['gpu_memory']
                print(f"   GPU Memory Allocated: {gpu_info.get('allocated_gb', 'N/A')} GB")
                print(f"   GPU Memory Reserved: {gpu_info.get('reserved_gb', 'N/A')} GB")
                print(f"   GPU Utilization: {gpu_info.get('utilization_percent', 'N/A')}%")
            
            if 'model_info' in stats:
                model_info = stats['model_info']
                print(f"   Parameters: {model_info.get('parameters', 'N/A')}")
                print(f"   Quantization: {model_info.get('quantization', 'N/A')}")
                print(f"   Context Window: {model_info.get('context_window', 'N/A')}")
                print(f"   Precision: {model_info.get('precision', 'N/A')}")
            
            return True
        else:
            print(f"❌ Stats endpoint returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Stats test failed: {str(e)}")
        return False

def test_6_integration_components():
    """Test 6: Check if integration components are ready"""
    print_step("6", "Testing SICA Integration Components")
    
    components_to_check = [
        ("base_agent/src/config.py", "Main configuration file"),
        ("base_agent/src/llm/", "LLM provider directory"),
        ("runner.py", "Main runner script"),
        ("base_agent/", "Base agent directory")
    ]
    
    all_good = True
    
    for component_path, description in components_to_check:
        path = Path(component_path)
        if path.exists():
            print(f"✅ {description}: Found")
        else:
            print(f"❌ {description}: Missing ({component_path})")
            all_good = False
    
    # Test if we can import key modules
    print("\n🔍 Testing Python imports...")
    
    try:
        # Add the base_agent to Python path
        import sys
        sys.path.append("base_agent")
        
        # Test basic imports
        from src.llm.api import create_completion
        print("✅ LLM API import: Success")
        
        from src.llm.base import Message
        print("✅ Message class import: Success")
        
    except ImportError as e:
        print(f"❌ Import test failed: {str(e)}")
        all_good = False
    except Exception as e:
        print(f"⚠️ Import test error: {str(e)}")
    
    return all_good

def main():
    """Run all tests"""
    print_header("Gemma 3 27B Simple Integration Test")
    
    print("🚀 Testing Gemma 3 27B integration with SICA step by step...")
    print("This will validate that everything is working before running the full system.")
    
    tests = [
        ("Server Health", test_1_server_health),
        ("Simple Generation", test_2_simple_generation),
        ("Coding Task", test_3_coding_task),
        ("Context Window", test_4_context_window),
        ("Server Stats", test_5_server_stats),
        ("Integration Components", test_6_integration_components)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    total_time = time.time() - start_time
    
    # Final summary
    print_header("Test Results Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 Tests Results: {passed}/{total} passed")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print()
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print()
    
    if passed == total:
        print("🎉 All tests passed! Gemma 3 27B is ready for SICA!")
        print("\n🚀 Next steps:")
        print("1. Your Gemma 3 27B integration is working perfectly")
        print("2. You can now run the full SICA system:")
        print("   python runner.py --test-gemma3  # Test runner integration")
        print("   python runner.py test --name gsm8k  # Test single benchmark")
        print("   python runner.py --id 1 --iterations 5  # Full SICA run")
        return 0
    elif passed >= 4:
        print("⚠️ Most tests passed - system should work with minor issues")
        print("💡 You can proceed with caution or fix the failing tests")
        return 0
    else:
        print("❌ Multiple tests failed - please address these issues:")
        print()
        for test_name, result in results:
            if not result:
                print(f"   - Fix: {test_name}")
        print()
        print("💡 Most common issues:")
        print("   - Gemma 3 server not running: python3 gemma_api_server.py")
        print("   - Missing integration files: Create config updates")
        print("   - GPU memory issues: Check nvidia-smi")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)
