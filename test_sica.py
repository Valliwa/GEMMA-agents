#!/usr/bin/env python3
"""
Test script for SICA (Self-Improving Coding Agent) with local Gemma integration
Tests the asyncio.timeout fix and verifies the agent functionality
"""

import os
import sys
import subprocess
import time
import json
import requests
from datetime import datetime

# Add the project path
project_path = "/home/icnfs/ma/v/vtw23/GEMMA/self_improving_coding_agent"
sys.path.insert(0, os.path.join(project_path, "base_agent", "src"))

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {text}")
    print(f"{'='*60}")

def print_status(text, status="INFO"):
    """Print status with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
    print(f"[{timestamp}] {emoji} {text}")

def test_environment():
    """Test the environment setup"""
    print_header("Testing Environment Setup")
    
    # Check Python version
    python_version = sys.version_info
    print_status(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print_status("Python 3.8+ required", "ERROR")
        return False
    
    # Check if project directory exists
    if not os.path.exists(project_path):
        print_status(f"Project directory not found: {project_path}", "ERROR")
        return False
    
    print_status("Environment check passed", "SUCCESS")
    return True

def test_gemma_server():
    """Test Gemma server connectivity"""
    print_header("Testing Gemma Server Connection")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print_status("Gemma server is healthy", "SUCCESS")
            print_status(f"Model: {health_data.get('model', 'unknown')}")
            return True
        else:
            print_status(f"Server returned status {response.status_code}", "ERROR")
            return False
            
    except requests.exceptions.ConnectionError:
        print_status("Cannot connect to Gemma server on localhost:8000", "ERROR")
        print_status("Please ensure your Gemma server is running")
        return False
    except Exception as e:
        print_status(f"Server test failed: {e}", "ERROR")
        return False

def test_simple_gemma_request():
    """Test a simple request to Gemma"""
    print_header("Testing Simple Gemma Request")
    
    try:
        payload = {
            "model": "gemma-3-27b-it",
            "messages": [{"role": "user", "content": "Say 'Hello from Gemma!' in exactly those words."}],
            "max_tokens": 20
        }
        
        print_status("Sending test request to Gemma...")
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print_status(f"Gemma response: {content}", "SUCCESS")
            return True
        else:
            print_status(f"Request failed with status {response.status_code}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Gemma request test failed: {e}", "ERROR")
        return False

def test_asyncio_fix():
    """Test that the asyncio.timeout fix was applied"""
    print_header("Testing AsyncIO Fix")
    
    try:
        # Import the callgraph manager to test the fix
        from callgraph.manager import CallGraphManager
        print_status("CallGraphManager imported successfully", "SUCCESS")
        
        # Check if asyncio.timeout was replaced
        import inspect
        source = inspect.getsource(CallGraphManager)
        
        if "asyncio.timeout" in source:
            print_status("asyncio.timeout still present - fix may not be applied", "ERROR")
            return False
        elif "asyncio.wait_for" in source or "timeout" not in source:
            print_status("AsyncIO fix appears to be applied", "SUCCESS")
            return True
        else:
            print_status("Cannot verify asyncio fix", "ERROR")
            return False
            
    except ImportError as e:
        print_status(f"Cannot import CallGraphManager: {e}", "ERROR")
        return False
    except Exception as e:
        print_status(f"AsyncIO test failed: {e}", "ERROR")
        return False

def run_simple_agent_test():
    """Run a simple agent test"""
    print_header("Running Simple Agent Test")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env["SICA_READ_ONLY"] = "true"
        env["SICA_DRY_RUN"] = "true"
        
        # Change to project directory
        os.chdir(project_path)
        
        print_status("Starting simple agent test...")
        print_status("Task: 'Say hello and list 3 files in the current directory'")
        
        # Run the agent with a simple task
        cmd = [
            sys.executable, "-m", "base_agent.agent", 
            "--server", 
            "-p", "Say hello and list 3 files in the current directory"
        ]
        
        start_time = time.time()
        
        # Run with timeout to prevent hanging
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
            duration = time.time() - start_time
            
            if process.returncode == 0:
                print_status(f"Agent test completed successfully in {duration:.1f}s", "SUCCESS")
                return True, stdout, stderr
            else:
                print_status(f"Agent test failed with return code {process.returncode}", "ERROR")
                return False, stdout, stderr
                
        except subprocess.TimeoutExpired:
            process.kill()
            print_status("Agent test timed out after 5 minutes", "ERROR")
            return False, "", "Timeout"
            
    except Exception as e:
        print_status(f"Agent test setup failed: {e}", "ERROR")
        return False, "", str(e)

def run_full_analysis_test():
    """Run the full base_agent directory analysis"""
    print_header("Running Full Directory Analysis Test")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env["SICA_READ_ONLY"] = "true"
        env["SICA_DRY_RUN"] = "true"
        
        # Change to project directory
        os.chdir(project_path)
        
        print_status("Starting full analysis test...")
        print_status("Task: 'Analyze the base_agent directory structure and suggest 3 specific improvements'")
        
        # Run the agent with the analysis task
        cmd = [
            sys.executable, "-m", "base_agent.agent", 
            "--server", 
            "-p", "Analyze the base_agent directory structure and suggest 3 specific improvements"
        ]
        
        start_time = time.time()
        
        # Run with longer timeout for analysis
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=600)  # 10 minute timeout
            duration = time.time() - start_time
            
            if process.returncode == 0:
                print_status(f"Full analysis completed successfully in {duration:.1f}s", "SUCCESS")
                return True, stdout, stderr
            else:
                print_status(f"Full analysis failed with return code {process.returncode}", "ERROR")
                return False, stdout, stderr
                
        except subprocess.TimeoutExpired:
            process.kill()
            print_status("Full analysis timed out after 10 minutes", "ERROR")
            return False, "", "Timeout"
            
    except Exception as e:
        print_status(f"Full analysis setup failed: {e}", "ERROR")
        return False, "", str(e)

def check_results():
    """Check the generated results"""
    print_header("Checking Generated Results")
    
    results_dir = os.path.join(project_path, "agent_work", "agent_outputs")
    
    if not os.path.exists(results_dir):
        print_status("Results directory not found", "ERROR")
        return False
    
    # Check for recent files
    files = []
    for root, dirs, filenames in os.walk(results_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            mtime = os.path.getmtime(filepath)
            files.append((filepath, mtime))
    
    # Sort by modification time
    files.sort(key=lambda x: x[1], reverse=True)
    
    print_status(f"Found {len(files)} result files")
    
    # Show the 5 most recent files
    for filepath, mtime in files[:5]:
        relative_path = os.path.relpath(filepath, results_dir)
        mod_time = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        file_size = os.path.getsize(filepath)
        print_status(f"  {relative_path} ({file_size} bytes, {mod_time})")
    
    return len(files) > 0

def main():
    """Run all tests"""
    print_header("SICA Agent Test Suite")
    print_status("Testing Self-Improving Coding Agent with Local Gemma")
    
    # Track test results
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Environment
    total_tests += 1
    if test_environment():
        tests_passed += 1
    
    # Test 2: Gemma server
    total_tests += 1
    if test_gemma_server():
        tests_passed += 1
    else:
        print_status("Skipping remaining tests - Gemma server not available", "ERROR")
        return
    
    # Test 3: Simple Gemma request
    total_tests += 1
    if test_simple_gemma_request():
        tests_passed += 1
    
    # Test 4: AsyncIO fix
    total_tests += 1
    if test_asyncio_fix():
        tests_passed += 1
    
    # Test 5: Simple agent test
    total_tests += 1
    success, stdout, stderr = run_simple_agent_test()
    if success:
        tests_passed += 1
        print_status("Simple agent test output (last 10 lines):")
        for line in stdout.split('\n')[-10:]:
            if line.strip():
                print(f"    {line}")
    else:
        print_status("Simple agent test failed")
        if stderr:
            print_status(f"Error: {stderr[:200]}...")
    
    # Test 6: Check results
    total_tests += 1
    if check_results():
        tests_passed += 1
    
    # Optional: Full analysis test (only if simple test passed)
    if tests_passed >= 4:  # Most basic tests passed
        print_status("Basic tests passed, running full analysis...")
        success, stdout, stderr = run_full_analysis_test()
        if success:
            print_status("🎉 FULL ANALYSIS TEST PASSED! 🎉", "SUCCESS")
        else:
            print_status("Full analysis test failed, but basic functionality works")
    
    # Summary
    print_header("Test Results Summary")
    print_status(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print_status("🎉 ALL TESTS PASSED! Your SICA agent is working perfectly! 🎉", "SUCCESS")
    elif tests_passed >= total_tests - 2:
        print_status("✨ Most tests passed! Your SICA agent is working well! ✨", "SUCCESS")
    else:
        print_status("⚠️ Some tests failed. Check the output above for details.", "ERROR")

if __name__ == "__main__":
    main()
