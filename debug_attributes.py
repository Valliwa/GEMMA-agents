#!/usr/bin/env python3
"""
Debug what attributes the agent framework expects from model objects
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'base_agent', 'src'))

def debug_model_usage():
    """Find where the 'fci' attribute is used in the codebase"""
    
    print("🔍 Searching for 'fci' attribute usage...")
    
    # Search for files that use .fci
    import glob
    import re
    
    base_agent_files = glob.glob("base_agent/**/*.py", recursive=True)
    
    fci_usage = []
    for file_path in base_agent_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if '.fci' in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if '.fci' in line:
                            fci_usage.append({
                                'file': file_path,
                                'line': i + 1,
                                'code': line.strip()
                            })
        except:
            continue
    
    print(f"Found {len(fci_usage)} places where .fci is used:")
    for usage in fci_usage[:10]:  # Show first 10
        print(f"  📁 {usage['file']}:{usage['line']}")
        print(f"     {usage['code']}")
        print()

def test_model_creation():
    """Test our model creation with detailed debugging"""
    
    print("🧪 Testing model creation...")
    
    try:
        from config import settings
        
        model = settings.MODEL
        print(f"✅ Model type: {type(model)}")
        print(f"✅ Model: {model}")
        
        # Test all expected attributes
        expected_attrs = ['fci', 'id', 'api_name', 'provider', 'function_calling_interface', 'preferred_arg_format']
        
        print("\n📋 Checking expected attributes:")
        for attr in expected_attrs:
            if hasattr(model, attr):
                value = getattr(model, attr)
                print(f"  ✅ {attr}: {value} (type: {type(value)})")
            else:
                print(f"  ❌ {attr}: MISSING")
        
        print(f"\n📋 All model attributes:")
        for attr in sorted(dir(model)):
            if not attr.startswith('_'):
                try:
                    value = getattr(model, attr)
                    print(f"  • {attr}: {value}")
                except:
                    print(f"  • {attr}: <error accessing>")
                    
        return model
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agent_creation():
    """Test creating an agent with our model"""
    
    print("\n🤖 Testing agent creation...")
    
    try:
        from base_agent.src.agents.base_agent import BaseAgent
        
        print("Attempting to create BaseAgent...")
        agent = BaseAgent()
        print("✅ BaseAgent created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        
        # Check if it's specifically the fci error
        if "'ModelInfo' object has no attribute 'fci'" in str(e):
            print("🎯 Confirmed: The issue is the missing 'fci' attribute")
        
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debugging Model Attributes")
    print("=" * 50)
    
    # 1. Search for fci usage
    debug_model_usage()
    
    # 2. Test our model
    model = test_model_creation()
    
    # 3. Test agent creation  
    if model:
        test_agent_creation()
