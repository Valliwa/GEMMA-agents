#!/usr/bin/env python3
"""
Debug script to see what models are available in the agent framework
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'base_agent', 'src'))

def debug_models():
    """Debug available models"""
    print("🔍 Debugging available models...")
    
    try:
        from base_agent.src.llm.models import MODELS
        print(f"✅ Found {len(MODELS)} models:")
        
        for i, model in enumerate(MODELS[:5]):  # Show first 5 models
            print(f"  {i+1}. {model}")
            if hasattr(model, 'api_name'):
                print(f"     API Name: {model.api_name}")
            if hasattr(model, 'provider'):
                print(f"     Provider: {model.provider}")
            if hasattr(model, 'fci'):
                print(f"     FCI: {model.fci}")
            print()
            
        # Find a Claude model
        claude_models = [m for m in MODELS if hasattr(m, 'api_name') and 'claude' in m.api_name.lower()]
        if claude_models:
            print(f"✅ Found {len(claude_models)} Claude models")
            chosen_model = claude_models[0]
            print(f"🎯 Will use: {chosen_model}")
            print(f"   API Name: {chosen_model.api_name}")
            print(f"   Provider: {chosen_model.provider}")
            
            return chosen_model
        else:
            print("⚠️ No Claude models found, using first available model")
            return MODELS[0] if MODELS else None
            
    except ImportError as e:
        print(f"❌ Could not import MODELS: {e}")
        return None
    except Exception as e:
        print(f"❌ Error debugging models: {e}")
        return None

def debug_config():
    """Debug our config"""
    print("\n🔍 Debugging our config...")
    
    try:
        from config import settings
        print("✅ Config imported successfully")
        
        print(f"Default LLM: {settings.default_llm}")
        
        # Check model attributes
        model = settings.MODEL
        print(f"MODEL type: {type(model)}")
        
        if hasattr(model, 'fci'):
            print(f"MODEL.fci: {model.fci}")
        else:
            print("❌ MODEL missing 'fci' attribute")
            
        if hasattr(model, 'api_name'):
            print(f"MODEL.api_name: {model.api_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    model = debug_models()
    debug_config()
    
    print("\n🎯 Recommendations:")
    if model:
        print(f"✅ Use this model in your settings: {model}")
    else:
        print("❌ No suitable model found - check your imports")
