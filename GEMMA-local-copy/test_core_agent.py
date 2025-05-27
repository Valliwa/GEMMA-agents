import sys
sys.path.insert(0, '/home/icnfs/ma/v/vtw23/GEMMA/packages')
sys.path.insert(0, '.')

try:
    # Test basic imports first
    from base_agent.src.config import settings
    print("✅ Config imported")
    
    # Test Gemma connection
    import requests
    response = requests.get('http://localhost:8000/health')
    print("✅ Gemma connection:", response.json()['status'])
    
    # Test LLM creation
    llm = settings.get_llm()
    print("✅ LLM created:", type(llm))
    
    print("🎉 Core components working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
