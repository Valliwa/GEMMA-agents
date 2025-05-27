#!/usr/bin/env python3
"""
Web-enabled agent runner with Gemma integration
This script ensures the web interface starts properly
"""

import sys
import os
import asyncio

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_agent_path = os.path.join(current_dir, 'base_agent')
src_path = os.path.join(base_agent_path, 'src')

sys.path.insert(0, base_agent_path)
sys.path.insert(0, src_path)

print("🔧 Loading Gemma configuration...")

# Import config first to apply Gemma patches
from config import settings
print("✅ Gemma configuration loaded and patches applied")

# Set command line arguments for web server
sys.argv = [
    'agent',
    '--server', 'true',
    '--port', '8080',
    '-p', 'Write a python script for a bouncing yellow ball within a square, make sure to handle collision detection properly. Make the square slowly rotate. Implement it in python. Make sure the ball stays within the square.'
]

print("🌐 Starting agent with web interface...")
print("🔗 Web interface will be available at: http://localhost:8080")
print("📱 Open your browser and navigate to http://localhost:8080")
print("")

# Import and run the agent
try:
    from base_agent.agent import main as agent_main
    
    if asyncio.iscoroutinefunction(agent_main):
        print("🔄 Running async agent...")
        result = asyncio.run(agent_main())
    else:
        print("🔄 Running sync agent...")
        result = agent_main()
        
    print(f"✅ Agent completed with result: {result}")
    
except KeyboardInterrupt:
    print("\n⏹️ Agent stopped by user")
except Exception as e:
    print(f"❌ Error running agent: {e}")
    import traceback
    traceback.print_exc()
