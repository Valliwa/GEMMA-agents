#!/usr/bin/env python3
"""
Investigation script to check existing Gemma support in the framework
"""

import os
import sys
import glob
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_section(title):
    print(f"\n📋 {title}")
    print('-'*40)

def safe_read_file(filepath, max_lines=50):
    """Safely read a file and return its contents"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > max_lines:
                content = ''.join(lines[:max_lines])
                content += f"\n... (truncated, showing first {max_lines} lines) ..."
            else:
                content = ''.join(lines)
            return content
    except Exception as e:
        return f"Error reading file: {e}"

def find_files_with_content(directory, pattern, content_pattern=None):
    """Find files matching pattern, optionally containing specific content"""
    results = []
    try:
        for filepath in glob.glob(f"{directory}/**/{pattern}", recursive=True):
            if content_pattern:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if content_pattern.lower() in f.read().lower():
                            results.append(filepath)
                except:
                    pass
            else:
                results.append(filepath)
    except Exception as e:
        print(f"Error searching: {e}")
    return results

def investigate_gemma_support():
    """Main investigation function"""
    print_header("Investigating Native Gemma Support in Self-Improving Coding Agent")
    
    # Check if we're in the right directory
    if not os.path.exists("base_agent"):
        print("❌ Not in the correct directory. Please run from the self_improving_coding_agent root directory.")
        return False
    
    print("✅ Found base_agent directory")
    
    # 1. Look for GemmaProvider
    print_section("1. Checking for GemmaProvider Implementation")
    
    gemma_provider_files = find_files_with_content("base_agent", "*gemma*")
    if gemma_provider_files:
        print("✅ Found Gemma-related files:")
        for f in gemma_provider_files:
            print(f"   - {f}")
            
        # Check the main provider file
        provider_file = "base_agent/src/llm/providers/gemma_provider.py"
        if os.path.exists(provider_file):
            print(f"\n📄 Contents of {provider_file}:")
            print("-" * 40)
            content = safe_read_file(provider_file)
            print(content)
        else:
            print("⚠️ gemma_provider.py not found at expected location")
    else:
        print("❌ No Gemma-related files found")
    
    # 2. Check Provider enum
    print_section("2. Checking Provider Enum Definition")
    
    types_file = "base_agent/src/types/llm_types.py"
    if os.path.exists(types_file):
        print(f"✅ Found {types_file}")
        content = safe_read_file(types_file, max_lines=200)
        
        if "GEMMA" in content.upper():
            print("✅ GEMMA found in types file")
            # Extract relevant sections
            lines = content.split('\n')
            in_provider_class = False
            provider_lines = []
            
            for i, line in enumerate(lines):
                if "class Provider" in line:
                    in_provider_class = True
                    provider_lines.extend(lines[i:i+20])  # Get next 20 lines
                    break
            
            if provider_lines:
                print("\n📄 Provider enum definition:")
                print("-" * 30)
                for line in provider_lines:
                    if line.strip() and not line.startswith('class'):
                        print(line)
                    if 'GEMMA' in line.upper():
                        print(f"🎯 --> {line} <-- 🎯")
        else:
            print("❌ GEMMA not found in types file")
    else:
        print(f"❌ Types file not found: {types_file}")
    
    # 3. Check Model definitions
    print_section("3. Checking Model Definitions")
    
    if os.path.exists(types_file):
        content = safe_read_file(types_file, max_lines=500)
        
        # Look for Model class or enum
        if "class Model" in content or "Model =" in content:
            print("✅ Found Model definitions")
            
            if "GEMMA" in content.upper():
                print("✅ GEMMA models found in definitions")
                # Extract Model-related lines
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'GEMMA' in line.upper():
                        print(f"🎯 {line.strip()}")
            else:
                print("❌ No GEMMA models found in definitions")
        else:
            print("⚠️ Model definitions not found or in unexpected format")
    
    # 4. Check providers directory
    print_section("4. Checking Providers Directory")
    
    providers_dir = "base_agent/src/llm/providers"
    if os.path.exists(providers_dir):
        print(f"✅ Found providers directory: {providers_dir}")
        
        files = os.listdir(providers_dir)
        print("📁 Provider files:")
        for f in sorted(files):
            if f.endswith('.py'):
                print(f"   - {f}")
                if 'gemma' in f.lower():
                    print(f"   🎯 --> {f} <-- (Gemma-related!)")
    else:
        print(f"❌ Providers directory not found: {providers_dir}")
    
    # 5. Check imports in api.py
    print_section("5. Checking API.py Imports")
    
    api_file = "base_agent/src/llm/api.py"
    if os.path.exists(api_file):
        content = safe_read_file(api_file, max_lines=100)
        
        if "GemmaProvider" in content:
            print("✅ GemmaProvider is imported in api.py")
            
            # Find the import line
            lines = content.split('\n')
            for line in lines:
                if "GemmaProvider" in line:
                    print(f"📄 Import line: {line.strip()}")
        else:
            print("❌ GemmaProvider not found in api.py imports")
    
    # 6. Check configuration integration points
    print_section("6. Checking Configuration Integration")
    
    config_file = "base_agent/src/config.py"
    if os.path.exists(config_file):
        print("✅ Found existing config.py")
        content = safe_read_file(config_file, max_lines=50)
        
        if "GEMMA" in content.upper():
            print("✅ GEMMA references found in config.py")
        else:
            print("❌ No GEMMA references in existing config.py")
            print("💡 This means your custom config.py will override the existing one")
    else:
        print("⚠️ No existing config.py found - your custom config will be the only one")
    
    # 7. Summary and recommendations
    print_section("7. Summary and Recommendations")
    
    # Check what we found
    has_provider_import = os.path.exists("base_agent/src/llm/api.py")
    has_providers_dir = os.path.exists("base_agent/src/llm/providers")
    has_types_file = os.path.exists("base_agent/src/types/llm_types.py")
    
    if has_provider_import and has_providers_dir and has_types_file:
        print("✅ Framework structure looks complete")
        
        # Check if GemmaProvider actually exists
        gemma_provider_path = "base_agent/src/llm/providers/gemma_provider.py"
        if os.path.exists(gemma_provider_path):
            print("✅ GemmaProvider file exists - native support likely available!")
            print("\n💡 RECOMMENDATION: Try using native Gemma support first")
            print("   1. Check if Provider.GEMMA is defined in types")
            print("   2. Test if GemmaProvider works with your server")
            print("   3. Use your config.py as backup/enhancement")
        else:
            print("⚠️ GemmaProvider file missing - you may need to create it")
            print("\n💡 RECOMMENDATION: Create GemmaProvider or use config.py approach")
            print("   1. Either implement missing GemmaProvider")
            print("   2. Or rely on your comprehensive config.py patches")
    else:
        print("⚠️ Framework structure incomplete")
        print("\n💡 RECOMMENDATION: Use your config.py approach")
        print("   Your comprehensive patching approach is the best option")
    
    return True

if __name__ == "__main__":
    success = investigate_gemma_support()
    if not success:
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("🎯 NEXT STEPS:")
    print("1. Review the output above")
    print("2. Check if GemmaProvider needs to be implemented")
    print("3. Test native support if available")
    print("4. Fall back to config.py approach if needed")
    print("5. Run integration tests")
    print('='*60)
