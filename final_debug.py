#!/usr/bin/env python3
"""
Final debug of ModelManager initialization in container
"""
import os
import sys
import json

# Add src to path
sys.path.append('/app/src')

def debug_step_by_step():
    """Debug step by step"""
    
    print("🔍 Final ModelManager Debug")
    print("=" * 50)
    
    print(f"📍 Current working directory: {os.getcwd()}")
    
    # Test the exact initialization as in the API
    models_dir = 'models/saved_models'
    print(f"📁 Models directory parameter: {models_dir}")
    
    # Check absolute path
    abs_models_dir = os.path.abspath(models_dir)
    print(f"📍 Absolute models directory: {abs_models_dir}")
    print(f"📂 Directory exists: {os.path.exists(abs_models_dir)}")
    
    # Check registry file
    registry_file = os.path.join(models_dir, 'model_registry.json')
    abs_registry_file = os.path.abspath(registry_file)
    print(f"📋 Registry file: {registry_file}")
    print(f"📍 Absolute registry file: {abs_registry_file}")
    print(f"📋 Registry exists: {os.path.exists(abs_registry_file)}")
    
    if os.path.exists(abs_registry_file):
        with open(abs_registry_file, 'r') as f:
            registry = json.load(f)
        print(f"📋 Registry content: {len(registry)} models")
        for name, info in registry.items():
            print(f"   - {name}: {info['path']}")
    
    # Test ModelManager initialization
    print("\n🔧 Testing ModelManager initialization...")
    try:
        from models.model_manager import ModelManager
        
        manager = ModelManager(models_dir=models_dir)
        print(f"✅ ModelManager created successfully")
        print(f"📋 Found {len(manager.model_registry)} models in registry")
        
        # Test list models
        models = manager.list_models()
        print(f"📋 list_models() returned: {type(models)} with {len(models) if models else 0} items")
        
    except Exception as e:
        print(f"❌ Error creating ModelManager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()