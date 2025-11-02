#!/usr/bin/env python3
"""
Final debug - check exactly what's happening with ModelManager initialization
"""
import os
import sys

# Add src to path to import ModelManager
sys.path.append('/app/src')

from models.model_manager import ModelManager

def debug_model_manager():
    """Debug the ModelManager initialization"""
    
    print("🔍 Debugging ModelManager initialization...")
    
    # Initialize with the same parameters as the API
    models_dir = "saved_models"  # Relative path as used in API
    
    print(f"📁 Using models directory: {models_dir}")
    print(f"📍 Current working directory: {os.getcwd()}")
    
    # Check if directory exists
    abs_models_dir = os.path.join(os.getcwd(), models_dir)
    print(f"📍 Absolute models directory: {abs_models_dir}")
    print(f"📂 Directory exists: {os.path.exists(abs_models_dir)}")
    
    if os.path.exists(abs_models_dir):
        contents = os.listdir(abs_models_dir)
        print(f"📋 Directory contents: {contents}")
    
    # Check registry file
    registry_path = os.path.join(abs_models_dir, "model_registry.json")
    print(f"📋 Registry file path: {registry_path}")
    print(f"📋 Registry exists: {os.path.exists(registry_path)}")
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            import json
            registry = json.load(f)
            print(f"📋 Registry content: {registry}")
    
    # Try to initialize ModelManager
    try:
        manager = ModelManager(models_dir=models_dir)
        print(f"✅ ModelManager initialized successfully")
        print(f"📊 Registered models count: {len(manager.registered_models) if manager.registered_models else 0}")
        
        if hasattr(manager, 'registered_models') and manager.registered_models:
            for name, info in manager.registered_models.items():
                print(f"   📄 {name}: {info}")
        
        # Try to list models
        try:
            models = manager.list_models()
            print(f"📋 Listed models: {models}")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error initializing ModelManager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_manager()