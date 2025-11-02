#!/usr/bin/env python3
"""
Debug model loading issue
"""

import sys
import os

# Add the source paths
sys.path.append('/app/src/models')

from model_manager import ModelManager

def debug_model_loading():
    """Debug model loading"""
    print("🔍 Debugging model loading...")
    
    # Initialize ModelManager the same way as the API
    manager = ModelManager(
        models_dir='/app/models/saved_models',
        results_dir='/app/models/test_results'
    )
    
    print(f"📁 Models directory: {manager.models_dir}")
    print(f"📁 Results directory: {manager.results_dir}")
    
    # Check if registry exists
    registry_path = "/app/models/model_registry.json"
    print(f"📋 Registry path: {registry_path}")
    print(f"📋 Registry exists: {os.path.exists(registry_path)}")
    
    if os.path.exists(registry_path):
        import json
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        print(f"📋 Registry content: {list(registry.keys())}")
        
        # Try to manually set the registry
        manager.models_registry = registry
        print(f"📋 Updated manager registry: {len(manager.models_registry)} models")
        
        # Try to list models
        try:
            models = manager.list_available_models()
            print(f"✅ Available models: {models}")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
        
        # Try to load a specific model
        model_name = "demo_rf_model_20251101_222620"
        if model_name in registry:
            try:
                print(f"🔄 Attempting to load {model_name}...")
                model, info = manager.load_model(model_name)
                print(f"✅ Successfully loaded {model_name}")
                print(f"📊 Model type: {type(model)}")
                print(f"📊 Model info: {info}")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    debug_model_loading()