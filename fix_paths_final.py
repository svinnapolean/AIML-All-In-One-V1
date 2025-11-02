#!/usr/bin/env python3
"""
Final fix for model registry paths
"""

import json
import os

def fix_registry_paths():
    """Fix model registry with absolute paths"""
    
    registry_path = "/app/models/model_registry.json" 
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Fix paths to be absolute and correct
        for model_name, model_info in registry.items():
            # Convert to absolute path
            absolute_path = f"/app/models/saved_models/{model_name}"
            model_info['path'] = absolute_path
            print(f"Updated {model_name}: {absolute_path}")
        
        # Save fixed registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print("✅ Registry updated with absolute paths")
        return True
    else:
        print(f"❌ Registry not found: {registry_path}")
        return False

if __name__ == "__main__":
    fix_registry_paths()