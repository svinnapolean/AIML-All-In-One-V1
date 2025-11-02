#!/usr/bin/env python3
"""
Quick fix for model loading issue
Updates model registry paths for container environment
"""

import json
import os

def fix_model_registry():
    """Fix the model registry paths for the container environment"""
    
    # Read the existing registry
    registry_path = "/app/models/saved_models/model_registry.json"
    target_path = "/app/models/model_registry.json"
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Fix the paths (convert Windows paths to Unix paths)
        for model_name, model_info in registry.items():
            old_path = model_info['path']
            # Convert Windows path to Unix path and make it relative to /app/models/
            new_path = old_path.replace('\\', '/')
            model_info['path'] = new_path
            print(f"Fixed path for {model_name}: {old_path} -> {new_path}")
        
        # Save the fixed registry to the correct location
        with open(target_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Updated model registry saved to {target_path}")
        print(f"📋 Found {len(registry)} models:")
        for model_name in registry.keys():
            print(f"   - {model_name}")
        
        return True
    else:
        print(f"❌ Registry file not found: {registry_path}")
        return False

if __name__ == "__main__":
    fix_model_registry()