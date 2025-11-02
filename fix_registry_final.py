#!/usr/bin/env python3
"""
Fix the model registry paths to work with the API
"""
import os
import json

def fix_registry_paths():
    """Fix the model registry paths"""
    
    registry_path = "/app/models/saved_models/model_registry.json"
    
    print(f"🔧 Fixing registry paths in {registry_path}")
    
    # Load current registry
    if not os.path.exists(registry_path):
        print("❌ Registry file not found")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    print(f"📋 Found {len(registry)} models in registry")
    
    # Fix each model path
    fixed_registry = {}
    for model_name, info in registry.items():
        old_path = info['path']
        # Convert from Windows backslash to Unix forward slash
        # and make it relative to the models/saved_models directory
        new_path = model_name  # Just use model name as path since it's relative
        
        fixed_info = info.copy()
        fixed_info['path'] = new_path
        fixed_registry[model_name] = fixed_info
        
        print(f"✅ Fixed {model_name}: {old_path} -> {new_path}")
    
    # Save fixed registry
    with open(registry_path, 'w') as f:
        json.dump(fixed_registry, f, indent=2)
    
    print(f"💾 Registry updated with {len(fixed_registry)} models")
    
    # Verify the fixed registry
    print("\n📋 Updated registry:")
    for model_name, info in fixed_registry.items():
        print(f"   {model_name}: {info['path']}")

if __name__ == "__main__":
    fix_registry_paths()