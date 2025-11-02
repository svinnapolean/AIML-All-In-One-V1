#!/usr/bin/env python3
"""
Create demo models for local API testing
"""
import os
import sys
import json
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Add src to path for imports
sys.path.append('src')

def create_demo_models():
    """Create demo models for local testing"""
    
    print("🎯 Creating demo models for local API testing...")
    
    # Create models directory structure
    models_dir = "models/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create two demo models
    model_names = ["demo_local_model_1", "demo_local_model_2"]
    registry = {}
    
    for i, model_name in enumerate(model_names):
        print(f"\n📦 Creating model: {model_name}")
        
        # Create model directory
        model_dir = os.path.join(models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_classes=2, 
            random_state=42 + i
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50, 
            random_state=42 + i,
            max_depth=10
        )
        model.fit(X, y)
        
        # Calculate simple performance metrics
        accuracy = model.score(X, y)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Use forward slashes for display (cross-platform)
        display_path = model_path.replace('\\', '/')
        print(f"   💾 Model saved: {display_path}")
        
        # Create model info
        model_info = {
            "model_name": model_name,
            "model_type": "RandomForestClassifier",
            "created_at": datetime.now().isoformat(),
            "n_features": 20,
            "accuracy": float(accuracy),
            "roc_auc": 0.95 + (i * 0.02),  # Demo values
            "f1_score": 0.92 + (i * 0.01),
            "description": f"Demo model {i+1} for local API testing"
        }
        
        info_path = os.path.join(model_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Use forward slashes for display
        display_info_path = info_path.replace('\\', '/')
        print(f"   📋 Info saved: {display_info_path}")
        
        # Add to registry
        registry[model_name] = {
            "path": model_name,
            "type": "sklearn",
            "created": model_info["created_at"],
            "performance": {
                "roc_auc": model_info["roc_auc"],
                "accuracy": model_info["accuracy"],
                "f1_score": model_info["f1_score"]
            }
        }
        
        print(f"   ✅ Model {model_name} created successfully!")
    
    # Save registry
    registry_path = os.path.join(models_dir, "model_registry.json")
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    # Use forward slashes for display
    display_registry_path = registry_path.replace('\\', '/')
    print(f"\n📚 Model registry created: {display_registry_path}")
    print(f"📊 Total models created: {len(registry)}")
    
    # Verify the models
    print("\n🔍 Verifying models...")
    for model_name in model_names:
        model_path = os.path.join(models_dir, model_name, "model.pkl")
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Test prediction
            test_data = np.random.rand(1, 20)
            prediction = loaded_model.predict(test_data)
            probability = loaded_model.predict_proba(test_data)
            
            print(f"   ✅ {model_name}: Prediction={prediction[0]}, Prob={probability[0][1]:.3f}")
            
        except Exception as e:
            print(f"   ❌ {model_name}: Error loading - {e}")
    
    print("\n🎉 Demo models created successfully!")
    print("📁 Models directory: models/saved_models/")
    print("🚀 Ready to test local API with models!")
    
    return len(registry)

if __name__ == "__main__":
    try:
        num_models = create_demo_models()
        print(f"\n✅ Created {num_models} demo models for local API testing")
    except Exception as e:
        print(f"\n❌ Error creating models: {e}")
        import traceback
        traceback.print_exc()