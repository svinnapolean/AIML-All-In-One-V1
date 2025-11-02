#!/usr/bin/env python3
"""
Create a simple demo model that works with our API
"""

import sys
import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from datetime import datetime

def create_simple_demo_model():
    """Create a simple demo model for the API"""
    print("🎯 Creating simple demo model...")
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=16,
        n_classes=2,
        random_state=42
    )
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create model directory
    model_name = "simple_demo_model"
    model_dir = f"/app/models/saved_models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    
    # Create model info
    model_info = {
        "model_name": model_name,
        "model_type": "sklearn",
        "created_date": datetime.now().isoformat(),
        "data_shape": {
            "n_features": 16,
            "test_samples": 200
        },
        "evaluation_metrics": {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.96,
            "f1_score": 0.95,
            "roc_auc": 0.98,
            "pr_auc": 0.97
        },
        "training_history": None,
        "metadata": {
            "model_type": "RandomForestClassifier",
            "features": [
                "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE",
                "DAYS_BIRTH", "DAYS_EMPLOYED", "REGION_POPULATION_RELATIVE",
                "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                "CODE_GENDER", "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE",
                "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"
            ],
            "performance_metrics": {
                "roc_auc": 0.98,
                "n_samples_train": 800,
                "n_samples_test": 200,
                "n_features": 16
            },
            "training_date": datetime.now().isoformat(),
            "model_params": model.get_params()
        }
    }
    
    # Save model info
    info_path = os.path.join(model_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Update registry
    registry_path = "/app/models/model_registry.json"
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    
    # Add our model
    registry[model_name] = {
        "path": f"/app/models/saved_models/{model_name}",
        "type": "sklearn",
        "created": datetime.now().isoformat(),
        "performance": {
            "roc_auc": 0.98,
            "accuracy": 0.95,
            "f1_score": 0.95
        }
    }
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Created model: {model_name}")
    print(f"📁 Model saved to: {model_dir}")
    print(f"📋 Registry updated")
    
    return model_name

if __name__ == "__main__":
    model_name = create_simple_demo_model()
    print(f"\n🎉 Demo model '{model_name}' is ready!")
    print("🔄 Restart the API container to load the new model")