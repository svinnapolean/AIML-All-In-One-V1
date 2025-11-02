"""
Test Advanced Models Training

This script tests the advanced model training using synthetic data
since the loan dataset isn't available in this demo environment.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.models.fast_deep_learning import fast_train_model, create_fast_models
from src.models.model_manager import ModelManager

def create_synthetic_loan_data(n_samples=10000, n_features=20, random_state=42):
    """Create synthetic loan default data for testing"""
    print("🎲 Creating synthetic loan default dataset...")
    
    # Create synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=random_state,
        flip_y=0.1  # Add some noise
    )
    
    # Create feature names similar to loan data
    feature_names = [
        'income_ratio', 'debt_to_income', 'credit_score', 'loan_amount_ratio',
        'employment_years', 'property_value', 'down_payment_ratio', 'age',
        'education_level', 'marital_status', 'dependents', 'location_risk',
        'previous_defaults', 'account_balance', 'loan_term', 'interest_rate',
        'collateral_value', 'income_stability', 'payment_history', 'risk_score'
    ][:n_features]
    
    # Scale features to reasonable ranges
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to DataFrame
    df = pd.DataFrame(X_scaled, columns=feature_names)
    
    print(f"✅ Created dataset with {n_samples} samples and {n_features} features")
    print(f"   📊 Target distribution: {np.bincount(y)} (class 0: {np.sum(y==0)}, class 1: {np.sum(y==1)})")
    
    return df, y, feature_names

def test_advanced_models():
    """Test training of advanced models"""
    print("🚀 Testing Advanced Models Training")
    print("=" * 50)
    
    # Create synthetic data
    X, y, feature_names = create_synthetic_loan_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"   Training: {X_train.shape}")
    print(f"   Testing: {X_test.shape}")
    
    # Initialize ModelManager
    model_manager = ModelManager()
    
    # Test advanced models
    advanced_models = ['autoencoder_classifier', 'lightgbm_classifier']
    results = {}
    
    for model_name in advanced_models:
        print(f"\\n{'='*50}")
        print(f"🚀 Training {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            model, auc_score, pr_auc, training_time = fast_train_model(
                model_name, X_train, X_test, y_train, y_test, 
                epochs=10,  # Reduced for quick testing
                batch_size=512
            )
            
            if model is not None:
                results[model_name] = {
                    'model': model,
                    'auc_score': auc_score,
                    'pr_auc': pr_auc,
                    'training_time': training_time,
                    'feature_names': feature_names
                }
                
                # Save model
                metadata = {
                    'roc_auc': auc_score,
                    'pr_auc': pr_auc,
                    'training_time': training_time,
                    'feature_names': feature_names,
                    'model_type': model_name,
                    'dataset': 'synthetic_loan_data'
                }
                
                model_manager.save_model(
                    model=model,
                    model_name=f"advanced_{model_name}",
                    X_test=X_test,
                    y_test=y_test,
                    model_type=model_name,
                    metadata=metadata
                )
                
                print(f"✅ {model_name} trained and saved successfully!")
                print(f"   🎯 ROC AUC: {auc_score:.4f}")
                print(f"   📈 PR AUC: {pr_auc:.4f}")
                print(f"   ⏱️ Training time: {training_time:.2f}s")
            else:
                print(f"❌ {model_name} training failed - model is None")
                results[model_name] = {'error': 'Model training returned None'}
                
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Summary
    print(f"\\n{'='*60}")
    print("🏆 ADVANCED MODELS TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name:25}: ✅ AUC={result['auc_score']:.4f}, Time={result['training_time']:.1f}s")
        else:
            print(f"{model_name:25}: ❌ ERROR - {result['error']}")
    
    return results

if __name__ == "__main__":
    results = test_advanced_models()