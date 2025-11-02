#!/usr/bin/env python3
"""
Quick Demo Model Training
Creates a sample model for testing the deployed API
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add source paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))

from home_loan_data import HomeLoanData
from model_manager import ModelManager

def create_sample_data():
    """Create a small sample dataset for demo purposes"""
    np.random.seed(42)
    
    # Create sample loan data
    n_samples = 1000
    
    data = {
        'AMT_CREDIT': np.random.uniform(50000, 500000, n_samples),
        'AMT_ANNUITY': np.random.uniform(5000, 50000, n_samples),
        'AMT_INCOME_TOTAL': np.random.uniform(30000, 200000, n_samples),
        'AMT_GOODS_PRICE': np.random.uniform(40000, 450000, n_samples),
        'CODE_GENDER': np.random.choice(['M', 'F'], n_samples),
        'DAYS_BIRTH': np.random.randint(-25000, -6000, n_samples),  # Age 16-68
        'DAYS_EMPLOYED': np.random.randint(-15000, 1000, n_samples),  # Employment history
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples),
        'NAME_INCOME_TYPE': np.random.choice(['Working', 'Commercial associate', 'Pensioner', 'State servant'], n_samples),
        'NAME_EDUCATION_TYPE': np.random.choice(['Secondary / secondary special', 'Higher education', 'Incomplete higher'], n_samples),
        'NAME_FAMILY_STATUS': np.random.choice(['Married', 'Single / not married', 'Civil marriage'], n_samples),
        'NAME_HOUSING_TYPE': np.random.choice(['House / apartment', 'Rented apartment', 'With parents'], n_samples),
        'REGION_POPULATION_RELATIVE': np.random.uniform(0.001, 0.1, n_samples),
        'EXT_SOURCE_1': np.random.uniform(0.1, 0.9, n_samples),
        'EXT_SOURCE_2': np.random.uniform(0.1, 0.9, n_samples),
        'EXT_SOURCE_3': np.random.uniform(0.1, 0.9, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic target based on some logical rules
    # Higher risk for: lower income, higher credit amount, unemployed, etc.
    risk_score = (
        (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']) * 0.3 +  # Debt-to-income ratio
        (df['DAYS_EMPLOYED'] > 0).astype(int) * 0.2 +  # Unemployed
        (df['EXT_SOURCE_1'] < 0.3).astype(int) * 0.2 +  # Poor external score
        np.random.uniform(0, 0.3, n_samples)  # Random factor
    )
    
    # Convert to binary target (1 = default, 0 = no default)
    df['TARGET'] = (risk_score > 0.5).astype(int)
    
    return df

def train_demo_model():
    """Train a demo model for testing the API"""
    print("🚀 Starting demo model training...")
    
    # Create output directory
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('test_results', exist_ok=True)
    
    # Generate sample data
    print("📊 Generating sample loan data...")
    df = create_sample_data()
    
    # Save sample data
    df.to_csv('sample_loan_data.csv', index=False)
    print(f"✅ Generated {len(df)} loan applications")
    
    # Initialize model manager
    print("🔧 Initializing model manager...")
    manager = ModelManager(
        models_dir='saved_models',
        results_dir='test_results'
    )
    
    # Initialize HomeLoanData with the sample data
    print("📈 Initializing loan data processor...")
    loan_data = HomeLoanData('sample_loan_data.csv')
    
    # Process the data
    print("⚙️ Processing loan data...")
    loan_data.load_dataset()
    loan_data.check_null_values()
    loan_data.analyze_target_distribution()
    loan_data.balance_dataset()
    loan_data.encode_columns()
    
    # Train model and get results
    print("🎯 Training and evaluating model...")
    results = loan_data.train_model_and_calculate_metrics()
    
    # Get the processed data from the class
    X = loan_data.X
    y = loan_data.y
    
    # Split data manually
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a Random Forest model 
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    roc_auc = results['roc_auc']
    print(f"🎯 Model Performance: ROC AUC = {roc_auc:.4f}")
    
    # Save the model using ModelManager
    model_name = f"demo_rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create model metadata
    model_metadata = {
        'model_type': 'RandomForestClassifier',
        'features': list(X_train.columns),
        'performance_metrics': {
            'roc_auc': float(roc_auc),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': len(X_train.columns)
        },
        'training_date': datetime.now().isoformat(),
        'model_params': model.get_params()
    }
    
    # Save model
    model_path = manager.save_model(
        model=model,
        model_name=model_name,
        X_test=X_test,
        y_test=y_test,
        model_type='sklearn',
        metadata=model_metadata
    )
    
    print(f"✅ Model saved: {model_name}")
    print(f"📁 Model path: {model_path}")
    
    print("🎉 Demo model training completed successfully!")
    print(f"📊 Model statistics:")
    print(f"   - ROC AUC: {roc_auc:.4f}")
    print(f"   - Features: {len(X_train.columns)}")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    return model_name, model_metadata

if __name__ == "__main__":
    try:
        model_name, metadata = train_demo_model()
        print("\n🚀 Demo model is ready for API testing!")
        print(f"📋 Use model name: {model_name}")
        print("\n💡 Next steps:")
        print("1. Copy the saved model to the Docker container")
        print("2. Test the API endpoints with the trained model")
        print("3. Run validation pipeline again")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()