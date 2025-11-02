# Model Manager - Complete ML Model Lifecycle Management

A comprehensive model management system for machine learning models, providing complete lifecycle management from training to deployment.

## 🎯 Purpose

The Model Manager provides a robust, enterprise-grade solution for managing machine learning models throughout their entire lifecycle. It handles model persistence, versioning, testing, comparison, and deployment preparation.

**Key Benefits:**
- 🔄 **Reproducible Model Management** - Consistent save/load workflows
- 📊 **Performance Tracking** - Comprehensive metrics storage and comparison
- 🎯 **Best Model Selection** - Systematic model comparison and ranking
- 🚀 **Production Ready** - Streamlined deployment workflows
- 📈 **Progress Monitoring** - Track model improvements over time

## 🚀 Features

### 📁 **Model Persistence**
- **Complete Model Storage**: Saves TensorFlow/Keras models with metadata
- **Preprocessing Pipeline**: Stores scalers, encoders, and preprocessing objects
- **Metadata Management**: Tracks performance metrics, creation time, model type
- **Version Control**: Automatic model versioning and registry management

### 🔬 **Model Testing & Evaluation**
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC, PR AUC
- **Visualization**: Automatic generation of ROC curves, confusion matrices
- **Test Result Storage**: Timestamped results with detailed reports
- **Consistent Evaluation**: Uses same preprocessing pipeline as training

### 📊 **Model Comparison**
- **Multi-Model Analysis**: Compare multiple models side-by-side
- **Performance Ranking**: Automatically identifies best performing models
- **Comparison Visualizations**: Side-by-side performance charts
- **Decision Support**: Clear recommendations for model selection

### 📋 **Model Registry**
- **Central Catalog**: Maintains registry of all trained models
- **Quick Overview**: List all models with key performance metrics
- **Model Discovery**: Easy browsing of available models
- **Metadata Search**: Find models by performance criteria

## 📊 **Usage Examples**

### 🔧 **Command Line Interface**

#### **Train and Save Model**
```bash
# Train with default architecture and save
python fast_deep_learning.py --save_model --model_name loan_model_v1

# Train with specific architecture
python fast_deep_learning.py --model lightweight_dnn --save_model --model_name loan_model_v2

# Train with custom parameters
python fast_deep_learning.py --epochs 100 --batch_size 4096 --save_model --model_name loan_model_v3
```

#### **Test Saved Models**
```bash
# Test a specific model
python fast_deep_learning.py --test_model loan_model_v1

# Test returns comprehensive metrics:
# - Accuracy, Precision, Recall, F1-Score
# - ROC AUC, PR AUC
# - Confusion Matrix
# - Visualization plots
```

#### **Compare Multiple Models**
```bash
# Compare performance of multiple models
python fast_deep_learning.py --compare_models loan_model_v1 loan_model_v2 loan_model_v3

# Comparison includes:
# - Side-by-side metrics table
# - Performance ranking
# - Best model identification
# - Comparison visualizations
```

#### **List All Models**
```bash
# View all saved models
python fast_deep_learning.py --list_models

# Shows for each model:
# - Model name and type
# - Creation timestamp
# - Key performance metrics (ROC AUC, Accuracy, F1)
```

### 🐍 **Python API Usage**

#### **Basic Setup**
```python
from model_manager import ModelManager
import tensorflow as tf
import numpy as np

# Initialize ModelManager
model_manager = ModelManager()
print("ModelManager initialized successfully!")
```

#### **1. Save Model Method - `save_model()`**
```python
# Example: Save a trained TensorFlow model
import tensorflow as tf
from tensorflow import keras

# Assume you have a trained model and test data
# model = your_trained_model
# X_test, y_test = your_test_data

# Save the model with comprehensive metadata
model_info = model_manager.save_model(
    model=model,                    # Your trained TensorFlow/Keras model
    model_name="loan_predictor_v1", # Descriptive name for the model
    X_test=X_test,                  # Test features for evaluation
    y_test=y_test,                  # Test labels for evaluation
    model_type='tensorflow',        # Model framework type
    metadata={                      # Optional: Additional information
        'architecture': 'deep_neural_network',
        'training_epochs': 50,
        'batch_size': 2048,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'dataset_size': len(X_test),
        'features': X_test.shape[1]
    }
)

print(f"Model saved successfully!")
print(f"Model path: {model_info['model_path']}")
print(f"ROC AUC: {model_info['test_results']['roc_auc']:.4f}")
```

#### **2. Load Model Method - `load_model()`**
```python
# Example: Load a previously saved model
try:
    loaded_model, model_info = model_manager.load_model("loan_predictor_v1")
    
    print(f"✅ Model loaded successfully!")
    print(f"Model type: {model_info['model_type']}")
    print(f"Created: {model_info['created_at']}")
    print(f"Original ROC AUC: {model_info['test_results']['roc_auc']:.4f}")
    
    # Use the loaded model for predictions
    predictions = loaded_model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
except FileNotFoundError:
    print("❌ Model not found! Check model name.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
```

#### **3. Test Model Method - `test_model()`**
```python
# Example: Test a saved model on new data
test_results = model_manager.test_model(
    model_name="loan_predictor_v1",
    X_test=X_test,
    y_test=y_test
)

print("📊 Test Results:")
print(f"Accuracy: {test_results['accuracy']:.4f}")
print(f"Precision: {test_results['precision']:.4f}")
print(f"Recall: {test_results['recall']:.4f}")
print(f"F1 Score: {test_results['f1_score']:.4f}")
print(f"ROC AUC: {test_results['roc_auc']:.4f}")
print(f"PR AUC: {test_results['pr_auc']:.4f}")

# Test results are automatically saved with timestamp
print(f"Results saved to: {test_results['results_file']}")
print(f"Plots saved to: {test_results['plots_file']}")
```

#### **4. Compare Models Method - `compare_models()`**
```python
# Example: Compare multiple models
model_names = ["loan_predictor_v1", "loan_predictor_v2", "loan_predictor_v3"]

comparison_results = model_manager.compare_models(
    model_names=model_names,
    X_test=X_test,
    y_test=y_test
)

print("🔍 Model Comparison Results:")
print("\n📊 Performance Summary:")
print(comparison_results['comparison_table'])

print(f"\n🏆 Best ROC AUC: {comparison_results['best_roc_auc']['model']} "
      f"({comparison_results['best_roc_auc']['score']:.4f})")

print(f"🏆 Best F1 Score: {comparison_results['best_f1']['model']} "
      f"({comparison_results['best_f1']['score']:.4f})")

# Access individual model results
for model_name in model_names:
    results = comparison_results['individual_results'][model_name]
    print(f"\n📈 {model_name}:")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 Score: {results['f1_score']:.4f}")
```

#### **5. List Models Method - `list_models()`**
```python
# Example: View all saved models and their performance
models = model_manager.list_models()

print(f"📋 Total Models: {len(models)}")
print("\n" + "="*80)

for model_name, info in models.items():
    print(f"🔸 {model_name}")
    print(f"   Type: {info['model_type']}")
    print(f"   Created: {info['created_at']}")
    print(f"   ROC AUC: {info['test_results']['roc_auc']:.4f}")
    print(f"   Accuracy: {info['test_results']['accuracy']:.4f}")
    print(f"   F1 Score: {info['test_results']['f1_score']:.4f}")
    print()

# Find best performing model
if models:
    best_model = max(models.items(), 
                    key=lambda x: x[1]['test_results']['roc_auc'])
    print(f"🏆 Best Model: {best_model[0]} "
          f"(ROC AUC: {best_model[1]['test_results']['roc_auc']:.4f})")
```

#### **6. Complete Workflow Example**
```python
# Complete example: Train, save, test, and compare models
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from home_loan_data import HomeLoanData

# Initialize ModelManager
model_manager = ModelManager()

# Load and prepare data
loan_data = HomeLoanData()
X_train, X_test, y_train, y_test = loan_data.load_and_prepare_data()

print("🚀 Training multiple models...")

# Train Model 1: Simple DNN
model1 = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model1.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)

# Save Model 1
model_manager.save_model(
    model=model1,
    model_name="simple_dnn",
    X_test=X_test,
    y_test=y_test,
    model_type='tensorflow',
    metadata={'architecture': 'simple_dnn', 'layers': 3, 'epochs': 20}
)

# Train Model 2: Deep DNN
model2 = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model2.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)

# Save Model 2
model_manager.save_model(
    model=model2,
    model_name="deep_dnn",
    X_test=X_test,
    y_test=y_test,
    model_type='tensorflow',
    metadata={'architecture': 'deep_dnn', 'layers': 4, 'epochs': 20}
)

print("💾 Models saved successfully!")

# Test individual models
print("\n🔬 Testing individual models...")
test1 = model_manager.test_model("simple_dnn", X_test, y_test)
test2 = model_manager.test_model("deep_dnn", X_test, y_test)

# Compare all models
print("\n🔍 Comparing all models...")
comparison = model_manager.compare_models(
    ["simple_dnn", "deep_dnn"], 
    X_test, 
    y_test
)

# List all models
print("\n📋 All saved models:")
all_models = model_manager.list_models()

# Select best model for deployment
best_model_name = comparison['best_roc_auc']['model']
best_model, _ = model_manager.load_model(best_model_name)

print(f"\n🚀 Best model '{best_model_name}' ready for deployment!")
```

#### **7. Error Handling Examples**
```python
# Example: Robust error handling
try:
    # Attempt to load a model that might not exist
    model, info = model_manager.load_model("nonexistent_model")
except FileNotFoundError:
    print("❌ Model not found. Available models:")
    available_models = model_manager.list_models()
    for name in available_models.keys():
        print(f"  - {name}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

try:
    # Attempt to test a model with invalid data
    results = model_manager.test_model("my_model", None, None)
except ValueError as e:
    print(f"❌ Data validation error: {e}")
except Exception as e:
    print(f"❌ Testing error: {e}")
```

## 📁 **File Structure**

```
saved_models/
├── model_registry.json          # Central registry of all models
├── loan_model_v1/
│   ├── model.h5                 # TensorFlow model file
│   └── model_info.json          # Model metadata and metrics
├── loan_model_v2/
│   ├── model.h5
│   └── model_info.json
└── ...

test_results/
├── loan_model_v1_test_results_20251101_213556.json
├── model_comparison_20251101_214302.json
└── plots/
    ├── loan_model_v1_test_plots_20251101_213555.png
    └── model_comparison_20251101_214301.png
```

## 📊 **Stored Information**

### **Model Registry (`model_registry.json`)**
```json
{
  "models": {
    "loan_model_v1": {
      "model_type": "tensorflow",
      "created_at": "2025-11-01T21:34:38.194237",
      "model_path": "saved_models/loan_model_v1",
      "test_results": {
        "roc_auc": 0.5000,
        "accuracy": 0.5000,
        "f1_score": 0.6667
      }
    }
  }
}
```

### **Model Metadata (`model_info.json`)**
```json
{
  "model_name": "loan_model_v1",
  "model_type": "tensorflow",
  "created_at": "2025-11-01T21:34:38.194237",
  "test_results": {
    "accuracy": 0.5000,
    "precision": 0.5000,
    "recall": 1.0000,
    "f1_score": 0.6667,
    "roc_auc": 0.5000,
    "pr_auc": 0.5000
  },
  "metadata": {
    "architecture": "fast_dnn",
    "training_time": 219.8,
    "data_shape": [565372, 120]
  }
}
```

### **Test Results (`test_results.json`)**
```json
{
  "model_name": "loan_model_v1",
  "test_timestamp": "2025-11-01T21:35:56.123456",
  "metrics": {
    "accuracy": 0.5000,
    "precision": 0.5000,
    "recall": 1.0000,
    "f1_score": 0.6667,
    "roc_auc": 0.5000,
    "pr_auc": 0.5000
  },
  "confusion_matrix": [[28269, 28269], [0, 56537]],
  "plots_saved": "test_results/plots/loan_model_v1_test_plots_20251101_213555.png"
}
```

## 🎯 **Model Testing Strategy**

### **Why Store and Test Each Model?**

1. **🔄 Reproducibility**: Exact model recreation for consistent results
2. **📊 Performance Tracking**: Compare different architectures and hyperparameters
3. **🎯 Best Model Selection**: Scientific approach to choosing optimal model
4. **📈 Progress Monitoring**: Track improvements over experiments
5. **🚀 Production Deployment**: Load best model for real-world predictions

### **Complete Testing Workflow**

1. **Train Multiple Models**
   ```bash
   python fast_deep_learning.py --model fast_dnn --save_model --model_name fast_model
   python fast_deep_learning.py --model lightweight_dnn --save_model --model_name light_model
   python fast_deep_learning.py --model efficient_tabnet --save_model --model_name tabnet_model
   ```

2. **Test Each Model Individually**
   ```bash
   python fast_deep_learning.py --test_model fast_model
   python fast_deep_learning.py --test_model light_model
   python fast_deep_learning.py --test_model tabnet_model
   ```

3. **Compare All Models**
   ```bash
   python fast_deep_learning.py --compare_models fast_model light_model tabnet_model
   ```

4. **Select Best Model**
   ```bash
   python fast_deep_learning.py --list_models
   # Choose the model with highest ROC AUC or F1 score
   ```

## 🏆 **Example Model Comparison Results**

| Model | ROC AUC | Accuracy | F1 Score | Architecture | Training Time |
|-------|---------|-----------|----------|--------------|---------------|
| `loan_model_v1` | 0.5000 | 50.00% | 0.6667 | Fast DNN | 219.8s |
| `loan_model_v2` | **0.7519** | **70.26%** | **0.6971** | Lightweight DNN | 100.4s |

**Best Model**: `loan_model_v2` - Higher performance with faster training! 🏆

## 🛠 **Advanced Features**

### **Automatic Model Versioning**
- Models are automatically versioned by name
- Registry tracks all model versions
- Easy rollback to previous versions

### **Comprehensive Visualizations**
- ROC curves with AUC scores
- Precision-Recall curves
- Confusion matrices with detailed metrics
- Model comparison charts

### **Production Ready**
- Models saved in standard formats (HDF5 for TensorFlow)
- Preprocessing pipelines preserved for deployment
- Metadata for model governance and compliance
- Easy integration with deployment pipelines

### **Error Handling**
- Robust error handling for model loading/saving
- Validation of model compatibility
- Clear error messages for troubleshooting

## 🚀 **Getting Started**

1. **Import the Model Manager**
   ```python
   from model_manager import ModelManager
   model_manager = ModelManager()
   ```

2. **Train and Save Your First Model**
   ```bash
   python fast_deep_learning.py --save_model --model_name my_first_model
   ```

3. **Test Your Model**
   ```bash
   python fast_deep_learning.py --test_model my_first_model
   ```

4. **View All Models**
   ```bash
   python fast_deep_learning.py --list_models
   ```

## 📈 **Best Practices**

1. **Naming Convention**: Use descriptive names like `loan_dnn_v1`, `loan_lightgbm_v2`
2. **Regular Testing**: Test models immediately after training
3. **Comparison Studies**: Always compare multiple approaches
4. **Documentation**: Add metadata about experiments and configurations
5. **Cleanup**: Remove underperforming models to maintain registry

---

The Model Manager provides everything you need for professional ML model management, from initial experimentation to production deployment! 🚀✨