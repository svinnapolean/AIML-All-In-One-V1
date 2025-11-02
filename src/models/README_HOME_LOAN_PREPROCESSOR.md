# Home Loan Data Preprocessor for Deep Learning

A comprehensive data preprocessing class specifically designed for Home Loan Default Prediction using Deep Learning models.

## 🎯 Problem Statement

For a safe and secure lending experience, it's important to analyze the past data. This preprocessor prepares historical loan data for deep learning models to predict the chance of default for future loans. The dataset is highly imbalanced and includes many features that make this problem challenging.

**Objective:** Prepare data for deep learning model that predicts whether or not an applicant will be able to repay a loan using historical data.

**Domain:** Finance  
**Analysis:** Data preprocessing for deep learning prediction model

## 🚀 Features

- **Robust Data Loading:** Error handling and comprehensive dataset exploration
- **Advanced Data Cleaning:** Handles missing values, removes redundant features
- **Feature Engineering:** Optimized for deep learning models
- **Imbalanced Dataset Handling:** SMOTE balancing for highly skewed target distribution
- **Deep Learning Ready:** Outputs data ready for TensorFlow/Keras models
- **Comprehensive Reporting:** Detailed analysis and statistics at each step

## 📊 Key Capabilities

### 1. Data Loading & Exploration
- Load CSV datasets with error handling
- Dataset shape, memory usage, and type analysis
- Target distribution analysis (identifies imbalanced data)
- Missing value comprehensive analysis

### 2. Data Cleaning & Preprocessing
- **Missing Value Handling:**
  - Drops columns with excessive missing values (configurable threshold)
  - Median imputation for numeric features
  - Mode imputation for categorical features
  
- **Feature Engineering:**
  - Label encoding for categorical variables
  - Correlation analysis and removal of highly correlated features
  - Constant/near-constant feature removal
  
### 3. Advanced Preprocessing
- **Robust Scaling:** Uses RobustScaler (more robust to outliers than StandardScaler)
- **Train/Test Split:** Stratified splitting to preserve target distribution
- **SMOTE Balancing:** Handles highly imbalanced datasets using Synthetic Minority Oversampling

### 4. Deep Learning Preparation
- Returns data in format ready for TensorFlow/Keras
- Provides both balanced training data and original test data
- Includes preprocessing objects for future use
- Comprehensive data package with metadata

## 📁 File Structure

```
src/models/
├── home_loan_preprocessor.py      # Main preprocessor class
├── demo_home_loan_preprocessor.py # Demonstration script
└── README_HOME_LOAN_PREPROCESSOR.md # This file
```

## 🔧 Installation & Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

**Required Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning preprocessing
- `imbalanced-learn` - SMOTE balancing
- `matplotlib`, `seaborn` - Visualization (optional)

## 📚 Usage

### Basic Usage

```python
from home_loan_preprocessor import HomeLoanPreprocessor

# Initialize preprocessor
preprocessor = HomeLoanPreprocessor('data/loan_data.csv')

# Run complete pipeline
deep_learning_data = preprocessor.run_complete_pipeline()

# Get preprocessed data for deep learning
X_train = deep_learning_data['X_train']  # Balanced and scaled training features
y_train = deep_learning_data['y_train']  # Balanced training target
X_test = deep_learning_data['X_test']    # Scaled testing features  
y_test = deep_learning_data['y_test']    # Original testing target
```

### Step-by-Step Usage

```python
# Initialize
preprocessor = HomeLoanPreprocessor('data/loan_data.csv')

# Step 1: Load data
preprocessor.load_data()

# Step 2: Analyze missing values
missing_summary = preprocessor.analyze_missing_values()

# Step 3: Clean data
clean_data = preprocessor.clean_data(
    missing_threshold=70,       # Drop columns with >70% missing
    correlation_threshold=0.95  # Remove highly correlated features
)

# Step 4: Prepare features and target
X, y = preprocessor.prepare_features_target()

# Step 5: Split data
X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.2)

# Step 6: Scale features
X_train_scaled, X_test_scaled = preprocessor.scale_features()

# Step 7: Balance data with SMOTE
X_train_balanced, y_train_balanced = preprocessor.balance_data()

# Step 8: Get final data package
final_data = preprocessor.get_data_for_deep_learning()
```

### Configuration Options

```python
# Run with custom parameters
deep_learning_data = preprocessor.run_complete_pipeline(
    missing_threshold=80,           # Higher tolerance for missing values
    correlation_threshold=0.90,     # Lower correlation threshold
    test_size=0.25,                # 25% for testing
    sampling_strategy='minority'    # Different SMOTE strategy
)
```

## 🧠 Deep Learning Integration

### TensorFlow/Keras Example

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Get preprocessed data
data = preprocessor.run_complete_pipeline()
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# Create deep learning model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(), 
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    class_weight='balanced',  # Additional balancing
    verbose=1
)
```

## 📊 Data Package Contents

The `get_data_for_deep_learning()` method returns a comprehensive dictionary:

```python
{
    # Training data (balanced and scaled)
    'X_train': array,           # Balanced training features
    'y_train': array,           # Balanced training target
    
    # Testing data (scaled, original distribution)
    'X_test': array,            # Scaled testing features
    'y_test': array,            # Original testing target
    
    # Original training data (for comparison)
    'X_train_original': array,  # Unbalanced training features
    'y_train_original': array,  # Unbalanced training target
    
    # Feature information
    'feature_names': list,      # List of feature names
    'n_features': int,          # Number of features
    
    # Data shapes
    'train_shape': tuple,       # Training data shape
    'test_shape': tuple,        # Testing data shape
    
    # Preprocessing objects (for future use)
    'scaler': RobustScaler,     # Fitted scaler
    'label_encoders': dict,     # Label encoders for categorical features
    'imputers': dict,           # Fitted imputers
    
    # Dataset information
    'data_info': dict          # Comprehensive dataset metadata
}
```

## 🎯 Key Benefits for Deep Learning

1. **Imbalanced Data Handling:** SMOTE balancing addresses the typical 90%+ vs <10% loan default ratio
2. **Robust Scaling:** RobustScaler handles outliers better than StandardScaler for financial data
3. **Feature Selection:** Removes redundant and constant features that can hurt deep learning performance
4. **Missing Value Strategy:** Comprehensive imputation preserves maximum information
5. **Ready-to-Use:** Direct integration with TensorFlow/Keras without additional preprocessing

## 🔍 Performance Considerations

- **Memory Efficient:** Uses RobustScaler which is more memory efficient than StandardScaler
- **SMOTE Optimization:** Configurable k_neighbors for optimal synthetic sample generation
- **Feature Reduction:** Automatic removal of redundant features reduces model complexity
- **Stratified Splitting:** Preserves target distribution in train/test splits

## 🛠️ Troubleshooting

### Common Issues

1. **File Not Found Error:**
   ```python
   # Ensure correct path
   preprocessor = HomeLoanPreprocessor('correct/path/to/loan_data.csv')
   ```

2. **Memory Issues with Large Datasets:**
   ```python
   # Increase missing threshold to remove more columns
   data = preprocessor.run_complete_pipeline(missing_threshold=50)
   ```

3. **SMOTE Errors:**
   ```python
   # Reduce k_neighbors if you have few minority samples
   preprocessor.balance_data(k_neighbors=3)
   ```

### Dataset Requirements

- **Target Column:** Must be named 'TARGET' with binary values (0=No Default, 1=Default)
- **File Format:** CSV format with headers
- **Missing Values:** Can handle any percentage of missing values
- **Data Types:** Supports numeric and categorical features

## 📈 Expected Output

```
🏠 Home Loan Preprocessor Initialized
🎯 Objective: Prepare data for deep learning loan default prediction
============================================================

📊 Loading Home Loan Dataset...
✅ Dataset loaded successfully!
   📏 Shape: (5000, 17)
   💾 Memory Usage: 0.65 MB
   🎯 Target Column: TARGET

🎯 Target Distribution (Imbalanced Dataset):
   0 (Payers (No Default)): 4,600 (92.00%)
   1 (Defaulters): 400 (8.00%)
   📊 Imbalance Ratio: 11.50:1 (Payers:Defaulters)

🔍 Analyzing Missing Values...
✅ Missing Value Analysis Complete!
   📊 Total columns: 17
   ❌ Columns with missing values: 4
   ✅ Columns without missing values: 13

🧹 Cleaning Data for Deep Learning...
✅ Dataset is now clean and ready for deep learning!

🎯 Preparing Features and Target...
✅ Features and target prepared!
   📊 Features shape: (5000, 16)
   🎯 Target shape: (5000,)

✂️  Splitting Data (Test Size: 20.0%)...
✅ Data split completed!
   📊 Training set: (4000, 16)
   📊 Testing set: (1000, 16)

⚖️  Scaling Features (RobustScaler)...
✅ Feature scaling completed!

⚖️  Balancing Imbalanced Dataset (SMOTE)...
✅ SMOTE Balancing Completed!
📊 Balanced Training Distribution:
   0 (Payers): 3,680 (50.00%)
   1 (Defaulters): 3,680 (50.00%)

🚀 Preparing Final Data for Deep Learning...
✅ Deep Learning Data Package Ready!
   🔢 Training samples: 7,360
   🔢 Testing samples: 1,000
   📋 Features: 16
   ⚖️  Balanced: ✅ (SMOTE applied)
   📏 Scaled: ✅ (RobustScaler)
   🧹 Cleaned: ✅ (Missing values handled)

🎉 HOME LOAN PREPROCESSING PIPELINE COMPLETED!
🚀 Data is now ready for Deep Learning models
============================================================
```

## 📝 Notes

- The preprocessor is specifically designed for financial/loan data with typical characteristics
- SMOTE balancing is essential for loan default prediction due to natural class imbalance
- RobustScaler is preferred over StandardScaler for financial data with outliers
- All preprocessing objects are preserved for applying the same transformations to new data

## 🤝 Integration with Existing Models

This preprocessor integrates seamlessly with the existing deep learning models in the project:

- `deep_learning_only.py` - Use the output directly with TabNet, DNN, Wide & Deep models
- `fast_deep_learning.py` - Replace the FastPreprocessor with this more comprehensive solution
- `home_loan_data.py` - Can be used alongside for additional analysis capabilities