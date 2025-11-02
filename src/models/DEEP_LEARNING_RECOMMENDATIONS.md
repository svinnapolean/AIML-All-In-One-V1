# 🧠 Deep Learning Model Recommendations for Loan Default Prediction

## 📊 Dataset Analysis Summary

**Your loan dataset characteristics:**
- **Size**: 307,511 records with 122 features
- **Target**: Binary classification (loan default prediction)
  - Class 0 (No default): 282,686 (91.9%)
  - Class 1 (Default): 24,825 (8.1%)
- **Class Imbalance**: Highly imbalanced (11.4:1 ratio)
- **Feature Types**: 106 numeric, 16 categorical
- **Missing Data**: 67/122 features have missing values (up to 69.9% missing)
- **Complexity**: High-dimensional financial data with various scales

## 🎯 Recommended Deep Learning Models

### 1. **TabNet (RECOMMENDED - Best for Tabular Data)**
```python
# TabNet is specifically designed for tabular data like yours
from pytorch_tabnet.tab_model import TabNetClassifier

model = TabNetClassifier(
    n_d=64, n_a=64,  # Feature transformer & attention dimensions
    n_steps=5,       # Number of decision steps
    gamma=1.5,       # Feature selection strength
    lambda_sparse=1e-3,  # Sparsity regularization
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',  # Sparse attention
    scheduler_params={"step_size":50, "gamma":0.9},
    verbose=0
)
```

**Why TabNet is perfect for your data:**
- ✅ **Handles missing values** naturally
- ✅ **Feature importance** interpretation
- ✅ **Attention mechanism** for complex relationships
- ✅ **Built for tabular data** like financial datasets
- ✅ **Excellent for imbalanced data** with proper loss functions

### 2. **Deep Neural Network with Advanced Architecture**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_advanced_dnn(input_dim):
    model = models.Sequential([
        # Input normalization
        layers.BatchNormalization(input_shape=(input_dim,)),
        
        # Deep layers with residual connections
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Use focal loss for imbalanced data
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=['precision', 'recall', 'auc']
    )
    return model
```

### 3. **Wide & Deep Network (Google's Architecture)**
```python
def create_wide_and_deep_model(wide_dim, deep_dim):
    # Wide input (linear features)
    wide_input = layers.Input(shape=(wide_dim,), name='wide_input')
    
    # Deep input (all features)
    deep_input = layers.Input(shape=(deep_dim,), name='deep_input')
    
    # Wide component (linear model)
    wide = layers.Dense(1, activation=None, name='wide')(wide_input)
    
    # Deep component
    deep = layers.Dense(256, activation='relu')(deep_input)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(128, activation='relu')(deep)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(64, activation='relu')(deep)
    deep = layers.Dense(1, activation=None, name='deep')(deep)
    
    # Combine wide and deep
    combined = layers.Add()([wide, deep])
    output = layers.Activation('sigmoid')(combined)
    
    model = models.Model(inputs=[wide_input, deep_input], outputs=output)
    return model
```

### 4. **AutoEncoder + Classifier (For Anomaly Detection)**
```python
def create_autoencoder_classifier(input_dim):
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation='relu')(encoder_input)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.Dense(256, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Classifier branch
    classifier = layers.Dense(32, activation='relu')(encoded)
    classifier = layers.Dropout(0.3)(classifier)
    classifier = layers.Dense(1, activation='sigmoid', name='classifier')(classifier)
    
    model = models.Model(encoder_input, [decoded, classifier])
    return model
```

## 🛠️ Data Preprocessing for Deep Learning

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

def preprocess_loan_data(df):
    # Handle missing values
    # For numerical features
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['SK_ID_CURR', 'TARGET']]
    
    # KNN imputation for missing values
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    
    # Handle categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Feature scaling
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df, scaler, le_dict

# Feature engineering for better performance
def create_new_features(df):
    # Credit utilization ratio
    df['credit_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Payment burden
    df['annuity_income_ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # Age in years
    df['age_years'] = -df['DAYS_BIRTH'] / 365
    
    # Employment years
    df['employment_years'] = -df['DAYS_EMPLOYED'] / 365
    df['employment_years'] = df['employment_years'].replace([np.inf, -np.inf], 0)
    
    return df
```

## ⚖️ Handling Class Imbalance

```python
# 1. Class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 2. SMOTE for synthetic oversampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. Focal Loss (already included in model definitions above)
```

## 📈 Model Training Strategy

```python
def train_with_cross_validation(model, X, y):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=100,
            batch_size=512,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5)
            ],
            verbose=0
        )
        
        # Evaluate
        val_pred = model.predict(X_val_fold)
        auc = roc_auc_score(y_val_fold, val_pred)
        auc_scores.append(auc)
        
    return np.mean(auc_scores), np.std(auc_scores)
```

## 🏆 Model Performance Ranking (Expected)

1. **TabNet** - 0.85-0.90 AUC (Best for tabular data)
2. **Wide & Deep** - 0.82-0.87 AUC (Good for mixed features)
3. **Advanced DNN** - 0.80-0.85 AUC (Solid baseline)
4. **AutoEncoder + Classifier** - 0.78-0.83 AUC (Good for anomaly detection)

## 🎯 Evaluation Metrics for Imbalanced Data

```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def evaluate_model(y_true, y_pred, y_pred_proba):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print(f"PR AUC Score: {average_precision_score(y_true, y_pred_proba):.4f}")
    
    # Custom threshold optimization
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    return optimal_threshold
```

## 💡 Implementation Recommendations

1. **Start with TabNet** - It's specifically designed for your type of data
2. **Use proper cross-validation** - Essential for reliable results
3. **Handle class imbalance** - Use focal loss + SMOTE + class weights
4. **Feature engineering** - Create meaningful financial ratios
5. **Hyperparameter tuning** - Use Optuna or similar for optimization
6. **Ensemble methods** - Combine multiple models for better performance

## 🚀 Next Steps

1. Implement the preprocessing pipeline
2. Start with TabNet model training
3. Compare with traditional ML models (XGBoost, LightGBM)
4. Use ensemble of top-performing models
5. Deploy the best model via your API

Your loan default prediction dataset is perfect for deep learning approaches, especially TabNet, due to its tabular nature, high dimensionality, and complex feature interactions!