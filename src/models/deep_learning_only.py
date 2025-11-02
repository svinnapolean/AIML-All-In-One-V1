"""
Deep Learning Models for Loan Default Prediction

This module implements deep learning models specifically for your loan dataset:
1. TabNet (Recommended for tabular data)
2. Advanced Deep Neural Network  
3. Wide & Deep Network
4. AutoEncoder + Classifier

Usage:
    python deep_learning_only.py --model tabnet --epochs 100
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanDataPreprocessor:
    """Advanced preprocessor for loan default prediction data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_names = None
        self.median_values = {}
    
    def fit_transform(self, df):
        """Fit preprocessor and transform data"""
        logger.info("Starting data preprocessing...")
        
        # Separate features and target
        X = df.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
        y = df['TARGET'] if 'TARGET' in df.columns else None
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=True)
        
        # Feature engineering
        X = self._create_features(X)
        
        # Encode categorical variables
        X = self._encode_categorical(X, fit=True)
        
        # Scale numerical features
        X = self._scale_features(X, fit=True)
        
        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        return X, y
    
    def _handle_missing_values(self, X, fit=True):
        """Handle missing values using KNN imputation for numeric, median for speed"""
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Handle numerical features - use median for speed on large dataset
        for col in numeric_features:
            if fit:
                median_val = X[col].median()
                self.median_values[col] = median_val
            else:
                median_val = self.median_values.get(col, 0)
            X[col] = X[col].fillna(median_val)
        
        # Fill categorical features
        for col in categorical_features:
            X[col] = X[col].fillna('Unknown')
        
        return X
    
    def _create_features(self, X):
        """Create advanced features for deep learning"""
        # Credit utilization ratio
        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['credit_income_ratio'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
        
        # Payment burden
        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['annuity_income_ratio'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
        
        # Age in years
        if 'DAYS_BIRTH' in X.columns:
            X['age_years'] = -X['DAYS_BIRTH'] / 365
        
        # Employment years
        if 'DAYS_EMPLOYED' in X.columns:
            X['employment_years'] = np.where(X['DAYS_EMPLOYED'] > 0, 0, -X['DAYS_EMPLOYED'] / 365)
        
        # Income per family member
        if 'AMT_INCOME_TOTAL' in X.columns and 'CNT_FAM_MEMBERS' in X.columns:
            X['income_per_person'] = X['AMT_INCOME_TOTAL'] / (X['CNT_FAM_MEMBERS'] + 1)
        
        return X
    
    def _encode_categorical(self, X, fit=True):
        """Encode categorical variables"""
        categorical_features = X.select_dtypes(include=['object']).columns
        
        for col in categorical_features:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    X[col] = X[col].astype(str)
                    mask = X[col].isin(le.classes_)
                    X.loc[~mask, col] = 'Unknown'
                    X[col] = le.transform(X[col])
        
        return X
    
    def _scale_features(self, X, fit=True):
        """Scale numerical features for deep learning"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


class DeepLearningModels:
    """Deep Learning Models for Loan Default Prediction"""
    
    @staticmethod
    def create_tabnet_model(input_dim):
        """Create TabNet model - Best for tabular data"""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            import torch
            
            model = TabNetClassifier(
                n_d=64, n_a=64,
                n_steps=5,
                gamma=1.5,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type='entmax',
                scheduler_params={"step_size": 50, "gamma": 0.9},
                verbose=0,
                seed=42
            )
            logger.info("✅ TabNet model created successfully")
            return model
        except ImportError as e:
            logger.error(f"❌ TabNet not available: {e}")
            logger.error("Install with: pip install pytorch-tabnet torch")
            return None
    @staticmethod
    def create_advanced_dnn(input_dim):
        """Create Advanced Deep Neural Network"""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            
            # First hidden block
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            
            # Second hidden block
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.BatchNormalization(),
            
            # Third hidden block
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Fourth hidden block
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with appropriate loss for imbalanced data
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        logger.info("✅ Advanced DNN model created successfully")
        return model
        return model
    @staticmethod
    def create_wide_and_deep(input_dim):
        """Create Wide & Deep Network"""
        wide_dim = min(20, input_dim // 4)  # Use subset for wide component
        
        # Inputs
        wide_input = keras.layers.Input(shape=(wide_dim,), name='wide_input')
        deep_input = keras.layers.Input(shape=(input_dim,), name='deep_input')
        
        # Wide component (linear)
        wide = keras.layers.Dense(1, activation=None, name='wide')(wide_input)
        
        # Deep component
        deep = keras.layers.Dense(256, activation='relu')(deep_input)
        deep = keras.layers.BatchNormalization()(deep)
        deep = keras.layers.Dropout(0.3)(deep)
        
        deep = keras.layers.Dense(128, activation='relu')(deep)
        deep = keras.layers.BatchNormalization()(deep)
        deep = keras.layers.Dropout(0.3)(deep)
        
        deep = keras.layers.Dense(64, activation='relu')(deep)
        deep = keras.layers.Dropout(0.2)(deep)
        
        deep = keras.layers.Dense(1, activation=None, name='deep')(deep)
        
        # Combine wide and deep
        combined = keras.layers.Add()([wide, deep])
        output = keras.layers.Activation('sigmoid')(combined)
        
        model = keras.Model(inputs=[wide_input, deep_input], outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        logger.info("✅ Wide & Deep model created successfully")
        return model
        return model
    @staticmethod
    def create_autoencoder_classifier(input_dim):
        """Create AutoEncoder + Classifier for anomaly detection + classification"""
        # Input
        input_layer = keras.layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = keras.layers.Dense(256, activation='relu')(input_layer)
        encoded = keras.layers.BatchNormalization()(encoded)
        encoded = keras.layers.Dropout(0.2)(encoded)
        
        encoded = keras.layers.Dense(128, activation='relu')(encoded)
        encoded = keras.layers.BatchNormalization()(encoded)
        encoded = keras.layers.Dropout(0.2)(encoded)
        
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(128, activation='relu')(encoded)
        decoded = keras.layers.Dense(256, activation='relu')(decoded)
        decoded = keras.layers.Dense(input_dim, activation='sigmoid', name='decoder')(decoded)
        
        # Classifier
        classifier = keras.layers.Dense(32, activation='relu')(encoded)
        classifier = keras.layers.Dropout(0.3)(classifier)
        classifier = keras.layers.Dense(1, activation='sigmoid', name='classifier')(classifier)
        
        model = keras.Model(inputs=input_layer, outputs=[decoded, classifier])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={'decoder': 'mse', 'classifier': 'binary_crossentropy'},
            loss_weights={'decoder': 0.1, 'classifier': 1.0},
            metrics={'classifier': ['accuracy', 'precision', 'recall', 'AUC']}
        )
        
        logger.info("✅ AutoEncoder + Classifier model created successfully")
        return model
        return model


def load_and_prepare_data(data_path='loan_data/loan_data.csv'):
    """Load and prepare the loan dataset"""
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Basic data info
        if 'TARGET' in df.columns:
            target_counts = df['TARGET'].value_counts()
            logger.info("Target distribution:")
            for class_val, count in target_counts.items():
                logger.info(f"  Class {class_val}: {count:,} ({count/len(df)*100:.1f}%)")
        else:
            logger.error("TARGET column not found in dataset")
            return None
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def train_deep_learning_model(model_name, X_train, X_test, y_train, y_test, epochs=100):
    """Train and evaluate a deep learning model"""
    logger.info(f"🚀 Training {model_name.upper()} model...")
    
    input_dim = X_train.shape[1]
    models = DeepLearningModels()
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Create model
    if model_name == 'tabnet':
        model = models.create_tabnet_model(input_dim)
        if model is None:
            return None
        
        # TabNet training
        model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_test.values, y_test.values)],
            max_epochs=epochs,
            patience=15,
            batch_size=1024,
            virtual_batch_size=256,
            eval_metric=['auc']
        )
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test.values)[:, 1]
        
    elif model_name == 'dnn':
        model = models.create_advanced_dnn(input_dim)
        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True, 
                monitor='val_auc',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=7, 
                factor=0.5, 
                monitor='val_auc',
                mode='max',
                min_lr=1e-6
            )
        ]
        
        
        # Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=1024,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        y_pred_proba = model.predict(X_test, batch_size=1024).flatten()
        
    elif model_name == 'wide_deep':
        model = models.create_wide_and_deep(input_dim)
        
        # Prepare data for Wide & Deep
        wide_features = X_train.iloc[:, :20]  # First 20 features for wide
        wide_test = X_test.iloc[:, :20]
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True,
                monitor='val_auc',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=7, 
                factor=0.5,
                monitor='val_auc',
                mode='max'
            )
        ]
        
        history = model.fit(
            [wide_features, X_train], y_train,
            [wide_features, X_train], y_train,
            validation_data=([wide_test, X_test], y_test),
            epochs=epochs,
            batch_size=1024,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        y_pred_proba = model.predict([wide_test, X_test], batch_size=1024).flatten()
        
    elif model_name == 'autoencoder':
        model = models.create_autoencoder_classifier(input_dim)
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True,
                monitor='val_classifier_auc',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=7, 
                factor=0.5,
                monitor='val_classifier_auc',
                mode='max'
            )
        ]
        
        history = model.fit(
        history = model.fit(
            X_train, [X_train, y_train],  # Reconstruct input + classify
            validation_data=(X_test, [X_test, y_test]),
            epochs=epochs,
            batch_size=1024,
            callbacks=callbacks,
            verbose=1
        )
        
        predictions = model.predict(X_test, batch_size=1024)
        y_pred_proba = predictions[1].flatten()  # Get classifier output
    
    else:
        logger.error(f"❌ Unknown model: {model_name}")
        return None
    
    # Evaluate model
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    logger.info(f"\n🎯 {model_name.upper()} RESULTS:")
    logger.info(f"ROC AUC: {auc_score:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return model, auc_score, pr_auc


def main():
    parser = argparse.ArgumentParser(description='Train DEEP LEARNING models for loan default prediction')
    parser.add_argument('--model', type=str, default='dnn', 
                       choices=['tabnet', 'dnn', 'wide_deep', 'autoencoder'],
                       help='Deep learning model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--data_path', type=str, default='loan_data/loan_data.csv',
                       help='Path to the loan dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    logger.info("🔥 DEEP LEARNING FOR LOAN DEFAULT PREDICTION 🔥")
    logger.info("=" * 60)
    
    # Load data
    df = load_and_prepare_data(args.data_path)
    if df is None:
        return
    
    # Preprocess data
    preprocessor = LoanDataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    if y is None:
        logger.error("No TARGET column found in dataset")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    # Train model
    result = train_deep_learning_model(args.model, X_train, X_test, y_train, y_test, args.epochs)
    
    if result is not None:
        model, auc_score, pr_auc = result
        logger.info(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"🏆 Final ROC AUC: {auc_score:.4f}")
        logger.info(f"🏆 Final PR AUC: {pr_auc:.4f}")
        
        # Save model
        if args.model == 'tabnet':
            try:
                model.save_model(f'loan_default_{args.model}_model')
                logger.info(f"💾 Model saved as: loan_default_{args.model}_model")
            except:
                logger.warning("Could not save TabNet model")
        else:
            model.save(f'loan_default_{args.model}_model.h5')
            logger.info(f"💾 Model saved as: loan_default_{args.model}_model.h5")


if __name__ == "__main__":
    main()