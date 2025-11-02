"""
Fast Deep Learning for Loan Default Prediction

This optimized version provides fast training with multiple performance improvements:
1. Uses HomeLoanData class for comprehensive preprocessing and SMOTE balancing
2. Data sampling for quick iterations  
3. Optimized architectures
4. Advanced data cleaning and feature engineering
5. Progressive training options

Usage:
    python fast_deep_learning.py --model fast_dnn --epochs 50
    python fast_deep_learning.py --model lightweight_dnn --sample_ratio 0.3  # 30% of data for speed
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from .home_loan_data import HomeLoanData
from .model_manager import ModelManager
from datetime import datetime
import argparse
import logging
import time
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastPreprocessor:
    """Fast preprocessor optimized for speed"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.median_values = {}
    
    def fit_transform(self, df, sample_ratio=1.0):
        """Fast preprocessing with optional sampling"""
        start_time = time.time()
        logger.info(f"Starting FAST preprocessing (sample_ratio={sample_ratio})...")
        
        # Sample data if requested
        if sample_ratio < 1.0:
            n_samples = int(len(df) * sample_ratio)
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {n_samples:,} records for faster training")
        
        # Separate features and target
        X = df.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
        y = df['TARGET'] if 'TARGET' in df.columns else None
        
        # Fast missing value handling (median only)
        X = self._fast_fill_missing(X)
        
        # Essential feature engineering only
        X = self._create_key_features(X)
        
        # Fast categorical encoding
        X = self._fast_encode_categorical(X)
        
        # Standard scaling
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Fast preprocessing complete in {processing_time:.2f}s. Shape: {X.shape}")
        return X, y
    
    def _fast_fill_missing(self, X):
        """Ultra-fast missing value filling"""
        # Numerical: median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = X[col].median()
            self.median_values[col] = median_val
            X[col] = X[col].fillna(median_val)
        
        # Categorical: 'Unknown'
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('Unknown')
        
        return X
    
    def _create_key_features(self, X):
        """Create only the most important features"""
        # Credit to income ratio
        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['credit_income_ratio'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
        
        # Age in years
        if 'DAYS_BIRTH' in X.columns:
            X['age_years'] = -X['DAYS_BIRTH'] / 365
        
        return X
    
    def _fast_encode_categorical(self, X):
        """Fast categorical encoding"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Keep only top categories, encode rest as 'Other'
            top_categories = X[col].value_counts().head(10).index
            X[col] = X[col].where(X[col].isin(top_categories), 'Other')
            
            # Label encode
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        return X


def create_fast_models():
    """Create optimized models for speed"""
    def fast_dnn(input_dim):
        """Fast Deep Neural Network - optimized for speed"""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        return model
        
    def lightweight_dnn(input_dim):
        """Lightweight model for very fast training"""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.002),
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        return model
    
    def autoencoder_classifier(input_dim):
        """Autoencoder-based feature extraction with classifier"""
        # Autoencoder architecture
        encoding_dim = max(32, input_dim // 4)  # Compressed representation
        
        # Encoder
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = keras.layers.Dropout(0.2)(encoded)
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder (for pretraining)
        decoded = keras.layers.Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Create autoencoder for pretraining
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Create encoder model
        encoder = keras.Model(input_layer, encoded)
        
        # Classifier on top of encoded features
        classifier_input = keras.layers.Input(shape=(encoding_dim,))
        x = keras.layers.Dense(64, activation='relu')(classifier_input)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        
        classifier = keras.Model(classifier_input, output)
        classifier.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        
        # Return both models for training
        return {
            'autoencoder': autoencoder,
            'encoder': encoder,
            'classifier': classifier,
            'encoding_dim': encoding_dim
        }
    
    def lightgbm_classifier(input_dim):
        """LightGBM gradient boosting classifier"""
        # LightGBM parameters optimized for loan default prediction
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42,
            'force_col_wise': True,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        return params
    
    def efficient_tabnet(input_dim):
        """Optimized TabNet for faster training"""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            
            model = TabNetClassifier(
                n_d=32, n_a=32,  # Reduced from 64 for speed
                n_steps=3,       # Reduced from 5 for speed
                gamma=1.5,
                lambda_sparse=1e-3,
                optimizer_params=dict(lr=3e-2),  # Higher learning rate
                mask_type='entmax',
                scheduler_params={"step_size": 30, "gamma": 0.9},
                verbose=0,
                seed=42
            )
            return model
        except ImportError:
            logger.error("TabNet not available. Use: pip install pytorch-tabnet torch")
            return None
    
    return {
        'fast_dnn': fast_dnn,
        'lightweight_dnn': lightweight_dnn,
        'autoencoder_classifier': autoencoder_classifier,
        'lightgbm_classifier': lightgbm_classifier,
        'efficient_tabnet': efficient_tabnet
    }


def load_and_prepare_data(data_path='loan_data/loan_data.csv', sample_ratio=1.0):
    """Load and prepare data using HomeLoanData for comprehensive preprocessing"""
    logger.info(f"Loading and preprocessing data from {data_path}")
    logger.info(f"Using HomeLoanData class for comprehensive data cleaning and SMOTE balancing")
    
    try:
        # Initialize the HomeLoanData class
        loan_data = HomeLoanData(data_path)
        
        # Apply sampling if requested for speed
        if sample_ratio < 1.0:
            logger.info(f"🔥 SPEED MODE: Will sample {sample_ratio*100:.0f}% of data for faster training")
            # Load dataset first to apply sampling
            if not loan_data.load_dataset():
                logger.error("Failed to load dataset")
                return None, None, None, None
            
            if loan_data.df is None:
                logger.error("Dataset is None after loading")
                return None, None, None, None
            
            original_rows = len(loan_data.df)
            n_samples = int(original_rows * sample_ratio)
            loan_data.df = loan_data.df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {n_samples:,} records from {original_rows:,} for faster training")
        
        # Run the complete analysis pipeline which includes:
        # - Data loading and exploration
        # - Missing value analysis and handling
        # - Data balancing with SMOTE
        # - Visualization and encoding
        # - Model training and evaluation
        logger.info("🧹 Running comprehensive analysis pipeline...")
        results = loan_data.analyze_complete_pipeline()
        
        if results is None or not hasattr(loan_data, 'X_balanced') or not hasattr(loan_data, 'y_balanced'):
            logger.error("Analysis pipeline failed or SMOTE balancing not completed")
            return None, None, None, None
        
        # Extract the preprocessed and balanced data
        X_train = loan_data.X_balanced          # Balanced training features (after SMOTE)
        y_train = loan_data.y_balanced          # Balanced training target (after SMOTE)
        X_test = loan_data.X_test               # Test features (scaled)
        y_test = loan_data.y_test               # Test target
        
        # Verify data is not None before converting
        if X_train is None or X_test is None or y_train is None or y_test is None:
            logger.error("One or more data components is None after extraction")
            return None, None, None, None
        
        # Convert to numpy arrays if they're pandas DataFrames/Series
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        
        logger.info("✅ Data preprocessing and balancing completed successfully!")
        logger.info(f"   📊 Training set (SMOTE balanced): {X_train.shape}")
        logger.info(f"   📊 Test set: {X_test.shape}")
        
        # Log target distribution
        import numpy as np
        # Ensure y_train and y_test are numpy arrays for np.unique
        y_train_array = np.asarray(y_train)
        y_test_array = np.asarray(y_test)
        
        unique_train, counts_train = np.unique(y_train_array, return_counts=True)
        unique_test, counts_test = np.unique(y_test_array, return_counts=True)
        
        logger.info("🎯 Target distribution:")
        logger.info(f"   Training (SMOTE balanced) - Class 0: {counts_train[0]:,}, Class 1: {counts_train[1]:,}")
        logger.info(f"   Testing (original) - Class 0: {counts_test[0]:,}, Class 1: {counts_test[1]:,}")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        logger.error(f"Make sure the HomeLoanData class is working correctly")
        return None, None, None, None


def fast_train_model(model_name, X_train, X_test, y_train, y_test, epochs=50, batch_size=2048):
    """Fast training with optimizations"""
    start_time = time.time()
    logger.info(f"🚀 FAST Training {model_name.upper()}...")
    
    input_dim = X_train.shape[1]
    models = create_fast_models()
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Create model
    if model_name in models:
        if model_name == 'efficient_tabnet':
            model = models[model_name](input_dim)
            if model is None:
                logger.warning("TabNet not available, falling back to fast_dnn")
                model = models['fast_dnn'](input_dim)
                model_name = 'fast_dnn'
        elif model_name == 'autoencoder_classifier':
            model_dict = models[model_name](input_dim)
            autoencoder = model_dict['autoencoder']
            encoder = model_dict['encoder']
            classifier = model_dict['classifier']
            encoding_dim = model_dict['encoding_dim']
        elif model_name == 'lightgbm_classifier':
            lgb_params = models[model_name](input_dim)
        else:
            model = models[model_name](input_dim)
    else:
        logger.error(f"Unknown model: {model_name}")
        return None
    
    # Train based on model type
    if model_name == 'autoencoder_classifier':
        logger.info("🔄 Training Autoencoder (Unsupervised pre-training)...")
        # Pre-train autoencoder (unsupervised)
        autoencoder.fit(
            X_train, X_train,
            epochs=epochs//2,  # Use half epochs for pre-training
            batch_size=batch_size,
            validation_data=(X_test, X_test),
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        logger.info("🎯 Training Classifier on encoded features...")
        # Encode features
        X_train_encoded = encoder.predict(X_train, batch_size=batch_size)
        X_test_encoded = encoder.predict(X_test, batch_size=batch_size)
        
        # Train classifier on encoded features
        classifier.fit(
            X_train_encoded, y_train,
            validation_data=(X_test_encoded, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=8, restore_best_weights=True, monitor='val_auc', mode='max'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=4, factor=0.5, monitor='val_auc', mode='max'
                )
            ],
            verbose=1
        )
        
        # Get predictions
        y_pred_proba = classifier.predict(X_test_encoded, batch_size=batch_size).flatten()
        
        # Create composite model for saving
        model = {
            'autoencoder': autoencoder,
            'encoder': encoder,
            'classifier': classifier,
            'type': 'autoencoder_classifier'
        }
        
    elif model_name == 'lightgbm_classifier':
        logger.info("🌟 Training LightGBM (Gradient Boosting)...")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train LightGBM model
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=epochs * 10,  # More iterations for gradient boosting
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'eval'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Get predictions
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        
    elif model_name == 'efficient_tabnet' and 'TabNet' in str(type(model)):
        # Fast TabNet training
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=epochs,
            patience=8,  # Reduced patience
            batch_size=batch_size,
            virtual_batch_size=512,  # Larger virtual batch
            eval_metric=['auc']
        )
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fast TensorFlow training
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=8,  # Reduced patience for speed
                restore_best_weights=True,
                monitor='val_auc',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=4,  # Faster LR reduction
                factor=0.5,
                monitor='val_auc',
                mode='max'
            )
        ]
        
        
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,  # Larger batch size for speed
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        y_pred_proba = model.predict(X_test, batch_size=batch_size).flatten()
    
    # Evaluate
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    training_time = time.time() - start_time
    
    logger.info(f"\n🎯 {model_name.upper()} RESULTS (Training time: {training_time:.1f}s):")
    logger.info(f"ROC AUC: {auc_score:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")
    logger.info(f"Training Speed: {training_time:.1f} seconds")
    
    return model, auc_score, pr_auc, training_time


def train_advanced_models(data_loader=None):
    """Train Autoencoder and LightGBM models with comparison"""
    print("\n🔧 Training Advanced Models (Autoencoder + LightGBM)")
    
    if data_loader is None:
        data_loader = HomeLoanData()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = data_loader.load_and_prepare_data()
    
    # Model configurations
    advanced_models = ['autoencoder_classifier', 'lightgbm_classifier']
    results = {}
    
    for model_name in advanced_models:
        print(f"\n{'='*50}")
        print(f"🚀 Training {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            model, auc_score, pr_auc, training_time = fast_train_model(
                model_name, X_train, X_test, y_train, y_test, 
                epochs=30,  # Reduced for faster training
                batch_size=1024
            )
            
            results[model_name] = {
                'model': model,
                'auc_score': auc_score,
                'pr_auc': pr_auc,
                'training_time': training_time,
                'feature_names': feature_names
            }
            
            print(f"\n✅ {model_name} completed!")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Compare results
    print(f"\n{'='*60}")
    print("🏆 ADVANCED MODELS COMPARISON")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name:25}: AUC={result['auc_score']:.4f}, PR-AUC={result['pr_auc']:.4f}, Time={result['training_time']:.1f}s")
        else:
            print(f"{model_name:25}: ERROR - {result['error']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FAST Deep Learning for Loan Default Prediction')
    parser.add_argument('--model', type=str, default='fast_dnn',
                       choices=['fast_dnn', 'lightweight_dnn', 'efficient_tabnet'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                       help='Ratio of data to use (0.1 = 10% for testing)')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size (larger = faster)')
    parser.add_argument('--data_path', type=str, default='loan_data/loan_data.csv')
    
    # Model management arguments
    parser.add_argument('--save_model', action='store_true', 
                       help='Save trained model for future testing')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name to save the model (default: auto-generated)')
    parser.add_argument('--test_model', type=str, default=None,
                       help='Test a previously saved model')
    parser.add_argument('--compare_models', nargs='+', default=None,
                       help='Compare multiple saved models')
    parser.add_argument('--list_models', action='store_true',
                       help='List all saved models')
    
    args = parser.parse_args()
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Handle model management commands
    if args.list_models:
        model_manager.list_models()
        return
    
    if args.test_model:
        logger.info(f"🔬 Testing saved model: {args.test_model}")
        # Load test data
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            args.data_path, 
            sample_ratio=args.sample_ratio
        )
        
        if X_test is not None and y_test is not None:
            results = model_manager.test_model(args.test_model, X_test, y_test)
            return
        else:
            logger.error("Failed to load test data")
            return
    
    if args.compare_models:
        logger.info(f"🔍 Comparing models: {args.compare_models}")
        # Load test data
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            args.data_path, 
            sample_ratio=args.sample_ratio
        )
        
        if X_test is not None and y_test is not None:
            comparison = model_manager.compare_models(args.compare_models, X_test, y_test)
            return
        else:
            logger.error("Failed to load test data")
            return
    
    # Training mode
    logger.info("⚡ FAST DEEP LEARNING FOR LOAN DEFAULT PREDICTION ⚡")
    logger.info("=" * 60)
    logger.info("🏠 Using HomeLoanData class for comprehensive preprocessing and SMOTE balancing")
    
    if args.sample_ratio < 1.0:
        logger.info(f"🔥 SPEED MODE: Using {args.sample_ratio*100:.0f}% of data for faster training")
    
    # Load and preprocess data using HomeLoanData
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        args.data_path, 
        sample_ratio=args.sample_ratio
    )
    
    if X_train is None or X_test is None or y_train is None or y_test is None:
        logger.error("Failed to load and preprocess data using HomeLoanData")
        return
    
    logger.info(f"🚀 Data ready for training:")
    logger.info(f"   Training set (SMOTE balanced): {X_train.shape}")
    logger.info(f"   Test set: {X_test.shape}")
    
    # Fast training
    result = fast_train_model(
        args.model, X_train, X_test, y_train, y_test, 
        epochs=args.epochs, batch_size=args.batch_size
    )
    
    if result is not None:
        model, auc_score, pr_auc, training_time = result
        logger.info(f"\n🎉 FAST TRAINING COMPLETED!")
        logger.info(f"🏆 ROC AUC: {auc_score:.4f}")
        logger.info(f"🏆 PR AUC: {pr_auc:.4f}")
        logger.info(f"⚡ Speed: {training_time:.1f} seconds")
        
        # Save model if requested
        if args.save_model:
            model_name = args.model_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare training metadata
            training_metadata = {
                'model_type': args.model,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'sample_ratio': args.sample_ratio,
                'training_time': training_time,
                'data_path': args.data_path
            }
            
            # Save model
            model_info = model_manager.save_model(
                model=model,
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
                model_type='tensorflow',
                metadata=training_metadata
            )
            
            logger.info(f"💾 Model saved as: {model_name}")
            logger.info(f"📊 Test the saved model with: --test_model {model_name}")
        
        # Performance recommendations
        if training_time > 300:  # 5 minutes
            logger.info(f"\n💡 SPEED TIPS:")
            logger.info(f"   • Try: --sample_ratio 0.3 (30% of data)")
            logger.info(f"   • Try: --model lightweight_dnn")
            logger.info(f"   • Try: --batch_size 4096")


if __name__ == "__main__":
    main()