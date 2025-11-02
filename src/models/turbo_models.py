"""
Performance-Optimized Advanced Models Training

This module provides highly optimized training for:
1. Fast Autoencoder with reduced complexity
2. Turbo LightGBM with optimized hyperparameters
3. XGBoost as additional fast alternative
4. Performance benchmarking and comparison
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import time
import logging
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow for optimal performance
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
tf.config.threading.set_intra_op_parallelism_threads(0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TurboModelTrainer:
    """Ultra-fast model trainer with performance optimizations"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.setup_tensorflow()
        
    def setup_tensorflow(self):
        """Configure TensorFlow for maximum performance"""
        if self.use_gpu:
            # Enable GPU if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    logger.info("🚀 GPU acceleration enabled")
                except RuntimeError as e:
                    logger.warning(f"GPU setup failed: {e}")
        
        # Enable mixed precision for faster training
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        logger.info("⚡ Mixed precision enabled")
    
    def create_turbo_autoencoder(self, input_dim: int) -> Dict[str, Any]:
        """Create ultra-fast autoencoder with minimal complexity"""
        # Reduced encoding dimension for speed
        encoding_dim = max(8, input_dim // 4)  # Smaller bottleneck
        
        # Input layer
        input_layer = keras.layers.Input(shape=(input_dim,))
        
        # Encoder - simplified single layer
        encoded = keras.layers.Dense(
            encoding_dim, 
            activation='relu',
            kernel_initializer='he_normal'
        )(input_layer)
        encoded = keras.layers.Dropout(0.2)(encoded)
        
        # Decoder - single layer
        decoded = keras.layers.Dense(
            input_dim, 
            activation='sigmoid',
            kernel_initializer='he_normal'
        )(encoded)
        
        # Autoencoder model
        autoencoder = keras.Model(input_layer, decoded, name='turbo_autoencoder')
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Higher LR for speed
            loss='mse',
            metrics=['mae']
        )
        
        # Encoder model
        encoder = keras.Model(input_layer, encoded, name='turbo_encoder')
        
        # Classifier - simplified
        classifier_input = keras.layers.Input(shape=(encoding_dim,))
        classifier_output = keras.layers.Dense(
            32, activation='relu', kernel_initializer='he_normal'
        )(classifier_input)
        classifier_output = keras.layers.Dropout(0.3)(classifier_output)
        classifier_output = keras.layers.Dense(
            1, activation='sigmoid', dtype='float32'  # Ensure float32 output
        )(classifier_output)
        
        classifier = keras.Model(classifier_input, classifier_output, name='turbo_classifier')
        classifier.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        
        return {
            'autoencoder': autoencoder,
            'encoder': encoder, 
            'classifier': classifier,
            'encoding_dim': encoding_dim,
            'type': 'turbo_autoencoder_classifier'
        }
    
    def create_turbo_lightgbm(self) -> Dict[str, Any]:
        """Create ultra-fast LightGBM with optimized parameters"""
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # Reduced for speed
            'learning_rate': 0.2,  # Higher for faster convergence
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,  # Use all cores
            'force_row_wise': True,  # Faster for small datasets
            'bin_construct_sample_cnt': 50000  # Reduce binning time
        }
    
    def create_turbo_xgboost(self) -> Dict[str, Any]:
        """Create XGBoost as alternative fast model"""
        try:
            import xgboost as xgb
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 4,  # Shallow trees for speed
                'learning_rate': 0.3,  # Higher for speed
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',  # Faster algorithm
                'verbosity': 0
            }
        except ImportError:
            logger.warning("XGBoost not available")
            return None
    
    def train_turbo_autoencoder(self, X_train, X_test, y_train, y_test, 
                               epochs=15, batch_size=2048) -> Tuple[Dict, float, float, float]:
        """Train autoencoder with maximum speed optimizations"""
        start_time = time.time()
        logger.info("🚀 Training Turbo Autoencoder...")
        
        model_dict = self.create_turbo_autoencoder(X_train.shape[1])
        autoencoder = model_dict['autoencoder']
        encoder = model_dict['encoder']
        classifier = model_dict['classifier']
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Fast autoencoder pre-training
        logger.info("⚡ Pre-training autoencoder...")
        autoencoder.fit(
            X_train, X_train,
            epochs=epochs//3,  # Reduced epochs
            batch_size=batch_size,
            validation_data=(X_test, X_test),
            verbose=0,  # Silent for speed
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )
        
        # Fast classifier training
        logger.info("🎯 Training classifier...")
        X_train_encoded = encoder.predict(X_train, batch_size=batch_size, verbose=0)
        X_test_encoded = encoder.predict(X_test, batch_size=batch_size, verbose=0)
        
        classifier.fit(
            X_train_encoded, y_train,
            validation_data=(X_test_encoded, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor='val_auc', mode='max'
                )
            ]
        )
        
        # Evaluate
        y_pred_proba = classifier.predict(X_test_encoded, batch_size=batch_size, verbose=0).flatten()
        auc_score = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Turbo Autoencoder: AUC={auc_score:.4f}, Time={training_time:.2f}s")
        
        return model_dict, auc_score, pr_auc, training_time
    
    def train_turbo_lightgbm(self, X_train, X_test, y_train, y_test,
                            num_boost_round=100) -> Tuple[Any, float, float, float]:
        """Train LightGBM with turbo optimizations"""
        start_time = time.time()
        logger.info("🌟 Training Turbo LightGBM...")
        
        params = self.create_turbo_lightgbm()
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Fast training
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            valid_names=['eval'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=0)  # Silent
            ]
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Turbo LightGBM: AUC={auc_score:.4f}, Time={training_time:.2f}s")
        
        return model, auc_score, pr_auc, training_time
    
    def train_turbo_xgboost(self, X_train, X_test, y_train, y_test,
                           num_boost_round=100) -> Tuple[Any, float, float, float]:
        """Train XGBoost with turbo optimizations"""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not available, skipping...")
            return None, 0, 0, 0
        
        start_time = time.time()
        logger.info("⚡ Training Turbo XGBoost...")
        
        params = self.create_turbo_xgboost()
        if params is None:
            return None, 0, 0, 0
        
        # Create DMatrix for speed
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Fast training
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False  # Silent
        )
        
        # Evaluate
        y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
        auc_score = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Turbo XGBoost: AUC={auc_score:.4f}, Time={training_time:.2f}s")
        
        return model, auc_score, pr_auc, training_time
    
    def benchmark_all_models(self, X_train, X_test, y_train, y_test) -> Dict[str, Dict]:
        """Benchmark all turbo models"""
        logger.info("🏁 Starting Turbo Model Benchmark")
        logger.info("=" * 60)
        
        results = {}
        
        # Turbo Autoencoder
        try:
            model, auc, pr_auc, time_taken = self.train_turbo_autoencoder(
                X_train, X_test, y_train, y_test, epochs=10, batch_size=4096
            )
            results['turbo_autoencoder'] = {
                'model': model,
                'auc': auc,
                'pr_auc': pr_auc,
                'time': time_taken,
                'type': 'turbo_autoencoder_classifier'
            }
        except Exception as e:
            logger.error(f"Turbo Autoencoder failed: {e}")
            results['turbo_autoencoder'] = {'error': str(e)}
        
        # Turbo LightGBM
        try:
            model, auc, pr_auc, time_taken = self.train_turbo_lightgbm(
                X_train, X_test, y_train, y_test, num_boost_round=50
            )
            results['turbo_lightgbm'] = {
                'model': model,
                'auc': auc,
                'pr_auc': pr_auc,
                'time': time_taken,
                'type': 'turbo_lightgbm_classifier'
            }
        except Exception as e:
            logger.error(f"Turbo LightGBM failed: {e}")
            results['turbo_lightgbm'] = {'error': str(e)}
        
        # Turbo XGBoost
        try:
            model, auc, pr_auc, time_taken = self.train_turbo_xgboost(
                X_train, X_test, y_train, y_test, num_boost_round=50
            )
            if model is not None:
                results['turbo_xgboost'] = {
                    'model': model,
                    'auc': auc,
                    'pr_auc': pr_auc,
                    'time': time_taken,
                    'type': 'turbo_xgboost_classifier'
                }
        except Exception as e:
            logger.error(f"Turbo XGBoost failed: {e}")
            results['turbo_xgboost'] = {'error': str(e)}
        
        # Performance Summary
        logger.info("\\n🏆 TURBO MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        for model_name, result in results.items():
            if 'error' not in result:
                logger.info(f"{model_name:20}: AUC={result['auc']:.4f}, Time={result['time']:.2f}s")
            else:
                logger.info(f"{model_name:20}: ERROR - {result['error']}")
        
        # Find fastest and best
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            fastest = min(valid_results.items(), key=lambda x: x[1]['time'])
            best_auc = max(valid_results.items(), key=lambda x: x[1]['auc'])
            
            logger.info(f"\\n⚡ FASTEST MODEL: {fastest[0]} ({fastest[1]['time']:.2f}s)")
            logger.info(f"🎯 BEST AUC: {best_auc[0]} ({best_auc[1]['auc']:.4f})")
        
        return results


def create_synthetic_data(n_samples=50000, n_features=20):
    """Create larger synthetic dataset for performance testing"""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42,
        flip_y=0.05
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return X_scaled, y, feature_names


def run_turbo_benchmark():
    """Run complete turbo model benchmark"""
    print("🚀 TURBO MODEL PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create test data
    print("📊 Creating test dataset...")
    X, y, feature_names = create_synthetic_data(n_samples=20000, n_features=20)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Dataset created: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Initialize trainer
    trainer = TurboModelTrainer()
    
    # Run benchmark
    results = trainer.benchmark_all_models(X_train, X_test, y_train, y_test)
    
    return results


if __name__ == "__main__":
    results = run_turbo_benchmark()