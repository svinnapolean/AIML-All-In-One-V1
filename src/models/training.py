"""
Model Training Module

This module provides comprehensive model training capabilities including:
- Data preprocessing and feature engineering
- Multiple algorithm support (Linear Regression, Random Forest, Neural Networks)
- Hyperparameter tuning with cross-validation
- Model persistence and versioning
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    algorithm: str = "random_forest"  # "linear", "random_forest", "neural_network"
    task_type: str = "regression"  # "regression" or "classification"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    hyperparameter_tuning: bool = True
    model_save_path: str = "models/trained_models"
    scaler_save_path: str = "models/scalers"
    
    # Algorithm-specific parameters
    random_forest_params: Dict[str, Any] = None
    linear_params: Dict[str, Any] = None
    neural_network_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.random_forest_params is None:
            self.random_forest_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        
        if self.linear_params is None:
            self.linear_params = {}
        
        if self.neural_network_params is None:
            self.neural_network_params = {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'alpha': [0.001, 0.01, 0.1],
                'max_iter': [1000]
            }


class ModelTrainer:
    """
    Comprehensive model trainer for various machine learning algorithms
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_names = None
        self.training_history = {}
        
        # Ensure save directories exist
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.scaler_save_path, exist_ok=True)
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess the input data
        
        Args:
            X: Feature matrix
            y: Target vector (optional for prediction)
            
        Returns:
            Preprocessed X and y
        """
        # Store feature names for later use
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Handle categorical features if needed
        if isinstance(X, pd.DataFrame):
            # For simplicity, assuming numerical features only
            # In practice, you'd handle categorical encoding here
            pass
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle target variable
        y_processed = None
        if y is not None:
            if self.config.task_type == "classification":
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = y.values if isinstance(y, pd.Series) else y
        
        return X_scaled, y_processed
    
    def _get_model(self) -> Any:
        """Get the appropriate model based on configuration"""
        if self.config.algorithm == "linear":
            if self.config.task_type == "regression":
                return LinearRegression(**self.config.linear_params)
            else:
                return LogisticRegression(**self.config.linear_params, random_state=self.config.random_state)
        
        elif self.config.algorithm == "random_forest":
            if self.config.task_type == "regression":
                return RandomForestRegressor(random_state=self.config.random_state)
            else:
                return RandomForestClassifier(random_state=self.config.random_state)
        
        elif self.config.algorithm == "neural_network":
            if self.config.task_type == "regression":
                return MLPRegressor(random_state=self.config.random_state)
            else:
                return MLPClassifier(random_state=self.config.random_state)
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """Get hyperparameter grid for tuning"""
        if self.config.algorithm == "random_forest":
            return self.config.random_forest_params
        elif self.config.algorithm == "neural_network":
            return self.config.neural_network_params
        else:
            return self.config.linear_params
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the model with the given data
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Training results and metrics
        """
        print(f"Starting training with {self.config.algorithm} for {self.config.task_type}")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Get base model
        self.model = self._get_model()
        
        # Hyperparameter tuning
        if self.config.hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            param_grid = self._get_param_grid()
            
            if param_grid:  # Only tune if parameters are provided
                grid_search = GridSearchCV(
                    self.model,
                    param_grid,
                    cv=self.config.cv_folds,
                    scoring='neg_mean_squared_error' if self.config.task_type == 'regression' else 'accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best parameters: {best_params}")
            else:
                self.model.fit(X_train, y_train)
                best_params = {}
        else:
            self.model.fit(X_train, y_train)
            best_params = {}
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=self.config.cv_folds,
            scoring='neg_mean_squared_error' if self.config.task_type == 'regression' else 'accuracy'
        )
        
        # Predictions and evaluation
        y_pred = self.model.predict(X_test)
        
        if self.config.task_type == "regression":
            test_score = mean_squared_error(y_test, y_pred)
            metric_name = "MSE"
        else:
            test_score = accuracy_score(y_test, y_pred)
            metric_name = "Accuracy"
        
        # Store training history
        self.training_history = {
            'algorithm': self.config.algorithm,
            'task_type': self.config.task_type,
            'best_params': best_params,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            f'test_{metric_name.lower()}': test_score,
            'feature_names': self.feature_names,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"Training completed!")
        print(f"CV {metric_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test {metric_name}: {test_score:.4f}")
        
        return self.training_history
    
    def save_model(self, model_name: str) -> str:
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_path = os.path.join(self.config.model_save_path, f"{model_name}.joblib")
        scaler_path = os.path.join(self.config.scaler_save_path, f"{model_name}_scaler.joblib")
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        
        # Save label encoder if exists
        if self.label_encoder is not None:
            encoder_path = os.path.join(self.config.scaler_save_path, f"{model_name}_encoder.joblib")
            joblib.dump(self.label_encoder, encoder_path)
        
        # Save training history
        history_path = os.path.join(self.config.model_save_path, f"{model_name}_history.joblib")
        joblib.dump(self.training_history, history_path)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_name: str):
        """Load a previously trained model"""
        model_path = os.path.join(self.config.model_save_path, f"{model_name}.joblib")
        scaler_path = os.path.join(self.config.scaler_save_path, f"{model_name}_scaler.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load label encoder if exists
        encoder_path = os.path.join(self.config.scaler_save_path, f"{model_name}_encoder.joblib")
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        
        # Load training history
        history_path = os.path.join(self.config.model_save_path, f"{model_name}_history.joblib")
        if os.path.exists(history_path):
            self.training_history = joblib.load(history_path)
        
        print(f"Model loaded from: {model_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Preprocess data (without target)
        X_processed, _ = self.preprocess_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Inverse transform if classification
        if self.config.task_type == "classification" and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions


# Example usage and utility functions
def create_sample_dataset(n_samples: int = 1000, task_type: str = "regression") -> Tuple[pd.DataFrame, pd.Series]:
    """Create a sample dataset for testing"""
    np.random.seed(42)
    
    # Generate features
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.uniform(0, 10, n_samples),
        'feature_4': np.random.exponential(2, n_samples)
    })
    
    if task_type == "regression":
        # Create target with some relationship to features
        y = (2 * X['feature_1'] + 
             1.5 * X['feature_2'] + 
             0.5 * X['feature_3'] + 
             np.random.normal(0, 0.5, n_samples))
        y = pd.Series(y, name='target')
    else:
        # Create binary classification target
        linear_combination = (2 * X['feature_1'] + 
                            1.5 * X['feature_2'] + 
                            0.5 * X['feature_3'])
        y = pd.Series((linear_combination > linear_combination.median()).astype(int), name='target')
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Creating sample dataset...")
    X, y = create_sample_dataset(n_samples=1000, task_type="regression")
    
    # Configure training
    config = TrainingConfig(
        algorithm="random_forest",
        task_type="regression",
        hyperparameter_tuning=True
    )
    
    # Train model
    trainer = ModelTrainer(config)
    results = trainer.train(X, y)
    
    # Save model
    model_path = trainer.save_model("sample_model")
    
    # Make predictions
    predictions = trainer.predict(X.head(10))
    print(f"Sample predictions: {predictions[:5]}")