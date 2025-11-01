"""
Model Testing Module

This module provides comprehensive model testing capabilities including:
- Model validation and performance testing
- A/B testing between different models
- Statistical significance testing
- Model robustness and stability testing
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy import stats


@dataclass
class TestingConfig:
    """Configuration for model testing"""
    test_data_path: str = "data/test"
    results_save_path: str = "results/testing"
    plots_save_path: str = "results/plots"
    cv_folds: int = 5
    confidence_level: float = 0.95
    random_state: int = 42
    
    # A/B testing parameters
    ab_test_size: int = 1000
    significance_level: float = 0.05
    
    def __post_init__(self):
        # Ensure save directories exist
        os.makedirs(self.results_save_path, exist_ok=True)
        os.makedirs(self.plots_save_path, exist_ok=True)


class ModelTester:
    """
    Comprehensive model testing framework
    """
    
    def __init__(self, config: TestingConfig):
        self.config = config
        self.test_results = {}
    
    def load_test_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data from file"""
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        return X, y
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model performance"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate classification model performance"""
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Add AUC if probabilities are provided
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    results['auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                else:  # Multi-class
                    results['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError:
                results['auc'] = None
        
        return results
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                           task_type: str = "regression") -> Dict[str, Any]:
        """Perform cross-validation testing"""
        if task_type == "regression":
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        else:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'confidence_interval': self._calculate_confidence_interval(scores)
            }
        
        return cv_results
    
    def _calculate_confidence_interval(self, scores: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for scores"""
        alpha = 1 - self.config.confidence_level
        n = len(scores)
        mean = scores.mean()
        std_err = scores.std() / np.sqrt(n)
        t_val = stats.t.ppf(1 - alpha/2, n-1)
        margin_error = t_val * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def test_model_stability(self, model, X: pd.DataFrame, y: pd.Series, 
                           n_iterations: int = 100) -> Dict[str, Any]:
        """Test model stability across multiple runs"""
        scores = []
        
        for i in range(n_iterations):
            # Add small random noise to test stability
            X_noise = X + np.random.normal(0, 0.01, X.shape)
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict(X_noise)
                    score = accuracy_score(y, y_pred)
                else:
                    y_pred = model.predict(X_noise)
                    score = r2_score(y, y_pred)
                
                scores.append(score)
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                continue
        
        scores = np.array(scores)
        
        return {
            'stability_scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'coefficient_variation': scores.std() / scores.mean() if scores.mean() != 0 else np.inf,
            'min_score': scores.min(),
            'max_score': scores.max()
        }
    
    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                      task_type: str = "regression") -> Dict[str, Any]:
        """Compare multiple models using statistical tests"""
        model_scores = {}
        
        # Get scores for each model
        for name, model in models.items():
            cv_results = self.cross_validate_model(model, X, y, task_type)
            
            if task_type == "regression":
                primary_metric = 'neg_mean_squared_error'
            else:
                primary_metric = 'accuracy'
            
            model_scores[name] = cv_results[primary_metric]['scores']
        
        # Perform pairwise statistical tests
        comparison_results = {}
        model_names = list(model_scores.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(model_scores[model1], model_scores[model2])
                
                comparison_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level,
                    'better_model': model1 if np.mean(model_scores[model1]) > np.mean(model_scores[model2]) else model2
                }
        
        return {
            'model_scores': {name: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            } for name, scores in model_scores.items()},
            'statistical_comparisons': comparison_results
        }
    
    def generate_test_report(self, model, X: pd.DataFrame, y: pd.Series, 
                           model_name: str, task_type: str = "regression") -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print(f"Generating test report for {model_name}...")
        
        # Basic predictions
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
        
        # Performance metrics
        if task_type == "regression":
            performance = self.evaluate_regression_model(y, y_pred)
        else:
            performance = self.evaluate_classification_model(y, y_pred, y_prob)
        
        # Cross-validation
        cv_results = self.cross_validate_model(model, X, y, task_type)
        
        # Stability testing
        stability_results = self.test_model_stability(model, X, y)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                'features': X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
                'importances': model.feature_importances_.tolist()
            }
        
        report = {
            'model_name': model_name,
            'task_type': task_type,
            'test_data_shape': X.shape,
            'performance_metrics': performance,
            'cross_validation': cv_results,
            'stability_analysis': stability_results,
            'feature_importance': feature_importance,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save report
        report_path = os.path.join(self.config.results_save_path, f"{model_name}_test_report.json")
        pd.Series(report).to_json(report_path, indent=2)
        
        print(f"Test report saved to: {report_path}")
        return report
    
    def plot_model_performance(self, model, X: pd.DataFrame, y: pd.Series, 
                             model_name: str, task_type: str = "regression"):
        """Generate performance visualization plots"""
        y_pred = model.predict(X)
        
        if task_type == "regression":
            self._plot_regression_performance(y, y_pred, model_name)
        else:
            self._plot_classification_performance(y, y_pred, model_name)
            
        # Feature importance plot
        if hasattr(model, 'feature_importances_'):
            self._plot_feature_importance(model, X, model_name)
    
    def _plot_regression_performance(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot regression performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_save_path, f"{model_name}_regression_performance.png"))
        plt.close()
    
    def _plot_classification_performance(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot classification performance"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Class distribution
        unique_classes, counts_true = np.unique(y_true, return_counts=True)
        _, counts_pred = np.unique(y_pred, return_counts=True)
        
        x = np.arange(len(unique_classes))
        width = 0.35
        
        axes[1].bar(x - width/2, counts_true, width, label='True', alpha=0.7)
        axes[1].bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.7)
        axes[1].set_xlabel('Classes')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Class Distribution')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(unique_classes)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_save_path, f"{model_name}_classification_performance.png"))
        plt.close()
    
    def _plot_feature_importance(self, model, X: pd.DataFrame, model_name: str):
        """Plot feature importance"""
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_save_path, f"{model_name}_feature_importance.png"))
        plt.close()


# Example usage
if __name__ == "__main__":
    from .training import ModelTrainer, TrainingConfig, create_sample_dataset
    
    # Create sample data
    X, y = create_sample_dataset(n_samples=1000, task_type="regression")
    
    # Train a model
    config = TrainingConfig(algorithm="random_forest", task_type="regression")
    trainer = ModelTrainer(config)
    trainer.train(X, y)
    
    # Test the model
    test_config = TestingConfig()
    tester = ModelTester(test_config)
    
    # Generate comprehensive test report
    report = tester.generate_test_report(trainer.model, X, y, "test_model", "regression")
    
    # Generate performance plots
    tester.plot_model_performance(trainer.model, X, y, "test_model", "regression")
    
    print("Testing completed!")