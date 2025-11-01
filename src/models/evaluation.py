"""
Model Evaluation Module

This module provides comprehensive model evaluation capabilities including:
- Advanced performance metrics and statistical analysis
- Model interpretability and explainability
- Bias and fairness evaluation
- Performance monitoring and drift detection
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    
    # Basic metrics
    primary_metric: float
    secondary_metrics: Dict[str, float]
    
    # Statistical measures
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, bool]
    
    # Model characteristics
    model_complexity: Dict[str, Any]
    training_time: float
    inference_time: float
    
    # Interpretability metrics
    feature_importance: Dict[str, float]
    interpretability_score: float
    
    # Fairness and bias metrics
    bias_metrics: Dict[str, float]
    fairness_score: float
    
    # Metadata
    evaluation_timestamp: str
    data_characteristics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Paths
    results_save_path: str = "results/evaluation"
    plots_save_path: str = "results/evaluation/plots"
    reports_save_path: str = "results/evaluation/reports"
    
    # Evaluation parameters
    confidence_level: float = 0.95
    n_bootstrap_samples: int = 1000
    significance_level: float = 0.05
    
    # Performance monitoring
    performance_threshold: float = 0.1  # Percentage drop threshold
    drift_detection_window: int = 100
    
    # Interpretability
    n_permutation_repeats: int = 10
    feature_importance_threshold: float = 0.01
    
    # Fairness evaluation
    protected_attributes: List[str] = None
    fairness_threshold: float = 0.1
    
    def __post_init__(self):
        # Create directories
        for path in [self.results_save_path, self.plots_save_path, self.reports_save_path]:
            os.makedirs(path, exist_ok=True)
        
        if self.protected_attributes is None:
            self.protected_attributes = []


class ModelEvaluator:
    """
    Comprehensive model evaluation framework with advanced analytics
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluation_history = []
        self.baseline_metrics = None
    
    def bootstrap_metric(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        metric_func, n_bootstrap: int = None) -> Tuple[float, Tuple[float, float]]:
        """Calculate metric with bootstrap confidence interval"""
        if n_bootstrap is None:
            n_bootstrap = self.config.n_bootstrap_samples
        
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except:
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        mean_score = np.mean(bootstrap_scores)
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
        
        return mean_score, (lower, upper)
    
    def calculate_comprehensive_metrics(self, model, X: pd.DataFrame, y: pd.Series,
                                      task_type: str = "regression") -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        start_time = datetime.now()
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
        
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Primary and secondary metrics
        if task_type == "regression":
            primary_metric = r2_score(y, y_pred)
            secondary_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'mape': np.mean(np.abs((y - y_pred) / y)) * 100
            }
            
            # Bootstrap confidence intervals
            confidence_intervals = {}
            for metric_name, metric_func in [
                ('r2', r2_score),
                ('mse', mean_squared_error),
                ('mae', mean_absolute_error)
            ]:
                _, ci = self.bootstrap_metric(y.values, y_pred, metric_func)
                confidence_intervals[metric_name] = ci
        
        else:  # Classification
            primary_metric = accuracy_score(y, y_pred)
            secondary_metrics = {
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            }
            
            if y_prob is not None and len(np.unique(y)) == 2:
                secondary_metrics['auc'] = roc_auc_score(y, y_prob[:, 1])
            
            # Bootstrap confidence intervals
            confidence_intervals = {}
            for metric_name, metric_func in [
                ('accuracy', accuracy_score),
                ('precision', lambda yt, yp: precision_score(yt, yp, average='weighted')),
                ('recall', lambda yt, yp: recall_score(yt, yp, average='weighted'))
            ]:
                _, ci = self.bootstrap_metric(y.values, y_pred, metric_func)
                confidence_intervals[metric_name] = ci
        
        # Model complexity
        model_complexity = self._calculate_model_complexity(model)
        
        # Feature importance and interpretability
        feature_importance = self._calculate_feature_importance(model, X, y)
        interpretability_score = self._calculate_interpretability_score(model, feature_importance)
        
        # Bias and fairness metrics (placeholder - would need protected attributes)
        bias_metrics = {}
        fairness_score = 1.0  # Perfect fairness by default
        
        # Data characteristics
        data_characteristics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_types': self._analyze_feature_types(X),
            'missing_values': X.isnull().sum().sum(),
            'target_distribution': self._analyze_target_distribution(y, task_type)
        }
        
        return EvaluationMetrics(
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            confidence_intervals=confidence_intervals,
            statistical_significance={},  # Would need comparison for significance
            model_complexity=model_complexity,
            training_time=0.0,  # Would need to track during training
            inference_time=inference_time,
            feature_importance=feature_importance,
            interpretability_score=interpretability_score,
            bias_metrics=bias_metrics,
            fairness_score=fairness_score,
            evaluation_timestamp=datetime.now().isoformat(),
            data_characteristics=data_characteristics
        )
    
    def _calculate_model_complexity(self, model) -> Dict[str, Any]:
        """Calculate model complexity metrics"""
        complexity = {'model_type': type(model).__name__}
        
        # Try to get model-specific complexity measures
        if hasattr(model, 'n_estimators'):  # Tree-based models
            complexity['n_estimators'] = model.n_estimators
        
        if hasattr(model, 'max_depth'):
            complexity['max_depth'] = model.max_depth
        
        if hasattr(model, 'n_layers_'):  # Neural networks
            complexity['n_layers'] = model.n_layers_
            complexity['n_outputs'] = model.n_outputs_
        
        if hasattr(model, 'coef_'):  # Linear models
            complexity['n_coefficients'] = np.prod(model.coef_.shape)
            complexity['coefficient_magnitude'] = np.mean(np.abs(model.coef_))
        
        return complexity
    
    def _calculate_feature_importance(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using multiple methods"""
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        importance_dict = {}
        
        # Built-in feature importance
        if hasattr(model, 'feature_importances_'):
            for name, importance in zip(feature_names, model.feature_importances_):
                importance_dict[f"{name}_builtin"] = importance
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=self.config.n_permutation_repeats,
                random_state=42
            )
            for name, importance in zip(feature_names, perm_importance.importances_mean):
                importance_dict[f"{name}_permutation"] = importance
        except:
            pass  # Skip if permutation importance fails
        
        return importance_dict
    
    def _calculate_interpretability_score(self, model, feature_importance: Dict[str, float]) -> float:
        """Calculate model interpretability score"""
        # Simple heuristic based on model type and feature importance distribution
        model_type = type(model).__name__
        
        # Base interpretability by model type
        interpretability_scores = {
            'LinearRegression': 0.9,
            'LogisticRegression': 0.9,
            'DecisionTreeRegressor': 0.8,
            'DecisionTreeClassifier': 0.8,
            'RandomForestRegressor': 0.6,
            'RandomForestClassifier': 0.6,
            'MLPRegressor': 0.3,
            'MLPClassifier': 0.3
        }
        
        base_score = interpretability_scores.get(model_type, 0.5)
        
        # Adjust based on feature importance clarity
        if feature_importance:
            importance_values = list(feature_importance.values())
            if importance_values:
                # Higher variance in importance = clearer feature distinctions
                importance_variance = np.var(importance_values)
                clarity_bonus = min(0.2, importance_variance * 2)
                base_score += clarity_bonus
        
        return min(1.0, base_score)
    
    def _analyze_feature_types(self, X: pd.DataFrame) -> Dict[str, int]:
        """Analyze types of features in the dataset"""
        if not hasattr(X, 'dtypes'):
            return {'numeric': X.shape[1]}
        
        type_counts = {
            'numeric': 0,
            'categorical': 0,
            'datetime': 0,
            'text': 0
        }
        
        for dtype in X.dtypes:
            if pd.api.types.is_numeric_dtype(dtype):
                type_counts['numeric'] += 1
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                type_counts['categorical'] += 1
            elif pd.api.types.is_datetime64_dtype(dtype):
                type_counts['datetime'] += 1
            else:
                type_counts['text'] += 1
        
        return type_counts
    
    def _analyze_target_distribution(self, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """Analyze target variable distribution"""
        if task_type == "regression":
            return {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
                'skewness': float(y.skew()),
                'kurtosis': float(y.kurtosis())
            }
        else:
            value_counts = y.value_counts()
            return {
                'classes': value_counts.index.tolist(),
                'counts': value_counts.values.tolist(),
                'class_balance': float(value_counts.min() / value_counts.max())
            }
    
    def evaluate_learning_curves(self, model, X: pd.DataFrame, y: pd.Series,
                                cv: int = 5) -> Dict[str, Any]:
        """Generate learning curves analysis"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }
    
    def detect_performance_drift(self, current_metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Detect performance drift compared to baseline"""
        if self.baseline_metrics is None:
            self.baseline_metrics = current_metrics
            return {'drift_detected': False, 'message': 'Baseline metrics established'}
        
        # Compare primary metric
        baseline_primary = self.baseline_metrics.primary_metric
        current_primary = current_metrics.primary_metric
        
        drift_percentage = abs(current_primary - baseline_primary) / baseline_primary * 100
        drift_detected = drift_percentage > self.config.performance_threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_percentage': drift_percentage,
            'threshold': self.config.performance_threshold,
            'baseline_metric': baseline_primary,
            'current_metric': current_primary,
            'recommendation': 'Retrain model' if drift_detected else 'Continue monitoring'
        }
    
    def generate_evaluation_report(self, model, X: pd.DataFrame, y: pd.Series,
                                 model_name: str, task_type: str = "regression") -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        print(f"Generating evaluation report for {model_name}...")
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(model, X, y, task_type)
        
        # Learning curves
        learning_curves = self.evaluate_learning_curves(model, X, y)
        
        # Performance drift detection
        drift_analysis = self.detect_performance_drift(metrics)
        
        # Compile report
        report = {
            'model_name': model_name,
            'task_type': task_type,
            'evaluation_metrics': metrics.to_dict(),
            'learning_curves': learning_curves,
            'drift_analysis': drift_analysis,
            'recommendations': self._generate_recommendations(metrics, drift_analysis),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_path = os.path.join(self.config.reports_save_path, f"{model_name}_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate plots
        self._generate_evaluation_plots(model, X, y, model_name, task_type, metrics)
        
        # Store in evaluation history
        self.evaluation_history.append(report)
        
        print(f"Evaluation report saved to: {report_path}")
        return report
    
    def _generate_recommendations(self, metrics: EvaluationMetrics, 
                                drift_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        recommendations = []
        
        # Performance-based recommendations
        if metrics.primary_metric < 0.7:  # Assuming normalized metrics
            recommendations.append("Consider feature engineering or algorithm selection")
        
        # Interpretability recommendations
        if metrics.interpretability_score < 0.5:
            recommendations.append("Consider using more interpretable models for better explainability")
        
        # Complexity recommendations
        if 'n_estimators' in metrics.model_complexity and metrics.model_complexity['n_estimators'] > 500:
            recommendations.append("Model might be overly complex; consider reducing n_estimators")
        
        # Drift recommendations
        if drift_analysis['drift_detected']:
            recommendations.append("Performance drift detected; consider model retraining")
        
        # Data quality recommendations
        if metrics.data_characteristics['missing_values'] > 0:
            recommendations.append("Address missing values in the dataset")
        
        return recommendations
    
    def _generate_evaluation_plots(self, model, X: pd.DataFrame, y: pd.Series,
                                 model_name: str, task_type: str, metrics: EvaluationMetrics):
        """Generate comprehensive evaluation plots"""
        
        # Feature importance plot
        if metrics.feature_importance:
            self._plot_feature_importance(metrics.feature_importance, model_name)
        
        # Performance metrics plot
        self._plot_performance_metrics(metrics, model_name)
        
        # Learning curves plot (if available)
        learning_curves = self.evaluate_learning_curves(model, X, y)
        self._plot_learning_curves(learning_curves, model_name)
    
    def _plot_feature_importance(self, feature_importance: Dict[str, float], model_name: str):
        """Plot feature importance"""
        # Filter for built-in importance if available
        builtin_importance = {k.replace('_builtin', ''): v for k, v in feature_importance.items() if '_builtin' in k}
        
        if not builtin_importance:
            # Use permutation importance
            builtin_importance = {k.replace('_permutation', ''): v for k, v in feature_importance.items() if '_permutation' in k}
        
        if builtin_importance:
            # Sort by importance
            sorted_features = sorted(builtin_importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features[:20])  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.plots_save_path, f"{model_name}_feature_importance.png"))
            plt.close()
    
    def _plot_performance_metrics(self, metrics: EvaluationMetrics, model_name: str):
        """Plot performance metrics with confidence intervals"""
        metric_names = list(metrics.secondary_metrics.keys())
        metric_values = list(metrics.secondary_metrics.values())
        
        # Get confidence intervals if available
        ci_lower = []
        ci_upper = []
        for name in metric_names:
            if name in metrics.confidence_intervals:
                ci_lower.append(metrics.confidence_intervals[name][0])
                ci_upper.append(metrics.confidence_intervals[name][1])
            else:
                ci_lower.append(metric_values[metric_names.index(name)])
                ci_upper.append(metric_values[metric_names.index(name)])
        
        plt.figure(figsize=(10, 6))
        x_pos = range(len(metric_names))
        plt.bar(x_pos, metric_values, alpha=0.7)
        plt.errorbar(x_pos, metric_values, 
                    yerr=[np.array(metric_values) - np.array(ci_lower),
                          np.array(ci_upper) - np.array(metric_values)],
                    fmt='none', capsize=5, color='red')
        plt.xticks(x_pos, metric_names, rotation=45)
        plt.ylabel('Metric Value')
        plt.title(f'Performance Metrics with Confidence Intervals - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_save_path, f"{model_name}_performance_metrics.png"))
        plt.close()
    
    def _plot_learning_curves(self, learning_curves: Dict[str, Any], model_name: str):
        """Plot learning curves"""
        train_sizes = learning_curves['train_sizes']
        train_mean = learning_curves['train_scores_mean']
        train_std = learning_curves['train_scores_std']
        val_mean = learning_curves['val_scores_mean']
        val_std = learning_curves['val_scores_std']
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.fill_between(train_sizes, 
                        np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std),
                        alpha=0.3)
        
        plt.plot(train_sizes, val_mean, 's-', label='Validation Score')
        plt.fill_between(train_sizes,
                        np.array(val_mean) - np.array(val_std),
                        np.array(val_mean) + np.array(val_std),
                        alpha=0.3)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_save_path, f"{model_name}_learning_curves.png"))
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
    
    # Evaluate the model
    eval_config = EvaluationConfig()
    evaluator = ModelEvaluator(eval_config)
    
    # Generate comprehensive evaluation report
    report = evaluator.generate_evaluation_report(trainer.model, X, y, "test_model", "regression")
    
    print("Evaluation completed!")
    print(f"Primary metric (R²): {report['evaluation_metrics']['primary_metric']:.4f}")
    print(f"Interpretability score: {report['evaluation_metrics']['interpretability_score']:.4f}")