"""
Advanced Tools for the Numerics AI Agent

This module provides specialized tools that extend the agent's capabilities:
- Model inference and prediction tools
- Advanced data analysis and statistical tools  
- Visualization and reporting tools
- Model management and comparison tools
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

# Local imports
from ..models.training import ModelTrainer, TrainingConfig, create_sample_dataset
from ..models.testing import ModelTester, TestingConfig  
from ..models.evaluation import ModelEvaluator, EvaluationConfig


logger = logging.getLogger(__name__)


class ModelInferenceTool:
    """
    Advanced model inference tool with batch processing and model comparison
    """
    
    def __init__(self, models_path: str = "models/trained_models"):
        self.models_path = models_path
        self.loaded_models = {}  # Cache for loaded models
    
    def predict_single(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with a single model"""
        try:
            # Load model if not cached
            if model_name not in self.loaded_models:
                trainer = ModelTrainer(TrainingConfig())
                trainer.load_model(model_name)
                self.loaded_models[model_name] = trainer
            
            trainer = self.loaded_models[model_name]
            
            # Convert input to DataFrame
            X = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = trainer.predict(X)[0]
            
            return {
                "model_name": model_name,
                "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                "input_features": input_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in single prediction: {e}")
            return {"error": str(e), "model_name": model_name}
    
    def predict_batch(self, model_name: str, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions"""
        try:
            # Load model if not cached
            if model_name not in self.loaded_models:
                trainer = ModelTrainer(TrainingConfig())
                trainer.load_model(model_name)
                self.loaded_models[model_name] = trainer
            
            trainer = self.loaded_models[model_name]
            
            # Convert input to DataFrame
            X = pd.DataFrame(input_data)
            
            # Make predictions
            predictions = trainer.predict(X)
            
            return {
                "model_name": model_name,
                "predictions": [float(p) if isinstance(p, (int, float, np.number)) else str(p) for p in predictions],
                "num_predictions": len(predictions),
                "input_features": input_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return {"error": str(e), "model_name": model_name}
    
    def compare_models(self, model_names: List[str], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare predictions from multiple models"""
        results = {}
        
        for model_name in model_names:
            result = self.predict_single(model_name, input_data)
            results[model_name] = result
        
        # Calculate statistics if all predictions are numeric
        predictions = []
        valid_models = []
        
        for model_name, result in results.items():
            if "prediction" in result and "error" not in result:
                try:
                    pred_value = float(result["prediction"])
                    predictions.append(pred_value)
                    valid_models.append(model_name)
                except:
                    continue
        
        summary = {
            "model_results": results,
            "input_features": input_data,
            "timestamp": datetime.now().isoformat()
        }
        
        if len(predictions) > 1:
            summary["prediction_statistics"] = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "range": float(np.max(predictions) - np.min(predictions)),
                "valid_models": valid_models
            }
        
        return summary
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = []
        if os.path.exists(self.models_path):
            for file in os.listdir(self.models_path):
                if file.endswith('.joblib') and not file.endswith('_history.joblib'):
                    models.append(file.replace('.joblib', ''))
        return models


class DataAnalysisTool:
    """
    Advanced data analysis tool with statistical testing and insights generation
    """
    
    def __init__(self, results_path: str = "results/analysis"):
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)
    
    def comprehensive_analysis(self, data: Union[str, pd.DataFrame], 
                             target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        try:
            # Load data if string path provided
            if isinstance(data, str):
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith('.json'):
                    df = pd.read_json(data)
                else:
                    # Try to parse as JSON string
                    df = pd.DataFrame(json.loads(data))
            else:
                df = data.copy()
            
            analysis = {
                "data_overview": self._basic_overview(df),
                "statistical_summary": self._statistical_summary(df),
                "missing_values_analysis": self._missing_values_analysis(df),
                "correlation_analysis": self._correlation_analysis(df),
                "distribution_analysis": self._distribution_analysis(df),
                "outlier_analysis": self._outlier_analysis(df),
                "timestamp": datetime.now().isoformat()
            }
            
            # Target-specific analysis
            if target_column and target_column in df.columns:
                analysis["target_analysis"] = self._target_analysis(df, target_column)
                analysis["feature_target_relationships"] = self._feature_target_analysis(df, target_column)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": str(e)}
    
    def _basic_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic data overview"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
    
    def _statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical summary"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        summary = numeric_df.describe()
        
        # Add additional statistics
        additional_stats = {}
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            additional_stats[col] = {
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "variance": float(col_data.var()),
                "coefficient_of_variation": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else None
            }
        
        return {
            "basic_statistics": summary.to_dict(),
            "additional_statistics": additional_stats
        }
    
    def _missing_values_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Missing values analysis"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            "missing_counts": missing_counts.to_dict(),
            "missing_percentages": missing_percentages.to_dict(),
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": missing_counts[missing_counts > 0].index.tolist(),
            "missing_patterns": self._missing_patterns(df)
        }
    
    def _missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns"""
        # Find common missing patterns
        missing_matrix = df.isnull()
        patterns = missing_matrix.value_counts().head(10)
        
        return {
            "most_common_patterns": [
                {
                    "pattern": pattern,
                    "count": int(count),
                    "percentage": float(count / len(df) * 100)
                }
                for pattern, count in patterns.items()
            ]
        }
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}
        
        corr_matrix = numeric_df.corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_correlations.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val),
                        "strength": "strong" if abs(corr_val) > 0.8 else "moderate"
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_correlations,
            "max_correlation": float(corr_matrix.abs().max().max()),
            "mean_correlation": float(corr_matrix.abs().mean().mean())
        }
    
    def _distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Distribution analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        distributions = {}
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            
            distributions[col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "mode": float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                "std": float(col_data.std()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "range": float(col_data.max() - col_data.min()),
                "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                "quartiles": {
                    "q1": float(col_data.quantile(0.25)),
                    "q2": float(col_data.quantile(0.5)),
                    "q3": float(col_data.quantile(0.75))
                },
                "distribution_type": self._classify_distribution(col_data)
            }
        
        return distributions
    
    def _classify_distribution(self, data: pd.Series) -> str:
        """Classify distribution type based on skewness and kurtosis"""
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "approximately_normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "unknown"
    
    def _outlier_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Outlier detection and analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            
            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            z_outliers = col_data[z_scores > 3]
            
            outliers[col] = {
                "iqr_outliers": {
                    "count": len(iqr_outliers),
                    "percentage": float(len(iqr_outliers) / len(col_data) * 100),
                    "values": iqr_outliers.tolist()[:10]  # Limit to first 10
                },
                "z_score_outliers": {
                    "count": len(z_outliers),
                    "percentage": float(len(z_outliers) / len(col_data) * 100),
                    "values": z_outliers.tolist()[:10]  # Limit to first 10
                },
                "bounds": {
                    "iqr_lower": float(lower_bound),
                    "iqr_upper": float(upper_bound)
                }
            }
        
        return outliers
    
    def _target_analysis(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable"""
        target_data = df[target_column].dropna()
        
        if pd.api.types.is_numeric_dtype(target_data):
            # Numeric target (regression)
            return {
                "type": "numeric",
                "statistics": {
                    "mean": float(target_data.mean()),
                    "median": float(target_data.median()),
                    "std": float(target_data.std()),
                    "min": float(target_data.min()),
                    "max": float(target_data.max()),
                    "skewness": float(target_data.skew()),
                    "kurtosis": float(target_data.kurtosis())
                },
                "distribution": self._classify_distribution(target_data),
                "outliers": len(target_data[np.abs((target_data - target_data.mean()) / target_data.std()) > 3])
            }
        else:
            # Categorical target (classification)
            value_counts = target_data.value_counts()
            return {
                "type": "categorical",
                "classes": value_counts.index.tolist(),
                "class_counts": value_counts.values.tolist(),
                "class_percentages": (value_counts / len(target_data) * 100).values.tolist(),
                "num_classes": len(value_counts),
                "most_common": value_counts.index[0],
                "class_balance": float(value_counts.min() / value_counts.max())
            }
    
    def _feature_target_analysis(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze relationships between features and target"""
        target_data = df[target_column]
        feature_relationships = {}
        
        for col in df.columns:
            if col == target_column:
                continue
            
            col_data = df[col]
            
            if pd.api.types.is_numeric_dtype(col_data) and pd.api.types.is_numeric_dtype(target_data):
                # Numeric-numeric relationship
                correlation = col_data.corr(target_data)
                feature_relationships[col] = {
                    "type": "numeric_correlation",
                    "correlation": float(correlation) if not pd.isna(correlation) else None,
                    "strength": self._correlation_strength(correlation) if not pd.isna(correlation) else "unknown"
                }
            
            # Could add categorical-numeric, categorical-categorical relationships here
        
        return feature_relationships
    
    def _correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"


class VisualizationTool:
    """
    Advanced visualization tool with multiple plot types and customization
    """
    
    def __init__(self, plots_path: str = "results/plots"):
        self.plots_path = plots_path
        os.makedirs(plots_path, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_comprehensive_plots(self, data: Union[str, pd.DataFrame], 
                                 target_column: Optional[str] = None) -> Dict[str, Any]:
        """Create a comprehensive set of visualizations"""
        try:
            # Load data if string path provided
            if isinstance(data, str):
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith('.json'):
                    df = pd.read_json(data)
                else:
                    df = pd.DataFrame(json.loads(data))
            else:
                df = data.copy()
            
            plots_created = []
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. Distribution plots
            distribution_plot = self._create_distribution_plots(df, timestamp)
            if distribution_plot:
                plots_created.append(distribution_plot)
            
            # 2. Correlation heatmap
            correlation_plot = self._create_correlation_heatmap(df, timestamp)
            if correlation_plot:
                plots_created.append(correlation_plot)
            
            # 3. Missing values heatmap
            missing_plot = self._create_missing_values_plot(df, timestamp)
            if missing_plot:
                plots_created.append(missing_plot)
            
            # 4. Target analysis plots
            if target_column and target_column in df.columns:
                target_plots = self._create_target_analysis_plots(df, target_column, timestamp)
                plots_created.extend(target_plots)
            
            # 5. Outlier plots
            outlier_plots = self._create_outlier_plots(df, timestamp)
            plots_created.extend(outlier_plots)
            
            return {
                "plots_created": plots_created,
                "total_plots": len(plots_created),
                "plots_directory": self.plots_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating comprehensive plots: {e}")
            return {"error": str(e)}
    
    def _create_distribution_plots(self, df: pd.DataFrame, timestamp: str) -> Optional[Dict[str, str]]:
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        # Create subplots
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                data = df[col].dropna()
                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_path, f"distributions_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_type": "distributions",
            "plot_path": plot_path,
            "description": f"Distribution plots for {len(numeric_cols)} numeric columns"
        }
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, timestamp: str) -> Optional[Dict[str, str]]:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_path, f"correlation_heatmap_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_type": "correlation_heatmap",
            "plot_path": plot_path,
            "description": f"Correlation heatmap for {numeric_df.shape[1]} numeric features"
        }
    
    def _create_missing_values_plot(self, df: pd.DataFrame, timestamp: str) -> Optional[Dict[str, str]]:
        """Create missing values visualization"""
        missing_data = df.isnull().sum()
        
        if missing_data.sum() == 0:
            return None
        
        plt.figure(figsize=(12, 6))
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        bars = plt.bar(range(len(missing_data)), missing_data.values)
        plt.xticks(range(len(missing_data)), missing_data.index, rotation=45)
        plt.ylabel('Number of Missing Values')
        plt.title('Missing Values by Column')
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = height / len(df) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{percentage:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_path, f"missing_values_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_type": "missing_values",
            "plot_path": plot_path,
            "description": f"Missing values visualization for {len(missing_data)} columns"
        }
    
    def _create_target_analysis_plots(self, df: pd.DataFrame, target_column: str, timestamp: str) -> List[Dict[str, str]]:
        """Create target variable analysis plots"""
        plots = []
        target_data = df[target_column].dropna()
        
        if pd.api.types.is_numeric_dtype(target_data):
            # Numeric target - create distribution plot
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(target_data, bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(target_data)
            plt.title(f'Box Plot of {target_column}')
            plt.ylabel(target_column)
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.plots_path, f"target_distribution_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots.append({
                "plot_type": "target_distribution",
                "plot_path": plot_path,
                "description": f"Distribution analysis of target variable {target_column}"
            })
        
        else:
            # Categorical target - create bar plot
            plt.figure(figsize=(10, 6))
            value_counts = target_data.value_counts()
            
            bars = plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            plt.ylabel('Count')
            plt.title(f'Class Distribution of {target_column}')
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                percentage = height / len(target_data) * 100
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{percentage:.1f}%',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.plots_path, f"target_classes_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots.append({
                "plot_type": "target_classes",
                "plot_path": plot_path,
                "description": f"Class distribution of target variable {target_column}"
            })
        
        return plots
    
    def _create_outlier_plots(self, df: pd.DataFrame, timestamp: str) -> List[Dict[str, str]]:
        """Create outlier detection plots"""
        plots = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return plots
        
        # Box plots for outlier detection
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                data = df[col].dropna()
                axes[i].boxplot(data)
                axes[i].set_title(f'Outliers in {col}')
                axes[i].set_ylabel(col)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_path, f"outliers_boxplots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "plot_type": "outliers_boxplots",
            "plot_path": plot_path,
            "description": f"Box plots for outlier detection in {len(numeric_cols)} numeric columns"
        })
        
        return plots


# Export tool functions for direct use in agent
def create_model_inference_tool(models_path: str = "models/trained_models"):
    """Create model inference tool function"""
    tool = ModelInferenceTool(models_path)
    
    def model_inference(model_name: str, input_data: str, mode: str = "single") -> str:
        """
        Make predictions using trained models
        
        Args:
            model_name: Name of the model to use
            input_data: JSON string with input features
            mode: 'single' for one prediction, 'batch' for multiple, 'compare' for model comparison
        """
        try:
            data = json.loads(input_data)
            
            if mode == "single":
                result = tool.predict_single(model_name, data)
            elif mode == "batch":
                result = tool.predict_batch(model_name, data)
            elif mode == "compare":
                models = data.get("models", [model_name])
                input_features = data.get("features", {})
                result = tool.compare_models(models, input_features)
            else:
                result = {"error": f"Unknown mode: {mode}"}
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    return model_inference


def create_data_analysis_tool(results_path: str = "results/analysis"):
    """Create data analysis tool function"""
    tool = DataAnalysisTool(results_path)
    
    def data_analysis(data_source: str, target_column: str = None, analysis_type: str = "comprehensive") -> str:
        """
        Perform statistical analysis on data
        
        Args:
            data_source: Path to data file or JSON data string
            target_column: Name of target column (optional)
            analysis_type: Type of analysis to perform
        """
        try:
            result = tool.comprehensive_analysis(data_source, target_column)
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    return data_analysis


def create_visualization_tool(plots_path: str = "results/plots"):
    """Create visualization tool function"""
    tool = VisualizationTool(plots_path)
    
    def visualization(data_source: str, target_column: str = None, plot_type: str = "comprehensive") -> str:
        """
        Create visualizations from data
        
        Args:
            data_source: Path to data file or JSON data string
            target_column: Name of target column (optional)
            plot_type: Type of plots to create
        """
        try:
            result = tool.create_comprehensive_plots(data_source, target_column)
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    return visualization