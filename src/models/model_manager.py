"""
Model Management and Testing System for Home Loan Default Prediction

This system provides comprehensive model management including:
1. Model saving with metadata and preprocessing objects
2. Model loading and prediction
3. Comprehensive model testing and evaluation
4. Model comparison and performance tracking
5. Automated testing pipelines

Usage:
    # Save model after training
    model_manager = ModelManager()
    model_manager.save_model(model, model_name, X_test, y_test, preprocessing_data)
    
    # Load and test model
    loaded_model = model_manager.load_model(model_name)
    results = model_manager.test_model(model_name, X_test, y_test)
    
    # Compare multiple models
    comparison = model_manager.compare_models(['model1', 'model2'], X_test, y_test)
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    """
    Comprehensive model management system for loan default prediction models
    """
    
    def __init__(self, models_dir='saved_models', results_dir='test_results'):
        """
        Initialize ModelManager
        
        Parameters:
        -----------
        models_dir : str
            Directory to save/load models
        results_dir : str
            Directory to save test results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        
        # Load model registry
        self.registry_file = os.path.join(self.models_dir, 'model_registry.json')
        self.model_registry = self._load_registry()
        
        print(f"🔧 ModelManager initialized")
        print(f"   📁 Models directory: {self.models_dir}")
        print(f"   📁 Results directory: {self.results_dir}")
        print(f"   📋 Registered models: {len(self.model_registry)}")
    
    def _load_registry(self):
        """Load or create model registry"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def save_model(self, model, model_name: str, X_test, y_test, 
                   model_type='tensorflow', preprocessing_data=None, 
                   training_history=None, metadata=None):
        """
        Save trained model with comprehensive metadata
        
        Parameters:
        -----------
        model : trained model object
        model_name : str
            Unique name for the model
        X_test, y_test : test data for evaluation
        model_type : str
            Type of model ('tensorflow', 'sklearn', 'pytorch', etc.)
        preprocessing_data : dict
            Preprocessing objects (scalers, encoders, etc.)
        training_history : dict
            Training history and metrics
        metadata : dict
            Additional metadata
        """
        print(f"💾 Saving model: {model_name}")
        
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate evaluation metrics
        evaluation_metrics = self._evaluate_model(model, X_test, y_test, model_type)
        
        # Prepare model metadata
        model_info = {
            'model_name': model_name,
            'model_type': model_type,
            'created_date': datetime.now().isoformat(),
            'data_shape': {
                'n_features': X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0]),
                'test_samples': len(X_test)
            },
            'evaluation_metrics': evaluation_metrics,
            'training_history': training_history,
            'metadata': metadata or {}
        }
        
        # Save model based on type
        if model_type.lower() == 'tensorflow':
            model_path = os.path.join(model_dir, 'model.h5')
            model.save(model_path)
            print(f"   ✅ TensorFlow model saved: {model_path}")
        elif model_type.lower() == 'sklearn':
            model_path = os.path.join(model_dir, 'model.pkl')
            joblib.dump(model, model_path)
            print(f"   ✅ Scikit-learn model saved: {model_path}")
        else:
            model_path = os.path.join(model_dir, 'model.pkl')
            pickle.dump(model, open(model_path, 'wb'))
            print(f"   ✅ Generic model saved: {model_path}")
        
        # Save preprocessing objects
        if preprocessing_data:
            preprocessing_path = os.path.join(model_dir, 'preprocessing.pkl')
            pickle.dump(preprocessing_data, open(preprocessing_path, 'wb'))
            print(f"   ✅ Preprocessing objects saved")
        
        # Save model info
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Update registry
        self.model_registry[model_name] = {
            'path': model_dir,
            'type': model_type,
            'created': model_info['created_date'],
            'performance': {
                'roc_auc': evaluation_metrics.get('roc_auc', 0),
                'accuracy': evaluation_metrics.get('accuracy', 0),
                'f1_score': evaluation_metrics.get('f1_score', 0)
            }
        }
        self._save_registry()
        
        print(f"   📊 ROC AUC: {evaluation_metrics.get('roc_auc', 0):.4f}")
        print(f"   📊 Accuracy: {evaluation_metrics.get('accuracy', 0):.4f}")
        print(f"   📊 F1 Score: {evaluation_metrics.get('f1_score', 0):.4f}")
        print(f"   ✅ Model '{model_name}' saved successfully!")
        
        return model_info
    
    def load_model(self, model_name: str):
        """
        Load saved model with preprocessing objects
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
            
        Returns:
        --------
        dict : Contains 'model', 'preprocessing', 'info'
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        print(f"📂 Loading model: {model_name}")
        
        # Get the relative path and join with base models directory
        relative_path = self.model_registry[model_name]['path']
        model_dir = os.path.join(self.models_dir, relative_path)
        model_type = self.model_registry[model_name]['type']
        
        # Load model info
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # Load model based on type
        if model_type.lower() == 'tensorflow':
            model_path = os.path.join(model_dir, 'model.h5')
            model = keras.models.load_model(model_path)
            print(f"   ✅ TensorFlow model loaded")
        elif model_type.lower() == 'sklearn':
            model_path = os.path.join(model_dir, 'model.pkl')
            model = joblib.load(model_path)
            print(f"   ✅ Scikit-learn model loaded")
        else:
            model_path = os.path.join(model_dir, 'model.pkl')
            model = pickle.load(open(model_path, 'rb'))
            print(f"   ✅ Generic model loaded")
        
        # Load preprocessing objects if available
        preprocessing_path = os.path.join(model_dir, 'preprocessing.pkl')
        preprocessing = None
        if os.path.exists(preprocessing_path):
            preprocessing = pickle.load(open(preprocessing_path, 'rb'))
            print(f"   ✅ Preprocessing objects loaded")
        
        result = {
            'model': model,
            'preprocessing': preprocessing,
            'info': model_info,
            'type': model_type
        }
        
        # Handle different model info structures for compatibility
        roc_auc = (model_info.get('evaluation_metrics', {}).get('roc_auc') or 
                   model_info.get('roc_auc', 0))
        created_date = (model_info.get('created_date') or 
                       model_info.get('created_at', 'Unknown'))
        
        print(f"   📊 Original ROC AUC: {roc_auc:.4f}")
        print(f"   📅 Created: {created_date}")
        
        return result
    
    def _evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluate model and return metrics"""
        try:
            # Get predictions
            if model_type.lower() == 'tensorflow':
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
            elif hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.predict(X_test)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='binary')),
                'recall': float(recall_score(y_test, y_pred, average='binary')),
                'f1_score': float(f1_score(y_test, y_pred, average='binary')),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
            }
            
            if hasattr(y_pred_proba, '__len__') and len(y_pred_proba) > 0:
                metrics['pr_auc'] = float(average_precision_score(y_test, y_pred_proba))
            
            return metrics
            
        except Exception as e:
            print(f"   ⚠️  Error evaluating model: {e}")
            return {'error': str(e)}
    
    def test_model(self, model_name: str, X_test, y_test, 
                   save_results=True, generate_plots=True):
        """
        Comprehensive testing of a saved model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to test
        X_test, y_test : test data
        save_results : bool
            Whether to save test results
        generate_plots : bool
            Whether to generate visualization plots
            
        Returns:
        --------
        dict : Comprehensive test results
        """
        print(f"🔬 Testing model: {model_name}")
        
        # Load model
        model_data = self.load_model(model_name)
        model = model_data['model']
        model_type = model_data['type']
        
        # Get predictions
        if model_type.lower() == 'tensorflow':
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
        elif hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate comprehensive metrics
        test_results = {
            'model_name': model_name,
            'test_date': datetime.now().isoformat(),
            'test_data_shape': {
                'samples': len(X_test),
                'features': X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0])
            },
            'metrics': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='binary')),
                'recall': float(recall_score(y_test, y_pred, average='binary')),
                'f1_score': float(f1_score(y_test, y_pred, average='binary')),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
                'pr_auc': float(average_precision_score(y_test, y_pred_proba))
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Print results
        print(f"   📊 Test Results for {model_name}:")
        print(f"      Accuracy: {test_results['metrics']['accuracy']:.4f}")
        print(f"      Precision: {test_results['metrics']['precision']:.4f}")
        print(f"      Recall: {test_results['metrics']['recall']:.4f}")
        print(f"      F1 Score: {test_results['metrics']['f1_score']:.4f}")
        print(f"      ROC AUC: {test_results['metrics']['roc_auc']:.4f}")
        print(f"      PR AUC: {test_results['metrics']['pr_auc']:.4f}")
        
        # Generate plots
        if generate_plots:
            self._generate_test_plots(model_name, y_test, y_pred, y_pred_proba, test_results)
        
        # Save results
        if save_results:
            results_file = os.path.join(
                self.results_dir, 
                f"{model_name}_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"   💾 Test results saved: {results_file}")
        
        return test_results
    
    def _generate_test_plots(self, model_name, y_test, y_pred, y_pred_proba, test_results):
        """Generate comprehensive test plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Test Results: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = test_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        axes[0,0].set_xticklabels(['No Default', 'Default'])
        axes[0,0].set_yticklabels(['No Default', 'Default'])
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC AUC = {test_results["metrics"]["roc_auc"]:.4f}')
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1,0].plot(recall, precision, label=f'PR AUC = {test_results["metrics"]["pr_auc"]:.4f}')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1,1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='No Default', density=True)
        axes[1,1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Default', density=True)
        axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Prediction Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(
            self.results_dir, 'plots',
            f"{model_name}_test_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Test plots saved: {plot_file}")
    
    def compare_models(self, model_names: List[str], X_test, y_test, 
                      save_comparison=True):
        """
        Compare multiple models on the same test dataset
        
        Parameters:
        -----------
        model_names : List[str]
            List of model names to compare
        X_test, y_test : test data
        save_comparison : bool
            Whether to save comparison results
            
        Returns:
        --------
        dict : Comparison results
        """
        print(f"🔍 Comparing {len(model_names)} models")
        
        comparison_results = {
            'comparison_date': datetime.now().isoformat(),
            'models': model_names,
            'test_data_shape': {
                'samples': len(X_test),
                'features': X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0])
            },
            'results': {}
        }
        
        all_predictions = {}
        
        for model_name in model_names:
            print(f"   🔬 Testing {model_name}...")
            
            try:
                # Test model
                results = self.test_model(model_name, X_test, y_test, 
                                        save_results=False, generate_plots=False)
                comparison_results['results'][model_name] = results['metrics']
                
                # Store predictions for ensemble analysis
                model_data = self.load_model(model_name)
                model = model_data['model']
                model_type = model_data['type']
                
                if model_type.lower() == 'tensorflow':
                    y_pred_proba = model.predict(X_test, verbose=0).flatten()
                elif hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = model.predict(X_test)
                
                all_predictions[model_name] = y_pred_proba
                
            except Exception as e:
                print(f"   ❌ Error testing {model_name}: {e}")
                comparison_results['results'][model_name] = {'error': str(e)}
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame(comparison_results['results']).T
        
        print(f"\n📊 Model Comparison Results:")
        print("=" * 80)
        if not metrics_df.empty:
            print(metrics_df.round(4).to_string())
            
            # Find best models
            if 'roc_auc' in metrics_df.columns:
                best_roc_auc = metrics_df['roc_auc'].idxmax()
                print(f"\n🏆 Best ROC AUC: {best_roc_auc} ({metrics_df.loc[best_roc_auc, 'roc_auc']:.4f})")
            
            if 'f1_score' in metrics_df.columns:
                best_f1 = metrics_df['f1_score'].idxmax()
                print(f"🏆 Best F1 Score: {best_f1} ({metrics_df.loc[best_f1, 'f1_score']:.4f})")
        
        # Generate comparison plots
        self._generate_comparison_plots(model_names, all_predictions, y_test, metrics_df)
        
        # Save comparison
        if save_comparison:
            comparison_file = os.path.join(
                self.results_dir,
                f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"\n💾 Comparison results saved: {comparison_file}")
        
        return comparison_results
    
    def _generate_comparison_plots(self, model_names, predictions, y_test, metrics_df):
        """Generate comparison visualization plots"""
        if len(model_names) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. ROC Curves Comparison
        axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        for model_name in model_names:
            if model_name in predictions:
                fpr, tpr, _ = roc_curve(y_test, predictions[model_name])
                roc_auc = metrics_df.loc[model_name, 'roc_auc'] if model_name in metrics_df.index else 0
                axes[0,0].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        axes[0,0].set_xlabel('False Positive Rate')
        axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('ROC Curves Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Metrics Comparison Bar Plot
        if not metrics_df.empty and len(metrics_df.columns) > 0:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
            
            if available_metrics:
                metrics_df[available_metrics].plot(kind='bar', ax=axes[0,1])
                axes[0,1].set_title('Metrics Comparison')
                axes[0,1].set_ylabel('Score')
                axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Prediction Distributions
        for i, model_name in enumerate(model_names[:4]):  # Limit to 4 models for clarity
            if model_name in predictions:
                axes[1,0].hist(predictions[model_name], bins=30, alpha=0.5, 
                              label=f'{model_name}', density=True)
        
        axes[1,0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Prediction Distributions')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Performance Radar Chart (if multiple metrics available)
        if not metrics_df.empty and len(metrics_df.columns) >= 3:
            available_metrics = [m for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] 
                               if m in metrics_df.columns]
            
            if len(available_metrics) >= 3:
                angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
                
                ax_radar = plt.subplot(2, 2, 4, projection='polar')
                
                for model_name in model_names[:3]:  # Limit to 3 models for clarity
                    if model_name in metrics_df.index:
                        values = metrics_df.loc[model_name, available_metrics].values
                        values = np.concatenate((values, [values[0]]))  # Complete the circle
                        ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_name)
                        ax_radar.fill(angles, values, alpha=0.1)
                
                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(available_metrics)
                ax_radar.set_ylim(0, 1)
                ax_radar.set_title('Performance Radar Chart')
                ax_radar.legend(bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save comparison plot
        plot_file = os.path.join(
            self.results_dir, 'plots',
            f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Comparison plots saved: {plot_file}")
    
    def list_models(self):
        """List all registered models"""
        if not self.model_registry:
            print("📋 No models registered")
            return {}
        
        print(f"📋 Registered Models ({len(self.model_registry)}):")
        print("=" * 80)
        
        for name, info in self.model_registry.items():
            print(f"🔸 {name}")
            print(f"   Type: {info['type']}")
            print(f"   Created: {info['created']}")
            print(f"   ROC AUC: {info['performance']['roc_auc']:.4f}")
            print(f"   Accuracy: {info['performance']['accuracy']:.4f}")
            print(f"   F1 Score: {info['performance']['f1_score']:.4f}")
            print()
        
        # Return the registry data for API use
        return self.model_registry
    
    def delete_model(self, model_name: str, confirm=True):
        """Delete a saved model"""
        if model_name not in self.model_registry:
            print(f"❌ Model '{model_name}' not found")
            return False
        
        if confirm:
            response = input(f"⚠️  Are you sure you want to delete model '{model_name}'? (y/N): ")
            if response.lower() != 'y':
                print("❌ Deletion cancelled")
                return False
        
        # Remove model directory
        model_dir = self.model_registry[model_name]['path']
        import shutil
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.model_registry[model_name]
        self._save_registry()
        
        print(f"✅ Model '{model_name}' deleted successfully")
        return True


# Example usage and testing functions
def example_usage():
    """Example of how to use the ModelManager"""
    print("📚 ModelManager Usage Example")
    print("=" * 50)
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # List existing models
    model_manager.list_models()
    
    print("\n💡 Usage Examples:")
    print("1. Save model after training:")
    print("   model_manager.save_model(model, 'my_model', X_test, y_test)")
    print()
    print("2. Load and test model:")
    print("   loaded_model = model_manager.load_model('my_model')")
    print("   results = model_manager.test_model('my_model', X_test, y_test)")
    print()
    print("3. Compare multiple models:")
    print("   comparison = model_manager.compare_models(['model1', 'model2'], X_test, y_test)")
    print()
    print("4. List all models:")
    print("   model_manager.list_models()")

if __name__ == "__main__":
    example_usage()