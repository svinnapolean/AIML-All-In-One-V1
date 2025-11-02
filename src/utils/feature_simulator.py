"""
Feature Simulator for handling missing features in test data
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureSimulator:
    """Simulates missing features for robust model testing"""
    
    def __init__(self, reference_features: List[str]):
        """
        Initialize with reference feature names from training data
        
        Args:
            reference_features: List of feature names from the training dataset
        """
        self.reference_features = reference_features
        self.feature_stats = {}
        
    def analyze_reference_data(self, reference_data: pd.DataFrame):
        """
        Analyze reference data to understand feature distributions
        
        Args:
            reference_data: DataFrame with training data to analyze
        """
        logger.info("📊 Analyzing reference data for feature simulation...")
        
        for feature in self.reference_features:
            if feature in reference_data.columns:
                col_data = reference_data[feature]
                
                # Calculate statistics based on data type
                if pd.api.types.is_numeric_dtype(col_data):
                    self.feature_stats[feature] = {
                        'type': 'numeric',
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'median': col_data.median(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'mode': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.median()
                    }
                else:
                    # Categorical data
                    value_counts = col_data.value_counts()
                    self.feature_stats[feature] = {
                        'type': 'categorical',
                        'mode': value_counts.index[0] if len(value_counts) > 0 else 0,
                        'value_counts': value_counts.to_dict(),
                        'unique_values': col_data.unique().tolist()
                    }
        
        logger.info(f"✅ Analyzed {len(self.feature_stats)} features for simulation")
    
    def simulate_missing_features(self, test_data: pd.DataFrame, 
                                method: str = 'statistical') -> pd.DataFrame:
        """
        Add missing features to test data using various simulation methods
        
        Args:
            test_data: DataFrame with potentially missing features
            method: Simulation method ('statistical', 'zero', 'median', 'random')
            
        Returns:
            DataFrame with all required features
        """
        result_data = test_data.copy()
        missing_features = []
        
        # Identify missing features
        for feature in self.reference_features:
            if feature not in result_data.columns:
                missing_features.append(feature)
        
        if not missing_features:
            logger.info("✅ No missing features found")
            return result_data[self.reference_features]  # Ensure correct order
        
        logger.info(f"🔧 Simulating {len(missing_features)} missing features using '{method}' method")
        
        for feature in missing_features:
            if feature in self.feature_stats:
                stats = self.feature_stats[feature]
                
                if method == 'statistical':
                    # Use statistical properties
                    if stats['type'] == 'numeric':
                        # Use normal distribution around mean
                        simulated_values = np.random.normal(
                            stats['mean'], 
                            stats['std'], 
                            len(result_data)
                        )
                        # Clip to reasonable bounds
                        simulated_values = np.clip(
                            simulated_values, 
                            stats['min'], 
                            stats['max']
                        )
                    else:
                        # Sample from categorical distribution
                        values = list(stats['value_counts'].keys())
                        probabilities = list(stats['value_counts'].values())
                        probabilities = np.array(probabilities) / sum(probabilities)
                        simulated_values = np.random.choice(
                            values, 
                            size=len(result_data), 
                            p=probabilities
                        )
                
                elif method == 'median':
                    # Use median/mode values
                    if stats['type'] == 'numeric':
                        simulated_values = [stats['median']] * len(result_data)
                    else:
                        simulated_values = [stats['mode']] * len(result_data)
                
                elif method == 'zero':
                    # Use zero/default values
                    simulated_values = [0] * len(result_data)
                
                elif method == 'random':
                    # Use random values within range
                    if stats['type'] == 'numeric':
                        simulated_values = np.random.uniform(
                            stats['min'], 
                            stats['max'], 
                            len(result_data)
                        )
                    else:
                        simulated_values = np.random.choice(
                            stats['unique_values'], 
                            size=len(result_data)
                        )
                
                result_data[feature] = simulated_values
                
            else:
                # No statistics available, use default approach
                logger.warning(f"⚠️ No statistics for {feature}, using zeros")
                result_data[feature] = 0
        
        # Ensure column order matches reference
        result_data = result_data[self.reference_features]
        
        logger.info(f"✅ Successfully simulated missing features: {missing_features}")
        return result_data
    
    def get_simulation_report(self, original_data: pd.DataFrame, 
                            simulated_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a report comparing original and simulated data
        
        Args:
            original_data: Original test data
            simulated_data: Data after feature simulation
            
        Returns:
            Dictionary with simulation statistics
        """
        report = {
            'original_features': len(original_data.columns),
            'simulated_features': len(simulated_data.columns),
            'added_features': len(simulated_data.columns) - len(original_data.columns),
            'missing_features_added': [],
            'simulation_summary': {}
        }
        
        # Identify what was added
        for feature in self.reference_features:
            if feature not in original_data.columns:
                report['missing_features_added'].append(feature)
                
                if feature in simulated_data.columns:
                    col_data = simulated_data[feature]
                    report['simulation_summary'][feature] = {
                        'mean': col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else None,
                        'unique_values': len(col_data.unique()),
                        'null_count': col_data.isnull().sum()
                    }
        
        return report


def create_feature_simulator_from_model_data(model_path: str) -> FeatureSimulator:
    """
    Create a FeatureSimulator from saved model metadata
    
    Args:
        model_path: Path to model directory containing metadata
        
    Returns:
        Configured FeatureSimulator instance
    """
    try:
        import json
        import os
        
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            feature_names = metadata.get('feature_names', [])
            simulator = FeatureSimulator(feature_names)
            
            # If training data statistics are saved, load them
            stats_path = os.path.join(model_path, 'feature_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    simulator.feature_stats = json.load(f)
                    
            return simulator
        else:
            logger.warning(f"No metadata found at {metadata_path}")
            return FeatureSimulator([])
            
    except Exception as e:
        logger.error(f"Error creating feature simulator: {str(e)}")
        return FeatureSimulator([])