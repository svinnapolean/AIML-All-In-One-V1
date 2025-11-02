"""
HomeLoanData Class for Comprehensive Loan Default Analysis

This class provides a complete pipeline for loan default prediction including:
1. Data loading and exploration
2. Null value analysis
3. Default rate analysis
4. Data balancing
5. Visualization
6. Encoding
7. Sensitivity calculation
8. ROC AUC calculation

Usage:
    loan_data = HomeLoanData('loan_data/loan_data.csv')
    loan_data.analyze_complete_pipeline()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class HomeLoanData:
    """
    Comprehensive class for Home Loan Default Prediction Analysis
    
    This class handles the complete machine learning pipeline from data loading
    to model evaluation with specific focus on loan default prediction.
    """
    
    def __init__(self, data_path='loan_data/loan_data.csv'):
        """
        Initialize the HomeLoanData class
        
        Parameters:
        data_path (str): Path to the loan dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_balanced = None
        self.y_balanced = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.is_balanced = False
        
        print("🏠 HomeLoanData class initialized")
        print("=" * 50)
    
    def load_dataset(self):
        """
        Task 1: Load the dataset
        """
        print("📊 TASK 1: Loading Dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {self.df.shape[1]}")
            print(f"   Rows: {self.df.shape[0]:,}")
            
            # Display basic info
            print(f"\n📋 Dataset Overview:")
            print(f"   First few columns: {list(self.df.columns[:5])}")
            print(f"   Target column: {'TARGET' if 'TARGET' in self.df.columns else 'Not found'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def check_null_values(self):
        """
        Task 2: Check for null values in the dataset
        """
        print("\n🔍 TASK 2: Checking for Null Values...")
        
        if self.df is None:
            print("❌ Dataset not loaded. Please run load_dataset() first.")
            return
        
        # Calculate null values
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100
        
        # Create null value summary
        null_summary = pd.DataFrame({
            'Column': null_counts.index,
            'Null_Count': null_counts.values,
            'Null_Percentage': null_percentages.values
        })
        
        # Filter columns with null values
        columns_with_nulls = null_summary[null_summary['Null_Count'] > 0].sort_values('Null_Percentage', ascending=False)
        
        print(f"✅ Null Value Analysis Complete!")
        print(f"   Total columns: {len(self.df.columns)}")
        print(f"   Columns with nulls: {len(columns_with_nulls)}")
        print(f"   Columns without nulls: {len(self.df.columns) - len(columns_with_nulls)}")
        
        if len(columns_with_nulls) > 0:
            print(f"\n📈 Top 10 Columns with Highest Null Percentages:")
            print(columns_with_nulls.head(10).to_string(index=False))
            
            # Summary statistics
            print(f"\n📊 Null Value Statistics:")
            print(f"   Maximum null percentage: {null_percentages.max():.2f}%")
            print(f"   Average null percentage: {null_percentages[null_percentages > 0].mean():.2f}%")
            print(f"   Median null percentage: {null_percentages[null_percentages > 0].median():.2f}%")
        else:
            print("✨ No null values found in the dataset!")
        
        return columns_with_nulls
    
    def analyze_target_distribution(self):
        """
        Task 3: Print the percentage of default to payer for TARGET column
        """
        print("\n🎯 TASK 3: Analyzing Target Distribution...")
        
        if self.df is None:
            print("❌ Dataset not loaded. Please run load_dataset() first.")
            return
        
        if 'TARGET' not in self.df.columns:
            print("❌ TARGET column not found in dataset.")
            return
        
        # Calculate target distribution
        target_counts = self.df['TARGET'].value_counts().sort_index()
        target_percentages = (target_counts / len(self.df)) * 100
        
        print(f"✅ Target Distribution Analysis:")
        print(f"   Total records: {len(self.df):,}")
        
        for class_val, count in target_counts.items():
            percentage = target_percentages[class_val]
            label = "Non-Default (Payers)" if class_val == 0 else "Default"
            print(f"   Class {class_val} ({label}): {count:,} ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        minority_class = target_counts.min()
        majority_class = target_counts.max()
        imbalance_ratio = majority_class / minority_class
        
        print(f"\n📊 Imbalance Analysis:")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"   Minority class size: {minority_class:,}")
        print(f"   Majority class size: {majority_class:,}")
        
        if imbalance_ratio > 2:
            print(f"   ⚠️  Dataset is IMBALANCED (ratio > 2:1)")
        else:
            print(f"   ✅ Dataset is relatively balanced")
        
        return target_counts, target_percentages
    
    def balance_dataset(self, method='smote', sampling_strategy='auto'):
        """
        Task 4: Balance the dataset if data is imbalanced
        
        Parameters:
        method (str): 'smote', 'undersample', 'smote_tomek'
        sampling_strategy (str): 'auto', 'minority', or float
        """
        print(f"\n⚖️ TASK 4: Balancing Dataset using {method.upper()}...")
        
        if self.df is None:
            print("❌ Dataset not loaded. Please run load_dataset() first.")
            return
        
        # Prepare features and target
        X = self.df.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
        y = self.df['TARGET']
        
        # Handle missing values quickly for balancing
        X_numeric = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(include=['object'])
        
        # Fill missing values
        for col in X_numeric.columns:
            X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())
        
        for col in X_categorical.columns:
            X_categorical[col] = X_categorical[col].fillna('Unknown')
            # Simple label encoding for balancing
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(X_categorical[col])
        
        # Combine features
        X_processed = pd.concat([X_numeric, X_categorical], axis=1)
        
        # Original distribution
        original_counts = y.value_counts().sort_index()
        print(f"📊 Original Distribution:")
        for class_val, count in original_counts.items():
            print(f"   Class {class_val}: {count:,}")
        
        # Apply balancing technique
        try:
            if method == 'smote':
                sampler = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
                self.X_balanced, self.y_balanced = sampler.fit_resample(X_processed, y)
                
            elif method == 'undersample':
                sampler = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
                self.X_balanced, self.y_balanced = sampler.fit_resample(X_processed, y)
                
            elif method == 'smote_tomek':
                sampler = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy)
                self.X_balanced, self.y_balanced = sampler.fit_resample(X_processed, y)
                
            else:
                print(f"❌ Unknown balancing method: {method}")
                return
            
            # New distribution
            new_counts = pd.Series(self.y_balanced).value_counts().sort_index()
            print(f"\n✅ Balanced Distribution ({method.upper()}):")
            for class_val, count in new_counts.items():
                print(f"   Class {class_val}: {count:,}")
            
            # Calculate change
            print(f"\n📈 Balancing Results:")
            for class_val in original_counts.index:
                original = original_counts[class_val]
                new = new_counts[class_val]
                change = ((new - original) / original) * 100
                print(f"   Class {class_val}: {original:,} → {new:,} ({change:+.1f}%)")
            
            self.is_balanced = True
            print(f"✅ Dataset successfully balanced using {method.upper()}!")
            
        except Exception as e:
            print(f"❌ Error during balancing: {e}")
            self.is_balanced = False
    
    def plot_data_distribution(self, figsize=(15, 6)):
        """
        Task 5: Plot the balanced or imbalanced data
        """
        print(f"\n📊 TASK 5: Plotting Data Distribution...")
        
        if self.df is None:
            print("❌ Dataset not loaded. Please run load_dataset() first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original distribution
        original_counts = self.df['TARGET'].value_counts().sort_index()
        
        # Plot 1: Original Distribution
        axes[0].bar(original_counts.index, original_counts.values, 
                   color=['skyblue', 'salmon'], alpha=0.7)
        axes[0].set_title('Original Data Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Target Class')
        axes[0].set_ylabel('Count')
        
        # Add value labels on bars
        for i, v in enumerate(original_counts.values):
            axes[0].text(i, v + 1000, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        total = len(self.df)
        for i, v in enumerate(original_counts.values):
            pct = (v / total) * 100
            axes[0].text(i, v/2, f'{pct:.1f}%', ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='white')
        
        # Plot 2: Balanced Distribution (if available)
        if self.is_balanced and self.y_balanced is not None:
            balanced_counts = pd.Series(self.y_balanced).value_counts().sort_index()
            
            axes[1].bar(balanced_counts.index, balanced_counts.values, 
                       color=['lightgreen', 'lightcoral'], alpha=0.7)
            axes[1].set_title('Balanced Data Distribution', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Target Class')
            axes[1].set_ylabel('Count')
            
            # Add value labels
            for i, v in enumerate(balanced_counts.values):
                axes[1].text(i, v + 1000, f'{v:,}', ha='center', va='bottom', fontweight='bold')
            
            # Add percentage labels
            total_balanced = len(self.y_balanced)
            for i, v in enumerate(balanced_counts.values):
                pct = (v / total_balanced) * 100
                axes[1].text(i, v/2, f'{pct:.1f}%', ha='center', va='center', 
                            fontsize=12, fontweight='bold', color='white')
        else:
            axes[1].text(0.5, 0.5, 'No Balanced Data\nRun balance_dataset()', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1].set_title('Balanced Data Distribution', fontsize=14, fontweight='bold')
        
        # Plot 3: Comparison
        if self.is_balanced and self.y_balanced is not None:
            classes = ['Non-Default (0)', 'Default (1)']
            original_values = original_counts.values
            balanced_values = balanced_counts.values
            
            x = np.arange(len(classes))
            width = 0.35
            
            axes[2].bar(x - width/2, original_values, width, label='Original', 
                       color='skyblue', alpha=0.7)
            axes[2].bar(x + width/2, balanced_values, width, label='Balanced', 
                       color='lightgreen', alpha=0.7)
            
            axes[2].set_title('Original vs Balanced Comparison', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Target Class')
            axes[2].set_ylabel('Count')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(classes)
            axes[2].legend()
            
            # Add value labels
            for i, (orig, bal) in enumerate(zip(original_values, balanced_values)):
                axes[2].text(i - width/2, orig + 1000, f'{orig:,}', 
                           ha='center', va='bottom', fontsize=10)
                axes[2].text(i + width/2, bal + 1000, f'{bal:,}', 
                           ha='center', va='bottom', fontsize=10)
        else:
            axes[2].text(0.5, 0.5, 'Balance data first\nto see comparison', 
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[2].set_title('Original vs Balanced Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Data distribution plots generated successfully!")
    
    def encode_columns(self):
        """
        Task 6: Encode the columns that are required for the model
        """
        print(f"\n🔧 TASK 6: Encoding Columns for Model...")
        
        if self.df is None:
            print("❌ Dataset not loaded. Please run load_dataset() first.")
            return
        
        # Use balanced data if available, otherwise original data
        if self.is_balanced and self.X_balanced is not None:
            print("📊 Using balanced dataset for encoding...")
            X = pd.DataFrame(self.X_balanced)
            y = pd.Series(self.y_balanced)
            data_source = "balanced"
        else:
            print("📊 Using original dataset for encoding...")
            X = self.df.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
            y = self.df['TARGET']
            data_source = "original"
        
        print(f"   Data source: {data_source}")
        print(f"   Shape: {X.shape}")
        
        # Separate numeric and categorical columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"   Numeric columns: {len(numeric_columns)}")
        print(f"   Categorical columns: {len(categorical_columns)}")
        
        # Handle missing values
        print("\n🔍 Handling missing values...")
        
        # Numeric columns - fill with median
        for col in numeric_columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"   ✅ Filled {col} nulls with median: {median_val:.2f}")
        
        # Categorical columns - fill with 'Unknown' and encode
        for col in categorical_columns:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna('Unknown')
                print(f"   ✅ Filled {col} nulls with 'Unknown'")
            
            # Label encode categorical columns
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            print(f"   ✅ Encoded {col} ({len(le.classes_)} categories)")
        
        # Scale numeric features
        print(f"\n⚖️ Scaling numeric features...")
        if len(numeric_columns) > 0:
            X[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])
            print(f"   ✅ Scaled {len(numeric_columns)} numeric columns")
        
        # Store processed data
        self.X = X
        self.y = y
        
        print(f"\n✅ Encoding completed successfully!")
        print(f"   Final shape: {self.X.shape}")
        print(f"   All columns are now numeric and ready for modeling")
        
        return X, y
    
    def train_model_and_calculate_metrics(self, test_size=0.2, random_state=42):
        """
        Tasks 7 & 8: Calculate sensitivity and ROC AUC
        """
        print(f"\n🤖 TASKS 7 & 8: Training Model and Calculating Metrics...")
        
        if self.X is None or self.y is None:
            print("❌ Data not encoded. Please run encode_columns() first.")
            return
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"📊 Data split completed:")
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        
        # Train a Random Forest model
        print(f"\n🌲 Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        print(f"✅ Model training completed!")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Task 7: Calculate Sensitivity (Recall/True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Task 8: Calculate ROC AUC
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f"\n🎯 TASK 7 - SENSITIVITY ANALYSIS:")
        print(f"   Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"   This means the model correctly identifies {sensitivity*100:.2f}% of actual defaults")
        
        print(f"\n📈 TASK 8 - ROC AUC ANALYSIS:")
        print(f"   ROC AUC Score: {roc_auc:.4f}")
        if roc_auc >= 0.9:
            print(f"   ⭐ Excellent model performance!")
        elif roc_auc >= 0.8:
            print(f"   ✅ Good model performance!")
        elif roc_auc >= 0.7:
            print(f"   👍 Fair model performance!")
        else:
            print(f"   ⚠️ Model needs improvement")
        
        print(f"\n📊 COMPLETE METRICS SUMMARY:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"   Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"   F1-Score: {f1_score:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"   True Negatives (TN): {tn:,}")
        print(f"   False Positives (FP): {fp:,}")
        print(f"   False Negatives (FN): {fn:,}")
        print(f"   True Positives (TP): {tp:,}")
        
        # Plot ROC Curve
        self.plot_roc_curve(y_pred_proba)
        
        return {
            'sensitivity': sensitivity,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'specificity': specificity,
            'f1_score': f1_score,
            'confusion_matrix': cm
        }
    
    def plot_roc_curve(self, y_pred_proba):
        """
        Plot ROC curve for visualization
        """
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("✅ ROC curve plotted successfully!")
    
    def analyze_complete_pipeline(self):
        """
        Run the complete analysis pipeline for all 8 tasks
        """
        print("🚀 STARTING COMPLETE HOME LOAN DATA ANALYSIS PIPELINE")
        print("=" * 60)
        
        # Task 1: Load dataset
        if not self.load_dataset():
            return
        
        # Task 2: Check null values
        null_analysis = self.check_null_values()
        
        # Task 3: Analyze target distribution
        target_analysis = self.analyze_target_distribution()
        
        # Task 4: Balance dataset
        self.balance_dataset(method='smote')
        
        # Task 5: Plot data distribution
        self.plot_data_distribution()
        
        # Task 6: Encode columns
        self.encode_columns()
        
        # Tasks 7 & 8: Calculate sensitivity and ROC AUC
        metrics = self.train_model_and_calculate_metrics()
        
        print(f"\n🎉 COMPLETE PIPELINE ANALYSIS FINISHED!")
        print("=" * 60)
        print("✅ All 8 tasks completed successfully!")
        
        return metrics


# Example usage and demonstration
if __name__ == "__main__":
    print("🏠 HomeLoanData Class Demonstration")
    print("=" * 50)
    
    # Initialize the class
    loan_analyzer = HomeLoanData('loan_data/loan_data.csv')
    
    # Run complete pipeline
    results = loan_analyzer.analyze_complete_pipeline()
    
    print(f"\n📋 Final Results Summary:")
    if results:
        print(f"   ROC AUC: {results['roc_auc']:.4f}")
        print(f"   Sensitivity: {results['sensitivity']:.4f}")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   F1-Score: {results['f1_score']:.4f}")