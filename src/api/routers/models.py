"""
Models Router

API endpoints for machine learning model operations including training, testing, and inference
"""

import os
import joblib
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel

from ...models.training import ModelTrainer, TrainingConfig
from ...models.testing import ModelTester
from ...models.evaluation import ModelEvaluator
from ...utils.feature_simulator import FeatureSimulator


router = APIRouter(prefix="/models", tags=["models"])


class TrainingRequest(BaseModel):
    """Request model for training a new model"""
    algorithm: str = "random_forest"
    task_type: str = "regression"
    test_size: float = 0.2
    hyperparameter_tuning: bool = True
    model_name: Optional[str] = None


class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    model_name: str
    features: List[float]


class TrainingResponse(BaseModel):
    """Response model for training results"""
    status: str
    model_name: str
    algorithm: str
    training_time: float
    metrics: Dict[str, Any]
    message: str


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_name: str
    prediction: float
    confidence: Optional[float]
    features_used: List[str]


@router.get("/", summary="List all trained models")
async def list_models():
    """Get a list of all trained models"""
    try:
        models_path = "models/trained_models"
        if not os.path.exists(models_path):
            return {"models": []}
        
        models = []
        for file in os.listdir(models_path):
            if file.endswith('.joblib'):
                model_name = file.replace('.joblib', '')
                model_path = os.path.join(models_path, file)
                stat = os.stat(model_path)
                
                models.append({
                    "name": model_name,
                    "file_size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.post("/train", response_model=TrainingResponse, summary="Train a new model")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(...)
):
    """Train a new machine learning model with uploaded data"""
    try:
        # Validate file type
        if not data_file.filename or not data_file.filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Generate model name if not provided
        model_name = request.model_name or f"{request.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create training configuration
        config = TrainingConfig(
            algorithm=request.algorithm,
            task_type=request.task_type,
            test_size=request.test_size,
            hyperparameter_tuning=request.hyperparameter_tuning
        )
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Save uploaded file temporarily
        temp_file_path = f"temp_{data_file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        try:
            # Train model in background
            background_tasks.add_task(
                _train_model_background,
                trainer,
                temp_file_path,
                model_name
            )
            
            return TrainingResponse(
                status="started",
                model_name=model_name,
                algorithm=request.algorithm,
                training_time=0.0,
                metrics={},
                message=f"Training started for model {model_name}"
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")


@router.post("/predict", response_model=PredictionResponse, summary="Make predictions")
async def predict(request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        model_path = f"models/trained_models/{request.model_name}.joblib"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        
        # Load model
        model = joblib.load(model_path)
        
        # Make prediction
        import numpy as np
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Get feature names if available
        feature_names = getattr(model, 'feature_names_in_', [f"feature_{i}" for i in range(len(request.features))])
        try:
            feature_list = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(len(request.features))]
        except:
            feature_list = [f"feature_{i}" for i in range(len(request.features))]
        
        return PredictionResponse(
            model_name=request.model_name,
            prediction=float(prediction),
            confidence=None,  # Could add confidence intervals for some models
            features_used=feature_list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@router.get("/{model_name}", summary="Get model details")
async def get_model_details(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model_path = f"models/trained_models/{model_name}.joblib"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Load model to get details
        model = joblib.load(model_path)
        
        # Get model metadata
        stat = os.stat(model_path)
        
        feature_names = getattr(model, 'feature_names_in_', [])
        try:
            feature_list = list(feature_names) if feature_names is not None else []
        except:
            feature_list = []
        
        details = {
            "name": model_name,
            "algorithm": type(model).__name__,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "feature_names": feature_list,
            "n_features": getattr(model, 'n_features_in_', 0)
        }
        
        return details
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model details: {str(e)}")


@router.delete("/{model_name}", summary="Delete a model")
async def delete_model(model_name: str):
    """Delete a trained model"""
    try:
        model_path = f"models/trained_models/{model_name}.joblib"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        os.remove(model_path)
        
        return {"message": f"Model {model_name} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


class AdvancedPredictionRequest(BaseModel):
    """Request model for advanced model predictions with missing feature handling"""
    model_name: str
    features: Dict[str, Any]  # Feature names and values
    simulation_method: str = "statistical"  # Method for handling missing features


class AdvancedPredictionResponse(BaseModel):
    """Response model for advanced predictions"""
    model_name: str
    prediction: float
    prediction_proba: Optional[float] = None
    simulation_report: Dict[str, Any]
    processing_time: float
    features_used: int
    features_simulated: int


@router.post("/advanced/predict", response_model=AdvancedPredictionResponse, 
            summary="Advanced prediction with missing feature simulation")
async def advanced_predict(request: AdvancedPredictionRequest):
    """
    Make predictions using advanced models with automatic missing feature simulation
    
    This endpoint can handle:
    - Autoencoder + Classifier models
    - LightGBM models  
    - Missing feature simulation using various methods
    - Comprehensive prediction reporting
    """
    try:
        import pandas as pd
        import numpy as np
        import time
        from ...utils.feature_simulator import create_feature_simulator_from_model_data
        from ...models.model_manager import ModelManager
        
        start_time = time.time()
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Load model
        model_data = model_manager.load_model(request.model_name)
        if not model_data:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        
        model = model_data['model']
        model_type = model_data.get('model_type', 'unknown')
        feature_names = model_data.get('feature_names', [])
        
        # Create input DataFrame from request features
        input_data = pd.DataFrame([request.features])
        
        # Create feature simulator
        model_path = os.path.join(model_manager.models_dir, request.model_name)
        feature_simulator = create_feature_simulator_from_model_data(model_path)
        
        # If no feature statistics available, create from feature names
        if not feature_simulator.feature_stats and feature_names:
            feature_simulator = FeatureSimulator(feature_names)
            # Set default statistics for basic simulation
            for feature in feature_names:
                if feature not in input_data.columns:
                    feature_simulator.feature_stats[feature] = {
                        'type': 'numeric',
                        'mean': 0.5,
                        'std': 0.2,
                        'median': 0.5,
                        'min': 0,
                        'max': 1,
                        'mode': 0.5
                    }
        
        # Simulate missing features
        original_features = len(input_data.columns)
        simulated_data = feature_simulator.simulate_missing_features(
            input_data, 
            method=request.simulation_method
        )
        
        # Generate simulation report
        simulation_report = feature_simulator.get_simulation_report(input_data, simulated_data)
        
        # Make prediction based on model type
        if model_type == 'autoencoder_classifier':
            # Handle autoencoder models
            encoder = model.get('encoder')
            classifier = model.get('classifier')
            
            if encoder and classifier:
                # Encode features first
                encoded_features = encoder.predict(simulated_data.values)
                prediction_proba = classifier.predict(encoded_features)[0][0]
                prediction = 1 if prediction_proba > 0.5 else 0
            else:
                raise HTTPException(status_code=500, detail="Invalid autoencoder model structure")
                
        elif model_type == 'lightgbm_classifier':
            # Handle LightGBM models
            prediction_proba = model.predict(simulated_data.values)[0]
            prediction = 1 if prediction_proba > 0.5 else 0
            
        else:
            # Handle sklearn and other models
            prediction_proba = model.predict_proba(simulated_data.values)[0][1]
            prediction = model.predict(simulated_data.values)[0]
        
        processing_time = time.time() - start_time
        
        return AdvancedPredictionResponse(
            model_name=request.model_name,
            prediction=float(prediction),
            prediction_proba=float(prediction_proba) if prediction_proba is not None else None,
            simulation_report=simulation_report,
            processing_time=processing_time,
            features_used=len(simulated_data.columns),
            features_simulated=simulation_report.get('added_features', 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making advanced prediction: {str(e)}")


@router.post("/advanced/train", summary="Train advanced models (Autoencoder + LightGBM)")
async def train_advanced_models(background_tasks: BackgroundTasks):
    """
    Train advanced models including Autoencoder and LightGBM
    
    This endpoint triggers training of:
    - Autoencoder + Classifier model
    - LightGBM Gradient Boosting model
    """
    try:
        from ...models.fast_deep_learning import train_advanced_models
        from ...models.model_manager import ModelManager
        
        # Add training to background tasks
        background_tasks.add_task(_train_advanced_models_background)
        
        return {
            "status": "training_started",
            "message": "Advanced model training started in background",
            "models": ["autoencoder_classifier", "lightgbm_classifier"],
            "estimated_time": "5-10 minutes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting advanced training: {str(e)}")


async def _train_advanced_models_background():
    """Background task for advanced model training"""
    try:
        from ...models.fast_deep_learning import train_advanced_models, load_and_prepare_data
        from ...models.model_manager import ModelManager
        
        print("🚀 Starting advanced model training...")
        
        # Load and prepare data using the function from fast_deep_learning
        X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
        
        # Train models
        results = train_advanced_models()
        
        # Save models using ModelManager
        model_manager = ModelManager()
        
        for model_name, result in results.items():
            if 'error' not in result:
                # Prepare metadata
                metadata = {
                    'roc_auc': result['auc_score'],
                    'pr_auc': result['pr_auc'],
                    'training_time': result['training_time'],
                    'feature_names': feature_names
                }
                
                model_manager.save_model(
                    model=result['model'],
                    model_name=f"advanced_{model_name}",
                    X_test=X_test,
                    y_test=y_test,
                    model_type=model_name,
                    metadata=metadata
                )
                print(f"✅ Saved {model_name} with AUC: {result['auc_score']:.4f}")
            else:
                print(f"❌ Failed to train {model_name}: {result['error']}")
                
        print("🎉 Advanced model training completed!")
        
    except Exception as e:
        print(f"❌ Error in background training: {str(e)}")


async def _train_model_background(trainer: ModelTrainer, data_file_path: str, model_name: str):
    """Background task for model training"""
    try:
        import pandas as pd
        
        # Load data
        if data_file_path.endswith('.csv'):
            data = pd.read_csv(data_file_path)
        else:
            data = pd.read_excel(data_file_path)
        
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Train model
        trainer.train(X, y)
        
        # Save model
        trainer.save_model(model_name)
        
        print(f"Model {model_name} trained successfully")
        
    except Exception as e:
        print(f"Error training model {model_name}: {str(e)}")