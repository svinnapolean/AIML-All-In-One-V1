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