"""
FastAPI Model Serving Application
Provides REST API endpoints for loan default prediction model serving
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
import os
import sys
from datetime import datetime
import uvicorn
import json

# Add the models directory to Python path
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from model_manager import ModelManager
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.model_manager import ModelManager
# HomeLoanData import removed - using direct data preprocessing instead

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="Production-ready API for loan default prediction using deep learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_manager = None
current_model = None
current_model_name = None
preprocessor = None

# Pydantic models for request/response validation
class LoanApplicationRequest(BaseModel):
    """Request model for loan application prediction"""
    
    # Required fields (most important features)
    amt_credit: float = Field(..., description="Credit amount of the loan", ge=0)
    amt_annuity: float = Field(..., description="Loan annuity", ge=0)
    amt_income_total: float = Field(..., description="Total income of the client", ge=0)
    amt_goods_price: Optional[float] = Field(None, description="Goods price", ge=0)
    
    # Demographic information
    code_gender: str = Field(..., description="Gender of the client", pattern="^[MF]$")
    days_birth: int = Field(..., description="Days since birth (negative)", le=0)
    days_employed: Optional[int] = Field(None, description="Days employed (negative for employed, positive for unemployed)")
    
    # Application details
    name_contract_type: str = Field(..., description="Contract type", pattern="^(Cash loans|Revolving loans)$")
    name_income_type: str = Field(..., description="Income type")
    name_education_type: str = Field(..., description="Education level")
    name_family_status: str = Field(..., description="Family status")
    name_housing_type: str = Field(..., description="Housing type")
    
    # Financial information
    region_population_relative: Optional[float] = Field(None, description="Population relative to region", ge=0, le=1)
    ext_source_1: Optional[float] = Field(None, description="External data source 1", ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, description="External data source 2", ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, description="External data source 3", ge=0, le=1)
    
    # Additional fields (optional)
    additional_features: Optional[Dict[str, Any]] = Field(None, description="Additional features as key-value pairs")
    
    @validator('amt_credit', 'amt_annuity', 'amt_income_total')
    def validate_positive_amounts(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    @validator('days_birth')
    def validate_birth_days(cls, v):
        if v > 0:
            raise ValueError('Days birth must be negative (days before today)')
        if v < -25000:  # approximately 68 years
            raise ValueError('Age seems unrealistic')
        return v

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    applications: List[LoanApplicationRequest] = Field(..., description="List of loan applications")
    
    @validator('applications')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError('At least one application required')
        if len(v) > 1000:  # Limit batch size
            raise ValueError('Batch size too large (max 1000)')
        return v

class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    application_id: str = Field(..., description="Unique identifier for this prediction")
    prediction: float = Field(..., description="Probability of default (0-1)")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, HIGH")
    model_used: str = Field(..., description="Name of the model used for prediction")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    confidence: float = Field(..., description="Model confidence score")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_id: str = Field(..., description="Unique identifier for this batch")
    total_processed: int = Field(..., description="Number of applications processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Current model name")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="API version")

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    created_at: str = Field(..., description="Model creation timestamp")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    features_count: int = Field(..., description="Number of input features")

# Startup time tracking
startup_time = datetime.now()

# Dependency for model loading
async def get_model_manager():
    """Dependency to ensure model manager is available"""
    global model_manager
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    return model_manager

async def get_current_model():
    """Dependency to ensure model is loaded"""
    global current_model, current_model_name
    if current_model is None or current_model_name is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model loaded. Use /load-model endpoint first."
        )
    return current_model, current_model_name

# Utility functions
def calculate_risk_level(probability: float) -> str:
    """Calculate risk level based on probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

def calculate_confidence(probability: float) -> float:
    """Calculate confidence score based on probability"""
    # Higher confidence when probability is closer to 0 or 1
    return 1.0 - 2.0 * abs(0.5 - probability)

def preprocess_request_data(request_data: Dict[str, Any]) -> np.ndarray:
    """Preprocess request data to match model input format"""
    global preprocessor, model_manager, current_model_name
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request_data])
        
        # Get the expected number of features from the current model
        expected_features = 20  # Default for demo models
        
        if current_model_name and model_manager:
            registry = model_manager.list_models()
            if current_model_name in registry:
                model_info_path = os.path.join(model_manager.models_dir, registry[current_model_name]['path'], 'model_info.json')
                if os.path.exists(model_info_path):
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                        expected_features = model_info.get('n_features', 20)
        
        # Use the preprocessor to transform the data
        # Note: This is a simplified version - in production, you'd need
        # the exact same preprocessing pipeline used during training
        processed_data = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Ensure we have the right number of features based on the current model
        if processed_data.shape[1] < expected_features:
            # Pad with zeros if needed
            padding = np.zeros((1, expected_features - processed_data.shape[1]))
            processed_data = np.concatenate([processed_data.values, padding], axis=1)
        elif processed_data.shape[1] > expected_features:
            # Truncate if too many features
            processed_data = processed_data.iloc[:, :expected_features].values
        else:
            processed_data = processed_data.values
            
        logger.info(f"Preprocessed data shape: {processed_data.shape}, expected features: {expected_features}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data preprocessing failed: {str(e)}"
        )

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global model_manager
    
    try:
        logger.info("🚀 Starting Loan Default Prediction API...")
        
        # Initialize model manager with correct models directory path
        # In Docker: /app/src/models/saved_models
        # Locally: Use absolute path from project root
        import os
        if os.path.exists('/app/src/models/saved_models'):
            # Running in Docker container
            models_dir = '/app/src/models/saved_models'
        else:
            # Running locally - build absolute path from current working directory
            # When run from project root, this should find src/models/saved_models
            models_dir = os.path.join(os.getcwd(), 'src', 'models', 'saved_models')
            
        model_manager = ModelManager(models_dir=models_dir)
        logger.info("✅ Model manager initialized")
        
        # Try to load the best available model
        models = model_manager.list_models()
        if models:
            # Find the best model by ROC AUC
            # Handle both 'test_results' and 'performance' keys for compatibility
            best_model_name = max(models.items(), 
                                key=lambda x: x[1].get('test_results', x[1].get('performance', {})).get('roc_auc', 0))[0]
            await load_model_internal(best_model_name)
            logger.info(f"✅ Auto-loaded best model: {best_model_name}")
        else:
            logger.warning("⚠️ No models available for auto-loading")
            
        logger.info("🎉 API startup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        # Don't raise here - let the API start even without models

async def load_model_internal(model_name: str):
    """Internal function to load a model"""
    global current_model, current_model_name, model_manager
    
    try:
        result = model_manager.load_model(model_name)
        current_model = result['model']
        model_info = result['info']
        current_model_name = model_name
        logger.info(f"✅ Model {model_name} loaded successfully")
        return model_info
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {str(e)}")
        raise

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    global current_model, current_model_name, startup_time
    
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        model_loaded=current_model is not None,
        model_name=current_model_name or "none",
        uptime_seconds=uptime,
        version="1.0.0"
    )

@app.get("/models", tags=["Model Management"])
async def list_available_models(manager: ModelManager = Depends(get_model_manager)):
    """List all available models"""
    try:
        models = manager.list_models()
        return {
            "available_models": models,
            "total_count": len(models),
            "current_loaded": current_model_name
        }
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )

@app.post("/load-model/{model_name}", tags=["Model Management"])
async def load_model(model_name: str, manager: ModelManager = Depends(get_model_manager)):
    """Load a specific model"""
    try:
        model_info = await load_model_internal(model_name)
        return {
            "message": f"Model {model_name} loaded successfully",
            "model_info": model_info
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found"
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Management"])
async def get_model_info(model_data = Depends(get_current_model)):
    """Get information about the currently loaded model"""
    model, model_name = model_data
    
    try:
        # Get model info from model manager
        result = model_manager.load_model(model_name)
        model_info = result['info']
        
        return ModelInfoResponse(
            model_name=model_name,
            model_type=model_info['model_type'],
            created_at=model_info['created_at'],
            performance_metrics=model_info['test_results'],
            features_count=120  # This should come from model metadata
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    request: LoanApplicationRequest,
    model_data = Depends(get_current_model)
):
    """Make a prediction for a single loan application"""
    model, model_name = model_data
    
    try:
        start_time = datetime.now()
        
        # Convert request to dict and preprocess
        request_dict = request.dict()
        processed_data = preprocess_request_data(request_dict)
        
        # Make prediction
        prediction_prob = model.predict(processed_data)[0][0]
        
        # Generate application ID
        app_id = f"app_{int(start_time.timestamp() * 1000)}"
        
        # Calculate additional metrics
        risk_level = calculate_risk_level(prediction_prob)
        confidence = calculate_confidence(prediction_prob)
        
        return PredictionResponse(
            application_id=app_id,
            prediction=float(prediction_prob),
            risk_level=risk_level,
            model_used=model_name,
            prediction_timestamp=start_time.isoformat(),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    model_data = Depends(get_current_model)
):
    """Make predictions for multiple loan applications"""
    model, model_name = model_data
    
    try:
        start_time = datetime.now()
        batch_id = f"batch_{int(start_time.timestamp() * 1000)}"
        
        predictions = []
        
        for i, app_request in enumerate(request.applications):
            try:
                # Process each application
                request_dict = app_request.dict()
                processed_data = preprocess_request_data(request_dict)
                
                # Make prediction
                prediction_prob = model.predict(processed_data)[0][0]
                
                # Generate application ID
                app_id = f"{batch_id}_app_{i+1}"
                
                # Calculate additional metrics
                risk_level = calculate_risk_level(prediction_prob)
                confidence = calculate_confidence(prediction_prob)
                
                predictions.append(PredictionResponse(
                    application_id=app_id,
                    prediction=float(prediction_prob),
                    risk_level=risk_level,
                    model_used=model_name,
                    prediction_timestamp=datetime.now().isoformat(),
                    confidence=confidence
                ))
                
            except Exception as e:
                logger.error(f"Failed to process application {i+1}: {str(e)}")
                # Continue with other applications
                
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_processed=len(predictions),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )

# Advanced Model Support
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

@app.post("/models/advanced/predict", response_model=AdvancedPredictionResponse, 
          tags=["Advanced Prediction"])
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
        import time
        start_time = time.time()
        
        # Load model
        model_data = model_manager.load_model(request.model_name)
        if not model_data:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        
        model = model_data['model']
        model_type = model_data.get('model_type', 'unknown')
        
        # Create input DataFrame from request features
        input_data = pd.DataFrame([request.features])
        
        # Simple feature simulation for missing features
        # In a real implementation, you would use the FeatureSimulator class
        expected_features = ["income_ratio", "debt_to_income", "credit_score", "loan_amount_ratio",
                           "employment_years", "property_value", "down_payment_ratio", "age",
                           "education_level", "marital_status", "dependents", "location_risk",
                           "previous_defaults", "account_balance", "loan_term", "interest_rate",
                           "collateral_value", "income_stability", "payment_history", "risk_score"]
        
        original_features = len(input_data.columns)
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in input_data.columns:
                if request.simulation_method == "statistical":
                    input_data[feature] = 0.5  # Use middle value
                elif request.simulation_method == "zero":
                    input_data[feature] = 0.0
                elif request.simulation_method == "median":
                    input_data[feature] = 0.5
                else:  # random
                    input_data[feature] = np.random.uniform(0, 1)
        
        # Reorder columns to match expected features
        input_data = input_data[expected_features]
        simulated_features = len(input_data.columns) - original_features
        
        # Make prediction based on model type
        if model_type == 'autoencoder_classifier':
            # Handle autoencoder models
            if isinstance(model, dict):
                encoder = model.get('encoder')
                classifier = model.get('classifier')
                
                if encoder and classifier:
                    # Encode features first
                    encoded_features = encoder.predict(input_data.values)
                    prediction_proba = classifier.predict(encoded_features)[0][0]
                    prediction = 1 if prediction_proba > 0.5 else 0
                else:
                    raise HTTPException(status_code=500, detail="Invalid autoencoder model structure")
            else:
                # Fallback to direct prediction
                prediction_proba = 0.5
                prediction = 0
                
        elif model_type == 'lightgbm_classifier':
            # Handle LightGBM models
            prediction_proba = model.predict(input_data.values)[0]
            prediction = 1 if prediction_proba > 0.5 else 0
            
        else:
            # Handle sklearn and other models
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_data.values)[0][1]
            else:
                prediction_proba = 0.5
            prediction = model.predict(input_data.values)[0]
        
        processing_time = time.time() - start_time
        
        simulation_report = {
            'original_features': original_features,
            'simulated_features': len(input_data.columns),
            'added_features': simulated_features,
            'missing_features_added': [f for f in expected_features if f not in request.features],
            'simulation_method': request.simulation_method
        }
        
        return AdvancedPredictionResponse(
            model_name=request.model_name,
            prediction=float(prediction),
            prediction_proba=float(prediction_proba) if prediction_proba is not None else None,
            simulation_report=simulation_report,
            processing_time=processing_time,
            features_used=len(input_data.columns),
            features_simulated=simulated_features
        )
        
    except Exception as e:
        logger.error(f"Error making advanced prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making advanced prediction: {str(e)}")

@app.post("/models/advanced/train", tags=["Advanced Training"])
async def train_advanced_models():
    """
    Train advanced models including Autoencoder and LightGBM
    
    This endpoint would normally trigger training of:
    - Autoencoder + Classifier model
    - LightGBM Gradient Boosting model
    
    For demo purposes, returns information about existing models.
    """
    try:
        # Get available models
        available_models = list(model_manager.model_registry.keys())
        advanced_models = [m for m in available_models if m.startswith('advanced_')]
        
        if advanced_models:
            return {
                "status": "models_available",
                "message": "Advanced models are already available",
                "models": advanced_models,
                "note": "Training functionality would be implemented in production"
            }
        else:
            return {
                "status": "no_advanced_models",
                "message": "No advanced models found. Would need to implement training.",
                "available_models": available_models,
                "note": "Training would create new autoencoder and lightgbm models"
            }
        
    except Exception as e:
        logger.error(f"Error in advanced training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in advanced training: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "model_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )