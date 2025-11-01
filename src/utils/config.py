"""
Configuration Management

Centralized configuration for the entire application
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="numerics_db")
    user: str = Field(default="postgres")
    password: str = Field(default="password")


class ModelConfig(BaseModel):
    """Model configuration"""
    default_algorithm: str = Field(default="random_forest")
    models_path: str = Field(default="models/trained_models")
    scalers_path: str = Field(default="models/scalers")
    max_models: int = Field(default=10)


class AgentConfig(BaseModel):
    """AI Agent configuration"""
    model_id: str = Field(default="openai/gpt-4.1-mini")
    github_token: Optional[str] = Field(default=None)
    max_conversation_turns: int = Field(default=50)
    temperature: float = Field(default=0.7)


class APIConfig(BaseModel):
    """API configuration"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    api_key: str = Field(default="dev-api-key-123")
    cors_origins: list = Field(default=["http://localhost:3000"])
    max_upload_size: int = Field(default=100 * 1024 * 1024)  # 100MB


class Config(BaseModel):
    """Main application configuration"""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    
    # Paths
    data_path: str = Field(default="data")
    results_path: str = Field(default="results")
    logs_path: str = Field(default="logs")
    uploads_path: str = Field(default="uploads")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override with environment variables
        self._load_from_env()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.debug = os.getenv("DEBUG", str(self.debug)).lower() == "true"
        
        # Paths
        self.data_path = os.getenv("DATA_PATH", self.data_path)
        self.results_path = os.getenv("RESULTS_PATH", self.results_path)
        self.logs_path = os.getenv("LOGS_PATH", self.logs_path)
        
        # API
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", str(self.api.port)))
        self.api.api_key = os.getenv("API_KEY", self.api.api_key)
        
        # Agent
        self.agent.github_token = os.getenv("GITHUB_TOKEN", self.agent.github_token)
        self.agent.model_id = os.getenv("MODEL_ID", self.agent.model_id)
        
        # Models
        self.models.models_path = os.getenv("MODELS_PATH", self.models.models_path)
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_path,
            self.results_path,
            self.logs_path,
            self.uploads_path,
            self.models.models_path,
            self.models.scalers_path,
            os.path.join(self.results_path, "plots"),
            os.path.join(self.results_path, "evaluation"),
            os.path.join(self.results_path, "analysis")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.dict()
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global configuration instance
_config = None


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file or environment"""
    global _config
    
    if _config is None:
        if config_file and os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            _config = Config(**config_data)
        else:
            _config = Config()
    
    return _config


def get_config() -> Config:
    """Get the current configuration"""
    return load_config()


# Configuration for different environments
DEVELOPMENT_CONFIG = {
    "environment": "development",
    "debug": True,
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"]
    }
}

PRODUCTION_CONFIG = {
    "environment": "production", 
    "debug": False,
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": []  # Set in environment
    }
}

TESTING_CONFIG = {
    "environment": "testing",
    "debug": True,
    "data_path": "test_data",
    "results_path": "test_results",
    "models": {
        "models_path": "test_models"
    }
}