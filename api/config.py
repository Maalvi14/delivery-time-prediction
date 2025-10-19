"""
Configuration for the FastAPI application.
"""

import os
from pathlib import Path
from typing import Optional


class APIConfig:
    """Configuration for the FastAPI application."""
    
    def __init__(self):
        # Model configuration
        self.model_path: str = os.getenv(
            "MODEL_PATH", 
            self._find_first_model()
        )
        
        # Server configuration
        self.host: str = os.getenv("API_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("API_PORT", "8000"))
        self.debug: bool = os.getenv("API_DEBUG", "false").lower() == "true"
        
        # API configuration
        self.title: str = "Delivery Time Prediction API"
        self.description: str = "API for predicting food delivery times using machine learning"
        self.version: str = "1.0.0"
        
        # CORS configuration
        self.cors_origins: list = os.getenv(
            "CORS_ORIGINS", 
            "http://localhost:3000,http://localhost:8080"
        ).split(",")
        
        # Logging configuration
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def _find_first_model(self) -> str:
        """
        Find the first available .pkl model file in the models directory.
        
        Returns:
            Path to the first model found, or default ridge_regression.pkl if none found
        """
        models_dir = Path(__file__).parent.parent / "models"
        
        # Look for .pkl files in the models directory
        pkl_files = list(models_dir.glob("*.pkl"))
        
        if pkl_files:
            # Sort by name for consistent ordering, return the first one
            first_model = sorted(pkl_files)[0]
            return str(first_model)
        
        # Fallback to default if no models found
        return str(models_dir / "ridge_regression.pkl")
        
    def validate_model_path(self) -> bool:
        """Validate that the model path exists."""
        return Path(self.model_path).exists()
    
    def get_model_path(self) -> str:
        """Get the absolute model path."""
        return str(Path(self.model_path).resolve())
    
    def get_model_name(self) -> str:
        """Get the name of the discovered model file."""
        return Path(self.model_path).stem
    
    def get_available_models(self) -> list:
        """Get list of all available model files."""
        models_dir = Path(__file__).parent.parent / "models"
        pkl_files = list(models_dir.glob("*.pkl"))
        return [f.stem for f in sorted(pkl_files)]


# Global configuration instance
config = APIConfig()
