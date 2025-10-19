"""
Predictor service for handling raw and processed input predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import logging
from pathlib import Path

from model_pipeline import DeliveryTimePredictor
from .models import OrderInputRaw


class PredictorService:
    """
    Service class to wrap DeliveryTimePredictor with API-specific functionality.
    
    This class handles:
    - Loading the predictor with preprocessing pipeline
    - Converting raw input to processed features
    - Making predictions
    - Error handling
    """
    
    def __init__(self, model_path: str, logger: logging.Logger = None):
        """
        Initialize the predictor service.
        
        Args:
            model_path: Path to the saved model
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = model_path
        self.predictor = None
        self.is_loaded = False
        
        self._load_predictor()
    
    def _load_predictor(self):
        """Load the predictor and validate it."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.predictor = DeliveryTimePredictor(model_path=self.model_path)
            self.is_loaded = True
            
            self.logger.info(f"Predictor service initialized with model: {self.predictor.model_name}")
            
            # Check if preprocessing pipeline is available
            if self.predictor.preprocessor is None:
                self.logger.warning("Preprocessing pipeline not available - only processed input will work")
            
        except Exception as e:
            self.logger.error(f"Failed to load predictor: {str(e)}")
            self.is_loaded = False
            raise
    
    def is_preprocessing_available(self) -> bool:
        """Check if preprocessing pipeline is available."""
        return (
            self.is_loaded and 
            self.predictor is not None and 
            self.predictor.preprocessor is not None
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded")
        
        return {
            "model_name": self.predictor.model_name,
            "feature_names": self.predictor.feature_names,
            "num_features": len(self.predictor.feature_names) if self.predictor.feature_names else 0,
            "model_type": type(self.predictor.model).__name__,
            "is_preprocessing_available": self.is_preprocessing_available()
        }
    
    def _raw_input_to_dataframe(self, order: OrderInputRaw) -> pd.DataFrame:
        """Convert raw order input to DataFrame."""
        order_dict = order.dict()
        
        # Remove None values
        order_dict = {k: v for k, v in order_dict.items() if v is not None}
        
        return pd.DataFrame([order_dict])
    
    def predict(self, order: OrderInputRaw) -> float:
        """
        Make prediction from raw order input.
        
        Args:
            order: Raw order input
            
        Returns:
            Predicted delivery time in minutes
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded")
        
        if not self.is_preprocessing_available():
            raise RuntimeError("Preprocessing pipeline not available")
        
        try:
            # Convert to DataFrame
            df = self._raw_input_to_dataframe(order)
            
            # Make prediction using the predictor's pipeline
            prediction = self.predictor.predict(df)[0]
            
            self.logger.info(f"Prediction made: {prediction:.2f} minutes")
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, orders: List[OrderInputRaw]) -> List[float]:
        """
        Make batch predictions from raw order inputs.
        
        Args:
            orders: List of raw order inputs
            
        Returns:
            List of predicted delivery times
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded")
        
        if not self.is_preprocessing_available():
            raise RuntimeError("Preprocessing pipeline not available")
        
        try:
            # Convert all orders to DataFrame
            order_dicts = []
            for order in orders:
                order_dict = order.dict()
                order_dict = {k: v for k, v in order_dict.items() if v is not None}
                order_dicts.append(order_dict)
            
            df = pd.DataFrame(order_dicts)
            
            # Make batch predictions
            predictions = self.predictor.predict(df)
            
            self.logger.info(f"Batch predictions made: {len(predictions)} orders")
            return [float(p) for p in predictions]
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
