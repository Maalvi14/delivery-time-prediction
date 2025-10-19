"""
Prediction and inference module.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any
import logging
import joblib
from pathlib import Path

from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer, FeatureSelector
from .config import Config


class DeliveryTimePredictor:
    """
    End-to-end predictor for delivery times.
    
    This class:
    - Loads trained model and preprocessing artifacts
    - Applies full preprocessing and feature engineering pipeline
    - Makes predictions on new data
    """
    
    def __init__(
        self,
        model_path: str = None,
        config: Config = None,
        logger: logging.Logger = None
    ):
        """
        Initialize the DeliveryTimePredictor.
        
        Args:
            model_path: Path to saved model
            config: Configuration object
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_name = None
        
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_selector = None
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """
        Load the trained model and artifacts.
        
        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model data
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_name = model_data['model_name']
        
        # Load preprocessing components if available
        self.preprocessor = model_data.get('preprocessor')
        self.feature_engineer = model_data.get('feature_engineer')
        self.feature_selector = model_data.get('feature_selector')
        
        self.logger.info(f"Loaded model: {self.model_name} from {model_path}")
        if self.preprocessor is not None:
            self.logger.info("Preprocessing pipeline components loaded")
    
    def fit_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = 'Delivery_Time_min'
    ):
        """
        Fit the preprocessing and feature engineering pipeline.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column
        """
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
        else:
            X = df.copy()
        
        # Fit preprocessor
        self.preprocessor = DataPreprocessor(self.config, self.logger)
        self.preprocessor.fit(df)
        
        # Fit feature engineer
        self.feature_engineer = FeatureEngineer(self.config, self.logger)
        self.feature_engineer.fit(X)
        
        # Fit feature selector
        self.feature_selector = FeatureSelector(self.config, self.logger)
        
        self.logger.info("Pipeline fitted successfully")
    
    def transform_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = 'Delivery_Time_min'
    ) -> pd.DataFrame:
        """
        Transform data through the full pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Transformed DataFrame
        """
        if self.preprocessor is None or self.feature_engineer is None:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # Store target if present
        if target_col in df.columns:
            target = df[target_col].copy()
            df_processed = df.drop(columns=[target_col])
        else:
            target = None
            df_processed = df.copy()
        
        # Apply preprocessing
        df_processed = self.preprocessor.transform(df_processed)
        
        # Apply feature engineering
        df_processed = self.feature_engineer.transform(df_processed)
        
        # Drop Order_ID if present
        if 'Order_ID' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Order_ID'])
        
        # Encode categorical variables
        df_processed = self.feature_selector.encode_categorical(df_processed)
        
        # Add target back if it was present
        if target is not None:
            df_processed[target_col] = target.values
        
        return df_processed
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions on new data.
        
        Args:
            data: Input data (DataFrame or dict)
            return_dataframe: If True, return DataFrame with predictions
            
        Returns:
            Predictions as array or DataFrame
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure we have a copy
        df = data.copy()
        
        # Apply pipeline transformations if available
        if self.preprocessor is not None and self.feature_engineer is not None:
            df_processed = self.transform_pipeline(df, target_col=None)
        else:
            df_processed = df.copy()
        
        # Ensure feature order matches training
        if self.feature_names:
            # Add missing columns with 0
            for col in self.feature_names:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            
            # Select and order columns
            df_processed = df_processed[self.feature_names]
        
        # Scale features
        X = self.scaler.transform(df_processed)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        if return_dataframe:
            result = data.copy()
            result['Predicted_Delivery_Time'] = predictions
            return result
        
        return predictions
        
    def predict_single(self, **kwargs) -> float:
        """
        Make prediction for a single order.
        
        Args:
            **kwargs: Order features as keyword arguments
            
        Returns:
            Predicted delivery time
        """
        predictions = self.predict(kwargs)
        return float(predictions[0])
    
    def get_prediction_explanation(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get prediction with explanation.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with prediction and metadata
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data_df = pd.DataFrame([data])
        else:
            data_df = data.copy()
        
        # Make prediction
        prediction = self.predict(data_df)[0]
        
        explanation = {
            'predicted_delivery_time': float(prediction),
            'model_name': self.model_name,
            'input_features': data if isinstance(data, dict) else data.iloc[0].to_dict(),
            'confidence_interval': {
                'lower': float(prediction * 0.9),  # Placeholder
                'upper': float(prediction * 1.1)   # Placeholder
            }
        }
        
        return explanation
    
    def batch_predict(
        self,
        data: pd.DataFrame,
        batch_size: int = 1000
    ) -> np.ndarray:
        """
        Make predictions on large dataset in batches.
        
        Args:
            data: Input DataFrame
            batch_size: Size of each batch
            
        Returns:
            Array of predictions
        """
        predictions = []
        
        n_batches = int(np.ceil(len(data) / batch_size))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            
            batch_data = data.iloc[start_idx:end_idx]
            batch_predictions = self.predict(batch_data)
            predictions.extend(batch_predictions)
            
            self.logger.info(
                f"Processed batch {i+1}/{n_batches} "
                f"({start_idx+1}-{end_idx}/{len(data)})"
            )
        
        return np.array(predictions)


class ModelInference:
    """
    Helper class for model inference operations.
    """
    
    @staticmethod
    def create_sample_input() -> Dict[str, Any]:
        """
        Create a sample input for testing.
        
        Returns:
            Dictionary with sample order data
        """
        return {
            'Order_ID': 1,
            'Distance_km': 10.5,
            'Weather': 'Clear',
            'Traffic_Level': 'Medium',
            'Time_of_Day': 'Evening',
            'Vehicle_Type': 'Bike',
            'Preparation_Time_min': 15,
            'Courier_Experience_yrs': 3.5
        }
    
    @staticmethod
    def validate_input(data: Dict[str, Any]) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'Distance_km',
            'Weather',
            'Traffic_Level',
            'Time_of_Day',
            'Vehicle_Type',
            'Preparation_Time_min',
            'Courier_Experience_yrs'
        ]
        
        for field in required_fields:
            if field not in data:
                return False
        
        return True
    
    @staticmethod
    def format_prediction_output(
        prediction: float,
        confidence: float = None
    ) -> str:
        """
        Format prediction for display.
        
        Args:
            prediction: Predicted delivery time
            confidence: Confidence score (optional)
            
        Returns:
            Formatted string
        """
        output = f"Predicted Delivery Time: {prediction:.1f} minutes"
        
        if confidence:
            output += f" (Confidence: {confidence:.2%})"
        
        return output

