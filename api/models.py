"""
Pydantic models for FastAPI request and response validation.
"""

from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class WeatherEnum(str, Enum):
    """Weather conditions enum."""
    CLEAR = "Clear"
    RAINY = "Rainy"
    SNOWY = "Snowy"
    FOGGY = "Foggy"


class TrafficLevelEnum(str, Enum):
    """Traffic level enum."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TimeOfDayEnum(str, Enum):
    """Time of day enum."""
    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"
    NIGHT = "Night"


class VehicleTypeEnum(str, Enum):
    """Vehicle type enum."""
    BIKE = "Bike"
    SCOOTER = "Scooter"
    CAR = "Car"


class OrderInputRaw(BaseModel):
    """Raw order input matching the original CSV format."""
    
    Order_ID: Optional[int] = Field(None, description="Order ID (optional)")
    Distance_km: float = Field(..., gt=0, description="Distance in kilometers")
    Weather: WeatherEnum = Field(..., description="Weather condition")
    Traffic_Level: TrafficLevelEnum = Field(..., description="Traffic level")
    Time_of_Day: TimeOfDayEnum = Field(..., description="Time of day")
    Vehicle_Type: VehicleTypeEnum = Field(..., description="Vehicle type")
    Preparation_Time_min: float = Field(..., ge=0, description="Preparation time in minutes")
    Courier_Experience_yrs: float = Field(..., ge=0, description="Courier experience in years")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Order_ID": 1001,
                "Distance_km": 10.5,
                "Weather": "Clear",
                "Traffic_Level": "Medium",
                "Time_of_Day": "Evening",
                "Vehicle_Type": "Bike",
                "Preparation_Time_min": 15.0,
                "Courier_Experience_yrs": 3.5
            }
        }




class PredictionResponse(BaseModel):
    """Single prediction response."""
    
    predicted_delivery_time: float = Field(..., description="Predicted delivery time in minutes")
    model_name: str = Field(..., description="Name of the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_delivery_time": 45.2,
                "model_name": "Ridge Regression"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_orders: int = Field(..., description="Total number of orders processed")
    model_name: str = Field(..., description="Name of the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "predicted_delivery_time": 45.2,
                        "model_name": "Ridge Regression"
                    },
                    {
                        "predicted_delivery_time": 38.7,
                        "model_name": "Ridge Regression"
                    }
                ],
                "total_orders": 2,
                "model_name": "Ridge Regression"
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Name of the loaded model")
    feature_names: List[str] = Field(..., description="List of feature names expected by the model")
    num_features: int = Field(..., description="Number of features")
    model_type: str = Field(..., description="Type of the model")
    is_preprocessing_available: bool = Field(..., description="Whether preprocessing pipeline is available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "Ridge Regression",
                "feature_names": [
                    "Distance_km",
                    "Preparation_Time_min",
                    "Courier_Experience_yrs",
                    "Estimated_Speed_kmh",
                    "Travel_Time_min",
                    "Total_Time_min",
                    "Is_Rush_Hour",
                    "Is_Bad_Weather",
                    "Is_High_Traffic",
                    "Experience_Level_Intermediate",
                    "Distance_Category_Medium",
                    "Weather_Traffic_Interaction",
                    "Vehicle_Traffic_Interaction"
                ],
                "num_features": 13,
                "model_type": "Ridge",
                "is_preprocessing_available": True
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    timestamp: str = Field(..., description="Current timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "Ridge Regression",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Model not found",
                "detail": "No trained model found at the specified path",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# Union type for flexible input handling
OrderInput = OrderInputRaw
