"""
FastAPI application for delivery time prediction.
"""

import logging
from datetime import datetime
from typing import List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import config
from .models import (
    OrderInputRaw, 
    PredictionResponse, 
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    ErrorResponse
)
from .predictor_service import PredictorService


# Global predictor service instance
predictor_service: PredictorService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global predictor_service
    
    # Startup
    try:
        logging.basicConfig(level=getattr(logging, config.log_level))
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Delivery Time Prediction API...")
        
        # Show available models
        available_models = config.get_available_models()
        logger.info(f"Available models: {available_models}")
        
        # Validate model path
        if not config.validate_model_path():
            logger.error(f"Model file not found at: {config.model_path}")
            raise FileNotFoundError(f"Model file not found at: {config.model_path}")
        
        # Show which model will be used
        selected_model = config.get_model_name()
        logger.info(f"Using model: {selected_model}")
        
        # Initialize predictor service
        predictor_service = PredictorService(config.get_model_path(), logger)
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title=config.title,
    description=config.description,
    version=config.version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request, exc):
    """Handle model not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="Model not found",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    """Handle runtime errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Prediction error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and model.
    """
    global predictor_service
    
    if predictor_service is None or not predictor_service.is_loaded:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    model_info = predictor_service.get_model_info()
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_info["model_name"],
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get model information.
    
    Returns details about the loaded model including feature names and metadata.
    """
    global predictor_service
    
    if predictor_service is None or not predictor_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not loaded"
        )
    
    model_info = predictor_service.get_model_info()
    
    return ModelInfoResponse(**model_info)


@app.get("/models/available")
async def get_available_models():
    """
    Get list of all available models in the models directory.
    
    Returns:
        List of available model names
    """
    available_models = config.get_available_models()
    current_model = config.get_model_name()
    
    return {
        "available_models": available_models,
        "current_model": current_model,
        "total_models": len(available_models)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(order: OrderInputRaw):
    """
    Make a single delivery time prediction.
    
    Accepts raw order data which will be automatically preprocessed and feature engineered.
    
    **Input Example:**
    ```json
    {
        "Order_ID": 1001,
        "Distance_km": 10.5,
        "Weather": "Clear",
        "Traffic_Level": "Medium",
        "Time_of_Day": "Evening",
        "Vehicle_Type": "Bike",
        "Preparation_Time_min": 15.0,
        "Courier_Experience_yrs": 3.5
    }
    ```
    """
    global predictor_service
    
    if predictor_service is None or not predictor_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not loaded"
        )
    
    try:
        # Make prediction
        predicted_time = predictor_service.predict(order)
        
        # Get model info
        model_info = predictor_service.get_model_info()
        
        return PredictionResponse(
            predicted_delivery_time=predicted_time,
            model_name=model_info["model_name"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(orders: List[OrderInputRaw]):
    """
    Make batch delivery time predictions.
    
    Accepts a list of raw order data which will be automatically preprocessed and feature engineered.
    
    **Example:**
    ```json
    [
        {
            "Order_ID": 1001,
            "Distance_km": 10.5,
            "Weather": "Clear",
            "Traffic_Level": "Medium",
            "Time_of_Day": "Evening",
            "Vehicle_Type": "Bike",
            "Preparation_Time_min": 15.0,
            "Courier_Experience_yrs": 3.5
        },
        {
            "Order_ID": 1002,
            "Distance_km": 5.2,
            "Weather": "Rainy",
            "Traffic_Level": "High",
            "Time_of_Day": "Morning",
            "Vehicle_Type": "Scooter",
            "Preparation_Time_min": 20.0,
            "Courier_Experience_yrs": 5.0
        }
    ]
    ```
    """
    global predictor_service
    
    if predictor_service is None or not predictor_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not loaded"
        )
    
    if not orders:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty orders list"
        )
    
    try:
        # Make batch predictions
        predictions = predictor_service.predict_batch(orders)
        
        # Get model info
        model_info = predictor_service.get_model_info()
        
        # Create response objects
        prediction_responses = []
        for prediction in predictions:
            prediction_responses.append(
                PredictionResponse(
                    predicted_delivery_time=prediction,
                    model_name=model_info["model_name"]
                )
            )
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_orders=len(orders),
            model_name=model_info["model_name"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Delivery Time Prediction API",
        "version": config.version,
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "available_models": "/models/available",
        "current_model": config.get_model_name()
    }
