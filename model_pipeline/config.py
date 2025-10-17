"""
Configuration module for model pipeline parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np


@dataclass
class Config:
    """Configuration for the delivery time prediction pipeline."""
    
    # Data paths (relative to project root by default)
    data_path: str = "data/Food_Delivery_Times.csv"
    model_save_path: str = "models"
    results_save_path: str = "results"
    
    # Data preprocessing
    missing_value_strategy: Dict[str, str] = field(default_factory=lambda: {
        'categorical': 'mode',
        'numerical': 'median'
    })
    
    outlier_method: str = 'iqr'  # 'iqr' or 'zscore'
    outlier_threshold: float = 1.5  # IQR multiplier
    
    # Feature engineering
    vehicle_speeds: Dict[str, int] = field(default_factory=lambda: {
        'Bike': 15,      # km/h average city speed
        'Scooter': 25,   # km/h average city speed
        'Car': 30        # km/h average city speed
    })
    
    bad_weather_conditions: List[str] = field(default_factory=lambda: [
        'Rainy', 'Snowy', 'Foggy'
    ])
    
    rush_hour_times: List[str] = field(default_factory=lambda: [
        'Morning', 'Evening'
    ])
    
    # Experience level bins
    experience_bins: List[float] = field(default_factory=lambda: [
        -np.inf, 2, 5, np.inf
    ])
    experience_labels: List[str] = field(default_factory=lambda: [
        'Novice', 'Intermediate', 'Expert'
    ])
    
    # Distance category bins
    distance_bins: List[float] = field(default_factory=lambda: [
        0, 5, 10, 15, np.inf
    ])
    distance_labels: List[str] = field(default_factory=lambda: [
        'Very_Short', 'Short', 'Medium', 'Long'
    ])
    
    # Model training
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Model hyperparameters
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'Linear Regression': {},
        'Ridge Regression': {'alpha': 1.0, 'random_state': 42},
        'Lasso Regression': {'alpha': 1.0, 'random_state': 42},
        'ElasticNet': {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42},
        'Decision Tree': {'max_depth': 10, 'random_state': 42},
        'Random Forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        },
        'Gradient Boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
        },
        'AdaBoost': {'n_estimators': 100, 'random_state': 42},
        'XGBoost': {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        },
        'LightGBM': {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        'K-Nearest Neighbors': {'n_neighbors': 5},
        'Support Vector Regression': {'kernel': 'rbf', 'C': 1.0}
    })
    
    # Feature selection
    correlation_threshold: float = 0.1
    p_value_threshold: float = 0.05
    feature_importance_threshold: float = 0.8
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        'r2', 'rmse', 'mae', 'mape'
    ])
    
    # Logging
    verbose: bool = True
    log_file: str = "logs/pipeline.log"


# Default configuration instance
default_config = Config()

