# Delivery Time Prediction Model Pipeline

A production-ready machine learning pipeline for predicting food delivery times.

## Overview

This package provides a complete end-to-end solution for:
- Data preprocessing and cleaning
- Feature engineering
- Model training and evaluation
- Model selection and persistence
- Making predictions on new data

## Installation

Install the required dependencies:

```bash
pip install -r ../requirements.txt
```

Or if using `uv`:
```bash
uv pip install -r ../requirements.txt
```

## Quick Start

### Training a Model

```python
from model_pipeline import DeliveryTimePipeline

# Initialize pipeline
pipeline = DeliveryTimePipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline(
    data_path='../data/Food_Delivery_Times.csv',
    save_model=True
)

# Access best model
best_model_name = results['best_model_name']
model_path = results['model_path']
```

### Making Predictions

```python
from model_pipeline import DeliveryTimePredictor

# Load trained model
predictor = DeliveryTimePredictor(model_path='../models/ridge_regression.pkl')

# Single prediction
delivery_time = predictor.predict_single(
    Distance_km=10.5,
    Weather='Clear',
    Traffic_Level='Medium',
    Time_of_Day='Evening',
    Vehicle_Type='Bike',
    Preparation_Time_min=15,
    Courier_Experience_yrs=3.5
)

print(f"Predicted delivery time: {delivery_time:.1f} minutes")
```

### Batch Predictions

```python
import pandas as pd
from model_pipeline import DeliveryTimePredictor

# Load new data
new_orders = pd.read_csv('new_orders.csv')

# Make predictions
predictor = DeliveryTimePredictor(model_path='../models/ridge_regression.pkl')
predictions = predictor.batch_predict(new_orders)

# Add predictions to dataframe
new_orders['Predicted_Time'] = predictions
```

## Module Structure

### `config.py`
Configuration parameters for the entire pipeline including:
- Data paths
- Preprocessing parameters
- Feature engineering settings
- Model hyperparameters
- Evaluation metrics

### `preprocessing.py`
`DataPreprocessor` class handles:
- Missing value imputation (mode for categorical, median for numerical)
- Outlier detection and capping using IQR method
- Data type conversions
- Data validation

### `feature_engineering.py`
`FeatureEngineer` class creates:
- Domain-based features (speed, travel time estimates)
- Binary indicators (rush hour, bad weather, high traffic)
- Categorical bins (experience level, distance category)
- Interaction features (weather×traffic, vehicle×traffic)

`FeatureSelector` class handles:
- Feature selection by correlation
- One-hot encoding of categorical variables

### `models.py`
`ModelTrainer` class manages:
- Initialization of 12 different regression models
- Model training and prediction
- Feature scaling
- Model persistence (save/load)

`ModelEvaluator` class provides:
- Model comparison tables
- Cross-validation
- Overfitting analysis
- Feature importance extraction

### `predict.py`
`DeliveryTimePredictor` class enables:
- Loading trained models
- End-to-end prediction pipeline
- Single and batch predictions
- Prediction explanations

### `pipeline.py`
`DeliveryTimePipeline` class orchestrates:
- Complete end-to-end workflow
- All pipeline stages from data loading to model saving
- Automatic logging and reporting

### `utils.py`
Utility functions for:
- Logging setup
- Data loading and saving
- Metric calculations
- Outlier detection

## Pipeline Workflow

The complete pipeline follows these steps:

1. **Data Loading**: Load raw CSV data
2. **Preprocessing**: Handle missing values and outliers
3. **Feature Engineering**: Create new features from existing ones
4. **Data Preparation**: Encode categorical variables, train-test split, scaling
5. **Model Training**: Train 12 different regression models
6. **Model Evaluation**: Compare models, analyze overfitting
7. **Model Selection**: Select best performing model
8. **Model Saving**: Persist model for future use

## Supported Models

The pipeline trains and compares:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Support Vector Regression

## Configuration

Customize the pipeline by modifying `Config` parameters:

```python
from model_pipeline import Config, DeliveryTimePipeline

# Create custom config
config = Config()
config.test_size = 0.3
config.random_state = 123
config.cv_folds = 10

# Use custom config
pipeline = DeliveryTimePipeline(config=config)
```

## Examples

See the `examples/` directory for:
- `train_model.py`: Complete training example
- `make_predictions.py`: Inference examples
- `custom_pipeline.py`: Using individual components

## Performance

Based on the food delivery dataset:
- Best Model: Ridge Regression
- Test R²: 0.8199
- Test RMSE: 8.98 minutes
- Test MAE: 6.04 minutes
- Test MAPE: 10.77%

## API Reference

### DeliveryTimePipeline

Main pipeline orchestrator.

**Methods:**
- `load_data(filepath)`: Load data from CSV
- `preprocess_data(df)`: Preprocess data
- `engineer_features(df)`: Engineer features
- `prepare_for_training(df)`: Prepare data for modeling
- `train_models()`: Train all models
- `evaluate_models()`: Evaluate and compare models
- `select_best_model(metric)`: Select best model
- `save_best_model(filepath)`: Save model to disk
- `run_full_pipeline(data_path, save_model)`: Run complete pipeline

### DeliveryTimePredictor

Prediction interface.

**Methods:**
- `load(model_path)`: Load trained model
- `predict(data)`: Make predictions on DataFrame or dict
- `predict_single(**kwargs)`: Predict single order
- `batch_predict(data, batch_size)`: Batch predictions
- `get_prediction_explanation(data)`: Get prediction with metadata

## Logging

The pipeline provides detailed logging:

```python
pipeline = DeliveryTimePipeline(verbose=True)
```

Logs are printed to console and can be saved to file by configuring `Config.log_file`.

## Error Handling

The pipeline includes comprehensive error handling and validation:
- Input data validation
- Missing required fields
- Invalid data types
- Model not fitted errors

## Testing

Run tests with:

```bash
pytest tests/
```

## License

See LICENSE file in repository root.

## Authors

Data Science Team

## Version

1.0.0

