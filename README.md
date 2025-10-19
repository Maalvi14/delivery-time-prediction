# Delivery Time Prediction

A comprehensive machine learning project for predicting food delivery times in urban settings. This project includes exploratory data analysis, feature engineering, model training and evaluation, and a production-ready ML pipeline.

## 📋 Project Overview

This project predicts delivery times for food orders based on various factors including:
- Distance to delivery location
- Weather conditions
- Traffic levels
- Time of day
- Vehicle type
- Restaurant preparation time
- Courier experience

**Best Model Performance:**
- Model: Ridge Regression
- Test R²: 0.8199
- Test RMSE: 8.98 minutes
- Test MAE: 6.04 minutes
- Test MAPE: 10.77%

## 🏗️ Project Structure

```
delivery-time-prediction/
├── data/                           # Data files
│   ├── Food_Delivery_Times.csv    # Raw dataset
│   └── model_comparison_results.csv
├── notebooks/                      # Jupyter notebooks
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   └── Assessment.ipynb           # Initial assessment
├── model_pipeline/                 # Production ML pipeline
│   ├── __init__.py
│   ├── config.py                  # Configuration parameters
│   ├── preprocessing.py           # Data preprocessing
│   ├── feature_engineering.py     # Feature engineering
│   ├── models.py                  # Model training & evaluation
│   ├── predict.py                 # Prediction interface
│   ├── pipeline.py                # Main pipeline orchestrator
│   ├── utils.py                   # Utility functions
│   ├── README.md                  # Pipeline documentation
│   └── examples/                  # Usage examples
│       ├── train_model.py
│       ├── make_predictions.py
│       └── custom_pipeline.py
├── api/                           # FastAPI application
│   ├── __init__.py
│   ├── app.py                     # Main FastAPI application
│   ├── models.py                  # Pydantic models
│   ├── predictor_service.py        # Predictor service wrapper
│   └── config.py                  # API configuration
├── models/                         # Saved models
├── results/                        # Model results
├── logs/                          # Pipeline logs
├── test_pipeline.py               # Pipeline test script
├── run_api.py                     # API server startup script
└── README.md                      # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd delivery-time-prediction
```

2. Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

### Train a Model

```python
from model_pipeline import DeliveryTimePipeline

# Initialize and run pipeline
pipeline = DeliveryTimePipeline()
results = pipeline.run_full_pipeline(
    data_path='data/Food_Delivery_Times.csv',
    save_model=True
)

print(f"Best Model: {results['best_model_name']}")
print(f"Model saved to: {results['model_path']}")
```

### Make Predictions

```python
from model_pipeline import DeliveryTimePredictor

# Load trained model
predictor = DeliveryTimePredictor(model_path='models/ridge_regression.pkl')

# Predict single order
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

## 🌐 API Usage

### Start the API Server

```bash
# Start the FastAPI server
python run_api.py
```

The server will start on `http://localhost:8000` by default. API documentation is available at `http://localhost:8000/docs`.

### Model Auto-Discovery

The API automatically discovers and uses the first available model in the `models/` directory:

- **Auto-discovery**: Scans `models/` directory for `.pkl` files
- **Consistent ordering**: Uses alphabetical sorting for predictable model selection
- **Fallback**: Falls back to `ridge_regression.pkl` if no models found
- **Override**: Can be overridden with `MODEL_PATH` environment variable

**Example scenarios:**
- If `models/` contains `[ridge_regression.pkl, random_forest.pkl, xgboost.pkl]` → Uses `ridge_regression.pkl`
- If `models/` contains `[random_forest.pkl, xgboost.pkl]` → Uses `random_forest.pkl`
- If `models/` is empty → Falls back to `ridge_regression.pkl` (will fail if not found)

### Configuration

Configure the API using environment variables:

```bash
# Model path (default: auto-discovers first .pkl file in models/ directory)
export MODEL_PATH="path/to/your/model.pkl"

# Server configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_DEBUG="false"

# CORS origins (comma-separated)
export CORS_ORIGINS="http://localhost:3000,http://localhost:8080"

# Logging
export LOG_LEVEL="INFO"
```

### API Endpoints

#### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Ridge Regression",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Model Information
```bash
GET /model/info
```

**Response:**
```json
{
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
  "is_preprocessing_available": true
}
```

#### Available Models
```bash
GET /models/available
```

**Response:**
```json
{
  "available_models": ["ridge_regression", "random_forest", "xgboost"],
  "current_model": "ridge_regression",
  "total_models": 3
}
```

#### Single Prediction (Raw Input)
```bash
POST /predict
```

**Request Body (Raw Input):**
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

**Response:**
```json
{
  "predicted_delivery_time": 45.2,
  "model_name": "Ridge Regression"
}
```


#### Batch Predictions
```bash
POST /predict/batch
```

**Request Body:**
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

**Response:**
```json
{
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
```

### Input Types

#### Raw Input (Recommended)
- **Use when:** You have original order data (distance, weather, etc.)
- **Advantage:** Automatic preprocessing and feature engineering
- **Required fields:** Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type, Preparation_Time_min, Courier_Experience_yrs
- **Optional fields:** Order_ID

### Error Handling

The API returns appropriate HTTP status codes:

- **200 OK:** Successful prediction
- **404 Not Found:** Model not loaded or endpoint not found
- **422 Unprocessable Entity:** Invalid input data
- **500 Internal Server Error:** Prediction failure

**Error Response Format:**
```json
{
  "error": "Model not found",
  "detail": "No trained model found at the specified path",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Example Usage with curl

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Distance_km": 10.5,
    "Weather": "Clear",
    "Traffic_Level": "Medium",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Bike",
    "Preparation_Time_min": 15.0,
    "Courier_Experience_yrs": 3.5
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "Distance_km": 10.5,
      "Weather": "Clear",
      "Traffic_Level": "Medium",
      "Time_of_Day": "Evening",
      "Vehicle_Type": "Bike",
      "Preparation_Time_min": 15.0,
      "Courier_Experience_yrs": 3.5
    },
    {
      "Distance_km": 5.2,
      "Weather": "Rainy",
      "Traffic_Level": "High",
      "Time_of_Day": "Morning",
      "Vehicle_Type": "Scooter",
      "Preparation_Time_min": 20.0,
      "Courier_Experience_yrs": 5.0
    }
  ]'
```

### Pretty-Printing CLI Output

For better readability when testing from the command line, you can pipe the output through `jq` to format the JSON response:

```bash
# Install jq (if not already installed)
# macOS: brew install jq
# Ubuntu/Debian: sudo apt install jq
# Windows: choco install jq

# Pretty-printed single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Distance_km": 10.5,
    "Weather": "Clear",
    "Traffic_Level": "Medium",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Bike",
    "Preparation_Time_min": 15.0,
    "Courier_Experience_yrs": 3.5
  }' | jq

# Pretty-printed batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "Distance_km": 10.5,
      "Weather": "Clear",
      "Traffic_Level": "Medium",
      "Time_of_Day": "Evening",
      "Vehicle_Type": "Bike",
      "Preparation_Time_min": 15.0,
      "Courier_Experience_yrs": 3.5
    }
  ]' | jq
```

**Alternative to jq:** You can also use `python -m json.tool` for pretty-printing:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Distance_km": 10.5, "Weather": "Clear", "Traffic_Level": "Medium", "Time_of_Day": "Evening", "Vehicle_Type": "Bike", "Preparation_Time_min": 15.0, "Courier_Experience_yrs": 3.5}' | python -m json.tool
```

### Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print("API Status:", response.json())

# Model info
response = requests.get(f"{BASE_URL}/model/info")
model_info = response.json()
print("Model:", model_info["model_name"])
print("Features:", model_info["num_features"])

# Single prediction
order_data = {
    "Distance_km": 10.5,
    "Weather": "Clear",
    "Traffic_Level": "Medium",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Bike",
    "Preparation_Time_min": 15.0,
    "Courier_Experience_yrs": 3.5
}

response = requests.post(f"{BASE_URL}/predict", json=order_data)
prediction = response.json()
print(f"Predicted delivery time: {prediction['predicted_delivery_time']:.1f} minutes")

# Batch prediction
orders_data = [
    {
        "Distance_km": 10.5,
        "Weather": "Clear",
        "Traffic_Level": "Medium",
        "Time_of_Day": "Evening",
        "Vehicle_Type": "Bike",
        "Preparation_Time_min": 15.0,
        "Courier_Experience_yrs": 3.5
    },
    {
        "Distance_km": 5.2,
        "Weather": "Rainy",
        "Traffic_Level": "High",
        "Time_of_Day": "Morning",
        "Vehicle_Type": "Scooter",
        "Preparation_Time_min": 20.0,
        "Courier_Experience_yrs": 5.0
    }
]

response = requests.post(f"{BASE_URL}/predict/batch", json=orders_data)
batch_predictions = response.json()
for i, pred in enumerate(batch_predictions["predictions"]):
    print(f"Order {i+1}: {pred['predicted_delivery_time']:.1f} minutes")
```

## 📊 Data Pipeline

The model pipeline consists of several stages:

### 1. Data Preprocessing
- **Missing Value Imputation**: Mode for categorical, median for numerical
- **Outlier Detection**: IQR method with 1.5x threshold
- **Outlier Treatment**: Winsorization (capping)
- **Type Conversion**: Categorical variables to category dtype

### 2. Feature Engineering
Creates 11 new features:
- **Domain Features**: Estimated speed, travel time, total time
- **Binary Indicators**: Rush hour, bad weather, high traffic
- **Categorical Bins**: Experience level, distance category
- **Interactions**: Weather×Traffic, Vehicle×Traffic

### 3. Model Training
Trains and compares 12 models:
- Linear Regression
- Ridge Regression ⭐ (Best)
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

### 4. Model Evaluation
- Train/test split (80/20)
- Cross-validation
- Overfitting analysis
- Feature importance analysis
- Multiple metrics (R², RMSE, MAE, MAPE)

## 🔧 Usage Examples

### Test the Pipeline

```bash
python test_pipeline.py
```

### Train with Custom Configuration

```python
from model_pipeline import Config, DeliveryTimePipeline

config = Config()
config.test_size = 0.3
config.random_state = 123
config.cv_folds = 10

pipeline = DeliveryTimePipeline(config=config)
results = pipeline.run_full_pipeline()
```

### Batch Predictions

```python
import pandas as pd
from model_pipeline import DeliveryTimePredictor

# Load new orders
new_orders = pd.read_csv('new_orders.csv')

# Make predictions
predictor = DeliveryTimePredictor(model_path='models/ridge_regression.pkl')
predictions = predictor.batch_predict(new_orders)

# Add to dataframe
new_orders['Predicted_Time'] = predictions
```

### Use Individual Components

```python
from model_pipeline import (
    DataPreprocessor,
    FeatureEngineer,
    ModelTrainer
)

# Preprocess
preprocessor = DataPreprocessor()
df_clean = preprocessor.fit_transform(df)

# Engineer features
engineer = FeatureEngineer()
df_engineered = engineer.fit_transform(df_clean)

# Train models
trainer = ModelTrainer()
results = trainer.train_all(X_train, y_train, X_test, y_test)
```

## 📈 Model Performance

| Rank | Model | Test R² | Test RMSE | Test MAE | MAPE (%) |
|------|-------|---------|-----------|----------|----------|
| 1 | Ridge Regression | 0.8199 | 8.98 | 6.04 | 10.77 |
| 2 | Linear Regression | 0.8193 | 9.00 | 6.06 | 10.83 |
| 3 | Lasso Regression | 0.8032 | 9.39 | 6.55 | 12.76 |
| 4 | LightGBM | 0.7900 | 9.70 | 6.90 | 12.47 |
| 5 | Random Forest | 0.7855 | 9.80 | 7.08 | 13.35 |

*Full results in `data/model_comparison_results.csv`*

## 📓 Notebooks

### EDA.ipynb
Comprehensive exploratory data analysis including:
- Dataset overview and inspection
- Data quality assessment
- Univariate, bivariate, and multivariate analysis
- Feature engineering exploration
- Model benchmarking
- Professional reporting

### Assessment.ipynb
Initial assessment and prototyping:
- Basic data inspection
- Preliminary preprocessing
- Feature correlation analysis

## 🛠️ API Reference

### DeliveryTimePipeline
Main pipeline orchestrator.

**Key Methods:**
- `load_data(filepath)` - Load data from CSV
- `preprocess_data(df)` - Clean and preprocess data
- `engineer_features(df)` - Create engineered features
- `train_models()` - Train all models
- `evaluate_models()` - Compare model performance
- `select_best_model()` - Select best performing model
- `run_full_pipeline()` - Execute complete pipeline

### DeliveryTimePredictor
Inference interface for predictions.

**Key Methods:**
- `predict(data)` - Make predictions on DataFrame
- `predict_single(**kwargs)` - Predict single order
- `batch_predict(data)` - Batch predictions with progress
- `get_prediction_explanation(data)` - Detailed prediction info

### Configuration
Customize pipeline behavior via `Config`:
- Data paths
- Preprocessing parameters
- Feature engineering settings
- Model hyperparameters
- Evaluation metrics

## 📚 Documentation

- **Pipeline Documentation**: `model_pipeline/README.md`
- **API Documentation**: See docstrings in each module
- **Examples**: `model_pipeline/examples/`

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_pipeline.py
```

This will:
1. Initialize the pipeline
2. Load and process data
3. Train models
4. Make test predictions
5. Verify all components work correctly

## 📦 Dependencies

Core dependencies:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- scipy

See `pyproject.toml` or `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## 📄 License

See LICENSE file for details.

## 👥 Authors

Data Science Team

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearch/RandomSearch
- [ ] Feature selection optimization
- [ ] Ensemble methods
- [x] Real-time prediction API
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Integration with delivery platforms
- [ ] API authentication and rate limiting
- [ ] Model versioning and rollback capabilities
- [ ] Prometheus metrics integration
- [ ] Docker containerization

## 📞 Support

For issues or questions:
1. Check the documentation in `model_pipeline/README.md`
2. Review examples in `model_pipeline/examples/`
3. Run `python test_pipeline.py` to diagnose issues
4. Open an issue on GitHub
