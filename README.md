# Delivery Time Prediction

A comprehensive machine learning project for predicting food delivery times in urban settings. This project addresses the critical business problem of late deliveries that hurt customer trust, increase support costs, and risk customer churn.


## 🚀 Quick Start

### Prerequisites

**macOS Users - Important:** You need to install `libomp` for LightGBM/XGBoost to work properly:
```bash
brew install libomp
```

### Installation

This project uses `uv` for dependency management. We strongly recommend using `uv run` instead of `python` or `pip` commands.

1. **Clone the repository:**
```bash
git clone <repository-url>
cd delivery-time-prediction
```

2. **Install dependencies with uv:**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# or install via Homebrew with
brew install uv

# Install project dependencies
uv sync
```

### Training a Model

**Recommended approach:** Use the provided training script with `uv run`:

```bash
# Train the model using the example script
uv run model_pipeline/examples/train_model.py
```

This will:
- Load the data from `data/Food_Delivery_Times.csv`
- Run the complete ML pipeline
- Train and compare 12 different models
- Save the best model to `models/`
- Generate performance reports




## 🌐 API Usage

### Start the API Server

```bash
# Start the FastAPI server
uv run run_api.py
```

The server will start on `http://localhost:8000` by default. API documentation is available at `http://localhost:8000/docs`.

### Model Auto-Discovery

The API automatically discovers and uses the first available model in the `models/` directory:
- **Auto-discovery**: Scans `models/` directory for `.pkl` files
- **Consistent ordering**: Uses alphabetical sorting for predictable model selection
- **Fallback**: Falls back to `ridge_regression.pkl` if no models found
- **Override**: Can be overridden with `MODEL_PATH` environment variable

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Model Information
```bash
GET /model/info
```

#### Single Prediction
```bash
POST /predict
```

**Request Body:**
```json
{
  "Distance_km": 10.5,
  "Weather": "Clear",
  "Traffic_Level": "Medium",
  "Time_of_Day": "Evening",
  "Vehicle_Type": "Bike",
  "Preparation_Time_min": 15.0,
  "Courier_Experience_yrs": 3.5
}
```

#### Batch Predictions
```bash
POST /predict/batch
```

### Example Usage with curl

```bash
# Health check
curl http://localhost:8000/health

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
  }' | jq
```

## Available Reports & Documentation

This project includes comprehensive analysis and documentation:

### 📈 Analysis Reports
- **[EDA Report](EDA_report.md)** - Complete exploratory data analysis with business insights
- **[Model Notes](model_notes.md)** - Detailed model development process and findings
- **[Pipeline Summary](PIPELINE_SUMMARY.md)** - Technical implementation overview
- **[Error Insights](error_insights.md)** - Analysis of prediction errors and patterns
- **[Explainability](explainability.md)** - Model interpretability and feature importance
- **[Strategic Reflections](strategic_reflections.md)** - Strategic insights and business recommendations

### 🗄️ SQL Analysis
- **[SQL Queries](sql/sql_queries.sql)** - Comprehensive SQL queries for data analysis
- **[SQL Insights](sql/sql_insights.md)** - Key findings from SQL analysis

### 📓 Jupyter Notebooks
- **[EDA.ipynb](notebooks/EDA.ipynb)** - Interactive exploratory data analysis

### 📁 Generated Images
All analysis plots are saved in `notebooks/images/` including:
- Feature distribution analysis
- Correlation heatmaps
- Model performance comparisons
- Error distribution analysis
- Feature importance plots



## Project Structure

```
delivery-time-prediction/
├── data/                                 # Data files
│   ├── Food_Delivery_Times.csv           # Raw dataset
│   └── model_comparison_results.csv      # Comparison Results of all 12 models
├── notebooks/                            # Jupyter notebooks
│   ├── EDA.ipynb                         # Exploratory Data Analysis
│   └── images/                           # Generated analysis plots
├── model_pipeline/                       # Production ML pipeline
│   ├── __init__.py
│   ├── config.py                         # Configuration parameters
│   ├── preprocessing.py                  # Data preprocessing
│   ├── feature_engineering.py            # Feature engineering
│   ├── models.py                         # Model training & evaluation
│   ├── predict.py                        # Prediction interface
│   ├── pipeline.py                       # Main pipeline orchestrator
│   ├── utils.py                          # Utility functions
│   ├── README.md                         # Pipeline documentation
│   └── examples/                         # Usage examples
│       ├── __init__.py
│       ├── train_model.py                # Training script
│       ├── make_predictions.py           # Prediction examples
│       └── custom_pipeline.py            # Custom pipeline examples
├── api/                                  # FastAPI application
│   ├── __init__.py
│   ├── app.py                            # Main FastAPI application
│   ├── models.py                         # Pydantic models
│   ├── predictor_service.py              # Predictor service wrapper
│   └── config.py                         # API configuration
├── models/                               # Saved models
│   └── ridge_regression.pkl              # Best performing model
├── results/                              # Model results
│   ├── model_comparison.csv              # Model comparison results
│   └── overfitting_analysis.csv          # Overfitting analysis results
├── sql/                                  # SQL analysis
│   ├── sql_queries.sql                   # SQL queries for data analysis
│   └── sql_insights.md                   # SQL analysis insights
├── tests/                                # Test files
│   ├── __init__.py
│   ├── test_pipeline.py                  # Pipeline tests
│   └── test_prediction.py                # Prediction tests
├── run_api.py                            # API server startup script
├── main.py                               # Main application entry point
├── pyproject.toml                        # Project dependencies
├── uv.lock                               # Dependency lock file
├── LICENSE                               # License file
├── EDA_report.md                         # EDA analysis report
├── error_insights.md                     # Error analysis insights
├── explainability.md                     # Model explainability report
├── model_notes.md                        # Model development notes
├── PIPELINE_SUMMARY.md                   # Pipeline summary
├── strategic_reflections.md              # Strategic insights
├── v1 DS Technical Assessment.pdf        # Technical assessment document
└── README.md                             # This file
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
uv run python test_pipeline.py
```

This will:
1. Initialize the pipeline
2. Load and process data
3. Train models
4. Make test predictions
5. Verify all components work correctly

## 📦 Dependencies

Core dependencies managed via `pyproject.toml`:
- pandas >= 2.3.3
- numpy (via pandas)
- scikit-learn >= 1.7.2
- xgboost >= 3.0.5
- lightgbm >= 4.6.0
- matplotlib >= 3.10.7
- seaborn >= 0.13.2
- scipy >= 1.16.2
- fastapi >= 0.119.0
- uvicorn >= 0.38.0

**Note:** This project requires Python >= 3.12


## 📈 Model Performance Comparison

| Rank | Model | Test R² | Test RMSE | Test MAE | MAPE (%) |
|------|-------|---------|-----------|----------|----------|
| 1 | Ridge Regression | 0.8199 | 8.98 | 6.04 | 10.77 |
| 2 | Linear Regression | 0.8193 | 9.00 | 6.06 | 10.83 |
| 3 | Lasso Regression | 0.8032 | 9.39 | 6.55 | 12.76 |
| 4 | LightGBM | 0.7900 | 9.70 | 6.90 | 12.47 |
| 5 | Random Forest | 0.7855 | 9.80 | 7.08 | 13.35 |

*Full and more detailed results available in `data/model_comparison_results.csv`* after running training pipeline.

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearch/RandomSearch
- [ ] Feature selection optimization
- [ ] Ensemble methods
- [x] Real-time prediction API
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Integration with delivery platforms
- [ ] Model versioning and rollback capabilities

## 🤝 Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## 📄 License

See LICENSE file for details.

## 👥 Authors

Data Science Team

## 📞 Support

For issues or questions:
1. Check the documentation in `model_pipeline/README.md`
2. Review examples in `model_pipeline/examples/`
3. Run `uv run python test_pipeline.py` to diagnose issues
4. Open an issue on GitHub
