import sys
from pathlib import Path

# Add parent directory to path to import model_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_pipeline import DeliveryTimePredictor

# Load the trained model
predictor = DeliveryTimePredictor(model_path='../models/ridge_regression.pkl')

# Make a single prediction
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