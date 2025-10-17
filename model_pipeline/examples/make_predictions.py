"""
Example: Making predictions with a trained model.

This script demonstrates how to use the DeliveryTimePredictor to make
predictions on new data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_pipeline import DeliveryTimePredictor


def predict_single_order():
    """Make a prediction for a single order."""
    
    print("="*80)
    print("SINGLE ORDER PREDICTION")
    print("="*80)
    
    # Load trained model
    model_path = 'models/ridge_regression.pkl'
    
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train a model first using train_model.py")
        return
    
    predictor = DeliveryTimePredictor(model_path=model_path)
    
    # Example order
    order = {
        'Order_ID': 1001,
        'Distance_km': 10.5,
        'Weather': 'Clear',
        'Traffic_Level': 'Medium',
        'Time_of_Day': 'Evening',
        'Vehicle_Type': 'Bike',
        'Preparation_Time_min': 15,
        'Courier_Experience_yrs': 3.5
    }
    
    print("\nOrder Details:")
    for key, value in order.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    predicted_time = predictor.predict_single(**order)
    
    print(f"\n✅ Predicted Delivery Time: {predicted_time:.1f} minutes")
    
    return predicted_time


def predict_multiple_orders():
    """Make predictions for multiple orders."""
    
    print("\n" + "="*80)
    print("BATCH PREDICTIONS")
    print("="*80)
    
    # Load trained model
    model_path = 'models/ridge_regression.pkl'
    
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        return
    
    predictor = DeliveryTimePredictor(model_path=model_path)
    
    # Example orders
    orders = pd.DataFrame([
        {
            'Order_ID': 1001,
            'Distance_km': 10.5,
            'Weather': 'Clear',
            'Traffic_Level': 'Medium',
            'Time_of_Day': 'Evening',
            'Vehicle_Type': 'Bike',
            'Preparation_Time_min': 15,
            'Courier_Experience_yrs': 3.5
        },
        {
            'Order_ID': 1002,
            'Distance_km': 5.2,
            'Weather': 'Rainy',
            'Traffic_Level': 'High',
            'Time_of_Day': 'Morning',
            'Vehicle_Type': 'Scooter',
            'Preparation_Time_min': 20,
            'Courier_Experience_yrs': 5.0
        },
        {
            'Order_ID': 1003,
            'Distance_km': 15.8,
            'Weather': 'Foggy',
            'Traffic_Level': 'Low',
            'Time_of_Day': 'Night',
            'Vehicle_Type': 'Car',
            'Preparation_Time_min': 10,
            'Courier_Experience_yrs': 2.0
        }
    ])
    
    print(f"\nPredicting for {len(orders)} orders...")
    
    # Make predictions
    predictions = predictor.predict(orders, return_dataframe=True)
    
    print("\nResults:")
    print(predictions[['Order_ID', 'Distance_km', 'Vehicle_Type', 'Predicted_Delivery_Time']].to_string(index=False))
    
    return predictions


def predict_from_csv():
    """Make predictions from a CSV file."""
    
    print("\n" + "="*80)
    print("PREDICTIONS FROM CSV")
    print("="*80)
    
    # Load trained model
    model_path = '../models/ridge_regression.pkl'
    
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        return
    
    predictor = DeliveryTimePredictor(model_path=model_path)
    
    # Load new orders from CSV
    csv_path = 'data/Food_Delivery_Times.csv'
    
    if not Path(csv_path).exists():
        print(f"\nError: Data file not found at {csv_path}")
        return
    
    new_orders = pd.read_csv(csv_path)
    
    # Take a sample
    sample_orders = new_orders.head(10).copy()
    
    # Remove actual delivery time (simulating new data)
    sample_orders_input = sample_orders.drop(columns=['Delivery_Time_min'], errors='ignore')
    
    print(f"\nLoaded {len(sample_orders_input)} sample orders from CSV")
    
    # Make predictions
    predictions = predictor.predict(sample_orders_input)
    
    # Add predictions to dataframe
    sample_orders_input['Predicted_Time'] = predictions
    
    # Compare with actual if available
    if 'Delivery_Time_min' in sample_orders.columns:
        sample_orders_input['Actual_Time'] = sample_orders['Delivery_Time_min'].values
        sample_orders_input['Error'] = abs(
            sample_orders_input['Predicted_Time'] - sample_orders_input['Actual_Time']
        )
        
        print("\nSample Predictions (with actual times for comparison):")
        print(sample_orders_input[['Order_ID', 'Distance_km', 'Predicted_Time', 'Actual_Time', 'Error']].to_string(index=False))
        
        print(f"\nMean Absolute Error: {sample_orders_input['Error'].mean():.2f} minutes")
    else:
        print("\nSample Predictions:")
        print(sample_orders_input[['Order_ID', 'Distance_km', 'Predicted_Time']].to_string(index=False))
    
    return sample_orders_input


def get_prediction_explanation():
    """Get detailed prediction explanation."""
    
    print("\n" + "="*80)
    print("PREDICTION EXPLANATION")
    print("="*80)
    
    # Load trained model
    model_path = '../models/ridge_regression.pkl'
    
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        return
    
    predictor = DeliveryTimePredictor(model_path=model_path)
    
    # Example order
    order = {
        'Distance_km': 12.0,
        'Weather': 'Rainy',
        'Traffic_Level': 'High',
        'Time_of_Day': 'Evening',
        'Vehicle_Type': 'Bike',
        'Preparation_Time_min': 18,
        'Courier_Experience_yrs': 2.5
    }
    
    # Get explanation
    explanation = predictor.get_prediction_explanation(order)
    
    print("\nPrediction Details:")
    print(f"  Model: {explanation['model_name']}")
    print(f"  Predicted Time: {explanation['predicted_delivery_time']:.1f} minutes")
    print(f"  Confidence Interval: [{explanation['confidence_interval']['lower']:.1f}, {explanation['confidence_interval']['upper']:.1f}] minutes")
    
    print("\nInput Features:")
    for key, value in explanation['input_features'].items():
        print(f"  {key}: {value}")
    
    return explanation


if __name__ == '__main__':
    # Single order prediction
    predict_single_order()
    
    # Multiple orders
    predict_multiple_orders()
    
    # From CSV
    predict_from_csv()
    
    # With explanation
    get_prediction_explanation()
    
    print("\n" + "="*80)
    print("✅ All predictions complete")
    print("="*80)

