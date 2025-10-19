"""
Quick test script to verify the model pipeline works correctly.
"""

import pandas as pd
from pathlib import Path

from model_pipeline import DeliveryTimePipeline, DeliveryTimePredictor


def test_full_pipeline():
    """Test the complete pipeline."""
    
    print("="*80)
    print("TESTING DELIVERY TIME PREDICTION PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = DeliveryTimePipeline(verbose=True)
    
    # Check if data exists
    data_path = Path('data/Food_Delivery_Times.csv')
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        return False
    
    try:
        # Run pipeline
        print("\n2. Running full pipeline...")
        results = pipeline.run_full_pipeline(
            data_path=str(data_path),
            save_model=True
        )
        
        print("\n3. Pipeline Results:")
        print(f"   ✅ Best Model: {results['best_model_name']}")
        print(f"   ✅ Model Path: {results['model_path']}")
        
        # Test prediction
        print("\n4. Testing predictions...")
        
        if results['model_path'] and Path(results['model_path']).exists():
            predictor = DeliveryTimePredictor(model_path=results['model_path'])
            
            # Test single prediction
            test_order = {
                'Order_ID': 9999,
                'Distance_km': 10.5,
                'Weather': 'Clear',
                'Traffic_Level': 'Medium',
                'Time_of_Day': 'Evening',
                'Vehicle_Type': 'Bike',
                'Preparation_Time_min': 15,
                'Courier_Experience_yrs': 3.5
            }
            
            predicted_time = predictor.predict_single(**test_order)
            print(f"   ✅ Single Prediction: {predicted_time:.1f} minutes")
            
            # Test batch prediction
            test_df = pd.DataFrame([test_order] * 3)
            predictions = predictor.predict(test_df)
            print(f"   ✅ Batch Predictions: {len(predictions)} predictions made")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - PIPELINE WORKING CORRECTLY")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_full_pipeline()
    
    if not success:
        print("\n❌ Tests failed. Please check the errors above.")
        exit(1)
    else:
        print("\n✅ Pipeline is ready for use")
        print("\nNext steps:")
        print("  - See model_pipeline/examples/ for usage examples")
        print("  - Run 'python model_pipeline/examples/train_model.py' to train")
        print("  - Run 'python model_pipeline/examples/make_predictions.py' to predict")
        exit(0)

