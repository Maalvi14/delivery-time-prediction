"""
Example: Training a delivery time prediction model.

This script demonstrates how to use the DeliveryTimePipeline to train
a complete model from raw data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_pipeline import DeliveryTimePipeline, Config


def main():
    """Train a delivery time prediction model."""
    
    # Initialize pipeline with default configuration
    print("Initializing pipeline...")
    pipeline = DeliveryTimePipeline(verbose=True)
    
    # Run complete pipeline
    print("\nRunning full pipeline...\n")
    # Resolve path relative to project root
    results = pipeline.run_full_pipeline(
        data_path='data/Food_Delivery_Times.csv',
        save_model=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Model: {results['best_model_name']}")
    print(f"Model saved to: {results['model_path']}")
    
    print("\nTop 5 Models:")
    top_5 = results['comparison_df'].head(5)
    print(top_5[['Rank', 'Model', 'Test_R²', 'Test_RMSE', 'Test_MAE']].to_string(index=False))
    
    print("\n✅ Training complete Model ready for predictions.")


def train_with_custom_config():
    """Train with custom configuration."""
    
    # Create custom configuration
    config = Config()
    config.test_size = 0.3
    config.random_state = 123
    config.cv_folds = 10
    
    # Initialize pipeline
    pipeline = DeliveryTimePipeline(config=config, verbose=True)
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        data_path='data/Food_Delivery_Times.csv',
        save_model=True
    )
    
    return results


if __name__ == '__main__':
    main()
    
    # Uncomment to try custom configuration:
    # train_with_custom_config()

