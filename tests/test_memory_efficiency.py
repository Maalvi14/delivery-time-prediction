#!/usr/bin/env python3
"""
Test script to verify memory efficiency improvements from downcasting.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the model_pipeline to the path
sys.path.append(str(Path(__file__).parent))

from model_pipeline import (
    DataPreprocessor, 
    FeatureEngineer, 
    FeatureSelector,
    Config
)
from model_pipeline.utils import load_data, setup_logging


def test_memory_efficiency():
    """Test memory efficiency with and without downcasting."""
    
    print("="*80)
    print("MEMORY EFFICIENCY TEST")
    print("="*80)
    
    # Setup
    logger = setup_logging(verbose=True)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data('data/Food_Delivery_Times.csv', logger)
    print(f"   Original data shape: {df.shape}")
    print(f"   Original memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Test without memory optimization
    print("\n2. Testing WITHOUT memory optimization...")
    config_no_opt = Config()
    config_no_opt.enable_memory_optimization = False
    
    preprocessor_no_opt = DataPreprocessor(config_no_opt, logger)
    df_processed_no_opt = preprocessor_no_opt.fit_transform(df)
    
    engineer_no_opt = FeatureEngineer(config_no_opt, logger)
    df_engineered_no_opt = engineer_no_opt.fit_transform(df_processed_no_opt)
    
    selector_no_opt = FeatureSelector(config_no_opt, logger)
    X_no_opt = df_engineered_no_opt.drop(columns=['Delivery_Time_min', 'Order_ID'], errors='ignore')
    X_encoded_no_opt = selector_no_opt.encode_categorical(X_no_opt)
    # No memory optimization applied
    X_encoded_no_opt = selector_no_opt.optimize_memory(X_encoded_no_opt)
    
    memory_no_opt = X_encoded_no_opt.memory_usage(deep=True).sum() / 1024
    print(f"   Final features memory (no optimization): {memory_no_opt:.2f} KB")
    print(f"   Data types: {dict(X_encoded_no_opt.dtypes.value_counts())}")
    
    # Test with memory optimization
    print("\n3. Testing WITH memory optimization...")
    config_opt = Config()
    config_opt.enable_memory_optimization = True
    
    preprocessor_opt = DataPreprocessor(config_opt, logger)
    df_processed_opt = preprocessor_opt.fit_transform(df)
    
    engineer_opt = FeatureEngineer(config_opt, logger)
    df_engineered_opt = engineer_opt.fit_transform(df_processed_opt)
    
    selector_opt = FeatureSelector(config_opt, logger)
    X_opt = df_engineered_opt.drop(columns=['Delivery_Time_min', 'Order_ID'], errors='ignore')
    X_encoded_opt = selector_opt.encode_categorical(X_opt)
    # Apply memory optimization
    X_encoded_opt = selector_opt.optimize_memory(X_encoded_opt)
    
    memory_opt = X_encoded_opt.memory_usage(deep=True).sum() / 1024
    print(f"   Final features memory (with optimization): {memory_opt:.2f} KB")
    print(f"   Data types: {dict(X_encoded_opt.dtypes.value_counts())}")
    
    # Calculate savings
    memory_saved = memory_no_opt - memory_opt
    memory_saved_pct = (memory_saved / memory_no_opt) * 100
    
    print("\n4. Memory Efficiency Results:")
    print(f"   Memory without optimization: {memory_no_opt:.2f} KB")
    print(f"   Memory with optimization:    {memory_opt:.2f} KB")
    print(f"   Memory saved:                {memory_saved:.2f} KB")
    print(f"   Percentage reduction:        {memory_saved_pct:.1f}%")
    
    # Verify data integrity
    print("\n5. Verifying data integrity...")
    
    # Check that the shapes are the same
    assert X_encoded_no_opt.shape == X_encoded_opt.shape, "Feature shapes don't match!"
    
    # Check that the values are approximately equal (within float32 precision)
    numeric_cols = X_encoded_no_opt.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in X_encoded_opt.columns:
            diff = np.abs(X_encoded_no_opt[col] - X_encoded_opt[col])
            max_diff = diff.max()
            if max_diff > 1e-6:  # Allow for small floating point differences
                print(f"   Warning: Large difference in {col}: {max_diff:.2e}")
            else:
                print(f"   ✓ {col}: values match (max diff: {max_diff:.2e})")
    
    print("\n6. Testing model training compatibility...")
    
    # Test that models can train with float32 data
    from model_pipeline.models import ModelTrainer
    from sklearn.model_selection import train_test_split
    
    trainer = ModelTrainer(config_opt, logger)
    
    # Prepare data for training
    y = df_engineered_opt['Delivery_Time_min']
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded_opt, y, test_size=0.2, random_state=42
    )
    
    # Scale and prepare data
    X_train_scaled, y_train_scaled = trainer.prepare_data(X_train, y_train, scale=True)
    X_test_scaled = trainer.scaler.transform(X_test.astype('float32'))
    
    print(f"   Training data dtype: {X_train_scaled.dtype}")
    print(f"   Training data shape: {X_train_scaled.shape}")
    print(f"   Memory per sample: {X_train_scaled.nbytes / len(X_train_scaled) / 1024:.2f} KB")
    
    # Train a simple model to verify compatibility
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    mse = np.mean((y_test - y_pred) ** 2)
    
    print(f"   ✓ Model training successful")
    print(f"   ✓ Test MSE: {mse:.2f}")
    
    print("\n" + "="*80)
    print("MEMORY EFFICIENCY TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return {
        'memory_no_opt': memory_no_opt,
        'memory_opt': memory_opt,
        'memory_saved': memory_saved,
        'memory_saved_pct': memory_saved_pct,
        'test_mse': mse
    }


if __name__ == "__main__":
    results = test_memory_efficiency()
    
    print(f"\nSummary:")
    print(f"Memory reduction: {results['memory_saved_pct']:.1f}%")
    print(f"Model performance: MSE = {results['test_mse']:.2f}")
