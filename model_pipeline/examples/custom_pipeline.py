"""
Example: Using individual pipeline components.

This script demonstrates how to use individual components of the pipeline
for more fine-grained control.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_pipeline import (
    Config,
    DataPreprocessor,
    FeatureEngineer,
    FeatureSelector,
    ModelTrainer,
    ModelEvaluator
)
from model_pipeline.utils import setup_logging, load_data


def custom_preprocessing_example():
    """Example of custom preprocessing."""
    
    print("="*80)
    print("CUSTOM PREPROCESSING")
    print("="*80)
    
    # Setup
    logger = setup_logging(verbose=True)
    config = Config()
    
    # Load data
    df = load_data('../data/Food_Delivery_Times.csv', logger)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config, logger)
    
    # Fit and transform
    df_clean = preprocessor.fit_transform(df)
    
    print("\nPreprocessing Summary:")
    summary = preprocessor.get_data_summary(df_clean)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return df_clean


def custom_feature_engineering_example():
    """Example of custom feature engineering."""
    
    print("\n" + "="*80)
    print("CUSTOM FEATURE ENGINEERING")
    print("="*80)
    
    # Setup
    logger = setup_logging(verbose=True)
    config = Config()
    
    # Modify config for custom vehicle speeds
    config.vehicle_speeds = {
        'Bike': 18,      # Faster bikes
        'Scooter': 28,   # Faster scooters
        'Car': 35        # Faster cars
    }
    
    # Load and preprocess data
    df = load_data('../data/Food_Delivery_Times.csv', logger)
    preprocessor = DataPreprocessor(config, logger)
    df_clean = preprocessor.fit_transform(df)
    
    # Engineer features
    engineer = FeatureEngineer(config, logger)
    df_engineered = engineer.fit_transform(df_clean)
    
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"After engineering: {df_engineered.shape[1]}")
    print(f"New features: {df_engineered.shape[1] - df.shape[1]}")
    
    print("\nNew feature columns:")
    new_cols = [col for col in df_engineered.columns if col not in df.columns]
    for col in new_cols:
        print(f"  - {col}")
    
    return df_engineered


def custom_model_training_example():
    """Example of training specific models."""
    
    print("\n" + "="*80)
    print("CUSTOM MODEL TRAINING")
    print("="*80)
    
    # Setup
    logger = setup_logging(verbose=True)
    config = Config()
    
    # Prepare data
    df = load_data('../data/Food_Delivery_Times.csv', logger)
    preprocessor = DataPreprocessor(config, logger)
    df_clean = preprocessor.fit_transform(df)
    
    engineer = FeatureEngineer(config, logger)
    df_engineered = engineer.fit_transform(df_clean)
    
    # Separate features and target
    X = df_engineered.drop(columns=['Delivery_Time_min', 'Order_ID'], errors='ignore')
    y = df_engineered['Delivery_Time_min']
    
    # Encode categorical
    selector = FeatureSelector(config, logger)
    X_encoded = selector.encode_categorical(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    # Train only specific models
    trainer = ModelTrainer(config, logger)
    
    # Keep only linear models
    models_to_train = {
        'Linear Regression': trainer.models['Linear Regression'],
        'Ridge Regression': trainer.models['Ridge Regression'],
        'Lasso Regression': trainer.models['Lasso Regression']
    }
    trainer.models = models_to_train
    
    # Scale and train
    X_train_scaled, y_train = trainer.prepare_data(X_train, y_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    results = trainer.train_all(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate
    evaluator = ModelEvaluator(config, logger)
    comparison = evaluator.create_comparison_table(results)
    
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    
    return results


def custom_evaluation_example():
    """Example of custom model evaluation."""
    
    print("\n" + "="*80)
    print("CUSTOM MODEL EVALUATION")
    print("="*80)
    
    # Setup
    logger = setup_logging(verbose=True)
    config = Config()
    config.cv_folds = 10  # More folds
    
    # Prepare data (abbreviated)
    df = load_data('../data/Food_Delivery_Times.csv', logger)
    preprocessor = DataPreprocessor(config, logger)
    df_clean = preprocessor.fit_transform(df)
    
    engineer = FeatureEngineer(config, logger)
    df_engineered = engineer.fit_transform(df_clean)
    
    X = df_engineered.drop(columns=['Delivery_Time_min', 'Order_ID'], errors='ignore')
    y = df_engineered['Delivery_Time_min']
    
    selector = FeatureSelector(config, logger)
    X_encoded = selector.encode_categorical(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=config.test_size, random_state=config.random_state
    )
    
    # Train models
    trainer = ModelTrainer(config, logger)
    X_train_scaled, y_train = trainer.prepare_data(X_train, y_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    # Train only a few models for speed
    models_subset = {
        'Ridge Regression': trainer.models['Ridge Regression'],
        'Random Forest': trainer.models['Random Forest'],
        'XGBoost': trainer.models['XGBoost']
    }
    trainer.models = models_subset
    
    results = trainer.train_all(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Custom evaluation
    evaluator = ModelEvaluator(config, logger)
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = evaluator.cross_validate_models(
        models_subset,
        X_train_scaled,
        y_train,
        cv=config.cv_folds
    )
    
    print("\nCross-Validation Results:")
    print(cv_results.to_string(index=False))
    
    # Overfitting analysis
    overfitting = evaluator.analyze_overfitting(results)
    
    print("\nOverfitting Analysis:")
    print(overfitting.to_string(index=False))
    
    # Feature importance
    best_model_name, best_model, _ = trainer.get_best_model(results)
    
    if hasattr(best_model, 'feature_importances_'):
        importance = evaluator.get_feature_importance(
            best_model,
            trainer.feature_names_,
            top_n=15
        )
        
        print(f"\nTop 15 Features for {best_model_name}:")
        print(importance.to_string(index=False))
    
    return results


if __name__ == '__main__':
    # Run examples
    custom_preprocessing_example()
    custom_feature_engineering_example()
    custom_model_training_example()
    custom_evaluation_example()
    
    print("\n" + "="*80)
    print("✅ All custom examples complete")
    print("="*80)

