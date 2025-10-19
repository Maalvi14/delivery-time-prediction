"""
Main pipeline orchestrator that coordinates all components.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from .config import Config
from .utils import setup_logging, load_data, save_results, print_section
from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer, FeatureSelector
from .models import ModelTrainer, ModelEvaluator
from .predict import DeliveryTimePredictor

from sklearn.model_selection import train_test_split


class DeliveryTimePipeline:
    """
    Complete end-to-end pipeline for delivery time prediction.
    
    This class orchestrates:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Model training and evaluation
    4. Model selection and saving
    5. Predictions
    """
    
    def __init__(self, config: Config = None, verbose: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object
            verbose: Whether to enable verbose logging
        """
        self.config = config or Config()
        self.logger = setup_logging(verbose=verbose)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config, self.logger)
        self.feature_engineer = FeatureEngineer(self.config, self.logger)
        self.feature_selector = FeatureSelector(self.config, self.logger)
        self.trainer = ModelTrainer(self.config, self.logger)
        self.evaluator = ModelEvaluator(self.config, self.logger)
        
        # Store pipeline state
        self.data_raw = None
        self.data_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        self.training_results = None
        self.best_model = None
        self.best_model_name = None
        
        self.logger.info("Pipeline initialized successfully")
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to data file (uses config if None)
            
        Returns:
            Loaded DataFrame
        """
        filepath = filepath or self.config.data_path
        
        print_section("LOADING DATA")
        self.data_raw = load_data(filepath, self.logger)
        
        print(f"\nDataset Info:")
        print(f"  Shape: {self.data_raw.shape[0]} rows × {self.data_raw.shape[1]} columns")
        print(f"  Memory: {self.data_raw.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        return self.data_raw
    
    def preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocess the data.
        
        Args:
            df: Input DataFrame (uses loaded data if None)
            
        Returns:
            Preprocessed DataFrame
        """
        df = df if df is not None else self.data_raw
        
        if df is None:
            raise ValueError("No data to preprocess. Call load_data() first.")
        
        print_section("PREPROCESSING DATA")
        
        # Fit and transform
        self.data_processed = self.preprocessor.fit_transform(df)
        
        # Print summary
        summary = self.preprocessor.get_data_summary(self.data_processed)
        print(f"\nPreprocessing Summary:")
        print(f"  Missing values: {summary['n_missing']} ({summary['missing_pct']:.2f}%)")
        print(f"  Numerical features: {summary['n_numerical']}")
        print(f"  Categorical features: {summary['n_categorical']}")
        
        return self.data_processed
    
    def engineer_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer features.
        
        Args:
            df: Input DataFrame (uses processed data if None)
            
        Returns:
            DataFrame with engineered features
        """
        df = df if df is not None else self.data_processed
        
        if df is None:
            raise ValueError("No data for feature engineering. Call preprocess_data() first.")
        
        print_section("ENGINEERING FEATURES")
        
        # Fit and transform
        df_engineered = self.feature_engineer.fit_transform(df)
        
        print(f"\nFeature Engineering Summary:")
        print(f"  Original features: {df.shape[1]}")
        print(f"  Engineered features: {df_engineered.shape[1]}")
        print(f"  New features: {df_engineered.shape[1] - df.shape[1]}")
        
        self.data_processed = df_engineered
        return df_engineered
    
    def prepare_for_training(
        self,
        df: pd.DataFrame = None,
        target_col: str = 'Delivery_Time_min'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame (uses processed data if None)
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        df = df if df is not None else self.data_processed
        
        if df is None:
            raise ValueError("No data to prepare. Call engineer_features() first.")
        
        print_section("PREPARING DATA FOR TRAINING")
        
        # Separate features and target
        X = df.drop(columns=[target_col, 'Order_ID'], errors='ignore')
        y = df[target_col]
        
        print(f"\nBefore encoding:")
        print(f"  Features: {X.shape}")
        print(f"  Target: {y.shape}")
        
        # Encode categorical variables
        X_encoded = self.feature_selector.encode_categorical(X, drop_first=True)
        
        # Apply memory optimization after all feature engineering is complete
        X_encoded = self.feature_selector.optimize_memory(X_encoded)
        
        print(f"\nAfter encoding:")
        print(f"  Features: {X_encoded.shape}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        print(f"\nData Split:")
        print(f"  Training: {len(self.X_train)} samples ({len(self.X_train)/len(X_encoded)*100:.1f}%)")
        print(f"  Test: {len(self.X_test)} samples ({len(self.X_test)/len(X_encoded)*100:.1f}%)")
        
        # Scale features
        self.X_train_scaled, self.y_train = self.trainer.prepare_data(
            self.X_train, self.y_train, scale=True
        )
        self.X_test_scaled = self.trainer.scaler.transform(self.X_test)
        
        print(f"\nFeatures scaled using StandardScaler")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train all models.
        
        Returns:
            Dictionary of training results
        """
        if self.X_train_scaled is None:
            raise ValueError("Data not prepared. Call prepare_for_training() first.")
        
        print_section("TRAINING MODELS")
        print(f"\nTraining {len(self.trainer.models)} models...\n")
        
        self.training_results = self.trainer.train_all(
            self.X_train_scaled,
            self.y_train,
            self.X_test_scaled,
            self.y_test
        )
        
        return self.training_results
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate and compare all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        if self.training_results is None:
            raise ValueError("No results to evaluate. Call train_models() first.")
        
        print_section("EVALUATING MODELS")
        
        # Create comparison table
        comparison_df = self.evaluator.create_comparison_table(self.training_results)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Analyze overfitting
        print("\n")
        overfitting_df = self.evaluator.analyze_overfitting(self.training_results)
        print("\nOverfitting Analysis:")
        print(overfitting_df.to_string(index=False))
        
        # Save results
        results_path = Path(self.config.results_save_path)
        results_path.mkdir(parents=True, exist_ok=True)
        
        comparison_df.to_csv(results_path / 'model_comparison.csv', index=False)
        overfitting_df.to_csv(results_path / 'overfitting_analysis.csv', index=False)
        
        self.logger.info(f"Results saved to {results_path}")
        
        return comparison_df
    
    def select_best_model(self, metric: str = 'R²') -> Tuple[str, Any]:
        """
        Select the best performing model.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_name, model)
        """
        if self.training_results is None:
            raise ValueError("No results available. Call train_models() first.")
        
        print_section("SELECTING BEST MODEL")
        
        model_name, model, metrics = self.trainer.get_best_model(
            self.training_results,
            metric=metric,
            use_test=True
        )
        
        self.best_model_name = model_name
        self.best_model = model
        
        print(f"\nBest Model: {model_name}")
        print(f"\nPerformance Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            print("\n")
            importance_df = self.evaluator.get_feature_importance(
                model,
                self.trainer.feature_names_,
                top_n=10
            )
            print("Top 10 Feature Importances:")
            print(importance_df.to_string(index=False))
        
        return model_name, model
    
    def save_best_model(self, filepath: str = None) -> str:
        """
        Save the best model to disk.
        
        Args:
            filepath: Path to save model (optional)
            
        Returns:
            Path where model was saved
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model() first.")
        
        print_section("SAVING MODEL")
        
        saved_path = self.trainer.save_model(
            self.best_model,
            self.best_model_name,
            filepath
        )
        
        print(f"\nModel saved to: {saved_path}")
        
        return saved_path
    
    def run_full_pipeline(
        self,
        data_path: str = None,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Args:
            data_path: Path to data file
            save_model: Whether to save the best model
            
        Returns:
            Dictionary with pipeline results
        """
        print("="*80)
        print("DELIVERY TIME PREDICTION PIPELINE".center(80))
        print("="*80)
        
        # 1. Load data
        self.load_data(data_path)
        
        # 2. Preprocess
        self.preprocess_data()
        
        # 3. Engineer features
        self.engineer_features()
        
        # 4. Prepare for training
        self.prepare_for_training()
        
        # 5. Train models
        self.train_models()
        
        # 6. Evaluate models
        comparison_df = self.evaluate_models()
        
        # 7. Select best model
        model_name, model = self.select_best_model()
        
        # 8. Save model
        if save_model:
            model_path = self.save_best_model()
        else:
            model_path = None
        
        # Prepare results
        results = {
            'best_model_name': model_name,
            'best_model': model,
            'model_path': model_path,
            'comparison_df': comparison_df,
            'training_results': self.training_results,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }
        
        print_section("PIPELINE COMPLETE")
        print(f"\n✅ Pipeline executed successfully")
        print(f"   Best Model: {model_name}")
        if model_path:
            print(f"   Model saved to: {model_path}")
        
        return results
    
    def create_predictor(self, model_path: str = None) -> DeliveryTimePredictor:
        """
        Create a predictor instance for inference.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            DeliveryTimePredictor instance
        """
        predictor = DeliveryTimePredictor(model_path, self.config, self.logger)
        
        # Fit the preprocessing pipeline if we have data
        if self.data_raw is not None:
            predictor.preprocessor = self.preprocessor
            predictor.feature_engineer = self.feature_engineer
            predictor.feature_selector = self.feature_selector
        
        return predictor

