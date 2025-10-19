"""
Model training, evaluation, and selection module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .config import Config
from .utils import calculate_metrics


class ModelTrainer:
    """
    Handles model training and management.
    
    This class:
    - Initializes multiple models
    - Trains models on scaled data
    - Tracks trained models and their parameters
    """
    
    def __init__(self, config: Config = None, logger: logging.Logger = None):
        """
        Initialize the ModelTrainer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)
        
        self.models = {}
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.feature_names_ = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models with their hyperparameters."""
        model_classes = {
            'Linear Regression': LinearRegression,
            'Ridge Regression': Ridge,
            'Lasso Regression': Lasso,
            'ElasticNet': ElasticNet,
            'Decision Tree': DecisionTreeRegressor,
            'Random Forest': RandomForestRegressor,
            'Gradient Boosting': GradientBoostingRegressor,
            'AdaBoost': AdaBoostRegressor,
            'XGBoost': XGBRegressor,
            'LightGBM': LGBMRegressor,
            'K-Nearest Neighbors': KNeighborsRegressor,
            'Support Vector Regression': SVR
        }
        
        for model_name, model_class in model_classes.items():
            params = self.config.model_params.get(model_name, {})
            self.models[model_name] = model_class(**params)
        
        self.logger.info(f"Initialized {len(self.models)} models")
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training (scaling, etc.).
        
        Args:
            X: Feature matrix
            y: Target variable
            scale: Whether to scale features
            
        Returns:
            Tuple of (X_scaled, y)
        """
        self.feature_names_ = X.columns.tolist()
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
            self.logger.info("Features scaled using StandardScaler")
        else:
            X_scaled = X.values
        
        return X_scaled, y.values
    
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all models and return results.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            
        Returns:
            Dictionary of training results
        """
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train)
                train_metrics = calculate_metrics(y_train, y_train_pred)
                
                result = {
                    'model': model,
                    'train_metrics': train_metrics
                }
                
                # Test metrics if provided
                if X_test is not None and y_test is not None:
                    y_test_pred = model.predict(X_test)
                    test_metrics = calculate_metrics(y_test, y_test_pred)
                    result['test_metrics'] = test_metrics
                    
                    self.logger.info(
                        f"{model_name} - Test R²: {test_metrics['R²']:.4f}, "
                        f"RMSE: {test_metrics['RMSE']:.2f}"
                    )
                else:
                    self.logger.info(
                        f"{model_name} - Train R²: {train_metrics['R²']:.4f}"
                    )
                
                results[model_name] = result
                self.trained_models[model_name] = model
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
        
        return results
    
    def get_best_model(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'R²',
        use_test: bool = True
    ) -> Tuple[str, Any, Dict[str, float]]:
        """
        Get the best performing model.
        
        Args:
            results: Training results dictionary
            metric: Metric to use for selection
            use_test: Use test metrics if True, else train metrics
            
        Returns:
            Tuple of (model_name, model, metrics)
        """
        metric_key = 'test_metrics' if use_test else 'train_metrics'
        
        # Find best model
        best_model_name = None
        best_score = -np.inf if metric == 'R²' else np.inf
        
        for model_name, result in results.items():
            if metric_key not in result:
                continue
            
            score = result[metric_key][metric]
            
            # Higher is better for R², lower is better for errors
            if metric == 'R²':
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError("No valid models found in results")
        
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name][metric_key]
        
        self.logger.info(
            f"Best model: {best_model_name} with {metric}={best_score:.4f}"
        )
        
        return best_model_name, best_model, best_metrics
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        filepath: str = None,
        preprocessor: Any = None,
        feature_engineer: Any = None,
        feature_selector: Any = None
    ) -> str:
        """
        Save a trained model to disk with preprocessing pipeline components.
        
        Args:
            model: Trained model
            model_name: Name of the model
            filepath: Path to save model (optional)
            preprocessor: Fitted preprocessor instance
            feature_engineer: Fitted feature engineer instance
            feature_selector: Fitted feature selector instance
            
        Returns:
            Path where model was saved
        """
        if filepath is None:
            filepath = Path(self.config.model_save_path) / f"{model_name.replace(' ', '_').lower()}.pkl"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model, scaler, and preprocessing components
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names_,
            'model_name': model_name,
            'preprocessor': preprocessor,
            'feature_engineer': feature_engineer,
            'feature_selector': feature_selector
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: str) -> Tuple[Any, StandardScaler, List[str], str, Any, Any, Any]:
        """
        Load a trained model from disk with preprocessing components.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Tuple of (model, scaler, feature_names, model_name, preprocessor, feature_engineer, feature_selector)
        """
        model_data = joblib.load(filepath)
        
        self.logger.info(f"Model loaded from {filepath}")
        
        return (
            model_data['model'],
            model_data['scaler'],
            model_data['feature_names'],
            model_data['model_name'],
            model_data.get('preprocessor'),
            model_data.get('feature_engineer'),
            model_data.get('feature_selector')
        )


class ModelEvaluator:
    """
    Handles model evaluation and comparison.
    """
    
    def __init__(self, config: Config = None, logger: logging.Logger = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)
    
    def create_comparison_table(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a comparison table of all models.
        
        Args:
            results: Training results dictionary
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, result in results.items():
            row = {'Model': model_name}
            
            # Add test metrics
            if 'test_metrics' in result:
                for metric, value in result['test_metrics'].items():
                    row[f'Test_{metric}'] = value
            
            # Add train metrics
            if 'train_metrics' in result:
                for metric, value in result['train_metrics'].items():
                    row[f'Train_{metric}'] = value
            
            # Calculate overfitting gap
            if 'test_metrics' in result and 'train_metrics' in result:
                row['Overfit_Gap'] = (
                    result['train_metrics']['R²'] - result['test_metrics']['R²']
                )
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by test R² if available
        if 'Test_R²' in df.columns:
            df = df.sort_values('Test_R²', ascending=False)
        
        # Add rank
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        return df.reset_index(drop=True)
    
    def cross_validate_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = None
    ) -> pd.DataFrame:
        """
        Perform cross-validation on all models.
        
        Args:
            models: Dictionary of models to evaluate
            X: Feature matrix
            y: Target variable
            cv: Number of CV folds (uses config if None)
            
        Returns:
            DataFrame with CV results
        """
        cv = cv or self.config.cv_folds
        cv_results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Cross-validating {model_name}...")
            
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring='r2',
                    n_jobs=-1
                )
                
                cv_results.append({
                    'Model': model_name,
                    'CV_Mean_R²': scores.mean(),
                    'CV_Std_R²': scores.std(),
                    'CV_Min_R²': scores.min(),
                    'CV_Max_R²': scores.max()
                })
                
                self.logger.info(
                    f"{model_name} - CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})"
                )
                
            except Exception as e:
                self.logger.error(f"Error in CV for {model_name}: {str(e)}")
        
        return pd.DataFrame(cv_results)
    
    def analyze_overfitting(
        self,
        results: Dict[str, Dict[str, Any]],
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Analyze overfitting for all models.
        
        Args:
            results: Training results dictionary
            threshold: Threshold for significant overfitting
            
        Returns:
            DataFrame with overfitting analysis
        """
        analysis = []
        
        for model_name, result in results.items():
            if 'train_metrics' not in result or 'test_metrics' not in result:
                continue
            
            train_r2 = result['train_metrics']['R²']
            test_r2 = result['test_metrics']['R²']
            gap = train_r2 - test_r2
            
            if gap > threshold:
                status = '🔴 Overfitting'
            elif gap > threshold / 2:
                status = '⚠️ Slight Overfitting'
            else:
                status = '✅ Good Generalization'
            
            analysis.append({
                'Model': model_name,
                'Train_R²': train_r2,
                'Test_R²': test_r2,
                'Gap': gap,
                'Status': status
            })
        
        df = pd.DataFrame(analysis).sort_values('Gap', ascending=False)
        return df.reset_index(drop=True)
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20
    ) -> Optional[pd.DataFrame]:
        """
        Get feature importance from tree-based models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance or None
        """
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Calculate cumulative importance
        importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
        
        return importance_df.head(top_n).reset_index(drop=True)

