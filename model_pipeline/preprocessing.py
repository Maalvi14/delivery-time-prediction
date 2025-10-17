"""
Data preprocessing module for cleaning and preparing raw data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .utils import cap_outliers, get_outlier_bounds_iqr
from .config import Config


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing operations.
    
    This class performs:
    - Missing value imputation
    - Outlier detection and handling
    - Data type conversions
    - Basic data validation
    """
    
    def __init__(self, config: Config = None, logger: logging.Logger = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)
        
        # Store fitted parameters for transform
        self.categorical_modes_ = {}
        self.numerical_medians_ = {}
        self.outlier_bounds_ = {}
        self.is_fitted_ = False
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting DataPreprocessor...")
        
        # Calculate modes for categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                self.categorical_modes_[col] = df[col].mode()[0]
        
        # Calculate medians for numerical columns
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # Exclude Order_ID if present
        num_cols = [col for col in num_cols if col not in ['Order_ID', 'Delivery_Time_min']]
        
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                self.numerical_medians_[col] = df[col].median()
        
        # Calculate outlier bounds for numerical columns
        for col in num_cols:
            if col in df.columns:
                lower, upper, n_outliers = get_outlier_bounds_iqr(
                    df[col],
                    threshold=self.config.outlier_threshold
                )
                self.outlier_bounds_[col] = (lower, upper)
        
        # Also calculate outlier bounds for target if present
        if 'Delivery_Time_min' in df.columns:
            lower, upper, n_outliers = get_outlier_bounds_iqr(
                df['Delivery_Time_min'],
                threshold=self.config.outlier_threshold
            )
            self.outlier_bounds_['Delivery_Time_min'] = (lower, upper)
        
        self.is_fitted_ = True
        self.logger.info("DataPreprocessor fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted parameters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted_:
            raise RuntimeError("DataPreprocessor must be fitted before transform")
        
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Convert data types
        df_clean = self._convert_data_types(df_clean)
        
        return df_clean
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data in one step.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using fitted strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed missing values
        """
        df_imputed = df.copy()
        
        # Report missing values
        missing_counts = df_imputed.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.info(f"Missing values found in {(missing_counts > 0).sum()} columns")
        
        # Fill categorical columns with mode
        for col, mode_value in self.categorical_modes_.items():
            if col in df_imputed.columns:
                n_missing = df_imputed[col].isnull().sum()
                if n_missing > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_value)
                    self.logger.info(f"Filled {n_missing} missing values in {col} with mode: {mode_value}")
        
        # Fill numerical columns with median
        for col, median_value in self.numerical_medians_.items():
            if col in df_imputed.columns:
                n_missing = df_imputed[col].isnull().sum()
                if n_missing > 0:
                    df_imputed[col] = df_imputed[col].fillna(median_value)
                    self.logger.info(f"Filled {n_missing} missing values in {col} with median: {median_value:.2f}")
        
        return df_imputed
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using winsorization with fitted bounds.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with capped outliers
        """
        df_capped = df.copy()
        
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in df_capped.columns:
                # Count outliers
                n_outliers = ((df_capped[col] < lower) | (df_capped[col] > upper)).sum()
                
                if n_outliers > 0:
                    self.logger.info(
                        f"Capping {n_outliers} outliers in {col}: "
                        f"bounds=[{lower:.2f}, {upper:.2f}]"
                    )
                
                # Cap values
                df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        
        return df_capped
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for categorical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with converted data types
        """
        df_converted = df.copy()
        
        # Convert categorical columns to category type
        categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        
        for col in categorical_cols:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype('category')
                self.logger.debug(f"Converted {col} to category type")
        
        return df_converted
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'n_missing': df.isnull().sum().sum(),
            'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'n_numerical': len(df.select_dtypes(include=['float64', 'int64']).columns),
            'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_kb': df.memory_usage(deep=True).sum() / 1024
        }
        
        return summary

