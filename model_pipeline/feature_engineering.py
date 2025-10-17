"""
Feature engineering module for creating and transforming features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from .config import Config


class FeatureEngineer:
    """
    Handles feature engineering and transformation operations.
    
    This class performs:
    - Domain-based feature creation
    - Binary indicator features
    - Categorical binning
    - Interaction features
    - Feature transformations
    """
    
    def __init__(self, config: Config = None, logger: logging.Logger = None):
        """
        Initialize the FeatureEngineer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)
        self.is_fitted_ = False
        self.feature_names_ = []
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineer (learns feature names).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting FeatureEngineer...")
        self.feature_names_ = df.columns.tolist()
        self.is_fitted_ = True
        self.logger.info("FeatureEngineer fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by engineering features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Create domain-based features
        df_engineered = self._create_domain_features(df_engineered)
        
        # Create binary indicator features
        df_engineered = self._create_binary_indicators(df_engineered)
        
        # Create categorical bins
        df_engineered = self._create_categorical_bins(df_engineered)
        
        # Create interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        self.logger.info(
            f"Feature engineering complete: "
            f"{df.shape[1]} → {df_engineered.shape[1]} features "
            f"({df_engineered.shape[1] - df.shape[1]} new features)"
        )
        
        return df_engineered
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data in one step.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        return self.fit(df).transform(df)
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain knowledge-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with domain features added
        """
        df_domain = df.copy()
        
        # Estimated Speed based on vehicle type
        if 'Vehicle_Type' in df_domain.columns:
            df_domain['Estimated_Speed_kmh'] = df_domain['Vehicle_Type'].map(
                self.config.vehicle_speeds
            )
            # Ensure numeric dtype (mapping from categorical can result in Categorical dtype)
            # Fill unknown vehicle types with the average configured speed
            default_speed = float(np.mean(list(self.config.vehicle_speeds.values())))
            df_domain['Estimated_Speed_kmh'] = (
                df_domain['Estimated_Speed_kmh']
                .astype('float64')
                .fillna(default_speed)
            )
            # Guard against zero or negative speeds
            df_domain['Estimated_Speed_kmh'] = df_domain['Estimated_Speed_kmh'].clip(lower=1e-6)
            self.logger.debug("Created Estimated_Speed_kmh feature")
        
        # Time per kilometer (speed factor)
        if 'Estimated_Speed_kmh' in df_domain.columns:
            df_domain['Time_per_km'] = 60.0 / df_domain['Estimated_Speed_kmh'].astype('float64')
            self.logger.debug("Created Time_per_km feature")
        
        # Estimated travel time
        if 'Distance_km' in df_domain.columns and 'Time_per_km' in df_domain.columns:
            df_domain['Estimated_Travel_Time'] = (
                df_domain['Distance_km'] * df_domain['Time_per_km']
            )
            self.logger.debug("Created Estimated_Travel_Time feature")
        
        # Total time estimate
        if ('Preparation_Time_min' in df_domain.columns and 
            'Estimated_Travel_Time' in df_domain.columns):
            df_domain['Total_Time_Estimate'] = (
                df_domain['Preparation_Time_min'] + df_domain['Estimated_Travel_Time']
            )
            self.logger.debug("Created Total_Time_Estimate feature")
        
        return df_domain
    
    def _create_binary_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with binary indicators added
        """
        df_binary = df.copy()
        
        # Rush hour indicator
        if 'Time_of_Day' in df_binary.columns:
            df_binary['Is_Rush_Hour'] = df_binary['Time_of_Day'].isin(
                self.config.rush_hour_times
            ).astype(int)
            self.logger.debug("Created Is_Rush_Hour feature")
        
        # Bad weather indicator
        if 'Weather' in df_binary.columns:
            df_binary['Is_Bad_Weather'] = df_binary['Weather'].isin(
                self.config.bad_weather_conditions
            ).astype(int)
            self.logger.debug("Created Is_Bad_Weather feature")
        
        # High traffic indicator
        if 'Traffic_Level' in df_binary.columns:
            df_binary['Is_High_Traffic'] = (
                df_binary['Traffic_Level'] == 'High'
            ).astype(int)
            self.logger.debug("Created Is_High_Traffic feature")
        
        return df_binary
    
    def _create_categorical_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical bins for continuous features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical bins added
        """
        df_binned = df.copy()
        
        # Experience level categories
        if 'Courier_Experience_yrs' in df_binned.columns:
            df_binned['Experience_Level'] = pd.cut(
                df_binned['Courier_Experience_yrs'],
                bins=self.config.experience_bins,
                labels=self.config.experience_labels
            )
            self.logger.debug("Created Experience_Level feature")
        
        # Distance categories
        if 'Distance_km' in df_binned.columns:
            df_binned['Distance_Category'] = pd.cut(
                df_binned['Distance_km'],
                bins=self.config.distance_bins,
                labels=self.config.distance_labels
            )
            self.logger.debug("Created Distance_Category feature")
        
        return df_binned
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between categorical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        df_interaction = df.copy()
        
        # Weather × Traffic interaction
        if 'Weather' in df_interaction.columns and 'Traffic_Level' in df_interaction.columns:
            df_interaction['Weather_Traffic'] = (
                df_interaction['Weather'].astype(str) + '_' + 
                df_interaction['Traffic_Level'].astype(str)
            )
            self.logger.debug("Created Weather_Traffic interaction feature")
        
        # Vehicle × Traffic interaction
        if 'Vehicle_Type' in df_interaction.columns and 'Traffic_Level' in df_interaction.columns:
            df_interaction['Vehicle_Traffic'] = (
                df_interaction['Vehicle_Type'].astype(str) + '_' + 
                df_interaction['Traffic_Level'].astype(str)
            )
            self.logger.debug("Created Vehicle_Traffic interaction feature")
        
        return df_interaction
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of engineered features.
        
        Returns:
            List of feature names
        """
        return self.feature_names_


class FeatureSelector:
    """
    Handles feature selection operations.
    """
    
    def __init__(self, config: Config = None, logger: logging.Logger = None):
        """
        Initialize the FeatureSelector.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)
        self.selected_features_ = []
    
    def select_by_correlation(
        self,
        df: pd.DataFrame,
        target: str = 'Delivery_Time_min',
        threshold: float = None
    ) -> List[str]:
        """
        Select features based on correlation with target.
        
        Args:
            df: Input DataFrame
            target: Target variable name
            threshold: Correlation threshold (uses config if None)
            
        Returns:
            List of selected feature names
        """
        threshold = threshold or self.config.correlation_threshold
        
        # Get numerical features only
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        if target not in numeric_df.columns:
            self.logger.warning(f"Target {target} not in numerical columns")
            return []
        
        # Calculate correlations with target
        correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
        
        # Remove target itself and low correlation features
        correlations = correlations.drop(target, errors='ignore')
        selected = correlations[correlations >= threshold].index.tolist()
        
        self.logger.info(
            f"Selected {len(selected)} features with correlation >= {threshold}"
        )
        
        return selected
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            df: Input DataFrame
            drop_first: Whether to drop first category to avoid multicollinearity
            
        Returns:
            DataFrame with encoded categorical variables
        """
        # Get categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            return df
        
        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
        
        self.logger.info(
            f"Encoded {len(cat_cols)} categorical features: "
            f"{df.shape[1]} → {df_encoded.shape[1]} features"
        )
        
        return df_encoded

