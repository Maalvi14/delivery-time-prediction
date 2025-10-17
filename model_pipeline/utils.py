"""
Utility functions for the model pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import logging
from pathlib import Path
import os


def setup_logging(log_file: str = None, verbose: bool = True) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        verbose: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('delivery_time_pipeline')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_outlier_bounds_iqr(
    data: pd.Series,
    threshold: float = 1.5
) -> Tuple[float, float, int]:
    """
    Get outlier bounds using IQR method.
    
    Args:
        data: Series of numerical data
        threshold: IQR multiplier (default 1.5)
        
    Returns:
        Tuple of (lower_bound, upper_bound, n_outliers)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    n_outliers = len(outliers)
    
    return lower_bound, upper_bound, n_outliers


def cap_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    threshold: float = 1.5,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Cap outliers in specified columns using winsorization.
    
    Args:
        df: Input dataframe
        columns: List of column names to process
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        logger: Logger instance
        
    Returns:
        DataFrame with capped outliers
    """
    df_capped = df.copy()
    
    for col in columns:
        if method == 'iqr':
            lower, upper, n_outliers = get_outlier_bounds_iqr(df[col], threshold)
            
            if logger and n_outliers > 0:
                logger.info(f"Capping {n_outliers} outliers in {col}: "
                          f"bounds=[{lower:.2f}, {upper:.2f}]")
            
            df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
    
    return df_capped


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_absolute_percentage_error
    )
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }


def save_results(
    results: Dict[str, Any],
    filepath: str,
    logger: logging.Logger = None
) -> None:
    """
    Save results to CSV file.
    
    Args:
        results: Dictionary or DataFrame of results
        filepath: Path to save file
        logger: Logger instance
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, dict):
        results = pd.DataFrame([results])
    
    results.to_csv(filepath, index=False)
    
    if logger:
        logger.info(f"Results saved to {filepath}")


def _resolve_project_path(path_str: str) -> Path:
    """
    Resolve paths relative to the project root (directory containing pyproject.toml),
    falling back to current working directory.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    
    # Try to find project root by looking for pyproject.toml upwards
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
            candidate = parent / path
            return candidate
    
    return cwd / path


def load_data(
    filepath: str,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        logger: Logger instance
        
    Returns:
        Loaded DataFrame
    """
    resolved = _resolve_project_path(filepath)
    df = pd.read_csv(resolved)
    
    if logger:
        logger.info(f"Loaded data from {resolved}: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def print_section(title: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str, width: int = 80) -> None:
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)

