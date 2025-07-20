"""
Data preprocessing utilities for fraud detection.

This module provides functions to clean, validate, and preprocess transaction data
for fraud detection analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', **kwargs) -> pd.DataFrame:
    """
    Handle missing values in the dataframe based on the specified strategy.
    
    Args:
        df: Input DataFrame with potential missing values.
        strategy: Strategy to handle missing values. Options:
                 - 'drop': Drop rows with missing values
                 - 'fill': Fill missing values (requires 'fill_values' parameter)
                 - 'interpolate': Use interpolation for numeric columns
        **kwargs: Additional arguments for the chosen strategy.
                 For 'fill' strategy, provide 'fill_values' dict {column: value}
                 
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df_clean = df.copy()
    
    # Check for missing values
    missing = df_clean.isnull().sum()
    missing = missing[missing > 0]
    
    if missing.empty:
        logger.info("No missing values found in the dataset.")
        return df_clean
    
    logger.warning(f"Found missing values in columns: {', '.join(missing.index)}")
    
    # Apply the specified strategy
    if strategy == 'drop':
        logger.info("Dropping rows with missing values.")
        df_clean = df_clean.dropna()
        
    elif strategy == 'fill':
        if 'fill_values' not in kwargs:
            raise ValueError("'fill_values' must be provided for 'fill' strategy")
        
        fill_values = kwargs['fill_values']
        logger.info(f"Filling missing values with: {fill_values}")
        df_clean = df_clean.fillna(fill_values)
        
    elif strategy == 'interpolate':
        logger.info("Interpolating missing values for numeric columns.")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(**kwargs)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    logger.info(f"Missing values handled. Remaining missing values: {df_clean.isnull().sum().sum()}")
    return df_clean

def preprocess_data(
    df: pd.DataFrame,
    ip_mapping: Optional[pd.DataFrame] = None,
    drop_duplicates: bool = True,
    convert_dtypes: bool = True,
    **missing_value_kwargs
) -> pd.DataFrame:
    """
    Preprocess transaction data for fraud detection.
    
    Args:
        df: Raw transaction DataFrame.
        ip_mapping: Optional DataFrame for IP to country mapping.
        drop_duplicates: Whether to drop duplicate rows.
        convert_dtypes: Whether to convert data types automatically.
        **missing_value_kwargs: Arguments passed to handle_missing_values.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for analysis.
    """
    logger.info("Starting data preprocessing...")
    df_processed = df.copy()
    
    # 1. Handle missing values
    df_processed = handle_missing_values(df_processed, **missing_value_kwargs)
    
    # 2. Convert data types
    if convert_dtypes:
        # Convert to more memory-efficient types
        type_conversions = {
            'user_id': 'category',
            'device_id': 'category',
            'source': 'category',
            'browser': 'category',
            'sex': 'category',
            'age': 'int8',
            'purchase_value': 'float32',
            'class': 'int8'
        }
        
        for col, dtype in type_conversions.items():
            if col in df_processed.columns:
                try:
                    df_processed[col] = df_processed[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
    
    # 3. Map IP addresses to countries if mapping is provided
    if ip_mapping is not None and 'ip_address' in df_processed.columns:
        logger.info("Mapping IP addresses to countries...")
        from .load_data import map_ip_to_country
        
        df_processed['country'] = df_processed['ip_address'].apply(
            lambda x: map_ip_to_country(x, ip_mapping)
        )
        
        # Convert country to category type
        df_processed['country'] = df_processed['country'].astype('category')
    
    # 4. Extract datetime features
    datetime_cols = ['signup_time', 'purchase_time']
    for col in datetime_cols:
        if col in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed[col]):
            # Extract time-based features
            prefix = col.replace('_time', '')
            df_processed[f'{prefix}_hour'] = df_processed[col].dt.hour.astype('int8')
            df_processed[f'{prefix}_dayofweek'] = df_processed[col].dt.dayofweek.astype('int8')
            df_processed[f'{prefix}_dayofmonth'] = df_processed[col].dt.day.astype('int8')
            df_processed[f'{prefix}_month'] = df_processed[col].dt.month.astype('int8')
    
    # 5. Calculate time since signup
    if all(col in df_processed.columns for col in ['signup_time', 'purchase_time']):
        df_processed['time_since_signup'] = (
            df_processed['purchase_time'] - df_processed['signup_time']
        ).dt.total_seconds() / 60  # in minutes
        
        # Convert to float32 to save memory
        df_processed['time_since_signup'] = df_processed['time_since_signup'].astype('float32')
    
    # 6. Drop duplicates if requested
    if drop_duplicates:
        initial_rows = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        removed = initial_rows - len(df_processed)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows.")
    
    logger.info("Data preprocessing completed successfully.")
    return df_processed

def detect_outliers(
    df: pd.DataFrame, 
    column: str, 
    method: str = 'iqr', 
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a numeric column.
    
    Args:
        df: Input DataFrame.
        column: Name of the column to analyze.
        method: Method for outlier detection ('iqr' or 'zscore').
        threshold: Threshold for outlier detection.
        
    Returns:
        pd.Series: Boolean mask of outlier rows.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")
    
    if method == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return ~df[column].between(lower_bound, upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return (z_scores.abs() > threshold)
        
    else:
        raise ValueError(f"Unknown method: {method}")
