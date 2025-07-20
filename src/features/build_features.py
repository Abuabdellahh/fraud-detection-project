"""
Feature engineering for fraud detection.

This module provides functions to create and select features that are
particularly relevant for detecting fraudulent transactions.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from transaction timestamps.
    
    Args:
        df: DataFrame containing transaction data with timestamp columns.
        
    Returns:
        pd.DataFrame: DataFrame with added time-based features.
    """
    logger.info("Creating time-based features...")
    
    df_features = df.copy()
    
    # Time since first transaction (in hours)
    if 'purchase_time' in df_features.columns:
        df_features['hours_since_first_purchase'] = (
            df_features['purchase_time'] - df_features['purchase_time'].min()
        ).dt.total_seconds() / 3600
    
    # Time of day features (sine/cosine encoding for cyclical nature)
    if 'purchase_hour' in df_features.columns:
        df_features['purchase_hour_sin'] = np.sin(2 * np.pi * df_features['purchase_hour']/24)
        df_features['purchase_hour_cos'] = np.cos(2 * np.pi * df_features['purchase_hour']/24)
    
    # Day of week features
    if 'purchase_dayofweek' in df_features.columns:
        df_features['purchase_dayofweek_sin'] = np.sin(2 * np.pi * df_features['purchase_dayofweek']/7)
        df_features['purchase_dayofweek_cos'] = np.cos(2 * np.pi * df_features['purchase_dayofweek']/7)
    
    # Weekend flag
    if 'purchase_dayofweek' in df_features.columns:
        df_features['is_weekend'] = df_features['purchase_dayofweek'].isin([5, 6]).astype(int)
    
    # Time since last transaction (per user)
    if 'user_id' in df_features.columns and 'purchase_time' in df_features.columns:
        df_features = df_features.sort_values(['user_id', 'purchase_time'])
        df_features['time_since_last_txn'] = df_features.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 60  # in minutes
        
        # Fill NA values (first transaction for each user) with a large value
        df_features['time_since_last_txn'] = df_features['time_since_last_txn'].fillna(24*60)  # 24 hours in minutes
    
    return df_features

def create_aggregated_features(df: pd.DataFrame, window: str = '24H') -> pd.DataFrame:
    """
    Create aggregated features over time windows.
    
    Args:
        df: DataFrame containing transaction data.
        window: Time window for aggregation (e.g., '24H' for 24 hours).
        
    Returns:
        pd.DataFrame: DataFrame with added aggregated features.
    """
    logger.info(f"Creating aggregated features with {window} window...")
    
    if 'purchase_time' not in df.columns or 'user_id' not in df.columns:
        logger.warning("Required columns not found for aggregation. Skipping...")
        return df
    
    df_agg = df.copy()
    df_agg = df_agg.sort_values('purchase_time')
    
    # Set index for time-based operations
    df_temp = df_agg.set_index('purchase_time')
    
    # User-level aggregations
    if 'purchase_value' in df_agg.columns:
        # Rolling window aggregations
        user_roll = df_temp.groupby('user_id')['purchase_value'].rolling(
            window=window, min_periods=1
        ).agg({
            'txn_count': 'count',
            'total_spend': 'sum',
            'avg_spend': 'mean',
            'std_spend': 'std',
            'min_spend': 'min',
            'max_spend': 'max',
            'median_spend': 'median',
            'spend_skew': lambda x: x.skew() if len(x) > 2 else 0
        }).reset_index()
        
        # Calculate time since last transaction
        user_roll['time_since_last_txn'] = user_roll.groupby('user_id')['purchase_time'].diff().dt.total_seconds().fillna(0)
        
        # Merge back with original data
        df_agg = df_agg.merge(
            user_roll.drop(columns=['purchase_value']),
            on=['user_id', 'purchase_time'],
            how='left'
        )
    
    return df_agg

def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral features for fraud detection.
    
    Args:
        df: DataFrame containing transaction data.
        
    Returns:
        pd.DataFrame: DataFrame with added behavioral features.
    """
    logger.info("Creating behavioral features...")
    
    df_behavior = df.copy()
    
    # Device usage patterns
    if 'device_id' in df_behavior.columns and 'user_id' in df_behavior.columns:
        # Number of unique devices per user (potential account takeover)
        devices_per_user = df_behavior.groupby('user_id')['device_id'].nunique().reset_index()
        devices_per_user.columns = ['user_id', 'unique_devices_per_user']
        df_behavior = df_behavior.merge(devices_per_user, on='user_id', how='left')
        
        # Flag for new device usage
        user_device_history = df_behavior.groupby('user_id')['device_id'].apply(set).to_dict()
        df_behavior['is_new_device'] = df_behavior.apply(
            lambda x: 1 if x['device_id'] not in user_device_history.get(x['user_id'], set()) else 0,
            axis=1
        )
    
    # Purchase velocity (transactions per time period)
    if 'purchase_time' in df_behavior.columns and 'user_id' in df_behavior.columns:
        df_behavior = df_behavior.sort_values(['user_id', 'purchase_time'])
        
        # Time since last transaction (in minutes)
        df_behavior['time_since_last_txn'] = df_behavior.groupby('user_id')['purchase_time'].diff().dt.total_seconds().fillna(0) / 60
        
        # Flag for unusually quick subsequent transactions
        df_behavior['rapid_transaction'] = (df_behavior['time_since_last_txn'] < 1).astype(int)  # Less than 1 minute
    
    # Purchase amount anomalies
    if 'purchase_value' in df_behavior.columns and 'user_id' in df_behavior.columns:
        # User's typical purchase amount statistics
        user_stats = df_behavior.groupby('user_id')['purchase_value'].agg([
            'mean', 'std', 'median', 'min', 'max'
        ]).add_prefix('user_purchase_').reset_index()
        
        df_behavior = df_behavior.merge(user_stats, on='user_id', how='left')
        
        # Flag for unusually large purchases
        df_behavior['is_large_purchase'] = (
            df_behavior['purchase_value'] > 
            (df_behavior['user_purchase_median'] + 3 * df_behavior['user_purchase_std'])
        ).astype(int)
        
        # Relative purchase amount
        df_behavior['purchase_ratio_to_avg'] = (
            df_behavior['purchase_value'] / df_behavior['user_purchase_mean']
        ).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df_behavior

def select_features(
    df: pd.DataFrame, 
    target_col: str = 'class',
    exclude_cols: List[str] = None,
    include_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features for model training.
    
    Args:
        df: DataFrame containing features and target.
        target_col: Name of the target column.
        exclude_cols: Columns to exclude from features.
        include_cols: Specific columns to include (if None, include all except excluded).
        
    Returns:
        Tuple containing:
            - DataFrame of features
            - Series of target values
    """
    logger.info("Selecting features for modeling...")
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Always exclude non-feature columns
    default_exclude = [
        'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address',
        'lower_bound_ip_address', 'upper_bound_ip_address', 'country'
    ]
    
    exclude_cols = list(set(exclude_cols + default_exclude))
    
    # Get target variable
    y = df[target_col].copy()
    
    # Select features
    if include_cols is not None:
        # Use only specified columns (minus any that should be excluded)
        features = [col for col in include_cols if col not in exclude_cols and col in df.columns]
    else:
        # Use all columns except excluded ones and target
        features = [col for col in df.columns if col not in exclude_cols + [target_col]]
    
    X = df[features].copy()
    
    # Convert categorical variables to dummy/indicator variables
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    logger.info(f"Selected {X.shape[1]} features for modeling.")
    return X, y

def create_features(
    df: pd.DataFrame,
    target_col: str = 'class',
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create all features for the fraud detection model.
    
    Args:
        df: Raw transaction data.
        target_col: Name of the target column.
        exclude_cols: Columns to exclude from features.
        
    Returns:
        Tuple containing:
            - DataFrame of features
            - Series of target values
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Create a copy to avoid modifying the original dataframe
    df_features = df.copy()
    
    # 1. Time-based features
    df_features = create_time_based_features(df_features)
    
    # 2. Aggregated features
    df_features = create_aggregated_features(df_features)
    
    # 3. Behavioral features
    df_features = create_behavioral_features(df_features)
    
    # 4. Select final features for modeling
    X, y = select_features(df_features, target_col=target_col, exclude_cols=exclude_cols)
    
    logger.info("Feature engineering completed successfully.")
    return X, y
