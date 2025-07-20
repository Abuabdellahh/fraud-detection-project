"""
Data loading utilities for fraud detection project.

This module provides functions to load and validate the transaction and IP mapping data.
"""

import os
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_transaction_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate transaction data from a CSV file.
    
    Args:
        file_path: Path to the transaction data CSV file.
        
    Returns:
        pd.DataFrame: Loaded and validated transaction data.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or missing required columns.
    """
    logger.info(f"Loading transaction data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transaction data file not found: {file_path}")
    
    # Define expected columns for validation
    expected_columns = [
        'user_id', 'signup_time', 'purchase_time', 'purchase_value',
        'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class'
    ]
    
    try:
        # Load the data with proper date parsing
        df = pd.read_csv(
            file_path,
            parse_dates=['signup_time', 'purchase_time'],
            infer_datetime_format=True
        )
        
        # Validate required columns
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        logger.info(f"Successfully loaded {len(df)} transactions")
        return df
        
    except Exception as e:
        logger.error(f"Error loading transaction data: {str(e)}")
        raise

def load_ip_mapping(file_path: str) -> pd.DataFrame:
    """
    Load and validate IP to country mapping data.
    
    Args:
        file_path: Path to the IP mapping CSV file.
        
    Returns:
        pd.DataFrame: Loaded and validated IP mapping data.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or missing required columns.
    """
    logger.info(f"Loading IP mapping data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"IP mapping file not found: {file_path}")
    
    expected_columns = [
        'lower_bound_ip_address', 'upper_bound_ip_address', 'country'
    ]
    
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert IP addresses to numeric for range queries
        for col in ['lower_bound_ip_address', 'upper_bound_ip_address']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype('float64')
                
        logger.info(f"Successfully loaded IP mappings for {len(df)} IP ranges")
        return df
        
    except Exception as e:
        logger.error(f"Error loading IP mapping data: {str(e)}")
        raise

def map_ip_to_country(ip_address: str, ip_mapping: pd.DataFrame) -> str:
    """
    Map an IP address to a country using the provided IP mapping.
    
    Args:
        ip_address: The IP address to map (as string).
        ip_mapping: DataFrame containing IP range to country mappings.
        
    Returns:
        str: Country code for the IP address, or 'Unknown' if not found.
    """
    try:
        # Convert IP to numeric for comparison
        ip_numeric = float(ip_address)
        
        # Find matching country using vectorized operations
        mask = (
            (ip_mapping['lower_bound_ip_address'] <= ip_numeric) & 
            (ip_mapping['upper_bound_ip_address'] >= ip_numeric)
        )
        
        matches = ip_mapping.loc[mask, 'country']
        return matches.iloc[0] if not matches.empty else 'Unknown'
        
    except (ValueError, AttributeError, IndexError):
        return 'Unknown'
