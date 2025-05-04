"""
Data preprocessing module for calculating returns and volatility metrics.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any
import logging
from pathlib import Path
from .config import config
from .logger import get_logger

logger = get_logger(__name__)

def calculate_returns(data: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate log returns from closing prices.
    
    Args:
        data: DataFrame with price data
        price_col: Column name for prices (default: 'Close')
    
    Returns:
        DataFrame with 'Returns' column added
    
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to calculate_returns")
        return pd.DataFrame()
    
    if price_col not in data.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    data = data.copy()
    
    # Ensure price data is numeric
    try:
        data[price_col] = data[price_col].astype(float)
    except Exception as e:
        logger.warning(f"Error converting {price_col} to float: {str(e)}")
        data[price_col] = data[price_col].replace('', np.nan).astype(float)
    
    # Calculate log returns
    data['Returns'] = np.log(data[price_col] / data[price_col].shift(1))
    
    logger.info(f"Calculated returns. NaN values: {data['Returns'].isna().sum()}")
    return data

def calculate_volatility(data: pd.DataFrame, window: Optional[int] = None, 
                        returns_col: str = 'Returns') -> pd.DataFrame:
    """
    Calculate realized volatility as rolling standard deviation of returns.
    
    Args:
        data: DataFrame with returns data
        window: Rolling window size (default: from config)
        returns_col: Column name for returns
    
    Returns:
        DataFrame with 'Volatility' column added
    
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to calculate_volatility")
        return pd.DataFrame()
    
    if returns_col not in data.columns:
        raise ValueError(f"Column '{returns_col}' not found in DataFrame")
    
    data = data.copy()
    window = window or config.get('vol_window')
    
    # Ensure returns are numeric
    try:
        data[returns_col] = data[returns_col].astype(float)
    except Exception as e:
        logger.warning(f"Error converting {returns_col} to float: {str(e)}")
        data[returns_col] = data[returns_col].replace('', np.nan).astype(float)
    
    # Calculate rolling volatility and annualize
    ann_factor = float(config.get('annualization_factor'))
    data['Volatility'] = data[returns_col].rolling(window=int(window)).std() * np.sqrt(ann_factor)
    
    logger.info(f"Calculated volatility with window={window}. NaN values: {data['Volatility'].isna().sum()}")
    return data

def calculate_advanced_volatility_metrics(data: pd.DataFrame, window: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate advanced volatility metrics including Parkinson, Garman-Klass, etc.
    
    Args:
        data: DataFrame with OHLC data
        window: Rolling window size (default: from config)
    
    Returns:
        DataFrame with additional volatility metrics
    
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to calculate_advanced_volatility_metrics")
        return pd.DataFrame()
    
    data = data.copy()
    window = window or config.get('vol_window')
    
    # Ensure OHLC data is numeric
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in data.columns:
            try:
                data[col] = data[col].astype(float)
            except Exception as e:
                logger.warning(f"Error converting {col} to float: {str(e)}")
                data[col] = data[col].replace('', np.nan).astype(float)
    
    # Parkinson volatility (using High-Low range)
    if all(col in data.columns for col in ['High', 'Low']):
        data['Parkinson_Vol'] = np.sqrt(
            1 / (4 * np.log(2)) * 
            ((np.log(data['High'] / data['Low']) ** 2).rolling(window=int(window)).mean())
        ) * np.sqrt(float(config.get('annualization_factor')))
        logger.info("Calculated Parkinson volatility")
    
    # Garman-Klass volatility
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        data['GK_Vol'] = np.sqrt(
            (0.5 * (np.log(data['High'] / data['Low']) ** 2) -
             (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open']) ** 2)).rolling(window=int(window)).mean()
        ) * np.sqrt(float(config.get('annualization_factor')))
        logger.info("Calculated Garman-Klass volatility")
    
    # Rogers-Satchell volatility
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        try:
            hl = np.log(data['High'] / data['Low'])
            co = np.log(data['Close'] / data['Open'])
            oc = np.log(data['Open'] / data['Close'])
            
            data['RS_Vol'] = np.sqrt(
                (hl * (hl - co) + co * (co - oc)).rolling(window=int(window)).mean()
            ) * np.sqrt(float(config.get('annualization_factor')))
            logger.info("Calculated Rogers-Satchell volatility")
        except Exception as e:
            logger.error(f"Error calculating Rogers-Satchell volatility: {str(e)}")
    
    return data

def calculate_volatility_ratio(data: pd.DataFrame, short_window: int = 10, 
                             long_window: int = 30) -> pd.DataFrame:
    """
    Calculate volatility ratio (short-term vol / long-term vol).
    
    Args:
        data: DataFrame with returns
        short_window: Short-term window
        long_window: Long-term window
    
    Returns:
        DataFrame with volatility ratio
    
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to calculate_volatility_ratio")
        return pd.DataFrame()
    
    data = data.copy()
    
    if 'Returns' not in data.columns:
        if 'Close' in data.columns:
            data = calculate_returns(data)
        else:
            raise ValueError("Required columns missing for volatility ratio calculation")
    
    # Ensure returns are numeric
    data['Returns'] = pd.to_numeric(data['Returns'], errors='coerce')
    
    # Calculate short and long-term volatility
    ann_factor = float(config.get('annualization_factor'))
    short_vol = data['Returns'].rolling(window=int(short_window)).std() * np.sqrt(ann_factor)
    long_vol = data['Returns'].rolling(window=int(long_window)).std() * np.sqrt(ann_factor)
    
    # Calculate ratio - handle division by zero
    data['Vol_Ratio'] = short_vol / long_vol.replace(0, np.nan)
    
    logger.info(f"Calculated volatility ratio ({short_window}/{long_window})")
    return data

def calculate_iv_rv_spread(data: pd.DataFrame, iv_col: str = 'ImpliedVolatility', 
                          rv_col: str = 'Volatility') -> pd.DataFrame:
    """
    Calculate spread between implied and realized volatility.
    
    Args:
        data: DataFrame with IV and RV
        iv_col: Column name for implied volatility
        rv_col: Column name for realized volatility
    
    Returns:
        DataFrame with IV-RV spread
    
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to calculate_iv_rv_spread")
        return pd.DataFrame()
    
    data = data.copy()
    
    if iv_col in data.columns and rv_col in data.columns:
        # Ensure both columns are numeric
        data[iv_col] = pd.to_numeric(data[iv_col], errors='coerce')
        data[rv_col] = pd.to_numeric(data[rv_col], errors='coerce')
        
        # Calculate spread and ratio
        data['IV_RV_Spread'] = data[iv_col] - data[rv_col]
        data['IV_RV_Ratio'] = data[iv_col] / data[rv_col].replace(0, np.nan)
        
        logger.info("Calculated IV-RV spread and ratio")
    else:
        logger.warning(f"Missing columns for IV-RV spread calculation: {iv_col} or {rv_col}")
    
    return data

def preprocess_for_modeling(data: pd.DataFrame, target_col: str = 'Volatility', 
                           features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Preprocess data for machine learning models.
    
    Args:
        data: DataFrame with all features
        target_col: Target variable column
        features: List of feature columns to include
    
    Returns:
        Preprocessed DataFrame ready for modeling
    
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to preprocess_for_modeling")
        return pd.DataFrame()
    
    data = data.copy()
    
    # Ensure target column exists
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Remove rows with NaN in target
    data = data.dropna(subset=[target_col])
    
    # Select features
    if features is None:
        # Default features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != target_col]
    
    # Ensure all columns are numeric
    for col in features + [target_col]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Forward fill then backward fill remaining NaNs
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Preprocessed data for modeling. Shape: {data.shape}")
    return data

def create_volatility_features(data: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Create lagged and moving average features for volatility.
    
    Args:
        data: DataFrame with volatility data
        lags: Number of lagged features to create
    
    Returns:
        DataFrame with additional features
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to create_volatility_features")
        return pd.DataFrame()
    
    data = data.copy()
    
    # Create lagged features
    for i in range(1, lags + 1):
        data[f'Vol_Lag_{i}'] = data['Volatility'].shift(i)
    
    # Create moving averages
    for window in [5, 10, 20]:
        data[f'Vol_MA_{window}'] = data['Volatility'].rolling(window=window).mean()
        data[f'Vol_EMA_{window}'] = data['Volatility'].ewm(span=window).mean()
    
    logger.info(f"Created volatility features with {lags} lags")
    return data

def normalize_features(data: pd.DataFrame, features: List[str], method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize features using specified method.
    
    Args:
        data: DataFrame with features
        features: List of feature columns to normalize
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        DataFrame with normalized features
    
    Raises:
        ValueError: If method is invalid or features are missing
    """
    if data is None or data.empty:
        logger.warning("Empty data provided to normalize_features")
        return pd.DataFrame()
    
    data = data.copy()
    
    # Validate features
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Features not found in DataFrame: {missing_features}")
    
    # Normalize features
    for feature in features:
        if method == 'zscore':
            mean = data[feature].mean()
            std = data[feature].std()
            data[f'{feature}_norm'] = (data[feature] - mean) / std
        elif method == 'minmax':
            min_val = data[feature].min()
            max_val = data[feature].max()
            data[f'{feature}_norm'] = (data[feature] - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Invalid normalization method: {method}")
    
    logger.info(f"Normalized {len(features)} features using {method} method")
    return data