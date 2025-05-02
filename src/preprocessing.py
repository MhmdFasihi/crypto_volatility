# src/preprocessing.py
"""
Data preprocessing module for calculating returns and volatility metrics.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
import logging
from config import VOL_WINDOW, ANNUALIZATION_FACTOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_returns(data: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate log returns from closing prices.
    
    Args:
        data: DataFrame with price data
        price_col: Column name for prices (default: 'Close')
    
    Returns:
        DataFrame with 'Returns' column added
    """
    if price_col not in data.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    data = data.copy()
    data['Returns'] = np.log(data[price_col] / data[price_col].shift(1))
    
    logger.info(f"Calculated returns. NaN values: {data['Returns'].isna().sum()}")
    return data

def calculate_volatility(data: pd.DataFrame, window: int = VOL_WINDOW, 
                        returns_col: str = 'Returns') -> pd.DataFrame:
    """
    Calculate realized volatility as rolling standard deviation of returns.
    
    Args:
        data: DataFrame with returns data
        window: Rolling window size
        returns_col: Column name for returns
    
    Returns:
        DataFrame with 'Volatility' column added
    """
    if returns_col not in data.columns:
        raise ValueError(f"Column '{returns_col}' not found in DataFrame")
    
    data = data.copy()
    # Calculate rolling volatility and annualize
    data['Volatility'] = data[returns_col].rolling(window=window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    
    logger.info(f"Calculated volatility with window={window}. NaN values: {data['Volatility'].isna().sum()}")
    return data

def calculate_advanced_volatility_metrics(data: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Calculate advanced volatility metrics including Parkinson, Garman-Klass, etc.
    
    Args:
        data: DataFrame with OHLC data
        window: Rolling window size
    
    Returns:
        DataFrame with additional volatility metrics
    """
    data = data.copy()
    
    # Parkinson volatility (using High-Low range)
    if all(col in data.columns for col in ['High', 'Low']):
        data['Parkinson_Vol'] = np.sqrt(
            1 / (4 * np.log(2)) * 
            ((np.log(data['High'] / data['Low']) ** 2).rolling(window=window).mean())
        ) * np.sqrt(ANNUALIZATION_FACTOR)
        logger.info("Calculated Parkinson volatility")
    
    # Garman-Klass volatility
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        data['GK_Vol'] = np.sqrt(
            (0.5 * (np.log(data['High'] / data['Low']) ** 2) -
             (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open']) ** 2)).rolling(window=window).mean()
        ) * np.sqrt(ANNUALIZATION_FACTOR)
        logger.info("Calculated Garman-Klass volatility")
    
    # Rogers-Satchell volatility
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        hl = np.log(data['High'] / data['Low'])
        co = np.log(data['Close'] / data['Open'])
        oc = np.log(data['Open'] / data['Close'])
        
        data['RS_Vol'] = np.sqrt(
            (hl * (hl - co) + co * (co - oc)).rolling(window=window).mean()
        ) * np.sqrt(ANNUALIZATION_FACTOR)
        logger.info("Calculated Rogers-Satchell volatility")
    
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
    """
    data = data.copy()
    
    if 'Returns' not in data.columns:
        data = calculate_returns(data)
    
    short_vol = data['Returns'].rolling(window=short_window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    long_vol = data['Returns'].rolling(window=long_window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    
    data['Vol_Ratio'] = short_vol / long_vol
    
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
    """
    data = data.copy()
    
    if iv_col in data.columns and rv_col in data.columns:
        data['IV_RV_Spread'] = data[iv_col] - data[rv_col]
        data['IV_RV_Ratio'] = data[iv_col] / data[rv_col]
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
    """
    data = data.copy()
    
    # Remove rows with NaN in target
    data = data.dropna(subset=[target_col])
    
    # Select features
    if features is None:
        # Default features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != target_col]
    
    # Forward fill then backward fill remaining NaNs
    data[features] = data[features].fillna(method='ffill').fillna(method='bfill')
    
    # Drop rows with any remaining NaNs
    initial_len = len(data)
    data = data.dropna(subset=features + [target_col])
    
    logger.info(f"Preprocessing complete. Dropped {initial_len - len(data)} rows with NaNs")
    
    return data

def create_volatility_features(data: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Create lagged volatility features for forecasting.
    
    Args:
        data: DataFrame with volatility
        lags: Number of lags to create
    
    Returns:
        DataFrame with lagged features
    """
    data = data.copy()
    
    if 'Volatility' not in data.columns:
        raise ValueError("'Volatility' column not found in DataFrame")
    
    for i in range(1, lags + 1):
        data[f'Vol_Lag_{i}'] = data['Volatility'].shift(i)
    
    # Add moving averages
    for window in [5, 10, 20]:
        data[f'Vol_MA_{window}'] = data['Volatility'].rolling(window=window).mean()
    
    # Add exponential moving averages
    for span in [5, 10, 20]:
        data[f'Vol_EMA_{span}'] = data['Volatility'].ewm(span=span, adjust=False).mean()
    
    logger.info(f"Created {lags} lagged features and moving averages")
    
    return data

def normalize_features(data: pd.DataFrame, features: List[str], method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize features for better model performance.
    
    Args:
        data: DataFrame with features
        features: List of features to normalize
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        DataFrame with normalized features
    """
    data = data.copy()
    
    for feature in features:
        if feature in data.columns:
            if method == 'zscore':
                data[f'{feature}_norm'] = (data[feature] - data[feature].mean()) / data[feature].std()
            elif method == 'minmax':
                min_val = data[feature].min()
                max_val = data[feature].max()
                data[f'{feature}_norm'] = (data[feature] - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
    
    logger.info(f"Normalized {len(features)} features using {method} method")
    
    return data