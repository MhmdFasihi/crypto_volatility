# tests/test_preprocessing.py
"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    calculate_returns,
    calculate_volatility,
    calculate_advanced_volatility_metrics,
    calculate_volatility_ratio,
    calculate_iv_rv_spread,
    preprocess_for_modeling,
    create_volatility_features,
    normalize_features
)

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = pd.DataFrame({
        'Close': [100, 105, 103, 108, 107, 110, 108, 112, 115, 113],
        'Open': [98, 104, 105, 107, 108, 109, 109, 110, 113, 114],
        'High': [101, 106, 106, 109, 110, 112, 111, 114, 116, 115],
        'Low': [97, 103, 102, 106, 106, 108, 107, 109, 112, 112]
    }, index=dates)
    return data

@pytest.fixture
def sample_volatility_data():
    """Create sample data with volatility."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = pd.DataFrame({
        'Volatility': [0.20, 0.22, 0.19, 0.21, 0.23, 0.25, 0.24, 0.22, 0.20, 0.18],
        'ImpliedVolatility': [0.22, 0.24, 0.21, 0.23, 0.25, 0.27, 0.26, 0.24, 0.22, 0.20]
    }, index=dates)
    return data

def test_calculate_returns(sample_price_data):
    """Test returns calculation."""
    result = calculate_returns(sample_price_data)
    
    assert 'Returns' in result.columns
    assert len(result) == len(sample_price_data)
    assert result['Returns'].iloc[0] != result['Returns'].iloc[0]  # First value should be NaN
    assert not result['Returns'].iloc[1:].isna().any()  # No NaN after first value
    
    # Check some calculated values
    expected_return = np.log(105/100)
    np.testing.assert_almost_equal(result['Returns'].iloc[1], expected_return, decimal=6)

def test_calculate_volatility(sample_price_data):
    """Test volatility calculation."""
    data_with_returns = calculate_returns(sample_price_data)
    result = calculate_volatility(data_with_returns, window=3)
    
    assert 'Volatility' in result.columns
    assert len(result) == len(data_with_returns)
    assert result['Volatility'].iloc[:2].isna().all()  # First window-1 values should be NaN
    assert not result['Volatility'].iloc[2:].isna().any()  # No NaN after window

def test_calculate_advanced_volatility_metrics(sample_price_data):
    """Test advanced volatility metrics calculation."""
    result = calculate_advanced_volatility_metrics(sample_price_data, window=3)
    
    # Check that all expected columns are present
    expected_columns = ['Parkinson_Vol', 'GK_Vol', 'RS_Vol']
    for col in expected_columns:
        assert col in result.columns
    
    # Check that values are calculated
    for col in expected_columns:
        assert not result[col].iloc[2:].isna().all()  # Should have some non-NaN values

def test_calculate_volatility_ratio(sample_price_data):
    """Test volatility ratio calculation."""
    data_with_returns = calculate_returns(sample_price_data)
    result = calculate_volatility_ratio(data_with_returns, short_window=2, long_window=4)
    
    assert 'Vol_Ratio' in result.columns
    assert len(result) == len(data_with_returns)
    # Should have NaN values where we don't have enough data
    assert result['Vol_Ratio'].iloc[:3].isna().all()

def test_calculate_iv_rv_spread(sample_volatility_data):
    """Test IV-RV spread calculation."""
    result = calculate_iv_rv_spread(sample_volatility_data)
    
    assert 'IV_RV_Spread' in result.columns
    assert 'IV_RV_Ratio' in result.columns
    
    # Check calculations
    expected_spread = sample_volatility_data['ImpliedVolatility'].iloc[0] - sample_volatility_data['Volatility'].iloc[0]
    np.testing.assert_almost_equal(result['IV_RV_Spread'].iloc[0], expected_spread, decimal=6)

def test_preprocess_for_modeling():
    """Test data preprocessing for modeling."""
    # Create sample data with some NaN values
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = pd.DataFrame({
        'Volatility': [0.20, np.nan, 0.19, 0.21, 0.23, 0.25, 0.24, np.nan, 0.20, 0.18],
        'Feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        'Feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }, index=dates)
    
    result = preprocess_for_modeling(data)
    
    # Check that NaN values are handled
    assert not result.isna().any().any()
    assert len(result) <= len(data)  # Some rows may be dropped

def test_create_volatility_features(sample_volatility_data):
    """Test volatility feature creation."""
    result = create_volatility_features(sample_volatility_data, lags=3)
    
    # Check lagged features
    for i in range(1, 4):
        assert f'Vol_Lag_{i}' in result.columns
    
    # Check moving averages
    for window in [5, 10, 20]:
        assert f'Vol_MA_{window}' in result.columns
        assert f'Vol_EMA_{window}' in result.columns

def test_normalize_features():
    """Test feature normalization."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Feature2': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }, index=dates)
    
    # Test z-score normalization
    result_zscore = normalize_features(data, ['Feature1', 'Feature2'], method='zscore')
    assert 'Feature1_norm' in result_zscore.columns
    assert 'Feature2_norm' in result_zscore.columns
    
    # Check that normalized values have mean ~0 and std ~1
    np.testing.assert_almost_equal(result_zscore['Feature1_norm'].mean(), 0, decimal=6)
    np.testing.assert_almost_equal(result_zscore['Feature1_norm'].std(), 1, decimal=6)
    
    # Test min-max normalization
    result_minmax = normalize_features(data, ['Feature1'], method='minmax')
    assert result_minmax['Feature1_norm'].min() == 0
    assert result_minmax['Feature1_norm'].max() == 1

def test_error_handling():
    """Test error handling in preprocessing functions."""
    # Test with missing column
    data = pd.DataFrame({'NotClose': [1, 2, 3]})
    with pytest.raises(ValueError):
        calculate_returns(data)
    
    # Test with invalid normalization method
    data = pd.DataFrame({'Feature1': [1, 2, 3]})
    with pytest.raises(ValueError):
        normalize_features(data, ['Feature1'], method='invalid')

if __name__ == '__main__':
    pytest.main([__file__])