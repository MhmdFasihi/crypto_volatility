# tests/test_preprocessing.py
"""
Tests for the preprocessing module.
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
    """Test calculate_returns function."""
    result = calculate_returns(sample_price_data, 'Close')
    
    assert isinstance(result, pd.DataFrame)
    assert 'Returns' in result.columns
    assert result['Returns'].isna().sum() == 1  # First row should be NaN
    assert not result['Returns'].iloc[1:].isna().any()
    assert np.allclose(result['Returns'].iloc[1:], 
                      np.log(sample_price_data['Close'].iloc[1:] / 
                            sample_price_data['Close'].iloc[:-1]))

def test_calculate_volatility(sample_price_data):
    """Test calculate_volatility function."""
    data_with_returns = calculate_returns(sample_price_data)
    result = calculate_volatility(data_with_returns, window=3)
    
    assert 'Volatility' in result.columns
    assert len(result) == len(data_with_returns)
    # Start from index 3 (not 2) since window is 3
    assert not result['Volatility'].iloc[3:].isna().any()  # No NaN after window

def test_calculate_advanced_volatility_metrics(sample_price_data):
    """Test calculate_advanced_volatility_metrics function."""
    result = calculate_advanced_volatility_metrics(sample_price_data, window=3)
    
    # Check that all expected columns are present
    expected_columns = ['Parkinson_Vol', 'GK_Vol', 'RS_Vol']
    for col in expected_columns:
        assert col in result.columns
    
    # Check that values are calculated
    for col in expected_columns:
        assert not result[col].iloc[2:].isna().all()  # Should have some non-NaN values

def test_calculate_volatility_ratio(sample_price_data):
    """Test calculate_volatility_ratio function."""
    data_with_returns = calculate_returns(sample_price_data)
    result = calculate_volatility_ratio(data_with_returns, short_window=2, long_window=4)
    
    assert 'Vol_Ratio' in result.columns
    assert len(result) == len(data_with_returns)
    # Should have NaN values where we don't have enough data
    assert result['Vol_Ratio'].iloc[:3].isna().all()

def test_calculate_iv_rv_spread(sample_volatility_data):
    """Test calculate_iv_rv_spread function."""
    result = calculate_iv_rv_spread(sample_volatility_data, 'ImpliedVolatility', 'Volatility')
    
    assert 'IV_RV_Spread' in result.columns
    assert 'IV_RV_Ratio' in result.columns
    
    # Check calculations
    expected_spread = sample_volatility_data['ImpliedVolatility'].iloc[0] - sample_volatility_data['Volatility'].iloc[0]
    np.testing.assert_almost_equal(result['IV_RV_Spread'].iloc[0], expected_spread, decimal=6)

def test_preprocess_for_modeling(sample_price_data, sample_volatility_data):
    """Test preprocess_for_modeling function."""
    result = preprocess_for_modeling(sample_price_data, sample_volatility_data)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['Returns', 'Volatility', 'ImpliedVolatility'])
    assert result.index.is_monotonic_increasing
    assert not result.isna().any()

def test_create_volatility_features(sample_volatility_data):
    """Test create_volatility_features function."""
    result = create_volatility_features(sample_volatility_data)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['Volatility', 'ImpliedVolatility', 'VolatilityRatio', 'IV_RV_Spread'])
    assert not result.isna().any()

def test_normalize_features(sample_volatility_data):
    """Test normalize_features function."""
    result = normalize_features(sample_volatility_data)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['Volatility', 'ImpliedVolatility'])
    assert not result.isna().any()
    assert all(result[col].between(0, 1).all() for col in ['Volatility', 'ImpliedVolatility'])

def test_calculate_returns_invalid_input():
    """Test calculate_returns with invalid input."""
    with pytest.raises(ValueError):
        calculate_returns(pd.DataFrame(), 'Close')
    
    with pytest.raises(ValueError):
        calculate_returns(pd.DataFrame({'Open': [1, 2, 3]}), 'Close')

def test_calculate_volatility_invalid_input():
    """Test calculate_volatility with invalid input."""
    with pytest.raises(ValueError):
        calculate_volatility(pd.DataFrame(), 'Returns')
    
    with pytest.raises(ValueError):
        calculate_volatility(pd.DataFrame({'Open': [1, 2, 3]}), 'Returns')

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