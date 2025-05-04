"""
Tests for the data acquisition module.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.data_acquisition import (
    get_data,
    get_multiple_tickers,
    validate_ticker,
    get_ticker_info,
    save_data_to_csv,
    load_data_from_csv,
    DeribitAPI
)

def test_get_data(mock_api_response):
    """Test get_data function."""
    with patch('yfinance.Ticker') as mock_ticker:
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [98, 99, 100],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2024-01-01', periods=3))
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function
        result = get_data('BTC-USD', '2024-01-01', '2024-01-03')
        
        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        assert len(result) == 3
        assert not result.isna().any()

def test_get_multiple_tickers(mock_api_response):
    """Test get_multiple_tickers function."""
    with patch('yfinance.Ticker') as mock_ticker:
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [98, 99, 100],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2024-01-01', periods=3))
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function
        tickers = ['BTC-USD', 'ETH-USD']
        result = get_multiple_tickers(tickers, '2024-01-01', '2024-01-03')
        
        assert isinstance(result, dict)
        assert all(ticker in result for ticker in tickers)
        assert all(isinstance(df, pd.DataFrame) for df in result.values())
        assert all(len(df) == 3 for df in result.values())

def test_validate_ticker():
    """Test validate_ticker function."""
    # Test valid ticker
    assert validate_ticker('BTC-USD') is True
    
    # Test invalid ticker
    assert validate_ticker('INVALID-TICKER') is False

def test_get_ticker_info(mock_api_response):
    """Test get_ticker_info function."""
    with patch('yfinance.Ticker') as mock_ticker:
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            'symbol': 'BTC-USD',
            'shortName': 'Bitcoin',
            'marketCap': 1000000000,
            'volume': 1000000
        }
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function
        result = get_ticker_info('BTC-USD')
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['symbol', 'shortName', 'marketCap', 'volume'])

def test_save_and_load_data(test_data_dir, sample_price_data):
    """Test save_data_to_csv and load_data_from_csv functions."""
    # Test saving data
    file_path = test_data_dir / 'test_data.csv'
    save_data_to_csv(sample_price_data, file_path)
    
    # Test loading data
    loaded_data = load_data_from_csv(file_path)
    
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.equals(sample_price_data)
    assert not loaded_data.isna().any()

def test_deribit_api(mock_api_response):
    """Test DeribitAPI class."""
    with patch('requests.get') as mock_get:
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_get.return_value = mock_response
        
        # Initialize API
        api = DeribitAPI()
        
        # Test get_trades_by_currency_and_time
        result = api.get_trades_by_currency_and_time('BTC', '2024-01-01', '2024-01-03')
        
        assert isinstance(result, dict)
        assert 'result' in result
        assert 'trades' in result['result']
        
        # Test get_implied_volatility_data
        result = api.get_implied_volatility_data('BTC', '2024-01-01', '2024-01-03')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ['timestamp', 'iv', 'price'])

def test_error_handling():
    """Test error handling in data acquisition functions."""
    # Test get_data with invalid dates
    with pytest.raises(ValueError):
        get_data('BTC-USD', '2024-01-03', '2024-01-01')  # End date before start date
    
    # Test get_multiple_tickers with empty list
    with pytest.raises(ValueError):
        get_multiple_tickers([], '2024-01-01', '2024-01-03')
    
    # Test save_data_to_csv with invalid path
    with pytest.raises(ValueError):
        save_data_to_csv(pd.DataFrame(), '/invalid/path/data.csv')
    
    # Test load_data_from_csv with non-existent file
    with pytest.raises(FileNotFoundError):
        load_data_from_csv('non_existent_file.csv')

def test_retry_mechanism():
    """Test retry mechanism in data acquisition functions."""
    with patch('yfinance.Ticker') as mock_ticker:
        # Setup mock to fail twice then succeed
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = [
            Exception("API Error"),
            Exception("API Error"),
            pd.DataFrame({
                'Open': [100],
                'High': [102],
                'Low': [98],
                'Close': [101],
                'Volume': [1000]
            }, index=pd.date_range(start='2024-01-01', periods=1))
        ]
        mock_ticker.return_value = mock_ticker_instance
        
        # Test function with retry
        result = get_data('BTC-USD', '2024-01-01', '2024-01-01')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert not result.isna().any() 