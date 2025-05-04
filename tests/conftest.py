"""
Test configuration and fixtures for the crypto volatility analysis system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.logger import setup_logging

# Set up logging for tests
setup_logging(log_level='INFO', log_dir='tests/logs')

@pytest.fixture(scope='session')
def test_data_dir():
    """Create and return test data directory."""
    data_dir = Path('tests/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

@pytest.fixture(scope='session')
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=dates)
    return data

@pytest.fixture(scope='session')
def sample_volatility_data():
    """Create sample volatility data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = pd.DataFrame({
        'Volatility': [0.20, 0.22, 0.19, 0.21, 0.23, 0.25, 0.24, 0.22, 0.20, 0.18],
        'ImpliedVolatility': [0.22, 0.24, 0.21, 0.23, 0.25, 0.27, 0.26, 0.24, 0.22, 0.20]
    }, index=dates)
    return data

@pytest.fixture(scope='session')
def sample_returns_data(sample_price_data):
    """Create sample returns data for testing."""
    data = sample_price_data.copy()
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    return data

@pytest.fixture(scope='session')
def mock_api_response():
    """Create mock API response data."""
    return {
        'result': {
            'trades': [
                {
                    'instrument_name': 'BTC-31DEC24-50000-C',
                    'price': 50000,
                    'mark_price': 50100,
                    'iv': 0.75,
                    'index_price': 50000,
                    'direction': 'buy',
                    'amount': 1,
                    'timestamp': int(datetime.now().timestamp() * 1000)
                }
            ]
        }
    }

@pytest.fixture(scope='session')
def test_config():
    """Create test configuration."""
    return {
        'data_dir': 'tests/data',
        'models_dir': 'tests/models',
        'logs_dir': 'tests/logs',
        'vol_window': 5,
        'annualization_factor': 365,
        'api': {
            'yfinance': {
                'rate_limit': 100,
                'retry_attempts': 2,
                'retry_delay': 1
            },
            'deribit': {
                'test_mode': True,
                'rate_limit': 50,
                'retry_attempts': 2,
                'retry_delay': 1
            }
        }
    }

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, test_config):
    """Set up test environment variables."""
    # Set test configuration
    for key, value in test_config.items():
        monkeypatch.setattr(config, key, value)
    
    # Set test environment variables
    monkeypatch.setenv('TESTING', 'True')
    monkeypatch.setenv('DEBUG', 'False')
    monkeypatch.setenv('LOG_LEVEL', 'INFO')
    
    # Create test directories
    for dir_name in ['data', 'models', 'logs']:
        Path(f'tests/{dir_name}').mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after tests
    if os.getenv('CLEANUP_TEST_FILES', 'True').lower() == 'true':
        for dir_name in ['data', 'models', 'logs']:
            test_dir = Path(f'tests/{dir_name}')
            if test_dir.exists():
                for file in test_dir.glob('*'):
                    if file.is_file():
                        file.unlink() 