"""
Test configuration and fixtures for the crypto volatility analysis system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime, timedelta
from src.config import Config, config

@pytest.fixture(scope='session')
def test_config():
    """Test configuration fixture."""
    return {
        'data_dir': 'tests/data',
        'models_dir': 'tests/models',
        'logs_dir': 'tests/logs',
        'vol_window': 30,
        'annualization_factor': 365,
        'api': {
            'yfinance': {
                'rate_limit': 100,
                'retry_attempts': 2,
                'retry_delay': 1
            },
            'deribit': {
                'rate_limit': 50,
                'retry_attempts': 2,
                'retry_delay': 1,
                'test_mode': True
            }
        },
        'model': {
            'default_window': 60,
            'train_test_split': 0.8,
            'validation_split': 0.2,
            'random_state': 42
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'max_size': 10485760,  # 10MB
            'backup_count': 5
        }
    }

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, test_config):
    """Set up test environment variables."""
    # Create test directories
    for dir_name in ['data_dir', 'models_dir', 'logs_dir']:
        os.makedirs(test_config[dir_name], exist_ok=True)

    # Update configuration
    config._config.update(test_config)

    # Clean up after tests
    yield
    for dir_name in ['data_dir', 'models_dir', 'logs_dir']:
        path = Path(test_config[dir_name])
        if path.exists():
            for file in path.glob('*'):
                file.unlink()
            path.rmdir()

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    prices = np.random.lognormal(mean=0.1, sigma=0.2, size=len(dates))
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, size=len(dates))
    }).set_index('Date')

@pytest.fixture
def sample_volatility_data(sample_price_data):
    """Generate sample volatility data for testing."""
    returns = np.log(sample_price_data['Close']).diff()
    volatility = returns.rolling(window=30).std() * np.sqrt(365)
    return pd.DataFrame({
        'Returns': returns,
        'Volatility': volatility
    })

@pytest.fixture
def sample_model_data():
    """Generate sample data for model testing."""
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    return X, y

@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample configuration file."""
    config_path = tmp_path / 'test_config.json'
    config_data = {
        'data_dir': 'test_data',
        'models_dir': 'test_models',
        'logs_dir': 'test_logs',
        'vol_window': 30
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return config_path 