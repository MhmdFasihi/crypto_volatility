# tests/test_config.py
"""
Test configuration for integration tests.
"""

import os
import tempfile
from datetime import datetime, timedelta
import pytest
import json
from pathlib import Path
from src.config import Config, load_config

# Test data configuration
TEST_TICKERS = ['BTC-USD', 'ETH-USD']
TEST_START_DATE = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')

# Test parameters
TEST_VOL_WINDOW = 10
TEST_LAGS = 3
TEST_HMM_STATES = 2
TEST_N_CLUSTERS = 2
TEST_ANOMALY_THRESHOLD = 2.0

# Test directories
TEST_DATA_DIR = tempfile.mkdtemp()
TEST_MODEL_DIR = tempfile.mkdtemp()

# Test environment
TEST_ENV = {
    'DEBUG': True,
    'LOG_LEVEL': 'DEBUG',
    'TEST_MODE': True
}

# Mock data generator
def generate_mock_data(ticker, days=30):
    """Generate mock price data for testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    trend = np.linspace(40000, 45000, days) if ticker == 'BTC-USD' else np.linspace(2000, 2500, days)
    noise = np.random.normal(0, trend * 0.02, days)
    prices = trend + noise
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    
    return data

# Cleanup function
def cleanup_test_dirs():
    """Clean up test directories."""
    import shutil
    
    for dir_path in [TEST_DATA_DIR, TEST_MODEL_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

def test_config_initialization(test_data_dir):
    """Test Config class initialization."""
    # Create test config file
    config_file = test_data_dir / 'test_config.json'
    test_config = {
        'data_dir': 'data',
        'models_dir': 'models',
        'logs_dir': 'logs',
        'api': {
            'yfinance': {
                'rate_limit': 100,
                'retry_attempts': 3
            }
        }
    }
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    # Initialize config
    config = Config(config_file=str(config_file))
    
    # Check configuration
    assert config.get('data_dir') == 'data'
    assert config.get('models_dir') == 'models'
    assert config.get('logs_dir') == 'logs'
    assert config.get('api.yfinance.rate_limit') == 100
    assert config.get('api.yfinance.retry_attempts') == 3

def test_config_default_values():
    """Test default configuration values."""
    config = Config()
    
    # Check default values
    assert config.get('data_dir') == 'data'
    assert config.get('models_dir') == 'models'
    assert config.get('logs_dir') == 'logs'
    assert config.get('vol_window') == 30
    assert config.get('annualization_factor') == 365

def test_config_environment_variables(monkeypatch):
    """Test environment variable overrides."""
    # Set environment variables
    monkeypatch.setenv('DATA_DIR', '/custom/data')
    monkeypatch.setenv('MODELS_DIR', '/custom/models')
    monkeypatch.setenv('LOGS_DIR', '/custom/logs')
    monkeypatch.setenv('VOL_WINDOW', '60')
    
    # Initialize config
    config = Config()
    
    # Check environment variable overrides
    assert config.get('data_dir') == '/custom/data'
    assert config.get('models_dir') == '/custom/models'
    assert config.get('logs_dir') == '/custom/logs'
    assert config.get('vol_window') == 60

def test_config_set_and_get():
    """Test setting and getting configuration values."""
    config = Config()
    
    # Set values
    config.set('test.key1', 'value1')
    config.set('test.key2', {'nested': 'value2'})
    
    # Get values
    assert config.get('test.key1') == 'value1'
    assert config.get('test.key2.nested') == 'value2'
    
    # Get with default
    assert config.get('non.existent.key', default='default') == 'default'

def test_config_save_and_load(test_data_dir):
    """Test saving and loading configuration."""
    config_file = test_data_dir / 'save_test_config.json'
    
    # Create and save config
    config = Config()
    config.set('test.key', 'value')
    config.save(str(config_file))
    
    # Load config
    new_config = Config(config_file=str(config_file))
    
    # Check loaded values
    assert new_config.get('test.key') == 'value'
    assert os.path.exists(config_file)

def test_config_path_handling():
    """Test path handling in configuration."""
    config = Config()
    
    # Set paths
    config.set('paths.data', 'data')
    config.set('paths.models', 'models')
    
    # Get absolute paths
    data_path = config.get_path('paths.data')
    models_path = config.get_path('paths.models')
    
    assert isinstance(data_path, Path)
    assert isinstance(models_path, Path)
    assert data_path.is_absolute()
    assert models_path.is_absolute()

def test_config_validation():
    """Test configuration validation."""
    config = Config()
    
    # Test invalid values
    with pytest.raises(ValueError):
        config.set('vol_window', -1)
    
    with pytest.raises(ValueError):
        config.set('annualization_factor', 0)
    
    # Test valid values
    config.set('vol_window', 30)
    config.set('annualization_factor', 365)
    assert config.get('vol_window') == 30
    assert config.get('annualization_factor') == 365

def test_config_merge():
    """Test merging configurations."""
    config1 = Config()
    config2 = Config()
    
    # Set different values
    config1.set('key1', 'value1')
    config1.set('nested.key1', 'nested1')
    config2.set('key2', 'value2')
    config2.set('nested.key2', 'nested2')
    
    # Merge configs
    config1.merge(config2)
    
    # Check merged values
    assert config1.get('key1') == 'value1'
    assert config1.get('key2') == 'value2'
    assert config1.get('nested.key1') == 'nested1'
    assert config1.get('nested.key2') == 'nested2'

def test_load_config_function(test_data_dir):
    """Test the load_config function."""
    # Create test config file
    config_file = test_data_dir / 'load_test_config.json'
    test_config = {
        'data_dir': 'data',
        'models_dir': 'models',
        'logs_dir': 'logs'
    }
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    # Load config
    config = load_config(str(config_file))
    
    # Check loaded values
    assert isinstance(config, Config)
    assert config.get('data_dir') == 'data'
    assert config.get('models_dir') == 'models'
    assert config.get('logs_dir') == 'logs'

def test_config_error_handling():
    """Test error handling in configuration."""
    # Test non-existent config file
    with pytest.raises(FileNotFoundError):
        Config(config_file='non_existent.json')
    
    # Test invalid JSON
    with pytest.raises(json.JSONDecodeError):
        config = Config()
        config._load_config('invalid_json')