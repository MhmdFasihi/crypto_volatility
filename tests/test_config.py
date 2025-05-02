# tests/test_config.py
"""
Test configuration for integration tests.
"""

import os
import tempfile
from datetime import datetime, timedelta

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