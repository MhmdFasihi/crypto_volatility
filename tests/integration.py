# tests/test_integration.py
"""
Integration tests for the crypto volatility analysis system.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_config import (
    TEST_TICKERS, TEST_START_DATE, TEST_END_DATE,
    TEST_MODEL_DIR, generate_mock_data, cleanup_test_dirs
)
from src.data_acquisition import get_data, get_combined_volatility_data
from src.preprocessing import calculate_returns, calculate_volatility
from src.forecasting import train_mlp, train_rnn, forecast_next_values, prepare_data
from src.clustering import cluster_tickers
from src.classification import train_hmm, predict_states, get_current_regime
from src.anomaly_detection import ensemble_anomaly_detection
from src.train_models import train_all_models
import src.config as config

# Override config for testing
config.MODEL_DIR = TEST_MODEL_DIR

@pytest.fixture(scope="module")
def setup_and_teardown():
    """Setup and teardown for integration tests."""
    yield
    cleanup_test_dirs()

@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    data = {}
    for ticker in TEST_TICKERS:
        data[ticker] = generate_mock_data(ticker)
    return data

@patch('src.data_acquisition.get_data')
def test_end_to_end_workflow(mock_get_data, mock_data):
    """Test the complete workflow from data acquisition to forecasting."""
    # Setup mock
    mock_get_data.side_effect = lambda ticker, start, end: mock_data.get(ticker, pd.DataFrame())
    
    # Step 1: Data Acquisition
    data = get_data('BTC-USD', TEST_START_DATE, TEST_END_DATE)
    assert not data.empty
    
    # Step 2: Preprocessing
    data = calculate_returns(data)
    data = calculate_volatility(data)
    assert 'Returns' in data.columns
    assert 'Volatility' in data.columns
    
    # Step 3: Model Training (MLP)
    X, y = prepare_data(data)
    if len(X) > 10:
        X_train, y_train = X[:20], y[:20]
        model, scaler = train_mlp(X_train.values, y_train.values)
        
        # Step 4: Forecasting
        recent_data = X_train.iloc[-1].values
        forecast = forecast_next_values(model, recent_data, scaler, n_ahead=5)
        assert len(forecast) == 5
        assert np.all(np.isfinite(forecast))

@patch('src.data_acquisition.get_data')
def test_clustering_integration(mock_get_data, mock_data):
    """Test clustering integration with multiple tickers."""
    # Setup mock
    mock_get_data.side_effect = lambda ticker, start, end: mock_data.get(ticker, pd.DataFrame())
    
    # Perform clustering
    cluster_mapping = cluster_tickers(TEST_TICKERS, n_clusters=2)
    
    assert len(cluster_mapping) == len(TEST_TICKERS)
    assert all(0 <= label < 2 for label in cluster_mapping.values())

@patch('src.data_acquisition.get_data')
def test_classification_integration(mock_get_data, mock_data):
    """Test HMM classification integration."""
    # Setup mock
    data = mock_data['BTC-USD']
    data = calculate_returns(data)
    data = calculate_volatility(data)
    
    # Train HMM
    model, scaler = train_hmm(data, n_states=3)
    
    # Predict states
    states = predict_states(model, data, scaler)
    assert len(states) == len(data)
    
    # Get current regime
    state_stats = {0: {'mean': 0.1, 'std': 0.01}, 
                   1: {'mean': 0.2, 'std': 0.02}, 
                   2: {'mean': 0.3, 'std': 0.03}}
    
    regime_name, state_idx, probability = get_current_regime(
        model, data, scaler, state_stats
    )
    assert isinstance(regime_name, str)
    assert 0 <= state_idx < 3
    assert 0 <= probability <= 1

@patch('src.data_acquisition.get_data')
def test_anomaly_detection_integration(mock_get_data, mock_data):
    """Test anomaly detection integration."""
    # Setup mock
    data = mock_data['BTC-USD']
    data = calculate_returns(data)
    data = calculate_volatility(data)
    
    # Detect anomalies
    result = ensemble_anomaly_detection(
        data,
        methods=['zscore', 'percentile'],
        threshold=0.5
    )
    
    assert 'Ensemble_Anomaly' in result.columns
    assert 'Ensemble_Score' in result.columns

@patch('src.data_acquisition.get_data')
def test_model_training_script(mock_get_data, mock_data):
    """Test the model training script integration."""
    # Setup mock
    mock_get_data.side_effect = lambda ticker, start, end: mock_data.get(ticker, pd.DataFrame())
    
    # Train models
    results = train_all_models(
        ticker='BTC-USD',
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        models_to_train=['mlp']
    )
    
    assert 'mlp' in results
    assert results['mlp']['status'] == 'success'

@patch('src.data_acquisition.get_data')
def test_error_handling_integration(mock_get_data):
    """Test error handling across modules."""
    # Test with empty data
    mock_get_data.return_value = pd.DataFrame()
    
    # Should handle empty data gracefully
    data = get_data('INVALID-TICKER', TEST_START_DATE, TEST_END_DATE)
    assert data.empty
    
    # Preprocessing should handle empty data
    result = calculate_returns(data)
    assert result.empty
    
    # Forecasting should handle insufficient data
    X, y = prepare_data(data)
    assert len(X) == 0

def test_configuration_integration():
    """Test configuration integration."""
    # Test config imports
    from src.config import VOL_WINDOW, LAGS, HMM_STATES
    
    assert isinstance(VOL_WINDOW, int)
    assert isinstance(LAGS, int)
    assert isinstance(HMM_STATES, int)

@pytest.mark.slow
def test_performance_integration(mock_data):
    """Test system performance with larger datasets."""
    # Generate larger dataset
    large_data = generate_mock_data('BTC-USD', days=365)
    
    # Measure preprocessing time
    import time
    start_time = time.time()
    
    data = calculate_returns(large_data)
    data = calculate_volatility(data)
    
    preprocessing_time = time.time() - start_time
    assert preprocessing_time < 1.0  # Should complete within 1 second
    
    # Measure model training time
    start_time = time.time()
    
    X, y = prepare_data(data)
    if len(X) > 100:
        X_train, y_train = X[:100], y[:100]
        model, scaler = train_mlp(X_train.values, y_train.values)
    
    training_time = time.time() - start_time
    assert training_time < 5.0  # Should complete within 5 seconds

def test_memory_usage():
    """Test memory usage of the system."""
    import psutil
    import gc
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run a complete workflow
    data = generate_mock_data('BTC-USD', days=100)
    data = calculate_returns(data)
    data = calculate_volatility(data)
    
    X, y = prepare_data(data)
    if len(X) > 50:
        X_train, y_train = X[:50], y[:50]
        model, scaler = train_mlp(X_train.values, y_train.values)
    
    # Check memory after operations
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Clean up
    gc.collect()
    
    # Memory increase should be reasonable
    assert memory_increase < 500  # Less than 500MB increase

if __name__ == '__main__':
    pytest.main([__file__, '-v'])