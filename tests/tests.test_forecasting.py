# tests/test_forecasting.py
"""
Unit tests for forecasting module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forecasting import (
    prepare_data,
    prepare_sequences,
    train_mlp,
    train_rnn,
    evaluate_model,
    save_model,
    load_model,
    forecast_next_values,
    ensemble_forecast
)

@pytest.fixture
def sample_volatility_series():
    """Create sample volatility time series."""
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
    data = pd.DataFrame({
        'Volatility': np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.1 + 0.2
    }, index=dates)
    return data

@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model saving/loading tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_prepare_data(sample_volatility_series):
    """Test data preparation for forecasting."""
    X, y = prepare_data(sample_volatility_series, lags=5)
    
    assert len(X) == len(y)
    assert X.shape[1] == 5  # Number of lags
    assert len(X) == len(sample_volatility_series) - 5  # Accounts for lags
    
    # Check column names
    expected_columns = [f'lag_{i}' for i in range(1, 6)]
    assert list(X.columns) == expected_columns

def test_prepare_sequences():
    """Test sequence preparation for RNN."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 11, 12])
    
    X_seq, y_seq = prepare_sequences(X, y)
    
    assert X_seq.shape == (3, 3, 1)  # (samples, timesteps, features)
    assert np.array_equal(y_seq, y)

def test_train_mlp(sample_volatility_series):
    """Test MLP training."""
    X, y = prepare_data(sample_volatility_series, lags=5)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    
    model, scaler = train_mlp(X_train.values, y_train.values)
    
    assert model is not None
    assert scaler is not None
    assert hasattr(model, 'predict')
    
    # Test prediction
    predictions = model.predict(scaler.transform(X_train.values))
    assert len(predictions) == len(X_train)

def test_train_rnn(sample_volatility_series):
    """Test RNN training."""
    X, y = prepare_data(sample_volatility_series, lags=5)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    
    # Prepare sequences
    X_train_seq, _ = prepare_sequences(X_train.values, y_train.values)
    
    model, scaler = train_rnn(X_train_seq, y_train.values, epochs=2, batch_size=8)
    
    assert model is not None
    assert scaler is not None
    assert hasattr(model, 'predict')
    
    # Test prediction
    predictions = model.predict(X_train_seq)
    assert len(predictions) == len(X_train)

def test_evaluate_model(sample_volatility_series):
    """Test model evaluation."""
    X, y = prepare_data(sample_volatility_series, lags=5)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train a simple MLP
    model, scaler = train_mlp(X_train.values, y_train.values)
    
    # Evaluate
    metrics = evaluate_model(model, X_test.values, y_test.values, scaler, is_rnn=False)
    
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics
    
    # Check that metrics are reasonable
    assert metrics['mae'] >= 0
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0

def test_save_and_load_model(sample_volatility_series, temp_model_dir):
    """Test model saving and loading."""
    # Set temporary model directory
    import src.config
    original_model_dir = src.config.MODEL_DIR
    src.config.MODEL_DIR = temp_model_dir
    
    try:
        X, y = prepare_data(sample_volatility_series, lags=5)
        X_train, y_train = X[:20], y[:20]
        
        # Train and save MLP
        model, scaler = train_mlp(X_train.values, y_train.values)
        save_model(model, scaler, 'test_mlp', is_rnn=False)
        
        # Load model
        loaded_model, loaded_scaler = load_model('test_mlp', is_rnn=False)
        
        assert loaded_model is not None
        assert loaded_scaler is not None
        
        # Check that predictions are the same
        original_pred = model.predict(scaler.transform(X_train.values))
        loaded_pred = loaded_model.predict(loaded_scaler.transform(X_train.values))
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        
    finally:
        # Restore original model directory
        src.config.MODEL_DIR = original_model_dir

def test_forecast_next_values(sample_volatility_series):
    """Test multi-step forecasting."""
    X, y = prepare_data(sample_volatility_series, lags=5)
    X_train, y_train = X[:20], y[:20]
    
    # Train model
    model, scaler = train_mlp(X_train.values, y_train.values)
    
    # Get recent data for forecasting
    recent_data = X_train.iloc[-1].values
    
    # Forecast next 5 values
    forecast = forecast_next_values(model, recent_data, scaler, is_rnn=False, n_ahead=5)
    
    assert len(forecast) == 5
    assert np.all(np.isfinite(forecast))  # No NaN or inf values

def test_ensemble_forecast(sample_volatility_series):
    """Test ensemble forecasting."""
    X, y = prepare_data(sample_volatility_series, lags=5)
    X_train, X_test = X[:20], X[20:]
    y_train, y_test = y[:20], y[20:]
    
    # Train multiple models
    mlp_model, mlp_scaler = train_mlp(X_train.values, y_train.values)
    
    # For testing, use the same model twice with different weights
    models = [mlp_model, mlp_model]
    scalers = [mlp_scaler, mlp_scaler]
    is_rnn = [False, False]
    weights = [0.7, 0.3]
    
    # Test ensemble
    ensemble_pred = ensemble_forecast(models, X_test.values, scalers, is_rnn, weights)
    
    assert len(ensemble_pred) == len(X_test)
    assert np.all(np.isfinite(ensemble_pred))

def test_error_handling():
    """Test error handling in forecasting functions."""
    # Test with empty data
    empty_data = pd.DataFrame({'Volatility': []})
    X, y = prepare_data(empty_data)
    assert len(X) == 0
    assert len(y) == 0
    
    # Test with invalid model type
    X_dummy = np.array([[1, 2], [3, 4]])
    y_dummy = np.array([5, 6])
    with pytest.raises(ValueError):
        train_rnn(X_dummy, y_dummy, model_type='InvalidRNN')

if __name__ == '__main__':
    pytest.main([__file__])