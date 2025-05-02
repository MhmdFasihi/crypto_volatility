# tests/test_anomaly_detection.py
"""
Unit tests for anomaly detection module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_detection import (
    detect_anomalies_zscore,
    detect_anomalies_isolation_forest,
    detect_anomalies_percentile,
    detect_anomalies_mahalanobis,
    create_anomaly_features,
    train_random_forest_detector,
    evaluate_anomaly_detector,
    get_anomaly_statistics,
    calculate_consecutive_anomalies,
    ensemble_anomaly_detection
)

@pytest.fixture
def sample_volatility_data():
    """Create sample volatility data with some outliers."""
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
    np.random.seed(42)
    
    # Create data with some outliers
    volatility = np.random.normal(0.2, 0.02, len(dates))
    volatility[10] = 0.4  # High outlier
    volatility[20] = 0.05  # Low outlier
    volatility[25] = 0.45  # Another high outlier
    
    data = pd.DataFrame({
        'Volatility': volatility,
        'Returns': np.random.normal(0, 0.01, len(dates))
    }, index=dates)
    
    return data

def test_detect_anomalies_zscore(sample_volatility_data):
    """Test Z-score anomaly detection."""
    result = detect_anomalies_zscore(sample_volatility_data, window=10, threshold=2.0)
    
    assert 'Z_Score' in result.columns
    assert 'Anomaly' in result.columns
    assert 'Anomaly_Direction' in result.columns
    
    # Check that we detected some anomalies
    assert result['Anomaly'].sum() > 0
    
    # Check that high outliers are detected
    assert result.loc[result.index[10], 'Anomaly'] == True
    assert result.loc[result.index[25], 'Anomaly'] == True

def test_detect_anomalies_isolation_forest(sample_volatility_data):
    """Test Isolation Forest anomaly detection."""
    features = ['Volatility', 'Returns']
    result = detect_anomalies_isolation_forest(
        sample_volatility_data, 
        features, 
        contamination=0.1
    )
    
    assert 'IF_Anomaly' in result.columns
    assert 'IF_Score' in result.columns
    
    # Check that we detected some anomalies
    assert result['IF_Anomaly'].sum() > 0

def test_detect_anomalies_percentile(sample_volatility_data):
    """Test percentile-based anomaly detection."""
    result = detect_anomalies_percentile(
        sample_volatility_data,
        lower_percentile=5,
        upper_percentile=95
    )
    
    assert 'Percentile_Anomaly' in result.columns
    assert 'Percentile_Direction' in result.columns
    
    # Check that we detected some anomalies
    assert result['Percentile_Anomaly'].sum() > 0

def test_detect_anomalies_mahalanobis(sample_volatility_data):
    """Test Mahalanobis distance anomaly detection."""
    features = ['Volatility', 'Returns']
    result = detect_anomalies_mahalanobis(sample_volatility_data, features)
    
    assert 'Mahalanobis_Distance' in result.columns
    assert 'Mahalanobis_Anomaly' in result.columns
    
    # Check that we detected some anomalies
    assert result['Mahalanobis_Anomaly'].sum() > 0

def test_create_anomaly_features(sample_volatility_data):
    """Test anomaly feature creation."""
    result = create_anomaly_features(sample_volatility_data)
    
    # Check for rate of change
    assert 'Vol_RoC' in result.columns
    
    # Check for rolling statistics
    for window in [5, 10, 20]:
        assert f'Vol_MA_{window}' in result.columns
        assert f'Vol_Std_{window}' in result.columns
        assert f'Vol_Skew_{window}' in result.columns
        assert f'Vol_Kurt_{window}' in result.columns
        assert f'Vol_Distance_MA_{window}' in result.columns
    
    # Check for volatility of volatility
    assert 'Vol_of_Vol' in result.columns

def test_train_random_forest_detector(sample_volatility_data):
    """Test Random Forest anomaly detector training."""
    # Create features and labels
    data = create_anomaly_features(sample_volatility_data)
    data = detect_anomalies_zscore(data)
    
    features = [col for col in data.columns if 'Vol_' in col and col != 'Volatility']
    
    # Train detector
    rf_model, scaler = train_random_forest_detector(data, features)
    
    assert rf_model is not None
    assert scaler is not None
    assert hasattr(rf_model, 'predict')
    assert hasattr(rf_model, 'feature_importances_')

def test_evaluate_anomaly_detector():
    """Test anomaly detector evaluation."""
    # Create dummy data
    y_test = np.array([0, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.2, 0.7])
    
    # Create dummy model
    class DummyModel:
        def predict(self, X):
            return y_pred
        
        def predict_proba(self, X):
            return np.vstack([1 - y_proba, y_proba]).T
    
    model = DummyModel()
    metrics = evaluate_anomaly_detector(model, None, y_test)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics
    
    # Check that metrics are in valid range
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        assert 0 <= metrics[metric] <= 1

def test_get_anomaly_statistics(sample_volatility_data):
    """Test anomaly statistics calculation."""
    data = detect_anomalies_zscore(sample_volatility_data)
    stats = get_anomaly_statistics(data)
    
    assert 'total_points' in stats
    assert 'anomaly_count' in stats
    assert 'anomaly_percentage' in stats
    assert 'consecutive_anomalies' in stats
    
    assert stats['total_points'] == len(data)
    assert stats['anomaly_count'] == data['Anomaly'].sum()

def test_calculate_consecutive_anomalies():
    """Test consecutive anomaly calculation."""
    # Test with simple pattern
    anomalies = pd.Series([False, True, True, False, True, True, True, False])
    max_consecutive = calculate_consecutive_anomalies(anomalies)
    assert max_consecutive == 3
    
    # Test with no anomalies
    anomalies = pd.Series([False, False, False])
    max_consecutive = calculate_consecutive_anomalies(anomalies)
    assert max_consecutive == 0
    
    # Test with all anomalies
    anomalies = pd.Series([True, True, True])
    max_consecutive = calculate_consecutive_anomalies(anomalies)
    assert max_consecutive == 3

def test_ensemble_anomaly_detection(sample_volatility_data):
    """Test ensemble anomaly detection."""
    result = ensemble_anomaly_detection(
        sample_volatility_data,
        methods=['zscore', 'percentile'],
        threshold=0.5
    )
    
    assert 'Ensemble_Score' in result.columns
    assert 'Ensemble_Anomaly' in result.columns
    
    # Check that ensemble detected some anomalies
    assert result['Ensemble_Anomaly'].sum() > 0
    
    # Check that ensemble score is between 0 and 1
    assert (result['Ensemble_Score'] >= 0).all()
    assert (result['Ensemble_Score'] <= 1).all()

def test_error_handling():
    """Test error handling in anomaly detection functions."""
    # Test with empty data
    empty_data = pd.DataFrame()
    result = detect_anomalies_zscore(empty_data)
    assert result.empty
    
    # Test with missing columns
    data = pd.DataFrame({'NotVolatility': [1, 2, 3]})
    # Should handle missing column gracefully
    result = detect_anomalies_zscore(data, column='Volatility')
    assert 'Z_Score' not in result.columns

if __name__ == '__main__':
    pytest.main([__file__])