# tests/test_clustering.py
"""
Unit tests for clustering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import (
    compute_dtw_distance,
    compute_distance_matrix,
    cluster_tickers,
    get_cluster_characteristics,
    find_optimal_clusters,
    get_similar_tickers
)

@pytest.fixture
def sample_time_series():
    """Create sample time series for testing."""
    # Create different patterns
    t = np.linspace(0, 4*np.pi, 100)
    series1 = np.sin(t) * 0.1 + 0.2
    series2 = np.sin(t + np.pi/4) * 0.1 + 0.2  # Phase shifted
    series3 = np.cos(t) * 0.15 + 0.25  # Different pattern
    series4 = np.sin(2*t) * 0.08 + 0.18  # Different frequency
    
    return [series1, series2, series3, series4]

@pytest.fixture
def mock_ticker_data():
    """Create mock data for ticker clustering."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    return {
        'BTC-USD': pd.DataFrame({
            'Close': [40000, 41000, 42000, 41500, 43000, 44000, 43500, 45000, 46000, 45500]
        }, index=dates),
        'ETH-USD': pd.DataFrame({
            'Close': [2000, 2100, 2200, 2150, 2300, 2400, 2350, 2500, 2600, 2550]
        }, index=dates),
        'XRP-USD': pd.DataFrame({
            'Close': [0.5, 0.52, 0.48, 0.51, 0.53, 0.49, 0.52, 0.54, 0.50, 0.53]
        }, index=dates)
    }

def test_compute_dtw_distance():
    """Test DTW distance computation."""
    # Test with identical series
    series1 = np.array([1, 2, 3, 4, 5])
    series2 = np.array([1, 2, 3, 4, 5])
    
    distance = compute_dtw_distance(series1, series2)
    assert distance == 0.0
    
    # Test with different series
    series3 = np.array([1, 3, 5, 7, 9])
    distance = compute_dtw_distance(series1, series3)
    assert distance > 0
    
    # Test different distance metrics
    distance_euclidean = compute_dtw_distance(series1, series3, 'euclidean')
    distance_manhattan = compute_dtw_distance(series1, series3, 'manhattan')
    distance_cosine = compute_dtw_distance(series1, series3, 'cosine')
    
    assert distance_euclidean != distance_manhattan
    assert distance_euclidean != distance_cosine

def test_compute_distance_matrix(sample_time_series):
    """Test distance matrix computation."""
    dist_matrix = compute_distance_matrix(sample_time_series)
    
    # Check matrix properties
    n = len(sample_time_series)
    assert dist_matrix.shape == (n, n)
    
    # Check diagonal is zero
    np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(n))
    
    # Check symmetry
    assert np.allclose(dist_matrix, dist_matrix.T)
    
    # Check that similar series have smaller distances
    # series1 and series2 are phase-shifted versions of the same pattern
    assert dist_matrix[0, 1] < dist_matrix[0, 2]  # series1 closer to series2 than series3

@patch('src.clustering.get_data')
def test_cluster_tickers(mock_get_data, mock_ticker_data):
    """Test ticker clustering."""
    # Setup mock
    def mock_get_data_side_effect(ticker, start_date, end_date):
        return mock_ticker_data.get(ticker, pd.DataFrame())
    
    mock_get_data.side_effect = mock_get_data_side_effect
    
    tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD']
    result = cluster_tickers(tickers, n_clusters=2)
    
    assert isinstance(result, dict)
    assert len(result) == len(tickers)
    assert all(ticker in result for ticker in tickers)
    assert all(0 <= label < 2 for label in result.values())
    
    # Test with invalid clustering method
    with pytest.raises(ValueError):
        cluster_tickers(tickers, clustering_method='invalid_method')

@patch('src.clustering.get_data')
def test_get_cluster_characteristics(mock_get_data, mock_ticker_data):
    """Test cluster characteristics calculation."""
    # Setup mock
    mock_get_data.side_effect = lambda ticker, start_date, end_date: mock_ticker_data.get(ticker, pd.DataFrame())
    
    cluster_mapping = {'BTC-USD': 0, 'ETH-USD': 0, 'XRP-USD': 1}
    stats = get_cluster_characteristics(['BTC-USD', 'ETH-USD', 'XRP-USD'], cluster_mapping)
    
    assert isinstance(stats, dict)
    assert len(stats) == 2  # Two clusters
    
    # Check that each cluster has expected stats
    for cluster_id, cluster_stats in stats.items():
        assert 'avg_volatility' in cluster_stats
        assert 'std_volatility' in cluster_stats
        assert 'median_volatility' in cluster_stats
        assert 'avg_return' in cluster_stats
        assert 'std_return' in cluster_stats
        assert 'num_tickers' in cluster_stats
        
        # Check that values are reasonable
        assert cluster_stats['num_tickers'] > 0
        assert isinstance(cluster_stats['avg_volatility'], float)

@patch('src.clustering.get_data')
def test_find_optimal_clusters(mock_get_data, mock_ticker_data):
    """Test optimal cluster finding."""
    # Setup mock
    mock_get_data.side_effect = lambda ticker, start_date, end_date: mock_ticker_data.get(ticker, pd.DataFrame())
    
    tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD']
    optimal_n, scores = find_optimal_clusters(tickers, max_clusters=3)
    
    assert isinstance(optimal_n, int)
    assert 2 <= optimal_n <= 3
    assert isinstance(scores, list)
    assert len(scores) <= 2  # max_clusters - 1
    
    # All scores should be between -1 and 1 (silhouette score range)
    assert all(-1 <= score <= 1 for score in scores)

def test_get_similar_tickers():
    """Test similar ticker retrieval."""
    cluster_mapping = {
        'BTC-USD': 0,
        'ETH-USD': 0,
        'XRP-USD': 1,
        'LTC-USD': 0,
        'BCH-USD': 1
    }
    
    # Test for ticker in cluster 0
    similar_tickers = get_similar_tickers('BTC-USD', cluster_mapping, max_similar=5)
    assert 'BTC-USD' not in similar_tickers
    assert 'ETH-USD' in similar_tickers
    assert 'LTC-USD' in similar_tickers
    assert 'XRP-USD' not in similar_tickers
    
    # Test with max_similar limit
    similar_tickers = get_similar_tickers('BTC-USD', cluster_mapping, max_similar=1)
    assert len(similar_tickers) == 1
    
    # Test with ticker not in mapping
    similar_tickers = get_similar_tickers('UNKNOWN-USD', cluster_mapping)
    assert len(similar_tickers) == 0

def test_error_handling():
    """Test error handling in clustering functions."""
    # Test invalid distance metric
    with pytest.raises(ValueError):
        compute_dtw_distance(np.array([1, 2]), np.array([3, 4]), 'invalid_metric')
    
    # Test with empty time series
    empty_series = []
    with pytest.raises(IndexError):
        compute_distance_matrix(empty_series)
    
    # Test clustering with less than 2 tickers
    with patch('src.clustering.get_data') as mock_get_data:
        mock_get_data.return_value = pd.DataFrame({'Close': [1, 2, 3]})
        result = cluster_tickers(['BTC-USD'], n_clusters=2)
        assert result == {}

if __name__ == '__main__':
    pytest.main([__file__])