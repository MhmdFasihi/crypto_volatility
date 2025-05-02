# tests/test_classification.py
"""
Unit tests for classification module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification import (
    train_hmm,
    predict_states,
    get_state_statistics,
    get_transition_matrix,
    classify_volatility_regime,
    get_regime_duration_statistics,
    get_current_regime,
    get_state_transition_probabilities
)

@pytest.fixture
def sample_volatility_data():
    """Create sample volatility data with distinct regimes."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    
    # Create data with three distinct volatility regimes
    volatility = []
    for i in range(len(dates)):
        if i < 20:  # Low volatility regime
            vol = np.random.normal(0.1, 0.01)
        elif i < 40:  # Medium volatility regime
            vol = np.random.normal(0.2, 0.02)
        else:  # High volatility regime
            vol = np.random.normal(0.3, 0.03)
        volatility.append(vol)
    
    data = pd.DataFrame({
        'Volatility': volatility
    }, index=dates)
    
    return data

def test_train_hmm(sample_volatility_data):
    """Test HMM training."""
    model, scaler = train_hmm(sample_volatility_data, n_states=3)
    
    assert model is not None
    assert scaler is not None
    assert model.n_components == 3
    assert hasattr(model, 'transmat_')
    assert hasattr(model, 'means_')
    
    # Check that the model has converged (or at least run)
    assert hasattr(model, 'monitor_')

def test_predict_states(sample_volatility_data):
    """Test state prediction."""
    model, scaler = train_hmm(sample_volatility_data, n_states=3)
    states = predict_states(model, sample_volatility_data, scaler)
    
    assert len(states) == len(sample_volatility_data)
    assert all(0 <= state < 3 for state in states)
    
    # Test that different volatility levels result in different states
    # (though we can't guarantee which state corresponds to which level)
    unique_states = np.unique(states)
    assert len(unique_states) <= 3

def test_get_state_statistics(sample_volatility_data):
    """Test state statistics extraction."""
    model, scaler = train_hmm(sample_volatility_data, n_states=3)
    stats = get_state_statistics(model, scaler)
    
    assert isinstance(stats, dict)
    assert len(stats) == 3
    
    for state_id, state_stats in stats.items():
        assert 'mean' in state_stats
        assert 'std' in state_stats
        assert 'stationary_prob' in state_stats
        
        # Check that values are reasonable
        assert isinstance(state_stats['mean'], float)
        assert isinstance(state_stats['std'], float)
        assert 0 <= state_stats['stationary_prob'] <= 1

def test_get_transition_matrix(sample_volatility_data):
    """Test transition matrix extraction."""
    model, scaler = train_hmm(sample_volatility_data, n_states=3)
    trans_mat = get_transition_matrix(model)
    
    assert trans_mat.shape == (3, 3)
    
    # Check that rows sum to 1 (probability distribution)
    row_sums = trans_mat.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(3))
    
    # Check that all values are probabilities
    assert np.all(trans_mat >= 0)
    assert np.all(trans_mat <= 1)

def test_classify_volatility_regime():
    """Test volatility regime classification."""
    state_stats = {
        0: {'mean': 0.1, 'std': 0.01},
        1: {'mean': 0.2, 'std': 0.02},
        2: {'mean': 0.3, 'std': 0.03}
    }
    
    # Test low volatility
    regime_name, state_idx = classify_volatility_regime(0.1, state_stats)
    assert regime_name == 'Low'
    assert state_idx == 0
    
    # Test medium volatility
    regime_name, state_idx = classify_volatility_regime(0.2, state_stats)
    assert regime_name == 'Medium'
    assert state_idx == 1
    
    # Test high volatility
    regime_name, state_idx = classify_volatility_regime(0.3, state_stats)
    assert regime_name == 'High'
    assert state_idx == 2

def test_get_regime_duration_statistics():
    """Test regime duration statistics calculation."""
    # Create a simple state sequence
    states = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 2, 2])
    
    duration_stats = get_regime_duration_statistics(states)
    
    assert isinstance(duration_stats, dict)
    assert len(duration_stats) == 3  # Three unique states
    
    # Check state 0 durations
    assert duration_stats[0]['mean_duration'] == 2.5  # (3 + 2) / 2
    assert duration_stats[0]['frequency'] == 2
    
    # Check state 1 durations
    assert duration_stats[1]['mean_duration'] == 2.5  # (2 + 3) / 2
    assert duration_stats[1]['frequency'] == 2
    
    # Check state 2 durations
    assert duration_stats[2]['mean_duration'] == 2.0
    assert duration_stats[2]['frequency'] == 1

def test_get_current_regime(sample_volatility_data):
    """Test current regime extraction."""
    model, scaler = train_hmm(sample_volatility_data, n_states=3)
    state_stats = get_state_statistics(model, scaler)
    
    regime_name, state_idx, probability = get_current_regime(
        model, sample_volatility_data, scaler, state_stats
    )
    
    assert isinstance(regime_name, str)
    assert isinstance(state_idx, (int, np.integer))
    assert isinstance(probability, float)
    assert 0 <= probability <= 1

def test_get_state_transition_probabilities(sample_volatility_data):
    """Test state transition probability calculation."""
    model, scaler = train_hmm(sample_volatility_data, n_states=3)
    
    # Test transition probabilities from state 0
    probs = get_state_transition_probabilities(model, from_state=0, time_horizon=5)
    
    assert isinstance(probs, dict)
    assert len(probs) == 3
    assert sum(probs.values()) == pytest.approx(1.0)  # Probabilities sum to 1
    assert all(0 <= p <= 1 for p in probs.values())

def test_error_handling():
    """Test error handling in classification functions."""
    # Test with empty data
    empty_data = pd.DataFrame({'Volatility': []})
    with pytest.raises(ValueError):
        train_hmm(empty_data)
    
    # Test with missing column
    wrong_data = pd.DataFrame({'NotVolatility': [1, 2, 3]})
    with pytest.raises(KeyError):
        train_hmm(wrong_data)

if __name__ == '__main__':
    pytest.main([__file__])