# src/classification.py
"""
Volatility classification module using Hidden Markov Models (HMM).
"""

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.config import HMM_STATES, VOL_WINDOW, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_hmm(data: pd.DataFrame, 
              feature_cols: List[str] = ['Volatility'],
              n_states: int = HMM_STATES,
              covariance_type: str = 'diag',
              n_iter: int = 1000) -> Tuple[GaussianHMM, StandardScaler]:
    """
    Train an HMM to classify volatility regimes.
    
    Args:
        data: DataFrame with volatility data
        feature_cols: List of feature columns to use
        n_states: Number of hidden states
        covariance_type: Covariance matrix type ('diag', 'full', 'tied', 'spherical')
        n_iter: Number of iterations for training
    
    Returns:
        Tuple of (trained HMM model, scaler)
    """
    # Extract features
    features = data[feature_cols].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=RANDOM_SEED
    )
    
    model.fit(features_scaled)
    
    # Log convergence
    if model.monitor_.converged:
        logger.info(f"HMM converged after {model.monitor_.iter} iterations")
    else:
        logger.warning(f"HMM did not converge after {n_iter} iterations")
    
    return model, scaler

def predict_states(model: GaussianHMM, 
                  data: pd.DataFrame, 
                  scaler: StandardScaler,
                  feature_cols: List[str] = ['Volatility']) -> np.ndarray:
    """
    Predict volatility states for given data.
    
    Args:
        model: Trained HMM model
        data: DataFrame with volatility data
        scaler: Feature scaler
        feature_cols: List of feature columns
    
    Returns:
        Array of predicted states
    """
    # Extract features with proper error handling
    features = data[feature_cols].dropna()
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict states
    states = model.predict(features_scaled)
    
    # Create a Series aligned to the original data
    state_series = pd.Series(index=data.index, dtype=int)
    state_series.loc[features.index] = states
    
    # Forward-fill NaN values (use the last known state)
    state_series = state_series.ffill().fillna(0).astype(int)
    
    # Ensure length matches
    if len(state_series) != len(data):
        import logging
        logging.warning(f"State length mismatch: {len(states)} vs {len(data)}")
        # Return states aligned with data index
        return state_series.reindex(data.index, fill_value=0).values
    
    return state_series.values

def get_state_statistics(model: GaussianHMM, 
                        scaler: StandardScaler) -> Dict[int, Dict[str, float]]:
    """
    Get statistical parameters for each state.
    
    Args:
        model: Trained HMM model
        scaler: Feature scaler
    
    Returns:
        Dictionary with state statistics
    """
    state_stats = {}
    
    # Get means and covariances in original scale
    means = scaler.inverse_transform(model.means_)
    
    for i in range(model.n_components):
        if model.covariance_type == 'diag':
            std = np.sqrt(model.covars_[i])
        elif model.covariance_type == 'full':
            std = np.sqrt(np.diag(model.covars_[i]))
        elif model.covariance_type == 'tied':
            std = np.sqrt(np.diag(model.covars_))
        elif model.covariance_type == 'spherical':
            std = np.sqrt(model.covars_[i])
        
        # Transform standard deviation to original scale
        std_transformed = std * scaler.scale_
        
        state_stats[i] = {
            'mean': float(means[i, 0]),
            'std': float(std_transformed[0]),
            'stationary_prob': float(model.get_stationary_distribution()[i])
        }
    
    return state_stats

def get_transition_matrix(model: GaussianHMM) -> np.ndarray:
    """
    Get the state transition probability matrix.
    
    Args:
        model: Trained HMM model
    
    Returns:
        Transition matrix
    """
    return model.transmat_

def classify_volatility_regime(volatility: float, 
                              state_stats: Dict[int, Dict[str, float]]) -> Tuple[str, int]:
    """
    Classify a volatility value into a regime based on state statistics.
    
    Args:
        volatility: Current volatility value
        state_stats: Dictionary with state statistics
    
    Returns:
        Tuple of (regime name, state index)
    """
    # Sort states by mean volatility
    sorted_states = sorted(state_stats.items(), key=lambda x: x[1]['mean'])
    
    # Calculate distance to each state mean
    distances = []
    for state, stats in sorted_states:
        distance = abs(volatility - stats['mean']) / stats['std']
        distances.append((state, distance))
    
    # Get closest state
    closest_state = min(distances, key=lambda x: x[1])[0]
    
    # Assign regime names based on sorted order
    regime_names = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
    regime_name = regime_names[closest_state] if closest_state < len(regime_names) else f'State_{closest_state}'
    
    return regime_name, closest_state

def get_regime_duration_statistics(states: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Calculate duration statistics for each regime.
    
    Args:
        states: Array of state predictions
    
    Returns:
        Dictionary with duration statistics
    """
    durations = {state: [] for state in np.unique(states)}
    
    current_state = states[0]
    current_duration = 1
    
    for i in range(1, len(states)):
        if states[i] == current_state:
            current_duration += 1
        else:
            durations[current_state].append(current_duration)
            current_state = states[i]
            current_duration = 1
    
    # Add the last duration
    durations[current_state].append(current_duration)
    
    # Calculate statistics
    duration_stats = {}
    for state, state_durations in durations.items():
        if state_durations:
            duration_stats[state] = {
                'mean_duration': np.mean(state_durations),
                'median_duration': np.median(state_durations),
                'max_duration': np.max(state_durations),
                'min_duration': np.min(state_durations),
                'frequency': len(state_durations)
            }
    
    return duration_stats

def visualize_states(data: pd.DataFrame, 
                    states: np.ndarray,
                    state_stats: Dict[int, Dict[str, float]],
                    save_path: Optional[str] = None) -> None:
    """
    Visualize volatility regimes over time.
    
    Args:
        data: DataFrame with volatility data
        states: Array of state predictions
        state_stats: Dictionary with state statistics
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Plot volatility with state colors
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    state_colors = [colors[state] if state < len(colors) else 'purple' for state in states]
    
    # Create a copy of data with aligned states
    plot_data = data.dropna(subset=['Volatility']).copy()
    if len(states) == len(plot_data):
        plot_data['State'] = states
    else:
        logger.warning(f"State length mismatch: {len(states)} vs {len(plot_data)}")
        plot_data['State'] = states[:len(plot_data)] if len(states) > len(plot_data) else np.pad(states, (0, len(plot_data) - len(states)), 'edge')
    
    # Plot volatility colored by state
    for state in np.unique(plot_data['State']):
        mask = plot_data['State'] == state
        color = colors[state] if state < len(colors) else 'purple'
        regime_name = ['Low', 'Medium', 'High', 'Very High', 'Extreme'][state] if state < 5 else f'State_{state}'
        
        ax1.scatter(plot_data.index[mask], plot_data['Volatility'][mask], 
                   c=color, label=f'{regime_name} Volatility', alpha=0.6, s=20)
    
    ax1.set_ylabel('Volatility')
    ax1.set_title('Volatility Regimes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot state sequence
    ax2.step(plot_data.index, plot_data['State'], where='post', 
             color='blue', linewidth=2)
    ax2.set_ylabel('State')
    ax2.set_xlabel('Date')
    ax2.set_yticks(range(len(state_stats)))
    ax2.set_yticklabels([f'State {i}' for i in range(len(state_stats))])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"State visualization saved to {save_path}")
    
    plt.close()

def get_current_regime(model: GaussianHMM, 
                      data: pd.DataFrame,
                      scaler: StandardScaler,
                      state_stats: Dict[int, Dict[str, float]],
                      feature_cols: List[str] = ['Volatility']) -> Tuple[str, int, float]:
    """
    Get the current volatility regime.
    
    Args:
        model: Trained HMM model
        data: DataFrame with volatility data
        scaler: Feature scaler
        state_stats: Dictionary with state statistics
        feature_cols: List of feature columns
    
    Returns:
        Tuple of (regime name, state index, probability)
    """
    # Get the most recent valid data point
    recent_data = data[feature_cols].dropna().tail(1)
    
    if recent_data.empty:
        return "Unknown", -1, 0.0
    
    # Scale data
    recent_scaled = scaler.transform(recent_data)
    
    # Predict state probabilities
    state_probs = model.predict_proba(recent_scaled)[-1]
    
    # Get most likely state
    current_state = np.argmax(state_probs)
    probability = state_probs[current_state]
    
    # Get regime name
    recent_volatility = recent_data.iloc[0, 0]
    regime_name, _ = classify_volatility_regime(recent_volatility, state_stats)
    
    return regime_name, current_state, probability

def train_multi_feature_hmm(data: pd.DataFrame,
                           feature_cols: List[str],
                           n_states: int = HMM_STATES) -> Tuple[GaussianHMM, StandardScaler]:
    """
    Train HMM with multiple features.
    
    Args:
        data: DataFrame with feature data
        feature_cols: List of feature columns to use
        n_states: Number of hidden states
    
    Returns:
        Tuple of (trained HMM model, scaler)
    """
    # Ensure all features are available
    available_features = [col for col in feature_cols if col in data.columns]
    
    if not available_features:
        raise ValueError("No valid features found in data")
    
    if len(available_features) < len(feature_cols):
        logger.warning(f"Using only available features: {available_features}")
    
    return train_hmm(data, available_features, n_states)

def get_state_transition_probabilities(model: GaussianHMM, 
                                     from_state: int, 
                                     time_horizon: int = 5) -> Dict[int, float]:
    """
    Calculate state transition probabilities over a time horizon.
    
    Args:
        model: Trained HMM model
        from_state: Starting state
        time_horizon: Number of steps to look ahead
    
    Returns:
        Dictionary of transition probabilities to each state
    """
    # Get transition matrix
    trans_mat = model.transmat_
    
    # Calculate multi-step transition probabilities
    multi_step_trans = np.linalg.matrix_power(trans_mat, time_horizon)
    
    # Get probabilities from the starting state
    probs = multi_step_trans[from_state]
    
    return {state: prob for state, prob in enumerate(probs)}