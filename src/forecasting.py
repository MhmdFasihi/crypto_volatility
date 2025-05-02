# src/forecasting.py
"""
Volatility forecasting module implementing MLP and RNN models.
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Tuple, Dict, Any, Optional, List
from src.config import (
    LAGS, MLP_HIDDEN_LAYERS, MLP_MAX_ITER, RNN_UNITS, 
    RNN_EPOCHS, RNN_BATCH_SIZE, RANDOM_SEED, TEST_SIZE,
    MODEL_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data: pd.DataFrame, target_col: str = 'Volatility', 
                lags: int = LAGS) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for forecasting.
    
    Args:
        data: DataFrame with volatility data
        target_col: Target column name
        lags: Number of lagged features
    
    Returns:
        Tuple of (X, y) for features and target
    """
    df = data[[target_col]].dropna().copy()
    
    # Create lagged features
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df[target_col].shift(i)
    
    # Drop rows with NaN (from lagging)
    df = df.dropna()
    
    # Define features and target
    feature_cols = [f'lag_{i}' for i in range(1, lags + 1)]
    X = df[feature_cols]
    y = df[target_col]
    
    logger.info(f"Prepared data with shape X: {X.shape}, y: {y.shape}")
    
    return X, y

def prepare_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for RNN input.
    
    Args:
        X: Feature array
        y: Target array
        sequence_length: Length of sequences (default: number of features)
    
    Returns:
        Tuple of reshaped (X, y) for RNN
    """
    if sequence_length is None:
        sequence_length = X.shape[1]
    
    # Reshape X for RNN input [samples, timesteps, features]
    X_seq = X.reshape(X.shape[0], sequence_length, 1)
    
    return X_seq, y

def train_mlp(X_train: np.ndarray, y_train: np.ndarray, 
              hidden_layers: Tuple = MLP_HIDDEN_LAYERS, 
              max_iter: int = MLP_MAX_ITER) -> Tuple[MLPRegressor, StandardScaler]:
    """
    Train an MLP model for volatility forecasting.
    
    Args:
        X_train: Training features
        y_train: Training target
        hidden_layers: Hidden layer sizes
        max_iter: Maximum iterations
    
    Returns:
        Tuple of (trained model, scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    model.fit(X_train_scaled, y_train)
    
    logger.info(f"Trained MLP with hidden layers: {hidden_layers}")
    
    return model, scaler

def train_rnn(X_train: np.ndarray, y_train: np.ndarray, 
              model_type: str = 'SimpleRNN',
              units: int = RNN_UNITS,
              epochs: int = RNN_EPOCHS,
              batch_size: int = RNN_BATCH_SIZE) -> Tuple[Sequential, StandardScaler]:
    """
    Train an RNN model for volatility forecasting.
    
    Args:
        X_train: Training features (reshaped for RNN)
        y_train: Training target
        model_type: Type of RNN ('SimpleRNN', 'LSTM', 'GRU')
        units: Number of RNN units
        epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        Tuple of (trained model, scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    # Build model
    model = Sequential()
    
    if model_type == 'SimpleRNN':
        model.add(SimpleRNN(units, activation='tanh', input_shape=(X_train.shape[1], 1)))
    elif model_type == 'LSTM':
        model.add(LSTM(units, activation='tanh', input_shape=(X_train.shape[1], 1)))
    elif model_type == 'GRU':
        model.add(GRU(units, activation='tanh', input_shape=(X_train.shape[1], 1)))
    else:
        raise ValueError(f"Unknown RNN type: {model_type}")
    
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    # Train model
    model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    logger.info(f"Trained {model_type} with {units} units")
    
    return model, scaler

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                  scaler: Optional[StandardScaler] = None,
                  is_rnn: bool = False) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        scaler: Feature scaler (if used)
        is_rnn: Whether the model is an RNN
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Scale features if scaler provided
    if scaler is not None:
        if is_rnn:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            X_test_scaled = scaler.transform(X_test_flat)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
        else:
            X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Make predictions
    if is_rnn:
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    else:
        y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    }
    
    logger.info(f"Model evaluation - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
    
    return metrics

def save_model(model: Any, scaler: StandardScaler, filename: str, is_rnn: bool = False):
    """
    Save the trained model and scaler.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        filename: File path to save (without extension)
        is_rnn: Whether the model is an RNN
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if is_rnn:
        model.save(os.path.join(MODEL_DIR, f"{filename}.h5"))
    else:
        joblib.dump(model, os.path.join(MODEL_DIR, f"{filename}.pkl"))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{filename}_scaler.pkl"))
    
    logger.info(f"Model saved to {MODEL_DIR}/{filename}")

def load_model(filename: str, is_rnn: bool = False) -> Tuple[Any, StandardScaler]:
    """
    Load a trained model and scaler.
    
    Args:
        filename: File path to load (without extension)
        is_rnn: Whether the model is an RNN
    
    Returns:
        Tuple of (loaded model, scaler)
    """
    if is_rnn:
        model = keras_load_model(os.path.join(MODEL_DIR, f"{filename}.h5"))
    else:
        model = joblib.load(os.path.join(MODEL_DIR, f"{filename}.pkl"))
    
    # Load scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{filename}_scaler.pkl"))
    
    logger.info(f"Model loaded from {MODEL_DIR}/{filename}")
    
    return model, scaler

def forecast_next_values(model: Any, X_recent: np.ndarray, scaler: StandardScaler, 
                        is_rnn: bool = False, n_ahead: int = 5) -> np.ndarray:
    """
    Forecast the next n values using the trained model.
    
    Args:
        model: Trained model
        X_recent: Recent feature values
        scaler: Feature scaler
        is_rnn: Whether the model is an RNN
        n_ahead: Number of steps to forecast ahead
    
    Returns:
        Array of forecasted values
    """
    forecasts = []
    current_input = X_recent.copy()
    
    for _ in range(n_ahead):
        # Scale input
        if is_rnn:
            input_flat = current_input.reshape(1, -1)
            input_scaled = scaler.transform(input_flat)
            input_scaled = input_scaled.reshape(1, current_input.shape[0], 1)
        else:
            input_scaled = scaler.transform(current_input.reshape(1, -1))
        
        # Predict
        if is_rnn:
            pred = model.predict(input_scaled, verbose=0)[0, 0]
        else:
            pred = model.predict(input_scaled)[0]
        
        forecasts.append(pred)
        
        # Update input for next prediction
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred
    
    return np.array(forecasts)

def cross_validate_time_series(model_func, X: np.ndarray, y: np.ndarray, 
                              n_splits: int = 5, **model_kwargs) -> List[Dict[str, float]]:
    """
    Perform time series cross-validation.
    
    Args:
        model_func: Function to create and train model
        X: Features
        y: Target
        n_splits: Number of CV splits
        **model_kwargs: Arguments for model function
    
    Returns:
        List of metrics for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        # Train model
        model, scaler = model_func(X_train_cv, y_train_cv, **model_kwargs)
        
        # Evaluate
        is_rnn = 'rnn' in str(model_func.__name__).lower()
        metrics = evaluate_model(model, X_test_cv, y_test_cv, scaler, is_rnn)
        metrics['fold'] = fold
        
        metrics_list.append(metrics)
        logger.info(f"Fold {fold}: MAE={metrics['mae']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in metrics_list])
        for metric in metrics_list[0].keys() if metric != 'fold'
    }
    
    logger.info(f"Cross-validation average MAE: {avg_metrics['mae']:.4f}")
    
    return metrics_list, avg_metrics

def ensemble_forecast(models: List[Any], X: np.ndarray, scalers: List[StandardScaler], 
                     is_rnn: List[bool], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Create ensemble forecast from multiple models.
    
    Args:
        models: List of trained models
        X: Input features
        scalers: List of scalers for each model
        is_rnn: List indicating if each model is RNN
        weights: Optional weights for each model
    
    Returns:
        Ensemble forecast
    """
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    predictions = []
    
    for model, scaler, rnn_flag in zip(models, scalers, is_rnn):
        # Scale input
        if rnn_flag:
            X_flat = X.reshape(X.shape[0], -1)
            X_scaled = scaler.transform(X_flat)
            X_scaled = X_scaled.reshape(X.shape)
        else:
            X_scaled = scaler.transform(X)
        
        # Predict
        if rnn_flag:
            pred = model.predict(X_scaled, verbose=0).flatten()
        else:
            pred = model.predict(X_scaled)
        
        predictions.append(pred)
    
    # Weighted average
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    return ensemble_pred