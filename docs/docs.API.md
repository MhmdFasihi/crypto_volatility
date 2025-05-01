# API Documentation

## Data Acquisition Module

### `get_data(ticker, start_date, end_date, interval='1d')`

Fetches historical price data for a cryptocurrency.

**Parameters:**
- `ticker` (str): Cryptocurrency ticker (e.g., 'BTC-USD')
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format
- `interval` (str): Data interval ('1d', '1h', etc.)

**Returns:**
- `pd.DataFrame`: Historical price data with OHLCV columns

**Example:**
```python
data = get_data('BTC-USD', '2024-01-01', '2024-12-31')
```

### `DeribitAPI` Class

Interface for Deribit options market data.

**Methods:**
- `get_implied_volatility_data(currency, days_back)`: Fetch IV data
- `get_aggregated_iv_by_expiry(currency, days_back)`: Get IV by expiry date

**Example:**
```python
deribit = DeribitAPI(test_mode=True)
iv_data = deribit.get_implied_volatility_data('BTC', days_back=30)
```

## Preprocessing Module

### `calculate_returns(data, price_col='Close')`

Calculates log returns from price data.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with price data
- `price_col` (str): Column name for prices

**Returns:**
- `pd.DataFrame`: DataFrame with 'Returns' column added

### `calculate_volatility(data, window=30, returns_col='Returns')`

Calculates realized volatility using rolling standard deviation.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with returns
- `window` (int): Rolling window size
- `returns_col` (str): Column name for returns

**Returns:**
- `pd.DataFrame`: DataFrame with 'Volatility' column added

### `calculate_advanced_volatility_metrics(data, window=30)`

Calculates Parkinson, Garman-Klass, and Rogers-Satchell volatility.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with OHLC data
- `window` (int): Rolling window size

**Returns:**
- `pd.DataFrame`: DataFrame with advanced volatility metrics

## Forecasting Module

### `train_mlp(X_train, y_train, hidden_layers=(50, 50), max_iter=1000)`

Trains a Multilayer Perceptron for volatility forecasting.

**Parameters:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training target
- `hidden_layers` (tuple): Hidden layer sizes
- `max_iter` (int): Maximum iterations

**Returns:**
- `tuple`: (trained model, scaler)

### `train_rnn(X_train, y_train, model_type='SimpleRNN', units=50)`

Trains an RNN model (SimpleRNN, LSTM, or GRU).

**Parameters:**
- `X_train` (np.ndarray): Training features (3D array)
- `y_train` (np.ndarray): Training target
- `model_type` (str): Type of RNN ('SimpleRNN', 'LSTM', 'GRU')
- `units` (int): Number of RNN units

**Returns:**
- `tuple`: (trained model, scaler)

### `forecast_next_values(model, X_recent, scaler, is_rnn=False, n_ahead=5)`

Generates multi-step ahead forecasts.

**Parameters:**
- `model`: Trained model
- `X_recent` (np.ndarray): Recent feature values
- `scaler`: Feature scaler
- `is_rnn` (bool): Whether model is RNN
- `n_ahead` (int): Steps to forecast ahead

**Returns:**
- `np.ndarray`: Forecasted values

## Classification Module

### `train_hmm(data, feature_cols=['Volatility'], n_states=3)`

Trains a Hidden Markov Model for regime classification.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with volatility data
- `feature_cols` (list): Features to use
- `n_states` (int): Number of hidden states

**Returns:**
- `tuple`: (trained HMM, scaler)

### `predict_states(model, data, scaler, feature_cols=['Volatility'])`

Predicts volatility states using trained HMM.

**Parameters:**
- `model`: Trained HMM
- `data` (pd.DataFrame): Data for prediction
- `scaler`: Feature scaler
- `feature_cols` (list): Feature columns

**Returns:**
- `np.ndarray`: Predicted states

### `get_current_regime(model, data, scaler, state_stats, feature_cols=['Volatility'])`

Gets the current volatility regime.

**Returns:**
- `tuple`: (regime name, state index, probability)

## Anomaly Detection Module

### `detect_anomalies_zscore(data, column='Volatility', window=30, threshold=2)`

Detects anomalies using rolling Z-Score.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with volatility data
- `column` (str): Column to analyze
- `window` (int): Rolling window size
- `threshold` (float): Z-Score threshold

**Returns:**
- `pd.DataFrame`: DataFrame with anomaly indicators

### `ensemble_anomaly_detection(data, methods=['zscore', 'isolation_forest'], threshold=0.5)`

Combines multiple anomaly detection methods.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with volatility data
- `methods` (list): Detection methods to use
- `threshold` (float): Consensus threshold

**Returns:**
- `pd.DataFrame`: DataFrame with ensemble results

## Clustering Module

### `cluster_tickers(tickers, start_date, end_date, n_clusters=3)`

Clusters cryptocurrencies based on volatility patterns.

**Parameters:**
- `tickers` (list): List of cryptocurrency tickers
- `start_date` (str): Start date
- `end_date` (str): End date
- `n_clusters` (int): Number of clusters

**Returns:**
- `dict`: Mapping of tickers to cluster labels

### `get_similar_tickers(ticker, cluster_mapping, max_similar=5)`

Finds similar cryptocurrencies within the same cluster.

**Parameters:**
- `ticker` (str): Target ticker
- `cluster_mapping` (dict): Cluster assignments
- `max_similar` (int): Maximum similar tickers to return

**Returns:**
- `list`: Similar ticker symbols