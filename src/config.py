# # src/config.py
# """
# Configuration parameters for the crypto volatility analysis project.
# """

# # List of cryptocurrency tickers to analyze
# TICKERS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD']

# # Rolling window for volatility calculation (in days)
# VOL_WINDOW = 30

# # Number of lagged features for forecasting models
# LAGS = 5

# # Number of clusters for cryptocurrency clustering
# N_CLUSTERS = 3

# # Number of volatility states for HMM classification
# HMM_STATES = 3

# # Z-Score threshold for anomaly detection
# ANOMALY_THRESHOLD = 2

# # Default date range for analysis (dynamic - last 365 days)
# from datetime import datetime, timedelta

# DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
# DEFAULT_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# # Data interval for yfinance (e.g., '1d' for daily, '1h' for hourly)
# DATA_INTERVAL = '1d'

# # Model parameters
# MLP_HIDDEN_LAYERS = (50, 50)
# MLP_MAX_ITER = 1000
# RNN_UNITS = 50
# RNN_EPOCHS = 50
# RNN_BATCH_SIZE = 32

# # File paths
# MODEL_DIR = 'models/'
# DATA_DIR = 'data/'

# # Random seed for reproducibility
# RANDOM_SEED = 42

# # Minimum data points required for model training
# MIN_DATA_POINTS = 100

# # Test set size for model evaluation
# TEST_SIZE = 0.2

# # Annualization factor for crypto (365 days since crypto trades 24/7)
# ANNUALIZATION_FACTOR = 365

# src/config.py
"""
Configuration parameters for the crypto volatility analysis project.
"""

# List of cryptocurrency tickers to analyze
TICKERS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD']

# Rolling window for volatility calculation (in days)
VOL_WINDOW = 30

# Number of lagged features for forecasting models
LAGS = 5

# Number of clusters for cryptocurrency clustering
N_CLUSTERS = 3

# Number of volatility states for HMM classification
HMM_STATES = 3

# Z-Score threshold for anomaly detection
ANOMALY_THRESHOLD = 2

# Default date range for analysis (dynamic - last 365 days)
from datetime import datetime, timedelta

DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
DEFAULT_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Data interval for yfinance (e.g., '1d' for daily, '1h' for hourly)
DATA_INTERVAL = '1d'

# Model parameters
MLP_HIDDEN_LAYERS = (50, 50)
MLP_MAX_ITER = 1000
RNN_UNITS = 50
RNN_EPOCHS = 50
RNN_BATCH_SIZE = 32

# File paths
from pathlib import Path
MODEL_DIR = Path('models')
DATA_DIR = 'data/'

# Random seed for reproducibility
RANDOM_SEED = 42

# Minimum data points required for model training
MIN_DATA_POINTS = 100

# Test set size for model evaluation
TEST_SIZE = 0.2

# Annualization factor for crypto (365 days since crypto trades 24/7)
ANNUALIZATION_FACTOR = 365