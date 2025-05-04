"""
Configuration parameters for the crypto volatility analysis project.
"""

# List of cryptocurrency tickers to analyze
TICKERS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD']

# Rolling window for volatility calculation (in days)
VOL_WINDOW = 30  # int

# Number of lagged features for forecasting models
LAGS = 5  # int

# Number of clusters for cryptocurrency clustering
N_CLUSTERS = 3  # int

# Number of volatility states for HMM classification
HMM_STATES = 3  # int

# Z-Score threshold for anomaly detection
ANOMALY_THRESHOLD = 2.0  # float

# Default date range for analysis (dynamic - last 365 days)
from datetime import datetime, timedelta

DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
DEFAULT_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Data interval for yfinance (e.g., '1d' for daily, '1h' for hourly)
DATA_INTERVAL = '1d'

# Model parameters
MLP_HIDDEN_LAYERS = (50, 50)  # tuple of ints
MLP_MAX_ITER = 1000  # int
RNN_UNITS = 50  # int
RNN_EPOCHS = 50  # int
RNN_BATCH_SIZE = 32  # int

# File paths
import os
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Random seed for reproducibility
RANDOM_SEED = 42  # int

# Minimum data points required for model training
MIN_DATA_POINTS = 100  # int

# Test set size for model evaluation
TEST_SIZE = 0.2  # float

# Annualization factor for crypto (365 days since crypto trades 24/7)
ANNUALIZATION_FACTOR = 365  # int

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)