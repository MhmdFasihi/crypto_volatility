"""
Configuration management module for the crypto volatility analysis system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, Callable
import json
from dotenv import load_dotenv
import logging

class Config:
    """Configuration class for managing application settings."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with optional config file."""
        self._config: Dict[str, Any] = {}
        self._load_defaults()
        if config_file:
            self.load(config_file)
        self._load_env_vars()

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._config = {
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            'vol_window': 30,
            'annualization_factor': 365,
            'api': {
                'yfinance': {
                    'rate_limit': 100,
                    'retry_attempts': 2,
                    'retry_delay': 1
                },
                'deribit': {
                    'rate_limit': 50,
                    'retry_attempts': 2,
                    'retry_delay': 1,
                    'test_mode': True
                }
            },
            'model': {
                'default_window': 60,
                'train_test_split': 0.8,
                'validation_split': 0.2,
                'random_state': 42,
                'forecasting': {
                    'lags': 5,
                    'mlp': {
                        'hidden_layers': [100, 50],
                        'max_iter': 1000
                    },
                    'rnn': {
                        'units': 50,
                        'epochs': 100,
                        'batch_size': 32
                    },
                    'test_size': 0.2
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'max_size': 10485760,  # 10MB
                'backup_count': 5
            }
        }

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            'DATA_DIR': ('data_dir', str),
            'MODELS_DIR': ('models_dir', str),
            'LOGS_DIR': ('logs_dir', str),
            'VOL_WINDOW': ('vol_window', int),
            'ANNUALIZATION_FACTOR': ('annualization_factor', int),
            'DEBUG': ('debug', lambda x: x.lower() == 'true'),
            'LOG_LEVEL': ('logging.level', str),
            'YFINANCE_API_KEY': ('api.yfinance.api_key', str),
            'DERIBIT_API_KEY': ('api.deribit.api_key', str),
            'DERIBIT_TEST_MODE': ('api.deribit.test_mode', lambda x: x.lower() == 'true')
        }

        for env_var, (config_path, cast_func) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                self.set(config_path, cast_func(value))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        try:
            parts = key.split('.')
            value = self._config
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        parts = key.split('.')
        config = self._config
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        config[parts[-1]] = value

    def load(self, config_file: str) -> None:
        """Load configuration from a file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self._config.update(config)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error loading config file: {e}")
            raise

    def save(self, config_file: Optional[str] = None) -> None:
        """Save configuration to a file."""
        if config_file is None:
            config_file = 'config.json'
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self._config, f, indent=4)
        except IOError as e:
            logging.error(f"Error saving config file: {e}")
            raise

    def get_path(self, key: str) -> Path:
        """Get a path from configuration."""
        path_str = self.get(f"{key}")
        if path_str is None:
            raise ValueError(f"Path not found in configuration: {key}")
        return Path(path_str)

    def merge(self, other: 'Config') -> None:
        """Merge another configuration into this one."""
        self._config.update(other._config)

    def __getattr__(self, name: str) -> Any:
        """Get configuration value using attribute access."""
        value = self.get(name)
        if value is None:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{name}'")
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Set configuration value using attribute access."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self.set(name, value)

# Create global configuration instance
config = Config()

def load_config(config_file: str) -> Config:
    """Load configuration from a file and return a new Config instance."""
    return Config(config_file)

# Export commonly used configuration values
DATA_DIR = config.get_path('data_dir')
MODELS_DIR = config.get_path('models_dir')
LOGS_DIR = config.get_path('logs_dir')
VOL_WINDOW = config.get('vol_window')
ANNUALIZATION_FACTOR = config.get('annualization_factor')

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# List of cryptocurrency tickers to analyze
TICKERS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BNB-USD']

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
DEFAULT_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')

# Data interval for yfinance (e.g., '1d' for daily, '1h' for hourly)
DATA_INTERVAL = '1d'

# Model parameters
MLP_HIDDEN_LAYERS = (50, 50)  # tuple of ints
MLP_MAX_ITER = 1000  # int
RNN_UNITS = 50  # int
RNN_EPOCHS = 50  # int
RNN_BATCH_SIZE = 32  # int

# Random seed for reproducibility
RANDOM_SEED = 42  # int

# Minimum data points required for model training
MIN_DATA_POINTS = 100  # int

# Test set size for model evaluation
TEST_SIZE = 0.2  # float