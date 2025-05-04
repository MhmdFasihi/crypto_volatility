"""
Configuration module for the crypto volatility analysis system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv
from .logger import get_logger

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for managing application settings."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        self.root_dir = Path(__file__).parent.parent
        self.config_file = self.root_dir / 'config.json'
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Default configuration
        self.config = {
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            'vol_window': 30,
            'annualization_factor': 365,
            'api': {
                'yfinance': {
                    'rate_limit': 2000,  # requests per hour
                    'retry_attempts': 3,
                    'retry_delay': 1
                },
                'deribit': {
                    'test_mode': True,
                    'rate_limit': 1000,  # requests per hour
                    'retry_attempts': 3,
                    'retry_delay': 1
                }
            },
            'model': {
                'default_window': 30,
                'train_test_split': 0.8,
                'validation_split': 0.1,
                'random_state': 42
            },
            'logging': {
                'level': 'INFO',
                'format': 'json',
                'max_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            }
        }
        
        # Load from config file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                logger.info("Configuration loaded from file")
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
        
        # Override with environment variables
        self._load_env_vars()
    
    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            'DEBUG': ('debug', bool),
            'LOG_LEVEL': ('logging.level', str),
            'YFINANCE_API_KEY': ('api.yfinance.api_key', str),
            'DERIBIT_API_KEY': ('api.deribit.api_key', str),
            'DERIBIT_API_SECRET': ('api.deribit.api_secret', str),
            'DATA_DIR': ('data_dir', str),
            'MODELS_DIR': ('models_dir', str),
            'LOGS_DIR': ('logs_dir', str),
            'VOL_WINDOW': ('vol_window', int),
            'ANNUALIZATION_FACTOR': ('annualization_factor', int)
        }
        
        for env_var, (config_path, type_cast) in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                try:
                    # Handle nested paths
                    if '.' in config_path:
                        parent, child = config_path.split('.')
                        if parent not in self.config:
                            self.config[parent] = {}
                        self.config[parent][child] = type_cast(value)
                    else:
                        self.config[config_path] = type_cast(value)
                except ValueError as e:
                    logger.error(f"Error casting environment variable {env_var}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        try:
            # Handle nested keys
            if '.' in key:
                value = self.config
                for k in key.split('.'):
                    value = value[k]
                return value
            return self.config.get(key, default)
        except KeyError:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving config file: {str(e)}")
    
    def get_path(self, key: str) -> Path:
        """
        Get configuration path.
        
        Args:
            key: Configuration key for path
        
        Returns:
            Path object
        """
        path_str = self.get(key)
        if path_str:
            return self.root_dir / path_str
        return self.root_dir / key

# Create global config instance
config = Config()

# Export commonly used values
DATA_DIR = config.get_path('data_dir')
MODELS_DIR = config.get_path('models_dir')
LOGS_DIR = config.get_path('logs_dir')
VOL_WINDOW = config.get('vol_window')
ANNUALIZATION_FACTOR = config.get('annualization_factor')
DEBUG = config.get('debug', False)
LOG_LEVEL = config.get('logging.level', 'INFO')

# List of cryptocurrency tickers to analyze
TICKERS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD']

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

# Random seed for reproducibility
RANDOM_SEED = 42  # int

# Minimum data points required for model training
MIN_DATA_POINTS = 100  # int

# Test set size for model evaluation
TEST_SIZE = 0.2  # float