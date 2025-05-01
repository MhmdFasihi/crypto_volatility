# Crypto Volatility Analysis Framework

A comprehensive Python framework for analyzing cryptocurrency market volatility using machine learning techniques. This project implements volatility forecasting, clustering, regime classification, and anomaly detection with an interactive Streamlit dashboard.

## Features

- **Volatility Forecasting**: ML-based predictions using MLP, RNN, LSTM, and GRU models
- **Volatility Clustering**: Group cryptocurrencies with similar volatility patterns using DTW
- **Regime Classification**: Identify market states using Hidden Markov Models (HMM)
- **Anomaly Detection**: Detect unusual volatility patterns with multiple algorithms
- **Interactive Dashboard**: Real-time visualization and analysis with Streamlit
- **Deribit Integration**: Options market data for implied volatility analysis

## Quick Start

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto_volatility_analysis.git
cd crypto_volatility_analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Train Models

Train models for a specific ticker:
```bash
python src/train_models.py --ticker BTC-USD --model all
```

Train specific model type:
```bash
python src/train_models.py --ticker ETH-USD --model mlp
```

#### 2. Launch Dashboard

Start the Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```

Access the dashboard at `http://localhost:8501`

#### 3. Run Tests

Execute all unit tests:
```bash
python tests/run_tests.py
```

## Project Structure

```
crypto_volatility_analysis/
├── data/                    # Downloaded data (optional)
├── models/                  # Saved ML models
├── src/                     # Source code
│   ├── data_acquisition.py  # Data fetching (yfinance + Deribit)
│   ├── preprocessing.py     # Data preprocessing and feature engineering
│   ├── forecasting.py       # Volatility forecasting models
│   ├── clustering.py        # Cryptocurrency clustering
│   ├── classification.py    # Volatility regime classification
│   ├── anomaly_detection.py # Anomaly detection algorithms
│   ├── dashboard.py         # Streamlit dashboard
│   ├── train_models.py      # Model training script
│   └── config.py            # Configuration parameters
├── tests/                   # Unit tests
├── notebooks/               # Exploratory analysis (optional)
├── requirements.txt         # Dependencies
├── pytest.ini              # Pytest configuration
└── README.md               # Documentation
```

## Core Components

### 1. Data Acquisition

- Fetches historical price data using yfinance
- Integrates Deribit API for options market data
- Supports multiple cryptocurrency pairs
- Handles data validation and error handling

### 2. Preprocessing

- Calculates log returns and realized volatility
- Implements advanced volatility metrics (Parkinson, Garman-Klass, Rogers-Satchell)
- Creates lagged features for ML models
- Handles missing data and normalization

### 3. Volatility Forecasting

- **MLP**: Multilayer Perceptron for short-term predictions
- **RNN/LSTM/GRU**: Recurrent networks for longer horizons
- Time series cross-validation
- Multi-step ahead forecasting
- Ensemble methods

### 4. Clustering

- Dynamic Time Warping (DTW) for similarity measurement
- Agglomerative and K-means clustering
- Silhouette score optimization
- Cluster visualization and analysis

### 5. Regime Classification

- Hidden Markov Models for state detection
- Automatic regime labeling (Low, Medium, High)
- Transition probability analysis
- Regime duration statistics

### 6. Anomaly Detection

- Z-Score based detection
- Isolation Forest algorithm
- Mahalanobis distance method
- Random Forest classifier
- Ensemble anomaly detection

### 7. Interactive Dashboard

- Real-time data visualization
- Model performance metrics
- Interactive parameter tuning
- Export functionality
- Multi-tab interface

## Configuration

Edit `src/config.py` to customize:

```python
# Default cryptocurrency tickers
TICKERS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD']

# Volatility calculation parameters
VOL_WINDOW = 30
LAGS = 5

# Model parameters
HMM_STATES = 3
N_CLUSTERS = 3
ANOMALY_THRESHOLD = 2
```

## API Reference

### Data Acquisition

```python
from src.data_acquisition import get_data

# Fetch price data
data = get_data('BTC-USD', '2024-01-01', '2024-12-31')

# Get combined volatility data (with Deribit IV)
data = get_combined_volatility_data('BTC-USD', start_date, end_date)
```

### Preprocessing

```python
from src.preprocessing import calculate_volatility

# Calculate realized volatility
data = calculate_volatility(data, window=30)

# Create features for modeling
data = create_volatility_features(data, lags=5)
```

### Forecasting

```python
from src.forecasting import train_mlp, forecast_next_values

# Train MLP model
model, scaler = train_mlp(X_train, y_train)

# Generate forecast
forecast = forecast_next_values(model, recent_data, scaler, n_ahead=5)
```

## Performance Metrics

The framework evaluates models using:

- **Forecasting**: MAE, MSE, RMSE, R², MAPE
- **Clustering**: Silhouette score
- **Classification**: State transition accuracy
- **Anomaly Detection**: Precision, Recall, F1, AUC

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Testing

Run tests with coverage:
```bash
pytest tests/ --cov=src
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Research papers on cryptocurrency volatility modeling
- Deribit API for options market data
- yfinance for historical price data
- Streamlit for dashboard framework

## Contact

For questions or support, please open an issue on GitHub.