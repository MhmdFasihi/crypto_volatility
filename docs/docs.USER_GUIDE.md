# User Guide

## Getting Started

### Installation

1. **System Requirements**
   - Python 3.8 or higher
   - 4GB RAM minimum (8GB recommended)
   - 2GB free disk space

2. **Setup Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### First Run

1. **Train Initial Models**
   ```bash
   python src/train_models.py --ticker BTC-USD --model all
   ```

2. **Launch Dashboard**
   ```bash
   streamlit run src/dashboard.py
   ```

3. **Access Dashboard**
   - Open browser at `http://localhost:8501`

## Dashboard Guide

### Overview Tab

The Overview tab provides:
- Current price and volatility metrics
- Historical price chart
- Realized vs implied volatility comparison
- Multiple volatility metrics visualization

**Key Features:**
- Real-time data updates
- Interactive charts with zoom/pan
- Customizable date ranges

### Forecasting Tab

Generate volatility predictions using:
- **MLP**: Best for 1-3 day forecasts
- **RNN**: Good for 5-7 day forecasts  
- **LSTM/GRU**: Best for 7+ day forecasts

**Steps:**
1. Select model type
2. Set forecast horizon
3. Click "Generate Forecast"
4. View predictions and confidence intervals

### Regime Analysis Tab

Identify market states:
- **Low Volatility**: Calm market conditions
- **Medium Volatility**: Normal trading
- **High Volatility**: Turbulent markets

**Features:**
- Current regime indicator
- State transition probabilities
- Historical regime visualization

### Anomaly Detection Tab

Detect unusual volatility patterns:
- **Z-Score Method**: Simple threshold-based
- **Ensemble Method**: Combines multiple algorithms

**Interpretation:**
- Red markers indicate anomalies
- Check anomaly statistics for patterns
- Review recent anomalies table

### Clustering Tab

Group similar cryptocurrencies:
1. Set number of clusters
2. Choose clustering method
3. Click "Run Clustering Analysis"
4. View similar cryptocurrencies

### Data Exploration Tab

- View raw data
- Download datasets
- Explore summary statistics

## Model Training

### Basic Training

Train specific model:
```bash
python src/train_models.py --ticker ETH-USD --model mlp
```

### Advanced Options

Include Deribit data:
```bash
python src/train_models.py --ticker BTC-USD --use-deribit
```

Use cross-validation:
```bash
python src/train_models.py --ticker BTC-USD --cv
```

### Batch Training

Train all tickers:
```bash
python src/train_models.py --model all
```

## Configuration

### Modify Settings

Edit `src/config.py`:

```python
# Add new tickers
TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD']

# Adjust volatility window
VOL_WINDOW = 20

# Change anomaly threshold
ANOMALY_THRESHOLD = 2.5
```

### Performance Tuning

For better performance:
- Reduce `VOL_WINDOW` for faster updates
- Lower `LAGS` for quicker training
- Decrease `HMM_STATES` for simpler models

## Troubleshooting

### Common Issues

1. **"No data available" error**
   - Check internet connection
   - Verify ticker symbol
   - Try different date range

2. **"Model not found" error**
   - Train the model first
   - Check model directory permissions

3. **Slow dashboard performance**
   - Reduce date range
   - Lower volatility window
   - Close other applications

### Error Messages

| Error | Solution |
|-------|----------|
| "Insufficient data" | Use longer date range |
| "Model convergence failed" | Try different parameters |
| "Connection timeout" | Check network/API status |

## Best Practices

### Data Management

1. **Regular Updates**
   - Retrain models monthly
   - Update data daily
   - Monitor API limits

2. **Backup Strategy**
   - Save trained models
   - Export important data
   - Version control configs

### Analysis Tips

1. **Volatility Forecasting**
   - Use ensemble methods for critical decisions
   - Consider multiple time horizons
   - Validate with backtesting

2. **Anomaly Detection**
   - Adjust thresholds based on market conditions
   - Investigate false positives
   - Combine with regime analysis

3. **Clustering Analysis**
   - Update clusters regularly
   - Validate with fundamental analysis
   - Use for portfolio diversification

## Advanced Usage

### Custom Models

Add new model architectures:

```python
# In src/forecasting.py
def train_custom_model(X_train, y_train):
    model = YourCustomModel()
    # Training logic here
    return model, scaler
```

### API Integration

Connect additional data sources:

```python
# In src/data_acquisition.py
def get_custom_data(ticker, source='your_api'):
    # Custom API integration
    pass
```

### Dashboard Customization

Add new visualization tabs:

```python
# In src/dashboard.py
with tab7:
    st.header("Custom Analysis")
    # Your custom visualizations
```

## Security Considerations

1. **API Keys**
   - Store in environment variables
   - Never commit to version control
   - Rotate regularly

2. **Data Privacy**
   - Local data storage only
   - No external data sharing
   - Secure file permissions

3. **Dashboard Access**
   - Run locally only
   - Use firewall for network access
   - Implement authentication if needed

## Performance Optimization

### Memory Management

- Process data in chunks
- Clear unused variables
- Use efficient data types

### Speed Improvements

- Enable caching in Streamlit
- Optimize database queries
- Use multiprocessing for training

### Resource Monitoring

```python
import psutil

# Monitor memory usage
memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_usage:.2f} MB")
```

## FAQ

**Q: How often should I retrain models?**
A: Monthly for stable performance, weekly for volatile markets.

**Q: Can I add custom indicators?**
A: Yes, modify the preprocessing module to add new features.

**Q: How do I handle missing data?**
A: The system automatically handles gaps, but extended periods may affect accuracy.

**Q: Is real-time analysis possible?**
A: Yes, with modifications to support streaming data.

## Support

For additional help:
- Open GitHub issues for bugs
- Check documentation updates
- Join community discussions

## Version History

- v1.0.0: Initial release
- v1.1.0: Added Deribit integration
- v1.2.0: Enhanced anomaly detection
- v1.3.0: Improved dashboard performance