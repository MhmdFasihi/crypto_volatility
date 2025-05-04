# src/data_acquisition.py
"""
Data acquisition module for fetching cryptocurrency data using yfinance with Deribit fallback.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Union, List
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import logging
import time
import random
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical price data for a given ticker with Deribit fallback.
    
    Args:
        ticker: Crypto ticker (e.g., 'BTC-USD')
        start_date: Start date (e.g., '2020-01-01')
        end_date: End date (e.g., '2025-05-01')
        interval: Data interval (e.g., '1d' for daily)
    
    Returns:
        DataFrame with price data
    """
    # Try to load from local cache first
    local_file = f"data/{ticker.replace('-', '_')}_{start_date}_{end_date}_{interval}.csv"
    if os.path.exists(local_file):
        try:
            data = pd.read_csv(local_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded cached data for {ticker} from {local_file}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    # Try yfinance with retries
    max_retries = 3
    for retry in range(max_retries):
        try:
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date} (Attempt {retry+1}/{max_retries})")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            
            if not data.empty:
                logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
                
                # Save to cache
                try:
                    os.makedirs("data", exist_ok=True)
                    data.to_csv(local_file)
                    logger.info(f"Saved data to cache: {local_file}")
                except Exception as e:
                    logger.warning(f"Failed to cache data: {e}")
                
                return data
            else:
                logger.warning(f"Empty data received for {ticker}, retry {retry+1}/{max_retries}")
                if retry < max_retries - 1:
                    backoff = (2 ** retry) + random.random()
                    logger.info(f"Backing off for {backoff:.2f} seconds")
                    time.sleep(backoff)
        except Exception as e:
            logger.warning(f"yfinance error: {e}, retry {retry+1}/{max_retries}")
            if retry < max_retries - 1:
                backoff = (2 ** retry) + random.random()
                logger.info(f"Backing off for {backoff:.2f} seconds")
                time.sleep(backoff)
    
    # If yfinance failed, try Deribit as fallback
    logger.info(f"yfinance failed for {ticker}, trying Deribit fallback")
    deribit_data = get_price_data_from_deribit(ticker, start_date, end_date, interval)
    
    if not deribit_data.empty:
        logger.info(f"Successfully fetched {len(deribit_data)} rows from Deribit for {ticker}")
        return deribit_data
    
    # If all methods fail, return empty DataFrame
    logger.error(f"All data acquisition methods failed for {ticker}")
    return pd.DataFrame()

def get_price_data_from_deribit(ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Get historical price data from Deribit API.
    
    Args:
        ticker: Crypto ticker (e.g., 'BTC-USD')
        start_date: Start date
        end_date: End date
        interval: Data interval
    
    Returns:
        DataFrame with price data
    """
    # Extract currency from ticker (e.g., 'BTC' from 'BTC-USD')
    currency = ticker.split('-')[0]
    
    try:
        # Calculate timestamps
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        days_back = (datetime.now() - start_datetime).days
        
        # Get data from Deribit
        deribit = DeribitAPI(test_mode=False)  # Use production API
        
        # Get trades data which contains price information
        response = deribit.get_trades_by_currency_and_time(
            currency=currency,
            start_timestamp=int(start_datetime.timestamp() * 1000),
            end_timestamp=int(end_datetime.timestamp() * 1000),
            count=20000  # Request a large number of trades
        )
        
        if 'result' in response and 'trades' in response['result']:
            trades = response['result']['trades']
            
            if trades:
                # Convert trades to DataFrame
                df = pd.DataFrame(trades)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Resample to the requested interval
                freq_map = {'1d': 'D', '1h': 'H', '1m': 'T', '1w': 'W'}
                freq = freq_map.get(interval, 'D')
                
                # Create OHLC data
                price_data = df['index_price'].resample(freq).ohlc()
                
                # Rename columns to match yfinance format
                price_data.columns = ['Open', 'High', 'Low', 'Close']
                
                # Add a placeholder volume column
                price_data['Volume'] = np.nan
                
                # Filter to the requested date range
                price_data = price_data.loc[start_datetime:end_datetime]
                
                return price_data
        
        logger.warning(f"No trade data found from Deribit for {ticker}")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error getting price data from Deribit: {e}")
        return pd.DataFrame()

def get_multiple_tickers(tickers: List[str], start_date: str, end_date: str, interval: str = '1d') -> dict:
    """
    Fetch data for multiple tickers.
    
    Args:
        tickers: List of crypto tickers
        start_date: Start date
        end_date: End date
        interval: Data interval
    
    Returns:
        Dictionary mapping tickers to DataFrames
    """
    data_dict = {}
    for ticker in tickers:
        data = get_data(ticker, start_date, end_date, interval)
        if not data.empty:
            data_dict[ticker] = data
    return data_dict

def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker exists and has data.
    
    Args:
        ticker: Crypto ticker to validate
    
    Returns:
        Boolean indicating if ticker is valid
    """
    try:
        # First try yfinance
        try:
            test_data = yf.Ticker(ticker).info
            if bool(test_data):
                return True
        except Exception:
            pass
        
        # Then try Deribit
        if '-USD' in ticker:
            currency = ticker.split('-')[0]
            deribit = DeribitAPI(test_mode=False)
            response = deribit.get_trades_by_currency_and_time(
                currency=currency,
                start_timestamp=int((datetime.now() - timedelta(days=1)).timestamp() * 1000),
                end_timestamp=int(datetime.now().timestamp() * 1000),
                count=1
            )
            return 'result' in response and 'trades' in response['result'] and len(response['result']['trades']) > 0
    except:
        return False
    
    return False

def get_ticker_info(ticker: str) -> dict:
    """
    Get detailed information about a ticker.
    
    Args:
        ticker: Crypto ticker
    
    Returns:
        Dictionary with ticker information
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.info
    except Exception as e:
        logger.error(f"Error getting info for {ticker}: {e}")
        return {}

def save_data_to_csv(data: pd.DataFrame, filename: str, data_dir: str = 'data/') -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        data: DataFrame to save
        filename: Name of the file
        data_dir: Directory to save the file
    """
    import os
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    data.to_csv(filepath)
    logger.info(f"Data saved to {filepath}")

def load_data_from_csv(filename: str, data_dir: str = 'data/') -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filename: Name of the file
        data_dir: Directory containing the file
    
    Returns:
        DataFrame with loaded data
    """
    import os
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Data loaded from {filepath}")
        return data
    else:
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()

class DeribitAPI:
    """
    Class to handle Deribit API interactions for implied volatility data.
    """
    
    def __init__(self, test_mode: bool = True):
        """
        Initialize Deribit API client.
        
        Args:
            test_mode: If True, use test.deribit.com, else use www.deribit.com
        """
        self.base_url = 'wss://test.deribit.com/ws/api/v2' if test_mode else 'wss://www.deribit.com/ws/api/v2'
        logger.info(f"Initialized Deribit API with URL: {self.base_url}")
    
    async def call_api(self, msg: dict) -> dict:
        """
        Make an API call to Deribit.
        
        Args:
            msg: Message dictionary to send
        
        Returns:
            API response as dictionary
        """
        try:
            async with websockets.connect(self.base_url) as websocket:
                await websocket.send(json.dumps(msg))
                response = await websocket.recv()
                return json.loads(response)
        except Exception as e:
            logger.error(f"Deribit API error: {e}")
            return {"error": str(e)}
    
    def get_trades_by_currency_and_time(self, currency: str, start_timestamp: int, 
                                       end_timestamp: int, count: int = 1000) -> dict:
        """
        Get trades for a currency within a time range.
        
        Args:
            currency: Currency (e.g., 'BTC', 'ETH')
            start_timestamp: Start timestamp in milliseconds
            end_timestamp: End timestamp in milliseconds
            count: Number of trades to retrieve
        
        Returns:
            API response with trades data
        """
        msg = {
            "jsonrpc": "2.0",
            "id": 1469,
            "method": "public/get_last_trades_by_currency_and_time",
            "params": {
                "currency": currency,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "count": count
            }
        }
        
        # Handle potential asyncio issues more gracefully
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.call_api(msg))
    
    def get_implied_volatility_data(self, currency: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get implied volatility data for a currency.
        
        Args:
            currency: Currency (e.g., 'BTC', 'ETH')
            days_back: Number of days to look back
        
        Returns:
            DataFrame with IV data
        """
        end_timestamp = int(datetime.now().timestamp() * 1000)
        start_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        try:
            response = self.get_trades_by_currency_and_time(
                currency=currency,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                count=10000  # Get more trades to ensure coverage
            )
            
            if 'result' in response and 'trades' in response['result']:
                trades = response['result']['trades']
                if trades:
                    # Convert to DataFrame
                    df = pd.DataFrame(trades)
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Select relevant columns
                    iv_df = df[['instrument_name', 'price', 'mark_price', 'iv', 'index_price', 'direction', 'amount']]
                    
                    logger.info(f"Retrieved {len(iv_df)} IV data points for {currency}")
                    return iv_df
                else:
                    logger.warning(f"No trades found for {currency}")
                    return pd.DataFrame()
            else:
                logger.error(f"Unexpected response format: {response}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting IV data for {currency}: {e}")
            return pd.DataFrame()
    
    def get_aggregated_iv_by_expiry(self, currency: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get aggregated implied volatility by expiry date.
        
        Args:
            currency: Currency (e.g., 'BTC', 'ETH')
            days_back: Number of days to look back
        
        Returns:
            DataFrame with aggregated IV by expiry
        """
        iv_data = self.get_implied_volatility_data(currency, days_back)
        
        if iv_data.empty:
            return pd.DataFrame()
        
        # Extract expiry date from instrument name
        iv_data['expiry'] = iv_data['instrument_name'].apply(
            lambda x: pd.to_datetime(x.split('-')[1], format='%d%b%y') if '-' in x else pd.NaT
        )
        
        # Aggregate by expiry and timestamp
        aggregated = iv_data.groupby(['expiry', pd.Grouper(freq='1H')]).agg({
            'iv': 'mean',
            'price': 'mean',
            'index_price': 'mean',
            'amount': 'sum'
        }).reset_index()
        
        return aggregated

def get_combined_volatility_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get combined realized and implied volatility data.
    
    Args:
        ticker: Crypto ticker (e.g., 'BTC-USD')
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with both realized and implied volatility
    """
    # Get price data (with Deribit fallback)
    price_data = get_data(ticker, start_date, end_date)
    
    if price_data.empty:
        logger.warning(f"No price data available for {ticker}")
        return pd.DataFrame()
    
    # Extract currency from ticker (e.g., 'BTC' from 'BTC-USD')
    currency = ticker.split('-')[0]
    
    # Get implied volatility from Deribit
    deribit = DeribitAPI(test_mode=False)  # Use production API
    iv_data = deribit.get_implied_volatility_data(currency)
    
    if not iv_data.empty:
        # Aggregate IV data to daily level
        daily_iv = iv_data.resample('D').agg({'iv': 'mean'})
        
        # Merge with price data
        combined_data = price_data.join(daily_iv, how='left')
        combined_data.rename(columns={'iv': 'ImpliedVolatility'}, inplace=True)
        return combined_data
    
    return price_data

def get_synthetic_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic data as a last resort when all real data sources fail.
    For testing and development purposes only.
    
    Args:
        ticker: Crypto ticker
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with synthetic price data
    """
    logger.warning(f"Generating synthetic data for {ticker}")
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Set base parameters based on ticker
    if 'BTC' in ticker:
        base_price = 30000
        volatility = 0.03
    elif 'ETH' in ticker:
        base_price = 2000
        volatility = 0.04
    elif 'XRP' in ticker:
        base_price = 0.5
        volatility = 0.05
    else:
        base_price = 100
        volatility = 0.03
    
    # Generate random walk for close prices
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, volatility, size=len(dates))
    close = base_price * np.cumprod(1 + returns)
    
    # Create OHLC data
    data = pd.DataFrame({
        'Open': close * np.random.uniform(0.99, 1.01, size=len(dates)),
        'High': close * np.random.uniform(1.01, 1.05, size=len(dates)),
        'Low': close * np.random.uniform(0.95, 0.99, size=len(dates)),
        'Close': close,
        'Volume': np.random.randint(1000, 10000, size=len(dates)) * close
    }, index=dates)
    
    # Make sure High >= Open, Close, Low and Low <= Open, Close
    for i in range(len(data)):
        row = data.iloc[i]
        high = max(row['Open'], row['Close'], row['High'])
        low = min(row['Open'], row['Close'], row['Low'])
        data.iloc[i, data.columns.get_loc('High')] = high
        data.iloc[i, data.columns.get_loc('Low')] = low
    
    return data