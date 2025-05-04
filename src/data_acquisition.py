"""
Data acquisition module for fetching cryptocurrency data using yfinance.
"""

import yfinance as yf
import pandas as pd
from typing import Union, List
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical price data for a given ticker.
    
    Args:
        ticker: Crypto ticker (e.g., 'BTC-USD')
        start_date: Start date (e.g., '2020-01-01')
        end_date: End date (e.g., '2025-05-01')
        interval: Data interval (e.g., '1d' for daily)
    
    Returns:
        DataFrame with price data
    """
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Using fixed past dates for testing
        import datetime
        today = datetime.datetime.now()
        fixed_end_date = today.strftime('%Y-%m-%d')
        fixed_start_date = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Use fixed dates first for debugging
        logger.info(f"Using historical date range: {fixed_start_date} to {fixed_end_date}")
        data = yf.download(ticker, start=fixed_start_date, end=fixed_end_date, interval=interval)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        # Ensure numeric columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
        logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
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
        test_data = yf.Ticker(ticker).info
        return bool(test_data)
    except:
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
        async with websockets.connect(self.base_url) as websocket:
            await websocket.send(json.dumps(msg))
            response = await websocket.recv()
            return json.loads(response)
    
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
        
        try:
            return asyncio.get_event_loop().run_until_complete(self.call_api(msg))
        except Exception as e:
            logger.error(f"Error calling Deribit API: {e}")
            return {"error": str(e)}
    
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
                    
                    # Ensure iv column is numeric
                    iv_df['iv'] = pd.to_numeric(iv_df['iv'], errors='coerce')
                    iv_df['price'] = pd.to_numeric(iv_df['price'], errors='coerce')
                    iv_df['mark_price'] = pd.to_numeric(iv_df['mark_price'], errors='coerce')
                    iv_df['index_price'] = pd.to_numeric(iv_df['index_price'], errors='coerce')
                    iv_df['amount'] = pd.to_numeric(iv_df['amount'], errors='coerce')
                    
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
    # Get realized volatility from yfinance
    price_data = get_data(ticker, start_date, end_date)
    
    # Extract currency from ticker (e.g., 'BTC' from 'BTC-USD')
    currency = ticker.split('-')[0]
    
    # Get implied volatility from Deribit
    deribit = DeribitAPI(test_mode=True)
    iv_data = deribit.get_implied_volatility_data(currency)
    
    if not iv_data.empty:
        # Aggregate IV data to daily level
        daily_iv = iv_data.resample('D').agg({'iv': 'mean'})
        
        # Rename for clarity
        daily_iv.rename(columns={'iv': 'ImpliedVolatility'}, inplace=True)
        
        # Merge with price data
        if not price_data.empty:
            combined_data = price_data.join(daily_iv, how='left')
            
            # Ensure ImpliedVolatility is numeric
            if 'ImpliedVolatility' in combined_data.columns:
                combined_data['ImpliedVolatility'] = pd.to_numeric(combined_data['ImpliedVolatility'], errors='coerce')
                
            return combined_data
    
    return price_data