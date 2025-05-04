"""
Data acquisition module for cryptocurrency data.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import websockets
import yfinance as yf
from dotenv import load_dotenv

from .config import config
from .logger import get_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = get_logger(__name__)

def get_data(ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical data for a crypto ticker.
    
    Args:
        ticker: Crypto ticker
        start_date: Start date
        end_date: End date
        interval: Data interval
    
    Returns:
        DataFrame with historical data
    """
    try:
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start > end:
            raise ValueError("Start date must be before end date")
        
        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        # Convert data to float type
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                try:
                    data[col] = data[col].astype(float)
                except Exception as e:
                    logger.warning(f"Error converting {col} to float: {str(e)}")
                    data[col] = data[col].replace('', np.nan).astype(float)
        
        # Validate data
        if data.isnull().any().any():
            logger.warning(f"Missing values found in data for {ticker}")
        
        logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}", exc_info=True)
        return pd.DataFrame()

class DeribitAPI:
    """Class for interacting with Deribit API."""
    
    def __init__(self):
        """Initialize Deribit API client."""
        self.base_url = "wss://test.deribit.com/ws/api/v2"
        self.api_key = os.getenv("DERIBIT_API_KEY")
        self.api_secret = os.getenv("DERIBIT_API_SECRET")
        self._ws = None
        self._loop = None
        logger.info(f"Initialized Deribit API with URL: {self.base_url}")

    async def _connect(self):
        """Establish WebSocket connection."""
        if self._ws is None:
            self._ws = await websockets.connect(self.base_url)
            logger.debug("WebSocket connection established")

    async def _disconnect(self):
        """Close WebSocket connection."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            logger.debug("WebSocket connection closed")

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send request to Deribit API."""
        try:
            await self._connect()
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }
            await self._ws.send(json.dumps(request))
            response = await self._ws.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error in API request: {str(e)}")
            raise

    def get_iv_data(self, symbol: str) -> pd.DataFrame:
        """Get implied volatility data for a symbol."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function
            result = loop.run_until_complete(self._get_iv_data_async(symbol))
            
            # Clean up
            loop.close()
            
            return result
        except Exception as e:
            logger.error(f"Error getting IV data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def _get_iv_data_async(self, symbol: str) -> pd.DataFrame:
        """Async implementation of get_iv_data."""
        try:
            # Get current price
            ticker = symbol.replace("-", "")
            response = await self._send_request("public/ticker", {"instrument_name": f"{ticker}-PERPETUAL"})
            
            if "result" not in response:
                logger.error(f"Invalid response for {symbol}: {response}")
                return pd.DataFrame()
            
            current_price = float(response["result"]["last_price"])
            
            # Get options chain
            response = await self._send_request("public/get_book_summary_by_currency", {
                "currency": ticker,
                "kind": "option"
            })
            
            if "result" not in response:
                logger.error(f"Invalid response for options chain: {response}")
                return pd.DataFrame()
            
            # Process options data
            options_data = []
            for option in response["result"]:
                try:
                    strike = float(option["instrument_name"].split("-")[2])
                    expiry = option["instrument_name"].split("-")[1]
                    option_type = option["instrument_name"].split("-")[3]
                    
                    # Calculate time to expiry
                    expiry_date = datetime.strptime(expiry, "%d%b%y")
                    tte = (expiry_date - datetime.now()).days / 365
                    
                    # Get option price
                    option_response = await self._send_request("public/ticker", {
                        "instrument_name": option["instrument_name"]
                    })
                    
                    if "result" in option_response:
                        option_price = float(option_response["result"]["last_price"])
                        
                        # Calculate implied volatility using Black-Scholes
                        iv = self._calculate_iv(
                            option_price,
                            current_price,
                            strike,
                            tte,
                            0.02,  # Risk-free rate
                            option_type == "C"
                        )
                        
                        options_data.append({
                            "strike": strike,
                            "expiry": expiry,
                            "type": option_type,
                            "price": option_price,
                            "iv": iv,
                            "tte": tte
                        })
                except Exception as e:
                    logger.warning(f"Error processing option {option['instrument_name']}: {str(e)}")
                    continue
            
            return pd.DataFrame(options_data)
        except Exception as e:
            logger.error(f"Error in _get_iv_data_async: {str(e)}")
            return pd.DataFrame()
        finally:
            await self._disconnect()

    def _calculate_iv(self, option_price: float, spot_price: float, strike: float,
                     tte: float, risk_free_rate: float, is_call: bool) -> float:
        """Calculate implied volatility using Black-Scholes."""
        try:
            # Simple implementation - in practice, use a proper IV calculation
            # This is just a placeholder
            return 0.5  # 50% IV as default
        except Exception as e:
            logger.error(f"Error calculating IV: {str(e)}")
            return np.nan

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
    try:
        # Get realized volatility from yfinance
        price_data = get_data(ticker, start_date, end_date)
        
        if price_data.empty:
            logger.warning(f"No price data found for {ticker}")
            return pd.DataFrame()
        
        # Extract currency from ticker (e.g., 'BTC' from 'BTC-USD')
        currency = ticker.split('-')[0]
        
        # Get implied volatility from Deribit
        deribit = DeribitAPI()
        iv_data = deribit.get_iv_data(ticker)
        
        if not iv_data.empty:
            # Aggregate IV data to daily level
            daily_iv = iv_data.groupby('expiry')['iv'].mean().reset_index()
            daily_iv['expiry'] = pd.to_datetime(daily_iv['expiry'], format='%d%b%y')
            daily_iv.set_index('expiry', inplace=True)
            daily_iv.rename(columns={'iv': 'ImpliedVolatility'}, inplace=True)
            
            # Merge with price data
            combined_data = price_data.join(daily_iv, how='left')
            
            # Ensure ImpliedVolatility is numeric
            if 'ImpliedVolatility' in combined_data.columns:
                combined_data['ImpliedVolatility'] = pd.to_numeric(combined_data['ImpliedVolatility'], errors='coerce')
            
            logger.info(f"Successfully combined volatility data for {ticker}")
            return combined_data
        
        return price_data
    except Exception as e:
        logger.error(f"Error getting combined volatility data: {str(e)}")
        return pd.DataFrame()

def fetch_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical data for a given ticker."""
    try:
        logger.info(f"Fetching historical data for {ticker}")
        data = yf.download(ticker, period=period, progress=False)
        logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_deribit_data(symbol: str) -> pd.DataFrame:
    """Get data from Deribit API."""
    try:
        api = DeribitAPI()
        return api.get_iv_data(symbol)
    except Exception as e:
        logger.error(f"Error getting Deribit data: {str(e)}")
        return pd.DataFrame()

def fetch_market_data(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch both historical and options data."""
    historical_data = fetch_historical_data(ticker, period)
    options_data = get_deribit_data(ticker)
    return historical_data, options_data