"""
Data Loader Utility

This module provides utilities for loading market data from various sources
including yfinance, Financial Modeling Prep (FMP), and local CSV files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
import requests
import os
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Data loader for market data from various sources
    """
    
    def __init__(self, fmp_api_key: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            fmp_api_key: API key for Financial Modeling Prep (optional)
        """
        self.fmp_api_key = fmp_api_key
        self.cache_dir = "data/cache"
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def load_from_yfinance(self, 
                          symbol: str, 
                          start_date: str, 
                          end_date: str,
                          interval: str = "1d") -> pd.DataFrame:
        """
        Load data from yfinance
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Standardize column names
            data.columns = [col.title() for col in data.columns]
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = np.nan
            
            print(f"Loaded {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def load_from_fmp(self, 
                      symbol: str, 
                      start_date: str, 
                      end_date: str) -> pd.DataFrame:
        """
        Load data from Financial Modeling Prep
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.fmp_api_key:
            print("FMP API key not provided")
            return pd.DataFrame()
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            params = {
                'apikey': self.fmp_api_key,
                'from': start_date,
                'to': end_date
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'historical' not in data:
                raise ValueError(f"No historical data found for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['historical'])
            
            # Standardize column names
            column_mapping = {
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.sort_index()
            
            print(f"Loaded {len(df)} data points for {symbol} from FMP")
            return df
            
        except Exception as e:
            print(f"Error loading data from FMP for {symbol}: {e}")
            return pd.DataFrame()
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = pd.read_csv(file_path)
            
            # Try to identify date column
            date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                data[date_columns[0]] = pd.to_datetime(data[date_columns[0]])
                data = data.set_index(date_columns[0])
            
            # Standardize column names
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adj close': 'Adj Close'
            }
            
            data.columns = [col.lower() for col in data.columns]
            data = data.rename(columns=column_mapping)
            
            print(f"Loaded {len(data)} data points from {file_path}")
            return data
            
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
    
    def get_sample_data(self, symbol: str = "AAPL", days: int = 252) -> pd.DataFrame:
        """
        Get sample data for testing
        
        Args:
            symbol: Stock symbol
            days: Number of days to fetch
            
        Returns:
            DataFrame with sample data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.load_from_yfinance(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    
    def save_to_csv(self, data: pd.DataFrame, file_path: str):
        """
        Save data to CSV file
        
        Args:
            data: DataFrame to save
            file_path: Output file path
        """
        try:
            data.to_csv(file_path)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")
    
    def get_multiple_symbols(self, 
                            symbols: List[str], 
                            start_date: str, 
                            end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            print(f"Loading data for {symbol}...")
            data = self.load_from_yfinance(symbol, start_date, end_date)
            if not data.empty:
                data_dict[symbol] = data
        
        return data_dict
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
        """
        if data.empty:
            print("Data is empty")
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        missing_values = data[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found: {missing_values.to_dict()}")
            return False
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        negative_prices = (data[price_columns] < 0).any().any()
        if negative_prices:
            print("Negative prices found")
            return False
        
        # Check for zero volumes
        zero_volumes = (data['Volume'] == 0).sum()
        if zero_volumes > len(data) * 0.1:  # More than 10% zero volumes
            print(f"High number of zero volumes: {zero_volumes}")
            return False
        
        print("Data validation passed")
        return True
    
    def resample_data(self, 
                     data: pd.DataFrame, 
                     frequency: str = "1D") -> pd.DataFrame:
        """
        Resample data to different frequency
        
        Args:
            data: Input DataFrame
            frequency: Target frequency (1D, 1H, 5T, etc.)
            
        Returns:
            Resampled DataFrame
        """
        try:
            resampled = data.resample(frequency).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Forward fill any missing values
            resampled = resampled.fillna(method='ffill')
            
            print(f"Resampled data to {frequency} frequency")
            return resampled
            
        except Exception as e:
            print(f"Error resampling data: {e}")
            return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to the data
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df


def create_sample_data():
    """Create sample price data for testing"""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Ensure High >= Low and High >= Open, Close
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load sample data
    data = loader.get_sample_data("AAPL", days=100)
    
    if not data.empty:
        # Validate data
        loader.validate_data(data)
        
        # Add technical indicators
        data_with_indicators = loader.add_technical_indicators(data)
        
        # Save to CSV
        loader.save_to_csv(data_with_indicators, "data/sample_prices.csv")
        
        print(f"Sample data shape: {data_with_indicators.shape}")
        print(f"Columns: {list(data_with_indicators.columns)}") 