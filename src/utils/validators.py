from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def validate_stock_data(data: pd.DataFrame) -> bool:
    """
    Validate stock data DataFrame
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        bool indicating if data is valid
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check required columns
    if not all(col in data.columns for col in required_columns):
        return False
        
    # Check for null values
    if data[required_columns].isnull().any().any():
        return False
        
    # Check for negative prices
    price_columns = ['Open', 'High', 'Low', 'Close']
    if (data[price_columns] <= 0).any().any():
        return False
        
    # Check for negative volume
    if (data['Volume'] < 0).any():
        return False
        
    # Check High >= Low
    if not (data['High'] >= data['Low']).all():
        return False
        
    # Check High >= Open and Close
    if not ((data['High'] >= data['Open']) & (data['High'] >= data['Close'])).all():
        return False
        
    # Check Low <= Open and Close
    if not ((data['Low'] <= data['Open']) & (data['Low'] <= data['Close'])).all():
        return False
        
    return True

def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """
    Validate date range for market data queries
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        bool indicating if date range is valid
    """
    # Check if dates are in correct order
    if start_date >= end_date:
        return False
        
    # Check if date range is not too long (e.g., max 2 years)
    if end_date - start_date > timedelta(days=730):
        return False
        
    # Check if end date is not in future
    if end_date > datetime.now():
        return False
        
    return True

def validate_ticker_symbol(ticker: str) -> bool:
    """
    Validate ticker symbol format
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        bool indicating if ticker format is valid
    """
    # Check length
    if not 1 <= len(ticker) <= 5:
        return False
        
    # Check if contains only valid characters
    if not ticker.replace('-', '').replace('.', '').isalnum():
        return False
        
    # Check if starts with letter
    if not ticker[0].isalpha():
        return False
        
    return True