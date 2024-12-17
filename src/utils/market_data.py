# src/utils/market_data.py

from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from utils.logging_config import logger


class MarketDataUtils:
    """Utility functions for market data processing"""

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()

    @staticmethod
    def calculate_volatility(returns: pd.Series, periods: int = 252) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(periods)

    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1.0

    @staticmethod
    def calculate_moving_averages(
        prices: pd.Series, windows: list
    ) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages"""
        return {
            f"MA_{window}": prices.rolling(window=window).mean() for window in windows
        }

    @staticmethod
    def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
