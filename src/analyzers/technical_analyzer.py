import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, Optional, Union, Tuple
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
import ta
from utils.logging_config import logger

class TechnicalAnalyzer:
    def __init__(self, polygon_connector):
        """Initialize technical analyzer with data connector"""
        self.market_data = polygon_connector
        self.market_indicators_cache = {}
        self.sector_performance_cache = {}

    def get_stock_metrics(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive stock metrics"""
        try:
            stock_data = self.market_data.get_stock_data(ticker, days=180)
            if not stock_data or stock_data['history'].empty:
                logger.warning(f"No data available for {ticker}")
                return None
                
            hist = stock_data['history']
            info = stock_data['info']
            
            if len(hist) < 2:
                logger.warning(f"Insufficient historical data for {ticker}")
                return None
                
            metrics = {}
            
            # Calculate core metrics
            price_metrics = self._calculate_price_metrics(hist)
            volume_metrics = self._calculate_volume_metrics(hist)
            
            if len(hist) >= 20:
                technical_metrics = self._calculate_technical_indicators(hist)
                metrics.update(technical_metrics)
                
            if len(hist) >= 14:
                volatility_metrics = self._calculate_volatility_metrics(hist)
                metrics.update(volatility_metrics)
                momentum_metrics = self._calculate_momentum_indicators(hist)
                metrics.update(momentum_metrics)
                
            # Get market context
            market_metrics = self._get_market_context(ticker)
            
            # Update all metrics
            metrics.update(price_metrics)
            metrics.update(volume_metrics)
            metrics.update(market_metrics)
            
            # Add fundamental metrics
            fundamental_metrics = {
                "market_cap": info.get("market_cap"),
                "shares_outstanding": info.get("share_class_shares_outstanding"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "description": info.get("description"),
                "exchange": info.get("primary_exchange"),
                "type": info.get("type")
            }
            metrics.update(fundamental_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {ticker}: {str(e)}")
            return None

    def _calculate_price_metrics(self, hist: pd.DataFrame) -> Dict:
        """Calculate price-based metrics"""
        metrics = {}
        try:
            if len(hist) < 2:
                return {}
                
            close = hist['Close']
            
            metrics['current_price'] = round(close.iloc[-1], 2)
            
            if len(close) >= 2:
                metrics['price_change_1d'] = round(
                    ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100, 2
                )
            
            if len(close) >= 5:
                metrics['price_change_1w'] = round(
                    ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100, 2
                )
                
            if len(close) >= 21:
                metrics['price_change_1m'] = round(
                    ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100, 2
                )
                
            if len(close) >= 20:
                metrics['price_trend'] = self._identify_price_trend(close)
                metrics['support_resistance'] = self._calculate_support_resistance(hist)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error in price metrics calculation: {str(e)}")
            return {}

    def _calculate_volume_metrics(self, hist: pd.DataFrame) -> Dict:
        """Calculate volume-based metrics"""
        metrics = {}
        try:
            volume = hist["Volume"]
            close = hist["Close"]

            metrics.update({
                "avg_volume_10d": round(volume.rolling(10).mean().iloc[-1], 2),
                "avg_volume_30d": round(volume.rolling(30).mean().iloc[-1], 2),
                "volume_price_trend": round((volume * close).pct_change().mean(), 4),
                "volume_momentum": round(ta.volume.force_index(close, volume), 2),
                "volume_trend": self._calculate_volume_trend(volume),
                "relative_volume": round(
                    volume.iloc[-1] / volume.rolling(20).mean().iloc[-1], 2
                ),
                "money_flow_index": round(
                    ta.volume.money_flow_index(
                        hist["High"], hist["Low"], close, volume, window=14
                    ),
                    2,
                )
            })

            return metrics

        except Exception as e:
            logger.error(f"Error in volume metrics calculation: {str(e)}")
            return {}

    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            close = hist["Close"]
            volume = hist["Volume"]

            return {
                "sma_20": round(close.rolling(window=20).mean().iloc[-1], 2),
                "ema_9": round(close.ewm(span=9).mean().iloc[-1], 2),
                "rsi": round(self._calculate_rsi(close), 2),
                "volatility": round(close.pct_change().std() * np.sqrt(252) * 100, 2),
                "bollinger_upper": round(
                    close.rolling(20).mean() + close.rolling(20).std() * 2, 2
                ).iloc[-1],
                "bollinger_lower": round(
                    close.rolling(20).mean() - close.rolling(20).std() * 2, 2
                ).iloc[-1],
                "volume_sma": round(volume.rolling(window=20).mean().iloc[-1], 2),
                "volume_ratio": round(
                    volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1], 2
                ),
                "macd": self._calculate_macd(close),
                "stochastic": self._calculate_stochastic(hist)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def _identify_price_trend(self, prices: pd.Series) -> str:
        """Identify price trend using multiple timeframes"""
        try:
            if len(prices) < 20:
                return "Insufficient Data"
                
            sma20 = prices.rolling(window=20).mean().iloc[-1]
            sma50 = prices.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else None
            sma200 = prices.rolling(window=200).mean().iloc[-1] if len(prices) >= 200 else None
            current_price = prices.iloc[-1]
            
            if sma200 is not None and sma50 is not None:
                if current_price > sma20 > sma50 > sma200:
                    return "Strong Uptrend"
                elif current_price < sma20 < sma50 < sma200:
                    return "Strong Downtrend"
            elif sma50 is not None:
                if current_price > sma20 > sma50:
                    return "Moderate Uptrend"
                elif current_price < sma20 < sma50:
                    return "Moderate Downtrend"
            else:
                if current_price > sma20:
                    return "Short-term Uptrend"
                elif current_price < sma20:
                    return "Short-term Downtrend"
                    
            return "Sideways"
            
        except Exception as e:
            logger.error(f"Error identifying price trend: {str(e)}")
            return "Unknown"

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            "macd_line": round(macd.iloc[-1], 2),
            "signal_line": round(signal.iloc[-1], 2),
            "histogram": round(macd.iloc[-1] - signal.iloc[-1], 2)
        }

    def _calculate_stochastic(self, hist: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        high_max = hist["High"].rolling(period).max()
        low_min = hist["Low"].rolling(period).min()
        k = 100 * (hist["Close"] - low_min) / (high_max - low_min)
        d = k.rolling(3).mean()
        return {
            "stoch_k": round(k.iloc[-1], 2),
            "stoch_d": round(d.iloc[-1], 2)
        }

    def _calculate_support_resistance(self, hist: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            close = hist["Close"]
            sma20 = close.rolling(window=20).mean().iloc[-1]
            sma50 = close.rolling(window=50).mean().iloc[-1]
            sma200 = close.rolling(window=200).mean().iloc[-1]
            
            local_min = close.rolling(window=20, center=True).min().iloc[-1]
            local_max = close.rolling(window=20, center=True).max().iloc[-1]
            
            return {
                "support_1": round(min(sma20, local_min), 2),
                "support_2": round(sma50, 2),
                "resistance_1": round(max(sma20, local_max), 2),
                "resistance_2": round(sma200, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {}

    def _calculate_volume_trend(self, volume: pd.Series) -> str:
        """Calculate volume trend characteristics"""
        try:
            current_vol = volume.iloc[-1]
            avg_vol_10d = volume.rolling(10).mean().iloc[-1]
            avg_vol_30d = volume.rolling(30).mean().iloc[-1]
            
            if current_vol > avg_vol_30d * 1.5:
                return "Very High"
            elif current_vol > avg_vol_10d * 1.2:
                return "Above Average"
            elif current_vol < avg_vol_30d * 0.5:
                return "Very Low"
            elif current_vol < avg_vol_10d * 0.8:
                return "Below Average"
            else:
                return "Average"
                
        except Exception as e:
            logger.error(f"Error calculating volume trend: {str(e)}")
            return "Unknown"

    def _get_market_context(self, ticker: str) -> Dict:
        """Get market context metrics"""
        try:
            metrics = {}
            
            sector_data = self._get_sector_performance(ticker)
            market_indicators = self._get_market_indicators()
            
            metrics.update({
                "sector_performance": sector_data,
                "market_indicators": market_indicators,
                "relative_strength": self._calculate_relative_strength(ticker),
                "beta": self._calculate_beta(ticker)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return {}

    def _calculate_beta(self, ticker: str) -> float:
        """Calculate beta coefficient"""
        try:
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=180)
            
            stock_data = self.market_data.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            spy_data = self.market_data.get_aggregates(
                ticker="SPY",
                multiplier=1,
                timespan="day",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            if stock_data is None or spy_data is None:
                return 1.0
                
            stock_returns = stock_data['close'].pct_change().dropna()
            market_returns = spy_data['close'].pct_change().dropna()
            
            if len(stock_returns) < 30:
                return 1.0
                
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            return round(min(max(beta, -10), 10), 2)
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0