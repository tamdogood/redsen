import io
import praw
import nltk
import pandas as pd
import numpy as np
import datetime as dt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from connectors.supabase_connector import CustomJSONEncoder, SupabaseConnector
from utils.logging_config import logger
from openai import OpenAI
import json
import os
from connectors.polygon_connector import PolygonConnector
import ta
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from typing import Dict, List, Optional, Tuple, Union
from scipy.signal import argrelextrema


load_dotenv()

# Download necessary NLTK resources
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)


class EnhancedStockAnalyzer:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        supabase_url: str,
        supabase_key: str,
    ):
        """Initialize the analyzer with API credentials"""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        self.db = SupabaseConnector(
            supabase_url=supabase_url, supabase_key=supabase_key
        )
        self.market_data = PolygonConnector(os.getenv("POLYGON_API_KEY", ""))
        # Initialize OpenAI client (do this in __init__ if used frequently)
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize caches
        self.stock_data_cache = {}
        self.market_indicators_cache = {}
        self.sector_performance_cache = {}

        # Add known invalid tickers/common words to filter out
        self.invalid_tickers = {
            "CEO",
            "IPO",
            "EPS",
            "GDP",
            "NYSE",
            "SEC",
            "USD",
            "ATH",
            "IMO",
            "PSA",
            "USA",
            "CDC",
            "WHO",
            "ETF",
            "YOLO",
            "FOMO",
            "FUD",
            "DOW",
            "LOL",
            "ANY",
            "ALL",
            "FOR",
            "ARE",
            "THE",
            "NOW",
            "NEW",
            "IRS",
            "FED",
            "NFT",
            "APE",
            "RH",
            "WSB",
            "I",
            "KSP",
            "CUDA",
            "NFT",
            "NFTs",
            "UK",
            "US",
            "EUR",
            "EU",
            "JP",
            "WSJ",
            "NYT",
            "STONK",
            "VR",
            "IoT",
            "FAQ",
            "ASAP",
            "DIY",
            "ROI",
            "KPI",
            "ROTH",
            "401K",
            "IRA",
            "BUY",
            "SELL",
            "MOASS",
            "END",
            "TODAY",
            "DOJ",
            "DOD",
            "AF",
            "CPU",
            "GPU",
            "THANK",
            "DOWN",
            "AVG",
            "MEAN",
            "FLOAT",
            "SOLD",
            "IS",
            "WE",
            "HOLD",
            "IV",
            "XXX",
            "FAV",
            "PE",
            "PS",
            "HELP",
            "SPAC",
            "FDA",
            "AWS",
            "LMAO",
            "DOGE",
            "HAS",
            "BAN",
            "LLC",
            "NEVER",
            "II",
            "P",
            "C",
            "YTD",
            "CFO",
            "LG",
            "TWS",
            "LFG",
            "WB",
            "COVID",
            "HYSA",
            "CHAT",
            "HOME",
        }

        self.market_indicators_cache = {}
        self.sector_performance_cache = {}

    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if a ticker symbol is valid using Polygon API"""
        if ticker in self.invalid_tickers:
            return False
            
        try:
            result = self.market_data.is_valid_ticker(ticker)
            return result
        except Exception as e:
            logger.warning(f"Ticker validation failed for {ticker}: {str(e)}")
            return False
            
    def extract_stock_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers using OpenAI with fallback to regex

        Args:
            text (str): Text to analyze for stock mentions

        Returns:
            List[str]: List of valid stock tickers
        """
        try:
            # First try regex extraction
            tickers = self._extract_tickers_with_regex(text)

            # If regex fails or returns no tickers, fallback to OpenAI
            # if not tickers:
            # tickers = self._extract_tickers_with_openai(text)

            # Validate tickers
            valid_tickers = []
            for ticker in tickers:
                if (
                    len(ticker) >= 1
                    and len(ticker) <= 5
                    and ticker not in self.invalid_tickers
                    and self.is_valid_ticker(ticker)
                ):
                    valid_tickers.append(ticker)

            return list(set(valid_tickers))  # Remove duplicates

        except Exception as e:
            logger.warning(
                f"Error in ticker extraction: {str(e)}. Falling back to regex."
            )
            return self._extract_tickers_with_regex(text)

    def _extract_tickers_with_openai(self, text: str) -> List[str]:
        """
        Use OpenAI to extract stock tickers from text

        Args:
            text (str): Text to analyze

        Returns:
            List[str]: Extracted stock tickers
        """
        try:

            # Prepare the prompt
            prompt = f"""Extract stock tickers from the following text. Return only the tickers in a JSON array format. 
            Rules:
            - Include tickers with or without $ prefix
            - Only include tickers 1-5 characters long
            - Exclude common words that look like tickers
            - If no valid tickers found, return empty array
            - Do not include explanation, only return the JSON array
            
            Text: {text}
            """

            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst that extracts stock tickers from text. Only respond with a JSON array of tickers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=25,
            )

            # Parse response
            try:
                tickers_text = response.choices[0].message.content.strip()
                # Handle potential JSON formatting issues
                tickers_text = tickers_text.replace("'", '"')
                if not tickers_text.startswith("["):
                    tickers_text = f"[{tickers_text}]"
                tickers = json.loads(tickers_text)

                # Clean tickers
                tickers = [ticker.strip("$") for ticker in tickers if ticker]
                return tickers

            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing OpenAI response: {str(e)}")
                return []

        except Exception as e:
            logger.warning(f"Error calling OpenAI API: {str(e)}")
            return []

    def _extract_tickers_with_regex(self, text: str) -> List[str]:
        """
        Fallback method using regex to extract stock tickers

        Args:
            text (str): Text to analyze

        Returns:
            List[str]: Extracted stock tickers
        """
        # Enhanced regex pattern
        patterns = [
            r"\$([A-Z]{1,5})\b",  # Matches tickers with $ prefix
            r"\b([A-Z]{1,5})\b(?!\.[A-Z]{1,2})",  # Matches uppercase words, excludes file extensions
        ]

        tickers = []
        for pattern in patterns:
            tickers.extend(re.findall(pattern, text))

        return list(set(tickers))  # Remove duplicates

    def get_stock_metrics(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive stock metrics with improved data validation"""
        try:
            stock_data = self.market_data.get_stock_data(ticker, days=180)
            if not stock_data or stock_data['history'].empty:
                logger.warning(f"No data available for {ticker}")
                return None
                
            hist = stock_data['history']
            info = stock_data['info']
            
            # Validate minimum data requirements
            if len(hist) < 2:
                logger.warning(f"Insufficient historical data for {ticker}")
                return None
                
            metrics = {}
            
            # Calculate metrics based on available data
            price_metrics = self._calculate_price_metrics(hist)
            volume_metrics = self._calculate_volume_metrics(hist)
            
            # Only calculate technical indicators if we have enough data
            if len(hist) >= 20:
                technical_metrics = self._calculate_technical_indicators(hist)
                metrics.update(technical_metrics)
                
            # Only calculate volatility metrics if we have enough data
            if len(hist) >= 14:
                volatility_metrics = self._calculate_volatility_metrics(hist)
                metrics.update(volatility_metrics)
                
            # Only calculate momentum indicators if we have enough data
            if len(hist) >= 14:
                momentum_metrics = self._calculate_momentum_indicators(hist)
                metrics.update(momentum_metrics)
                
            # Get market context
            market_metrics = self._get_market_context(ticker)
            
            # Update all available metrics
            metrics.update(price_metrics)
            metrics.update(volume_metrics)
            metrics.update(market_metrics)
            
            # Add fundamental metrics if available
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

    def _calculate_volume_metrics(self, hist: pd.DataFrame) -> Dict:
        """Calculate enhanced volume-based metrics"""
        metrics = {}
        try:
            volume = hist["Volume"]
            close = hist["Close"]

            # Calculate volume indicators
            metrics.update(
                {
                    "avg_volume_10d": round(volume.rolling(10).mean().iloc[-1], 2),
                    "avg_volume_30d": round(volume.rolling(30).mean().iloc[-1], 2),
                    "volume_price_trend": round(
                        (volume * close).pct_change().mean(), 4
                    ),
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
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error in volume metrics calculation: {str(e)}")

        return metrics

    def _calculate_volatility_metrics(self, hist: pd.DataFrame) -> Dict:
        """Calculate comprehensive volatility metrics with data validation"""
        metrics = {}
        try:
            if len(hist) < 2:  # Minimum required data points
                return {}
                
            close = hist['Close']
            
            # Calculate returns with validation
            returns = close.pct_change().dropna()
            
            if len(returns) > 0:
                # Daily volatility
                metrics['volatility_daily'] = round(returns.std() * np.sqrt(252) * 100, 2)
                
                # Weekly volatility (requires at least 5 data points)
                if len(returns) >= 5:
                    metrics['volatility_weekly'] = round(
                        returns.rolling(5).std().iloc[-1] * np.sqrt(52) * 100, 2
                    )
                    
                # Monthly volatility (requires at least 21 data points)
                if len(returns) >= 21:
                    metrics['volatility_monthly'] = round(
                        returns.rolling(21).std().iloc[-1] * np.sqrt(12) * 100, 2
                    )
            
            # Add volatility trend if enough data
            if len(returns) >= 20:
                metrics['volatility_trend'] = self._calculate_volatility_trend(returns)
                
            # Add Keltner channels if enough data
            if len(hist) >= 20:
                metrics['keltner_channels'] = self._calculate_keltner_channels(hist)
                
            # Add ATR if enough data
            if len(hist) >= 14:
                try:
                    metrics['atr'] = round(
                        ta.volatility.average_true_range(
                            hist['High'], hist['Low'], close
                        ), 4
                    )
                except:
                    metrics['atr'] = round(
                        (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1], 4
                    )
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error in volatility metrics calculation: {str(e)}")
            return {}
        
    def _get_market_context(self, ticker: str) -> Dict:
        """Get market context metrics using Polygon data"""
        try:
            # # Check cache first
            # cache_key = f"market_context_{ticker}"
            # cached_data = self.cache.get(cache_key)
            # if cached_data:
            #     return cached_data

            metrics = {}

            # Get sector performance
            sector_data = self._get_sector_performance(ticker)
            market_indicators = self._get_market_indicators()

            metrics.update({
                "sector_performance": sector_data,
                "market_indicators": market_indicators,
                "relative_strength": self._calculate_relative_strength(ticker),
                "beta": self._calculate_beta(ticker)
            })

            # Cache the results
            # self.cache.set(cache_key, metrics, ttl=3600)  # Cache for 1 hour
            return metrics

        except Exception as e:
            logger.error(f"Error in market context calculation: {str(e)}")
            return {}

    def _calculate_options_metrics(self, stock) -> Dict:
        """Calculate options-based indicators"""
        metrics = {}
        try:
            # Get options data
            options = stock.options
            if options:
                current_price = stock.history(period="1d")["Close"].iloc[-1]

                calls_data = []
                puts_data = []

                for date in options[:2]:  # Look at nearest two expiration dates
                    opt = stock.option_chain(date)
                    calls_data.extend(opt.calls.to_dict("records"))
                    puts_data.extend(opt.puts.to_dict("records"))

                metrics.update(
                    {
                        "put_call_ratio": len(puts_data) / max(len(calls_data), 1),
                        "implied_volatility": self._calculate_weighted_iv(
                            calls_data + puts_data
                        ),
                        "options_volume": sum(
                            c.get("volume", 0) for c in calls_data + puts_data
                        ),
                        "max_pain": self._calculate_max_pain(
                            calls_data, puts_data, current_price
                        ),
                    }
                )

        except Exception as e:
            logger.error(f"Error in options metrics calculation: {str(e)}")

        return metrics

    def _calculate_relative_strength(self, ticker: str) -> float:
        """Calculate relative strength compared to market using Polygon data"""
        try:
            # Get stock and market data
            stock_data = self.market_data.get_stock_data(ticker, days=180)
            spy_data = self.market_data.get_stock_data("SPY", days=180)

            if not stock_data or not spy_data:
                return 0.0

            stock_hist = stock_data['history']['Close']
            market_hist = spy_data['history']['Close']

            # Calculate returns
            stock_returns = stock_hist.pct_change()
            market_returns = market_hist.pct_change()

            relative_strength = (stock_returns.mean() / market_returns.mean()) * 100
            return round(relative_strength, 2)

        except Exception as e:
            logger.error(f"Error calculating relative strength: {str(e)}")
            return 0.0

    def _calculate_beta(self, ticker: str) -> float:
        """Calculate beta coefficient using Polygon data"""
        try:
            # Get one year of daily data
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=7)

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

            # Access the 'close' column (lowercase) directly from the DataFrame
            stock_returns = stock_data['close'].pct_change().dropna()
            market_returns = spy_data['close'].pct_change().dropna()

            # Align the data
            aligned_data = pd.concat(
                [stock_returns, market_returns],
                axis=1,
                join='inner'
            )
            aligned_data.columns = ['stock', 'market']

            if len(aligned_data) < 30:  # Require at least 30 data points
                return 1.0

            # Calculate beta
            covariance = aligned_data['stock'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()
            beta = covariance / market_variance if market_variance != 0 else 1.0

            return round(min(max(beta, -10), 10), 2)  # Clamp between -10 and 10

        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0  # Return neutral beta on error

    def _identify_price_trend(self, prices: pd.Series) -> str:
        """Identify price trend using multiple timeframes with data validation"""
        try:
            if len(prices) < 20:
                return "Insufficient Data"
                
            # Calculate SMAs based on available data
            sma20 = prices.rolling(window=min(20, len(prices))).mean().iloc[-1]
            sma50 = prices.rolling(window=min(50, len(prices))).mean().iloc[-1] if len(prices) >= 50 else None
            sma200 = prices.rolling(window=min(200, len(prices))).mean().iloc[-1] if len(prices) >= 200 else None
            current_price = prices.iloc[-1]
            
            # Determine trend based on available SMAs
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

    def _calculate_support_resistance(self, hist: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels using multiple methods"""
        try:
            close = hist["Close"]

            # Calculate potential levels using moving averages
            sma20 = close.rolling(window=20).mean().iloc[-1]
            sma50 = close.rolling(window=50).mean().iloc[-1]
            sma200 = close.rolling(window=200).mean().iloc[-1]

            # Find local mins and maxs
            local_min = close.rolling(window=20, center=True).min().iloc[-1]
            local_max = close.rolling(window=20, center=True).max().iloc[-1]

            return {
                "support_1": round(min(sma20, local_min), 2),
                "support_2": round(sma50, 2),
                "resistance_1": round(max(sma20, local_max), 2),
                "resistance_2": round(sma200, 2),
            }

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {}

    def _calculate_volume_trend(self, volume: pd.Series) -> str:
        """Calculate volume trend and characteristics"""
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

    def _calculate_volatility_trend(self, returns: pd.Series) -> str:
        """Calculate volatility trend and characteristics"""
        try:
            current_vol = returns.std()
            hist_vol_10d = returns.rolling(10).std().mean()
            hist_vol_30d = returns.rolling(30).std().mean()

            if current_vol > hist_vol_30d * 1.5:
                return "Extremely High"
            elif current_vol > hist_vol_10d * 1.2:
                return "Elevated"
            elif current_vol < hist_vol_30d * 0.5:
                return "Very Low"
            elif current_vol < hist_vol_10d * 0.8:
                return "Subdued"
            else:
                return "Normal"

        except Exception as e:
            logger.error(f"Error calculating volatility trend: {str(e)}")
            return "Unknown"

    def _calculate_keltner_channels(self, hist: pd.DataFrame) -> Dict:
        """Calculate Keltner Channels with data validation"""
        try:
            if len(hist) < 20:  # Minimum required data points
                return {
                    'keltner_middle': None,
                    'keltner_upper': None,
                    'keltner_lower': None
                }
                
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate the middle line (20-day EMA)
            middle_line = typical_price.ewm(span=20, adjust=False).mean()
            
            # Calculate ATR with validation
            try:
                atr = ta.volatility.average_true_range(high, low, close)
            except:
                atr = (high - low).rolling(window=14).mean()
            
            # Calculate upper and lower bands
            upper_band = middle_line + (2 * atr)
            lower_band = middle_line - (2 * atr)
            
            return {
                'keltner_middle': round(middle_line.iloc[-1], 2),
                'keltner_upper': round(upper_band.iloc[-1], 2),
                'keltner_lower': round(lower_band.iloc[-1], 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {str(e)}")
            return {
                'keltner_middle': None,
                'keltner_upper': None,
                'keltner_lower': None
            }

    def _calculate_weighted_iv(self, options_data: List[Dict]) -> float:
        """Calculate volume-weighted implied volatility"""
        try:
            total_volume = sum(opt.get("volume", 0) for opt in options_data)
            if total_volume == 0:
                return 0.0

            weighted_iv = (
                sum(
                    opt.get("impliedVolatility", 0) * opt.get("volume", 0)
                    for opt in options_data
                )
                / total_volume
            )

            return round(weighted_iv * 100, 2)

        except Exception as e:
            logger.error(f"Error calculating weighted IV: {str(e)}")
            return 0.0

    def _calculate_max_pain(
        self, calls: List[Dict], puts: List[Dict], current_price: float
    ) -> float:
        """Calculate options max pain price"""
        try:
            strike_prices = sorted(
                list(set([opt.get("strike", 0) for opt in calls + puts]))
            )

            min_pain = float("inf")
            max_pain_price = current_price

            for strike in strike_prices:
                total_pain = sum(
                    max(0, strike - opt.get("strike", 0)) * opt.get("openInterest", 0)
                    for opt in calls
                ) + sum(
                    max(0, opt.get("strike", 0) - strike) * opt.get("openInterest", 0)
                    for opt in puts
                )

                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_price = strike

            return round(max_pain_price, 2)

        except Exception as e:
            logger.error(f"Error calculating max pain: {str(e)}")
            return current_price

    def _get_sector_performance(self, ticker: str) -> Dict:
        """Get sector and industry performance metrics"""
        try:
            # if ticker in self.sector_performance_cache:
                # return self.sector_performance_cache[ticker]

            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get("sector", "")
            industry = info.get("industry", "")

            # Get sector ETF performance
            sector_etfs = {
                "Technology": "XLK",
                "Financial": "XLF",
                "Healthcare": "XLV",
                "Consumer Cyclical": "XLY",
                "Consumer Defensive": "XLP",
                "Energy": "XLE",
                "Industrial": "XLI",
                "Materials": "XLB",
                "Real Estate": "XLRE",
                "Utilities": "XLU",
                "Communication Services": "XLC",
            }

            sector_etf = sector_etfs.get(sector, "SPY")
            etf_data = yf.Ticker(sector_etf).history(period="1mo")

            sector_metrics = {
                "sector": sector,
                "industry": industry,
                "sector_performance_1m": round(
                    ((etf_data["Close"].iloc[-1] / etf_data["Close"].iloc[0]) - 1)
                    * 100,
                    2,
                ),
                "sector_relative_strength": self._calculate_relative_strength(
                    sector_etf
                ),
                "sector_volatility": round(
                    etf_data["Close"].pct_change().std() * np.sqrt(252) * 100, 2
                ),
            }

            # self.sector_performance_cache[ticker] = sector_metrics
            return sector_metrics

        except Exception as e:
            logger.error(f"Error getting sector performance: {str(e)}")
            return {}

    def _get_market_indicators(self) -> Dict:
        """Get broad market indicators using Polygon data with fallback options"""
        try:
            # Check cache
            # if cache_key in self.market_indicators_cache:
                # return self.market_indicators_cache[cache_key]

            # Get market data for SPY
            spy_data = self.market_data.get_stock_data("SPY", days=30)
            if not spy_data:
                return {}

            spy_hist = spy_data['history']

            # Calculate implied volatility using SPY data instead of VIX
            returns = spy_hist['Close'].pct_change().dropna()
            implied_volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility

            market_metrics = {
                "market_trend": self._identify_price_trend(spy_hist['Close']),
                "market_volatility": round(implied_volatility, 2),
                "market_momentum": round(ta.momentum.rsi(spy_hist['Close']), 2),
                "market_breadth": self._calculate_market_breadth(),
                "market_sentiment": self._calculate_market_sentiment_alternative(implied_volatility)
            }

            # self.market_indicators_cache[cache_key] = market_metrics
            return market_metrics

        except Exception as e:
            logger.error(f"Error getting market indicators: {str(e)}")
            return {}

    def _calculate_market_sentiment_alternative(self, volatility: float) -> str:
        """Calculate market sentiment based on SPY volatility instead of VIX"""
        # Typically, VIX values are about 1.2x the implied volatility calculated from SPY
        # So we'll adjust our thresholds accordingly
        adjusted_volatility = volatility * 1.2
        
        if adjusted_volatility >= 35:
            return "Extremely Fearful"
        elif adjusted_volatility >= 30:
            return "Fearful"
        elif adjusted_volatility >= 25:
            return "Neutral"
        elif adjusted_volatility >= 15:
            return "Complacent"
        else:
            return "Extremely Complacent"
        
    def _calculate_market_breadth(self) -> Dict:
        """Calculate market breadth using major index components"""
        try:
            major_tickers = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META",
                "NVDA", "BRK.B", "JPM", "JNJ", "V"
            ]

            advancing = 0
            declining = 0

            for ticker in major_tickers:
                try:
                    data = self.market_data.get_stock_data(ticker, days=5)
                    if data and len(data['history']) >= 2:
                        hist = data['history']
                        price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) 
                                    / hist['Close'].iloc[-2])
                        
                        if price_change > 0:
                            advancing += 1
                        elif price_change < 0:
                            declining += 1
                except Exception:
                    continue

            total_stocks = advancing + declining
            if total_stocks == 0:
                return {
                    "advance_decline_ratio": 1.0,
                    "breadth_trend": "Neutral",
                    "advancing_stocks": 0,
                    "declining_stocks": 0
                }

            ad_ratio = advancing / total_stocks if total_stocks > 0 else 1.0

            # Determine trend
            if ad_ratio > 0.65:
                trend = "Strongly Positive"
            elif ad_ratio > 0.55:
                trend = "Positive"
            elif ad_ratio < 0.35:
                trend = "Strongly Negative"
            elif ad_ratio < 0.45:
                trend = "Negative"
            else:
                trend = "Neutral"

            return {
                "advance_decline_ratio": round(ad_ratio, 2),
                "breadth_trend": trend,
                "advancing_stocks": advancing,
                "declining_stocks": declining
            }

        except Exception as e:
            logger.error(f"Error calculating market breadth: {str(e)}")
            return {
                "advance_decline_ratio": 1.0,
                "breadth_trend": "Neutral",
                "advancing_stocks": 0,
                "declining_stocks": 0
            }
                
    def _calculate_market_sentiment(self, vix_level: float) -> str:
        """Calculate market sentiment based on VIX"""
        if vix_level >= 30:
            return "Extremely Fearful"
        elif vix_level >= 25:
            return "Fearful"
        elif vix_level >= 20:
            return "Neutral"
        elif vix_level >= 15:
            return "Complacent"
        else:
            return "Extremely Complacent"

    def _calculate_composite_scores(self, metrics: Dict) -> Dict:
        """Calculate composite scores for different aspects of analysis"""
        try:
            scores = {}

            # Technical score components
            technical_components = {
                "trend": self._score_trend(metrics.get("price_trend", "")),
                "momentum": self._score_momentum(metrics),
                "volatility": self._score_volatility(metrics),
                "volume": self._score_volume(metrics),
            }

            # Fundamental score components
            fundamental_components = {
                "valuation": self._score_valuation(metrics),
                "growth": self._score_growth(metrics),
                "profitability": self._score_profitability(metrics),
            }

            # Calculate weighted scores
            scores["technical_score"] = round(
                sum(technical_components.values()) / len(technical_components), 2
            )
            scores["fundamental_score"] = round(
                sum(fundamental_components.values()) / len(fundamental_components), 2
            )

            # Overall score (0-100)
            scores["overall_score"] = round(
                (
                    scores["technical_score"] * 0.4
                    + scores["fundamental_score"] * 0.4
                    + metrics.get("sentiment_score", 0) * 0.2
                ),
                2,
            )

            return scores

        except Exception as e:
            logger.error(f"Error calculating composite scores: {str(e)}")
            return {}

    def _score_trend(self, trend: str) -> float:
        """Score trend strength (0-1)"""
        trend_scores = {
            "Strong Uptrend": 1.0,
            "Moderate Uptrend": 0.75,
            "Sideways": 0.5,
            "Moderate Downtrend": 0.25,
            "Strong Downtrend": 0.0,
        }
        return trend_scores.get(trend, 0.5)

    def _score_momentum(self, metrics: Dict) -> float:
        """
        Score momentum indicators with proper type handling
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            float: Momentum score between 0 and 1
        """
        try:
            scores = []
            
            # RSI score (30-70 range is neutral)
            try:
                rsi = metrics.get("rsi", 50)
                if isinstance(rsi, pd.Series):
                    rsi = rsi.iloc[-1]  # Get last value if it's a series
                rsi = float(rsi)
                
                if rsi > 70:
                    scores.append(1.0)
                elif rsi < 30:
                    scores.append(0.0)
                else:
                    scores.append(0.5 + (rsi - 50) / 40)
            except (TypeError, ValueError) as e:
                logger.debug(f"Error processing RSI: {str(e)}")
                scores.append(0.5)  # Neutral score on error
                
            # MACD score
            try:
                macd = metrics.get("macd", {})
                if isinstance(macd, dict):
                    macd_line = float(macd.get("macd_line", 0))
                    signal_line = float(macd.get("signal_line", 0))
                else:
                    macd_line = 0
                    signal_line = 0
                    
                if macd_line > signal_line:
                    scores.append(0.75)
                else:
                    scores.append(0.25)
            except (TypeError, ValueError) as e:
                logger.debug(f"Error processing MACD: {str(e)}")
                scores.append(0.5)
                
            # Momentum indicator
            try:
                momentum = metrics.get("momentum_score", 0)
                if isinstance(momentum, pd.Series):
                    momentum = momentum.iloc[-1]
                momentum = float(momentum)
                scores.append(0.5 + (momentum / 2))  # Normalize to 0-1 range
            except (TypeError, ValueError) as e:
                logger.debug(f"Error processing momentum score: {str(e)}")
                
            # Sentiment momentum if available
            try:
                sent_momentum = metrics.get("sentiment_momentum", 0)
                if isinstance(sent_momentum, pd.Series):
                    sent_momentum = sent_momentum.iloc[-1]
                sent_momentum = float(sent_momentum)
                
                if abs(sent_momentum) > 0:
                    scores.append(0.5 + (sent_momentum / 2))
            except (TypeError, ValueError) as e:
                logger.debug(f"Error processing sentiment momentum: {str(e)}")
                
            # Calculate final score
            if scores:
                return float(sum(scores) / len(scores))
            return 0.5  # Neutral score if no valid components
            
        except Exception as e:
            logger.error(f"Error scoring momentum: {str(e)}")
            return 0.5  # Return neutral score on error

    def _safe_float_conversion(self, value: Any, default: float = 0.0) -> float:
        """
        Safely convert a value to float
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            float: Converted value or default
        """
        try:
            if isinstance(value, pd.Series):
                if len(value) > 0:
                    return float(value.iloc[-1])
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _score_volatility(self, metrics: Dict) -> float:
        """Score volatility metrics (0-1)"""
        try:
            volatility = metrics.get("volatility_daily", 30)

            if volatility > 100:
                return 0.0  # Extremely volatile
            elif volatility < 10:
                return 1.0  # Very stable
            else:
                return 1 - (volatility / 100)

        except Exception as e:
            logger.error(f"Error scoring volatility: {str(e)}")
            return 0.5

    def _score_volume(self, metrics: Dict) -> float:
        """Score volume metrics (0-1)"""
        try:
            relative_volume = metrics.get("relative_volume", 1.0)

            if relative_volume > 2.0:
                return 1.0  # High volume is good
            elif relative_volume < 0.5:
                return 0.0  # Low volume is bad
            else:
                return relative_volume / 2.0

        except Exception as e:
            logger.error(f"Error scoring volume: {str(e)}")
            return 0.5

    def _score_valuation(self, metrics: Dict) -> float:
        """Score valuation metrics (0-1)"""
        try:
            pe_ratio = metrics.get("pe_ratio", 0.0)

            if pe_ratio <= 0:
                return 0.0
            elif pe_ratio > 100:
                return 0.0
            else:
                # Score based on distance from ideal PE ratio (15)
                return 1 - min(abs(pe_ratio - 15) / 15, 1)

        except Exception as e:
            logger.error(f"Error scoring valuation: {str(e)}")
            return 0.5

    def _score_growth(self, metrics: Dict) -> float:
        """Score growth metrics (0-1)"""
        try:
            revenue_growth = metrics.get("revenue_growth", 0)

            if revenue_growth > 0.5:
                return 1.0
            elif revenue_growth < -0.2:
                return 0.0
            else:
                return 0.5 + revenue_growth

        except Exception as e:
            logger.error(f"Error scoring growth: {str(e)}")
            return 0.5

    def _score_profitability(self, metrics: Dict) -> float:
        """Score profitability metrics (0-1)"""
        try:
            profit_margin = metrics.get("profit_margins", 0)

            if profit_margin > 0.2:
                return 1.0
            elif profit_margin < 0:
                return 0.0
            else:
                return profit_margin * 5

        except Exception as e:
            logger.error(f"Error scoring profitability: {str(e)}")
            return 0.5

    def _calculate_basic_metrics(self, hist: pd.DataFrame) -> Dict:
        """
        Calculate basic price and volume metrics

        Args:
            hist: Historical price data DataFrame
        Returns:
            Dict of calculated metrics
        """
        try:
            current_price = hist["Close"].iloc[-1]
            price_1w_ago = hist["Close"].iloc[-10]
            price_1d_ago = hist["Close"].iloc[-2]
            avg_volume = hist["Volume"].mean()

            # Verify we have valid numerical data
            if not (
                pd.notnull(current_price)
                and pd.notnull(price_1w_ago)
                and pd.notnull(avg_volume)
            ):
                logger.error("Invalid price or volume data")
                return {}

            return {
                "current_price": round(current_price, 2),
                "price_change_1w": round(
                    ((current_price - price_1w_ago) / price_1w_ago) * 100, 2
                ),
                "price_change_1d": round(
                    ((current_price - price_1d_ago) / price_1d_ago) * 100, 2
                ),
                "avg_volume": int(avg_volume),
                "volume_change": round(
                    ((hist["Volume"].iloc[-1] - avg_volume) / avg_volume) * 100, 2
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def get_fundamental_metrics(self, ticker: str) -> Dict:
        """Get fundamental metrics for a stock"""
        try:
            # Get financial ratios from Polygon
            ratios = self.market_data.get_financial_ratios(ticker)
            
            return {
                "market_cap": self.market_data.get_market_cap(ticker),
                "pe_ratio": ratios.get("pe_ratio"),
                "price_to_book": ratios.get("price_to_book"),
                "debt_to_equity": ratios.get("debt_to_equity"),
                "current_ratio": ratios.get("current_ratio"),
                "profit_margin": ratios.get("profit_margin"),
                "roe": ratios.get("roe")
            }
        except Exception as e:
            logger.warning(f"Error fetching fundamental metrics: {str(e)}")
            return {}

    def _get_market_metrics(self, stock) -> Dict:
        """Get market-related metrics for a stock"""
        try:
            info = stock.info
            return {
                "target_price": info.get("targetMeanPrice"),
                "recommendation": info.get("recommendationKey"),
                "analyst_count": info.get("numberOfAnalystOpinions"),
                "short_ratio": info.get("shortRatio"),
                "relative_volume": info.get("regularMarketVolume", 0)
                / (info.get("averageVolume", 1) or 1),
            }
        except Exception as e:
            logger.warning(f"Error fetching market metrics: {str(e)}")
            return {}

    def calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            close = hist["Close"]
            volume = hist["Volume"]

            return {
                # Trend Indicators
                "sma_20": round(close.rolling(window=20).mean().iloc[-1], 2),
                "ema_9": round(close.ewm(span=9).mean().iloc[-1], 2),
                "rsi": round(self._calculate_rsi(close), 2),
                # Volatility Indicators
                "volatility": round(close.pct_change().std() * np.sqrt(252) * 100, 2),
                # Volume Indicators
                "volume_sma": round(volume.rolling(window=20).mean().iloc[-1], 2),
                "volume_ratio": round(
                    volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1], 2
                ),
            }
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {str(e)}")
            return {}

    def _calculate_price_gaps(self, hist: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """Calculate and classify price gaps in the data"""
        try:
            gaps = {"gap_up": 0.0, "gap_down": 0.0, "gap_type": "None"}

            # Calculate gaps between daily candles
            high_prev = hist["High"].shift(1)
            low_prev = hist["Low"].shift(1)
            high_curr = hist["High"]
            low_curr = hist["Low"]

            # Check last 5 days for gaps
            for i in range(-5, 0):
                try:
                    # Gap up: Today's low higher than yesterday's high
                    if low_curr.iloc[i] > high_prev.iloc[i]:
                        gaps["gap_up"] = round(
                            ((low_curr.iloc[i] / high_prev.iloc[i]) - 1) * 100, 2
                        )
                        gaps["gap_type"] = "Up"
                        break

                    # Gap down: Today's high lower than yesterday's low
                    elif high_curr.iloc[i] < low_prev.iloc[i]:
                        gaps["gap_down"] = round(
                            ((high_curr.iloc[i] / low_prev.iloc[i]) - 1) * 100, 2
                        )
                        gaps["gap_type"] = "Down"
                        break
                except IndexError:
                    continue

            return gaps

        except Exception as e:
            logger.error(f"Error calculating price gaps: {str(e)}")
            return {"gap_up": 0.0, "gap_down": 0.0, "gap_type": "None"}

    def _identify_swing_points(
        self, hist: pd.DataFrame, window: int = 20
    ) -> Dict[str, float]:
        """Identify swing high and low points"""
        try:
            close = hist["Close"].values

            # Find local maxima and minima
            max_idx = argrelextrema(close, np.greater, order=window)[0]
            min_idx = argrelextrema(close, np.less, order=window)[0]

            # Get the most recent swing points
            recent_high = close[max_idx[-1]] if len(max_idx) > 0 else close[-1]
            recent_low = close[min_idx[-1]] if len(min_idx) > 0 else close[-1]

            return {
                "swing_high": round(float(recent_high), 2),
                "swing_low": round(float(recent_low), 2),
            }

        except Exception as e:
            logger.error(f"Error identifying swing points: {str(e)}")
            return {"swing_high": 0.0, "swing_low": 0.0}

    def _calculate_pivot_points(self, hist: pd.DataFrame) -> Dict[str, float]:
        """Calculate floor trader pivot points"""
        try:
            last_day = hist.iloc[-1]
            high = last_day["High"]
            low = last_day["Low"]
            close = last_day["Close"]

            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)

            return {
                "pivot": round(pivot, 2),
                "r1": round(r1, 2),
                "r2": round(r2, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2),
            }

        except Exception as e:
            logger.error(f"Error calculating pivot points: {str(e)}")
            return {"pivot": 0.0, "r1": 0.0, "r2": 0.0, "s1": 0.0, "s2": 0.0}

    def _calculate_price_metrics(self, hist: pd.DataFrame) -> Dict:
        """Calculate comprehensive price-based metrics with data validation"""
        metrics = {}
        try:
            if len(hist) < 2:  # Minimum required data points
                return {}
                
            close = hist['Close']
            
            # Base calculations with data validation
            metrics['current_price'] = round(close.iloc[-1], 2)
            
            # 1-day change (requires at least 2 data points)
            if len(close) >= 2:
                metrics['price_change_1d'] = round(
                    ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100, 2
                )
            
            # 1-week change (requires at least 5 data points)
            if len(close) >= 5:
                metrics['price_change_1w'] = round(
                    ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100, 2
                )
                
            # 1-month change (requires at least 21 data points)
            if len(close) >= 21:
                metrics['price_change_1m'] = round(
                    ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100, 2
                )
                
            # 3-month change (requires at least 63 data points)
            if len(close) >= 63:
                metrics['price_change_3m'] = round(
                    ((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]) * 100, 2
                )
            
            # Add price gaps if enough data
            if len(hist) >= 2:
                metrics['price_gaps'] = self._calculate_price_gaps(hist)
                
            # Add price trend if enough data
            if len(close) >= 20:  # Minimum for meaningful trend
                metrics['price_trend'] = self._identify_price_trend(close)
                
            # Add support/resistance if enough data
            if len(hist) >= 20:
                metrics['support_resistance'] = self._calculate_support_resistance(hist)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error in price metrics calculation: {str(e)}")
            return {}
           
    def _calculate_momentum_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate comprehensive momentum indicators"""
        metrics = {}
        try:
            close = hist["Close"]
            high = hist["High"]
            low = hist["Low"]

            # Initialize technical indicators
            macd = MACD(close)
            stoch = StochasticOscillator(high, low, close)
            bb = BollingerBands(close)

            metrics.update(
                {
                    "rsi": round(ta.momentum.rsi(close), 2),
                    "stoch_k": round(stoch.stoch(), 2),
                    "stoch_d": round(stoch.stoch_signal(), 2),
                    "macd_line": round(macd.macd(), 2),
                    "macd_signal": round(macd.macd_signal(), 2),
                    "macd_diff": round(macd.macd_diff(), 2),
                    "adx": round(ta.trend.adx(high, low, close), 2),
                    "cci": round(ta.trend.cci(high, low, close), 2),
                    "bb_upper": round(bb.bollinger_hband(), 2),
                    "bb_lower": round(bb.bollinger_lband(), 2),
                    "bb_middle": round(bb.bollinger_mavg(), 2),
                }
            )

        except Exception as e:
            logger.error(f"Error in momentum indicators calculation: {str(e)}")

        return metrics

    def analyze_subreddit_sentiment(
        self, subreddit_name: str, time_filter: str = "day", limit: int = 2
    ) -> pd.DataFrame:
        """Analyze subreddit with improved error handling and save posts by ticker"""
        sentiment_data = self._get_reddit_sentiment(subreddit_name, time_filter, limit)

        if sentiment_data.empty:
            logger.warning(
                f"No valid sentiment data found for subreddit: {subreddit_name}"
            )
            return pd.DataFrame()

        # Filter out rows with invalid tickers before processing
        valid_rows = []
        for idx, row in sentiment_data.iterrows():
            if self.is_valid_ticker(row["ticker"]):
                valid_rows.append(idx)

        sentiment_data = sentiment_data.loc[valid_rows]

        if sentiment_data.empty:
            logger.warning(f"No valid tickers found in subreddit: {subreddit_name}")
            return pd.DataFrame()

        # Add stock metrics in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self.get_stock_metrics, row["ticker"]): row["ticker"]
                for _, row in sentiment_data.iterrows()
            }

            # Collect valid results
            valid_tickers = set()
            for future in future_to_ticker:
                ticker = future_to_ticker[future]
                try:
                    metrics = future.result()
                    if metrics:
                        valid_tickers.add(ticker)
                        for key, value in metrics.items():
                            sentiment_data.loc[
                                sentiment_data["ticker"] == ticker, key
                            ] = value
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")

            # Filter out rows without valid metrics
            sentiment_data = sentiment_data[
                sentiment_data["ticker"].isin(valid_tickers)
            ]

        # Save posts by ticker
        # self.save_posts_by_ticker(sentiment_data, subreddit_name)

        return sentiment_data

    def _get_reddit_sentiment(
        self, subreddit_name: str, time_filter: str, limit: int
    ) -> pd.DataFrame:
        """Get Reddit sentiment analysis with enhanced scoring"""
        subreddit = self.reddit.subreddit(subreddit_name)
        sentiment_data = []
        post_data = []

        try:
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                submission_tickers = self.extract_stock_tickers(
                    submission.title + " " + submission.selftext
                )
                if not submission_tickers:
                    continue

                # First analyze submission sentiment
                submission_sentiment = self._analyze_text_sentiment(
                    submission.title + " " + submission.selftext
                )
                # Analyze comments with weighted scoring
                submission.comments.replace_more(
                    limit=int(os.getenv("REDDIT_COMMENT_LIMIT", 10))
                )
                comments = submission.comments.list()

                comment_data = []
                sentiment_scores = []
                weighted_scores = []  # Weight scores by comment score and awards
                bullish_comments = []
                bearish_comments = []

                for comment in comments:
                    try:
                        # Get base sentiment
                        comment_sentiment = self._analyze_text_sentiment(comment.body)
                        base_score = comment_sentiment["compound"]

                        # Calculate comment weight based on score and awards
                        comment_weight = self._calculate_comment_weight(comment)

                        weighted_sentiment = base_score * comment_weight
                        sentiment_scores.append(base_score)
                        weighted_scores.append(weighted_sentiment)

                        # Classify comment sentiment
                        if base_score > 0.1:
                            bullish_comments.append(weighted_sentiment)
                        elif base_score < -0.1:
                            bearish_comments.append(weighted_sentiment)

                        comment_data.append(
                            {
                                "author": str(comment.author),
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": dt.datetime.fromtimestamp(
                                    comment.created_utc
                                ),
                                "sentiment": comment_sentiment,
                                "weight": comment_weight,
                            }
                        )
                    except Exception as e:
                        continue

                # Calculate final sentiment metrics
                avg_base_sentiment = (
                    np.mean(sentiment_scores) if sentiment_scores else 0
                )
                avg_weighted_sentiment = (
                    np.mean(weighted_scores) if weighted_scores else 0
                )

                # Calculate bullish/bearish ratios using weighted scores
                total_bullish = sum(score for score in bullish_comments)
                total_bearish = abs(sum(score for score in bearish_comments))
                total_sentiment = total_bullish + total_bearish

                bullish_ratio = (
                    total_bullish / total_sentiment if total_sentiment > 0 else 0
                )
                bearish_ratio = (
                    total_bearish / total_sentiment if total_sentiment > 0 else 0
                )

                # Store post data
                post_info = {
                    "post_id": submission.id,
                    "title": submission.title,
                    "content": submission.selftext,
                    "url": submission.url,
                    "author": str(submission.author),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "upvote_ratio": submission.upvote_ratio,
                    "created_utc": dt.datetime.fromtimestamp(submission.created_utc),
                    "tickers": submission_tickers,
                    "submission_sentiment": submission_sentiment,
                    "avg_base_sentiment": avg_base_sentiment,
                    "avg_weighted_sentiment": avg_weighted_sentiment,
                    "comments": comment_data,
                    "subreddit": subreddit_name,
                }
                post_data.append(post_info)

                # Add sentiment data for each ticker
                for ticker in submission_tickers:
                    sentiment_data.append(
                        {
                            "ticker": ticker,
                            "title": submission.title,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "upvote_ratio": submission.upvote_ratio,
                            "comment_sentiment_avg": avg_weighted_sentiment,
                            "base_sentiment": avg_base_sentiment,
                            "submission_sentiment": submission_sentiment["compound"],
                            "bullish_comments_ratio": bullish_ratio,
                            "bearish_comments_ratio": bearish_ratio,
                            "sentiment_confidence": len(
                                sentiment_scores
                            ),  # Number of analyzed comments
                            "timestamp": dt.datetime.fromtimestamp(
                                submission.created_utc
                            ),
                            "post_id": submission.id,
                        }
                    )

            # Save to Supabase
            for post in post_data:
                self.db.save_post_data(post)

            return pd.DataFrame(sentiment_data)

        except Exception as e:
            logger.error(f"Error analyzing subreddit {subreddit_name}: {str(e)}")
            return pd.DataFrame()

    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Enhanced sentiment analysis combining VADER, LLM, and technical features

        Args:
            text: Text to analyze

        Returns:
            Dict containing combined sentiment scores and metadata
        """
        try:
            # Get base VADER sentiment
            vader_scores = self.sia.polarity_scores(text)

            # Get basic sentiment features
            basic_features = self._get_basic_sentiment_features(text)

            # # Get LLM sentiment if text is substantial enough (e.g., > 50 chars)
            # llm_sentiment = {}
            # if len(text) > 50:
            #     try:
            #         llm_sentiment = self._get_llm_sentiment(text)
            #     except Exception as e:
            #         logger.warning(f"LLM sentiment analysis failed: {str(e)}")
            #         llm_sentiment = {
            #             "score": 0,
            #             "features": {},
            #             "confidence": 0,
            #             "terms": [],
            #         }

            # Combine base scores with adjustments
            base_score = vader_scores["compound"]

            # Adjust sentiment based on features and context
            adjusted_score = self._adjust_sentiment_score(
                base_score, basic_features, text
            )

            # Calculate confidence score
            confidence_score = min(
                1.0,
                (
                    0.4 * (1 - vader_scores["neu"])  # VADER confidence
                    # + 0.4 * llm_sentiment.get("confidence", 0)  # LLM confidence
                    + 0.2
                    * (len(text) / 1000)  # Length-based confidence (max at 1000 chars)
                ),
            )

            # Combine all signals
            final_compound = (
                adjusted_score * 0.4  # Adjusted VADER score
                # + llm_sentiment.get("score", 0) * 0.4  # LLM score
                + base_score * 0.2  # Original VADER score
            )

            # Ensure the final score is within [-1, 1]
            final_compound = max(min(final_compound, 1.0), -1.0)

            return {
                "compound": final_compound,
                "confidence": confidence_score,
                "vader_scores": vader_scores,
                # "llm_sentiment": llm_sentiment,
                "features": basic_features,
                # "terms": llm_sentiment.get("terms", []),
                "pos": vader_scores["pos"],
                "neg": vader_scores["neg"],
                "neu": vader_scores["neu"],
            }

        except Exception as e:
            logger.error(f"Error in enhanced sentiment analysis: {str(e)}")
            # Fallback to basic VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            return {
                "compound": vader_scores["compound"],
                "confidence": 0.3,  # Low confidence for fallback
                "pos": vader_scores["pos"],
                "neg": vader_scores["neg"],
                "neu": vader_scores["neu"],
                "features": self._get_basic_sentiment_features(text),
            }

    def _adjust_sentiment_score(
        self, base_score: float, features: Dict, text: str
    ) -> float:
        """
        Adjust sentiment score based on financial context and features

        Args:
            base_score: Base sentiment score from VADER
            features: Dictionary of extracted features
            text: Original text

        Returns:
            float: Adjusted sentiment score
        """
        adjustment = 0.0

        # Adjust for financial-specific features
        if features["has_exclamation"]:
            adjustment += 0.1
        if features["has_dollar_sign"]:
            adjustment += 0.05
        if features["has_rocket"] or features["has_moon"]:
            adjustment += 0.15
        if features["has_buy_terms"]:
            adjustment += 0.1
        if features["has_sell_terms"]:
            adjustment -= 0.1
        if features["has_analysis_terms"]:
            adjustment += 0.05  # Slight boost for analytical content

        # Adjust for specific financial terms
        text_lower = text.lower()

        # Positive financial terms
        positive_terms = [
            "upgrade",
            "beat",
            "exceeded",
            "growth",
            "profit",
            "outperform",
            "strong",
            "higher",
            "up",
            "gain",
            "positive",
            "buy",
            "accumulate",
            "bullish",
        ]

        # Negative financial terms
        negative_terms = [
            "downgrade",
            "miss",
            "missed",
            "decline",
            "loss",
            "underperform",
            "weak",
            "lower",
            "down",
            "negative",
            "sell",
            "bearish",
            "warning",
        ]

        # Count term occurrences and adjust
        positive_count = sum(1 for term in positive_terms if term in text_lower)
        negative_count = sum(1 for term in negative_terms if term in text_lower)

        adjustment += (positive_count * 0.05) - (negative_count * 0.05)

        # Additional context adjustments
        if "bankruptcy" in text_lower or "bankrupt" in text_lower:
            adjustment -= 0.3
        if "merger" in text_lower or "acquisition" in text_lower:
            adjustment += 0.2
        if "scandal" in text_lower or "fraud" in text_lower:
            adjustment -= 0.3
        if "earnings" in text_lower:
            if any(term in text_lower for term in ["beat", "exceeded", "above"]):
                adjustment += 0.2
            elif any(term in text_lower for term in ["miss", "below", "disappoint"]):
                adjustment -= 0.2

        # Scale adjustment based on base score direction
        if base_score > 0:
            adjustment = min(adjustment, 1 - base_score)  # Don't exceed 1
        elif base_score < 0:
            adjustment = max(adjustment, -1 - base_score)  # Don't go below -1

        # Calculate final score
        final_score = base_score + adjustment

        # Ensure the final score is within [-1, 1]
        return max(min(final_score, 1.0), -1.0)

    def _get_basic_sentiment_features(self, text: str) -> Dict:
        """Extract basic sentiment features from text"""
        text_lower = text.lower()
        return {
            "has_exclamation": "!" in text,
            "has_dollar_sign": "$" in text,
            "has_rocket": "🚀" in text or "rocket" in text_lower,
            "has_moon": "🌙" in text_lower or "moon" in text_lower,
            "has_buy_terms": any(
                term in text_lower
                for term in [
                    "buy",
                    "long",
                    "bull",
                    "calls",
                    "bullish",
                    "upside",
                    "breakout",
                    "growth",
                    "strong",
                    "higher",
                    "accumulate",
                ]
            ),
            "has_sell_terms": any(
                term in text_lower
                for term in [
                    "sell",
                    "short",
                    "bear",
                    "puts",
                    "bearish",
                    "downside",
                    "breakdown",
                    "weak",
                    "lower",
                    "crash",
                    "dump",
                ]
            ),
            "has_analysis_terms": any(
                term in text_lower
                for term in [
                    "analysis",
                    "technical",
                    "fundamental",
                    "chart",
                    "pattern",
                    "trend",
                    "support",
                    "resistance",
                    "price target",
                    "valuation",
                    "forecast",
                ]
            ),
        }

    def _calculate_comment_weight(self, comment) -> float:
        """
        Calculate enhanced weight for a comment based on various factors

        Args:
            comment: Reddit comment object

        Returns:
            float: Weight between 0.5 and 2.0
        """
        # Base score weight
        if comment.score <= 0:
            score_weight = 0.5
        else:
            score_weight = min(1 + np.log1p(comment.score) / 10, 1.5)

        # Awards weight
        try:
            awards_count = (
                len(comment.all_awardings) if hasattr(comment, "all_awardings") else 0
            )
            awards_weight = min(1 + (awards_count * 0.1), 1.5)
        except:
            awards_weight = 1.0

        # Content quality weights
        length_weight = min(1 + (len(comment.body) / 1000), 1.2)

        # Check for quality indicators
        quality_indicators = {
            "has_numbers": bool(re.search(r"\d+", comment.body)),
            "has_links": "http" in comment.body.lower(),
            "has_analysis": any(
                term in comment.body.lower()
                for term in [
                    "analysis",
                    "research",
                    "report",
                    "study",
                    "data",
                    "evidence",
                    "source",
                    "reference",
                ]
            ),
        }

        quality_score = 1.0
        if quality_indicators["has_numbers"]:
            quality_score += 0.1
        if quality_indicators["has_links"]:
            quality_score += 0.1
        if quality_indicators["has_analysis"]:
            quality_score += 0.2

        # Calculate final weight
        final_weight = (
            score_weight * 0.4
            + awards_weight * 0.2
            + length_weight * 0.2
            + quality_score * 0.2
        )

        # Ensure weight is within bounds
        return min(max(final_weight, 0.5), 2.0)

    # def _get_llm_sentiment(self, text: str) -> Dict:
    #     """Get sentiment analysis from OpenAI with proper schema validation"""
    #     try:
    #         # Generate hash of the text for deduplication
    #         text_hash = hashlib.sha256(text.encode()).hexdigest()

    #         # Check cache
    #         existing_analysis = self.db.get_llm_sentiment(text_hash)
    #         if existing_analysis:
    #             return {
    #                 "score": existing_analysis["sentiment_score"],
    #                 "features": existing_analysis["features"],
    #                 "confidence": existing_analysis["confidence_score"],
    #                 "terms": existing_analysis["terms"],
    #             }

    #         start_time = time.time()

    #         system_message = {
    #             "role": "system",
    #             "content": """You are a financial analyst expert in market sentiment analysis.
    #             Analyze the given text and provide a structured sentiment analysis response.
    #             Ensure all numeric values are within their specified ranges.""",
    #         }

    #         user_message = {
    #             "role": "user",
    #             "content": f"""Analyze the sentiment of this financial text and provide:

    #             Text: {text}

    #             Return a JSON response with the following structure:
    #             {{
    #                 "score": <float between -1 and 1>,
    #             }}""",
    #         }

    #         # Call OpenAI API with standard JSON response
    #         response = self.openai_client.chat.completions.create(
    #             model="gpt-4",
    #             messages=[system_message, user_message],
    #             temperature=0,
    #             # response_format={"type": "json_object"},
    #         )

    #         processing_time = int((time.time() - start_time) * 1000)

    #         try:
    #             # Parse the response
    #             sentiment_result = json.loads(response.choices[0].message.content)

    #             # Validate against schema
    #             validated_result = SentimentAnalysisResult(**sentiment_result)

    #             # Prepare record for storage
    #             sentiment_data = {
    #                 "text_hash": text_hash,
    #                 "original_text": text,
    #                 "analysis_timestamp": dt.datetime.now().isoformat(),
    #                 "sentiment_score": validated_result.score,
    #                 "raw_response": response.model_dump(),
    #                 "processing_time_ms": processing_time,
    #                 "model_version": "gpt-4",
    #             }

    #             # Store in database
    #             self.db.save_llm_sentiment(sentiment_data)

    #             return validated_result.model_dump()

    #         except json.JSONDecodeError as e:
    #             logger.error(f"Error parsing OpenAI response: {str(e)}")
    #             return {"score": 0, "features": {}, "confidence": 0, "terms": []}
    #         except ValueError as e:
    #             logger.error(f"Schema validation error: {str(e)}")
    #             return {"score": 0, "features": {}, "confidence": 0, "terms": []}

    #     except Exception as e:
    #         logger.error(f"Error in LLM sentiment analysis: {str(e)}")
    #         return {"score": 0, "features": {}, "confidence": 0, "terms": []}

    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            close = hist["Close"]
            volume = hist["Volume"]

            return {
                # Trend Indicators
                "sma_20": round(close.rolling(window=20).mean().iloc[-1], 2),
                "ema_9": round(close.ewm(span=9).mean().iloc[-1], 2),
                "rsi": round(self._calculate_rsi(close), 2),
                # Volatility Indicators
                "volatility": round(close.pct_change().std() * np.sqrt(252) * 100, 2),
                "bollinger_upper": round(
                    close.rolling(20).mean() + close.rolling(20).std() * 2, 2
                ).iloc[-1],
                "bollinger_lower": round(
                    close.rolling(20).mean() - close.rolling(20).std() * 2, 2
                ).iloc[-1],
                # Volume Indicators
                "volume_sma": round(volume.rolling(window=20).mean().iloc[-1], 2),
                "volume_ratio": round(
                    volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1], 2
                ),
                # Momentum Indicators
                "macd": self._calculate_macd(close),
                "stochastic": self._calculate_stochastic(hist),
            }
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {str(e)}")
            return {}

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            "macd_line": round(macd.iloc[-1], 2),
            "signal_line": round(signal.iloc[-1], 2),
            "macd_histogram": round(macd.iloc[-1] - signal.iloc[-1], 2),
        }

    def _calculate_stochastic(self, hist: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        high_max = hist["High"].rolling(period).max()
        low_min = hist["Low"].rolling(period).min()
        k = 100 * (hist["Close"] - low_min) / (high_max - low_min)
        d = k.rolling(3).mean()
        return {"stoch_k": round(k.iloc[-1], 2), "stoch_d": round(d.iloc[-1], 2)}

    def generate_analysis_report(self, df: pd.DataFrame) -> str:
        """
        Generate insights from the analyzed data with timestamps

        Args:
            df (pd.DataFrame): Analyzed stock data

        Returns:
            str: Analysis report
        """
        if df.empty:
            return "No data available for analysis"

        # Get current timestamp
        analysis_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_start = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")

        report = []
        report.append("Stock Analysis Report")
        report.append("===================")
        report.append(f"Analysis Generated: {analysis_time}")
        report.append(
            f"Data Period: {data_start} to {dt.datetime.now().strftime('%Y-%m-%d')}"
        )
        report.append("===================\n")

        # Most discussed stocks
        top_discussed = df.nlargest(5, "num_comments")
        report.append("Most Discussed Stocks:")
        for _, row in top_discussed.iterrows():
            report.append(f"- {row['ticker']}: {row['num_comments']} comments")

        # Highest sentiment scores
        top_sentiment = df.nlargest(5, "comment_sentiment_avg")
        report.append("\nMost Positive Sentiment:")
        for _, row in top_sentiment.iterrows():
            report.append(f"- {row['ticker']}: {row['comment_sentiment_avg']:.2f}")

        # Price changes
        top_gainers = df.nlargest(3, "price_change_1w")
        report.append("\nTop Price Gainers (2 weeks):")
        for _, row in top_gainers.iterrows():
            report.append(
                f"- {row['ticker']}: {row['price_change_1w']}% (Current: ${row['current_price']})"
            )

        # Technical signals
        report.append("\nTechnical Signals:")
        for _, row in df.iterrows():
            if "rsi" in row and "sma_20" in row:
                if row["rsi"] > 70:
                    report.append(
                        f"- {row['ticker']}: Potentially overbought (RSI: {row['rsi']:.2f})"
                    )
                elif row["rsi"] < 30:
                    report.append(
                        f"- {row['ticker']}: Potentially oversold (RSI: {row['rsi']:.2f})"
                    )

        return "\n".join(report)

    def save_results(self, df: pd.DataFrame, filename: str = "stock_analysis.csv"):
        """
        Save results with enhanced metrics and detailed sentiment analysis

        Args:
            df (pd.DataFrame): Analysis results DataFrame
            filename (str): Base filename for saving results
        """
        if df.empty:
            logger.warning("No results to save")
            return

        def convert_to_json_serializable(obj):
            """Helper function to convert values to JSON serializable types"""
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (dt.datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif pd.isna(obj):
                return None
            return obj

        def safe_get(row, key, default=None):
            """Safely get a value from a row, with default if missing"""
            try:
                return convert_to_json_serializable(row.get(key, default))
            except:
                return default

        try:
            # Create a timestamp for this analysis
            current_time = dt.datetime.now()

            # Add timing information
            df["analysis_timestamp"] = current_time
            df["data_start_date"] = current_time - dt.timedelta(days=30)
            df["data_end_date"] = current_time

            # Create results directory with timestamp
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            results_dir = f"analysis_results/{timestamp_str}"
            os.makedirs(results_dir, exist_ok=True)

            # Prepare main analysis DataFrame
            analysis_df = df.copy()

            # Save main analysis CSV
            csv_path = f"{results_dir}/{filename}"
            analysis_df.to_csv(csv_path, index=False)

            # Generate and save detailed report
            report = self.generate_analysis_report(analysis_df)
            report_path = f"{results_dir}/analysis_report.txt"
            with open(report_path, "w") as f:
                f.write(report)

            # Save detailed sentiment data in JSON format
            sentiment_data = {
                "analysis_timestamp": current_time.isoformat(),
                "metadata": {
                    "total_stocks_analyzed": int(len(df)),
                    "average_sentiment": (
                        float(df["comment_sentiment_avg"].mean())
                        if "comment_sentiment_avg" in df.columns
                        else 0.0
                    ),
                    "total_comments_analyzed": (
                        int(df["num_comments"].sum())
                        if "num_comments" in df.columns
                        else 0
                    ),
                },
                "stocks": [],
            }

            for _, row in df.iterrows():
                print(row)
                stock_data = {
                    "ticker": str(row["ticker"]),
                    "sentiment_metrics": {
                        "comment_sentiment_avg": safe_get(
                            row, "comment_sentiment_avg", 0.0
                        ),
                        "bullish_ratio": safe_get(row, "bullish_comments_ratio", 0.0),
                        "bearish_ratio": safe_get(row, "bearish_comments_ratio", 0.0),
                    },
                    "price_metrics": {
                        "current_price": safe_get(row, "current_price"),
                        "price_change_1w": safe_get(row, "price_change_1w"),
                        "price_change_1d": safe_get(row, "price_change_1d"),
                        "volume_change": safe_get(row, "volume_change"),
                        "technical_indicators": {
                            "sma_20": safe_get(row, "sma_20"),
                            "rsi": safe_get(row, "rsi"),
                            "volatility": safe_get(row, "volatility"),
                        },
                    },
                    "fundamental_metrics": {
                        "market_cap": safe_get(row, "market_cap"),
                        "pe_ratio": safe_get(row, "pe_ratio"),
                    },
                    "reddit_metrics": {
                        "score": safe_get(row, "score", 0),
                        "num_comments": safe_get(row, "num_comments", 0),
                        "composite_score": safe_get(row, "composite_score", 0.0),
                    },
                }
                sentiment_data["stocks"].append(stock_data)

            # Save detailed JSON
            json_path = f"{results_dir}/detailed_analysis.json"
            with open(json_path, "w") as f:
                json.dump(sentiment_data, f, indent=2)

            # Save summary of extremes with converted values
            summary_data = {
                "most_bullish": [
                    {
                        "ticker": str(r["ticker"]),
                        "sentiment": safe_get(r, "comment_sentiment_avg", 0.0),
                    }
                    for _, r in df.nlargest(5, "comment_sentiment_avg")[
                        ["ticker", "comment_sentiment_avg"]
                    ].iterrows()
                ],
                "most_bearish": [
                    {
                        "ticker": str(r["ticker"]),
                        "sentiment": safe_get(r, "comment_sentiment_avg", 0.0),
                    }
                    for _, r in df.nsmallest(5, "comment_sentiment_avg")[
                        ["ticker", "comment_sentiment_avg"]
                    ].iterrows()
                ],
                "most_discussed": [
                    {"ticker": str(r["ticker"]), "comments": int(r["num_comments"])}
                    for _, r in df.nlargest(5, "num_comments")[
                        ["ticker", "num_comments"]
                    ].iterrows()
                ],
                "best_performing": [
                    {
                        "ticker": str(r["ticker"]),
                        "price_change": safe_get(r, "price_change_1w", 0.0),
                    }
                    for _, r in df.nlargest(5, "price_change_1w")[
                        ["ticker", "price_change_1w"]
                    ].iterrows()
                ],
                "worst_performing": [
                    {
                        "ticker": str(r["ticker"]),
                        "price_change": safe_get(r, "price_change_1w", 0.0),
                    }
                    for _, r in df.nsmallest(5, "price_change_1w")[
                        ["ticker", "price_change_1w"]
                    ].iterrows()
                ],
            }

            summary_path = f"{results_dir}/analysis_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)

            logger.info(
                f"Analysis results saved to directory: {results_dir}\n"
                f"- CSV: {filename}\n"
                f"- Report: analysis_report.txt\n"
                f"- Detailed Analysis: detailed_analysis.json\n"
                f"- Summary: analysis_summary.json"
            )

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise  # Re-raise the exception for debugging

    def save_results_to_storage(
        self, df: pd.DataFrame, bucket_name: str = "analysis-results"
    ) -> Dict:
        """
        Save analysis results to Supabase storage bucket with improved organization and metadata

        Args:
            df (pd.DataFrame): Analysis results DataFrame
            bucket_name (str): Name of the storage bucket

        Returns:
            Dict: Operation status and results
        """
        if os.getenv("DB_WRITE", False) != "true":
            logger.warning("DB_WRITE environment variable not set to 'true'")
            return {"success": False, "error": "DB_WRITE not enabled"}
        
        if df.empty:
            logger.warning("No results to save")
            return {"success": False, "error": "Empty DataFrame provided"}

        try:
            # Add timing information to DataFrame
            current_time = dt.datetime.now()
            df["analysis_timestamp"] = current_time
            df["data_start_date"] = current_time - dt.timedelta(days=30)
            df["data_end_date"] = current_time

            # Save results using new SupabaseConnector method
            storage_result = self.db.save_analysis_to_storage(df, bucket_name)

            if not storage_result["success"]:
                logger.error(
                    f"Failed to save to storage: {storage_result.get('error')}"
                )
                return storage_result

            # Generate additional analysis report
            report = self.generate_analysis_report(df)

            # Save report to storage
            report_result = self.db.save_to_storage(
                {
                    "content": report,
                    "path": f"{storage_result['base_path']}/analysis_report.txt",
                    "content_type": "text/plain",
                },
                bucket_name,
            )

            # Save enhanced sentiment data
            sentiment_data = self.prepare_sentiment_data(df)
            sentiment_result = self.db.save_to_storage(
                {
                    "content": json.dumps(sentiment_data, indent=2),
                    "path": f"{storage_result['base_path']}/sentiment_analysis.json",
                    "content_type": "application/json",
                },
                bucket_name,
            )

            # Combine all results
            results = {
                "success": True,
                "base_path": storage_result["base_path"],
                "timestamp": current_time.isoformat(),
                "files": {
                    **storage_result["upload_results"],
                    "report": report_result,
                    "sentiment": sentiment_result,
                },
                "analysis_metadata": {
                    "total_stocks": len(df),
                    "analysis_period": {
                        "start": df["data_start_date"].iloc[0].isoformat(),
                        "end": df["data_end_date"].iloc[0].isoformat(),
                    },
                    "metrics_generated": list(df.columns),
                },
            }

            logger.info(
                f"Analysis results successfully saved to storage bucket: {bucket_name}\n"
                f"Base path: {storage_result['base_path']}\n"
                f"Total files saved: {len(results['files'])}"
            )

            return results

        except Exception as e:
            logger.error(f"Error saving results to storage: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": dt.datetime.now().isoformat(),
            }

    def prepare_sentiment_data(self, df: pd.DataFrame) -> Dict:
        """
        Prepare enhanced sentiment analysis data for storage

        Args:
            df: Analysis DataFrame

        Returns:
            Dict containing organized sentiment data
        """
        sentiment_data = {
            "analysis_timestamp": dt.datetime.now(),
            "metadata": {
                "total_stocks_analyzed": int(len(df)),
                "average_sentiment": (
                    float(df["comment_sentiment_avg"].mean())
                    if "comment_sentiment_avg" in df.columns
                    else 0.0
                ),
                "total_comments_analyzed": (
                    int(df["num_comments"].sum()) if "num_comments" in df.columns else 0
                ),
                "sentiment_distribution": {
                    "bullish": (
                        float(df["bullish_comments_ratio"].mean())
                        if "bullish_comments_ratio" in df.columns
                        else 0.0
                    ),
                    "bearish": (
                        float(df["bearish_comments_ratio"].mean())
                        if "bearish_comments_ratio" in df.columns
                        else 0.0
                    ),
                    "neutral": (
                        float(
                            1
                            - df["bullish_comments_ratio"].mean()
                            - df["bearish_comments_ratio"].mean()
                        )
                        if all(
                            col in df.columns
                            for col in [
                                "bullish_comments_ratio",
                                "bearish_comments_ratio",
                            ]
                        )
                        else 0.0
                    ),
                },
            },
            "stocks": [
                {
                    "ticker": str(row["ticker"]),
                    "sentiment_metrics": {
                        "comment_sentiment_avg": float(
                            row.get("comment_sentiment_avg", 0)
                        ),
                        "bullish_ratio": float(row.get("bullish_comments_ratio", 0)),
                        "bearish_ratio": float(row.get("bearish_comments_ratio", 0)),
                        "sentiment_confidence": float(
                            row.get("sentiment_confidence", 0)
                        ),
                    },
                    "performance_metrics": {
                        "price_change_1w": float(row.get("price_change_1w", 0)),
                        "volume_change": float(row.get("volume_change", 0)),
                        "technical_score": float(row.get("technical_score", 0)),
                        "sentiment_score": float(row.get("sentiment_score", 0)),
                    },
                    "activity_metrics": {
                        "num_comments": int(row.get("num_comments", 0)),
                        "score": int(row.get("score", 0)),
                        "composite_score": float(row.get("composite_score", 0)),
                    },
                }
                for _, row in df.iterrows()
            ],
        }

        # Add correlations if possible
        try:
            sentiment_data["correlations"] = {
                "sentiment_price": (
                    float(df["comment_sentiment_avg"].corr(df["price_change_1w"]))
                    if all(
                        col in df.columns
                        for col in ["comment_sentiment_avg", "price_change_1w"]
                    )
                    else 0.0
                ),
                "sentiment_volume": (
                    float(df["comment_sentiment_avg"].corr(df["volume_change"]))
                    if all(
                        col in df.columns
                        for col in ["comment_sentiment_avg", "volume_change"]
                    )
                    else 0.0
                ),
                "sentiment_score": (
                    float(df["comment_sentiment_avg"].corr(df["composite_score"]))
                    if all(
                        col in df.columns
                        for col in ["comment_sentiment_avg", "composite_score"]
                    )
                    else 0.0
                ),
            }
        except Exception as e:
            logger.warning(f"Error calculating correlations: {str(e)}")
            sentiment_data["correlations"] = {}

        return json.loads(json.dumps(sentiment_data, cls=CustomJSONEncoder))

    # Modeling Functions
    def _calculate_prediction_features(self, hist: pd.DataFrame) -> Dict:
        """Calculate comprehensive features for prediction modeling"""
        try:
            features = {}
            close = hist['Close']
            volume = hist['Volume']
            high = hist['High']
            low = hist['Low']

            # Price Movement Features
            features.update({
                # Trend Strength Indicators
                'adx_14': round(ta.trend.adx(high, low, close, window=14), 2),
                'cci_20': round(ta.trend.cci(high, low, close, window=20), 2),
                'dpo_20': round(ta.trend.dpo(close, window=20), 2),
                'trix_30': round(ta.trend.trix(close, window=30), 2),
                
                # Momentum Features
                'rsi_14': round(ta.momentum.rsi(close, window=14), 2),
                'stoch_k': round(ta.momentum.stoch(high, low, close), 2),
                'stoch_d': round(ta.momentum.stoch_signal(high, low, close), 2),
                'williams_r': round(ta.momentum.williams_r(high, low, close), 2),
                'ultimate_osc': round(ta.momentum.ultimate_oscillator(high, low, close), 2),
                
                # Volatility Features
                'bbands_upper': round(ta.volatility.bollinger_hband(close), 2),
                'bbands_lower': round(ta.volatility.bollinger_lband(close), 2),
                'atr_14': round(ta.volatility.average_true_range(high, low, close), 4),
                'keltner_upper': self._calculate_keltner_channels(hist)['keltner_upper'],
                'keltner_lower': self._calculate_keltner_channels(hist)['keltner_lower'],
                
                # Volume Features
                'obv': round(ta.volume.on_balance_volume(close, volume), 2),
                'mfi_14': round(ta.volume.money_flow_index(high, low, close, volume), 2),
                'vwap': round(ta.volume.volume_weighted_average_price(high, low, close, volume), 2),
                
                # Price Pattern Features
                'price_gaps': self._calculate_price_gaps(hist),
                'swing_points': self._identify_swing_points(hist),
                'support_resistance': self._calculate_support_resistance(hist),
                
                # Custom Ratios and Indicators
                'close_to_high_ratio': round(close.iloc[-1] / high.rolling(20).max().iloc[-1], 4),
                'close_to_low_ratio': round(close.iloc[-1] / low.rolling(20).min().iloc[-1], 4),
                'volume_price_trend': round((volume * close).pct_change().mean(), 4),
                
                # Mean Reversion Features
                'zscore_20': round((close - close.rolling(20).mean()) / close.rolling(20).std(), 2).iloc[-1],
                'pct_from_sma50': round((close.iloc[-1] / close.rolling(50).mean().iloc[-1] - 1) * 100, 2),
                'pct_from_sma200': round((close.iloc[-1] / close.rolling(200).mean().iloc[-1] - 1) * 100, 2)
            })
            
            # Add Advanced Pattern Recognition
            features.update(self._identify_candlestick_patterns(hist))
            features.update(self._calculate_fibonacci_levels(hist))
            
            # Add Relative Strength Features
            features.update(self._calculate_relative_strength_features(hist))
            
            # Add Sentiment-Price Correlation Features
            features.update(self._calculate_sentiment_price_features(hist))
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating prediction features: {str(e)}")
            return {}

    def _identify_candlestick_patterns(self, hist: pd.DataFrame) -> Dict:
        """Identify candlestick patterns"""
        try:
            patterns = {}
            open_prices = hist['Open']
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
            
            # Single Candlestick Patterns
            patterns.update({
                'doji': ta.candles.doji(open_prices, high, low, close),
                'hammer': ta.candles.hammer(open_prices, high, low, close),
                'shooting_star': ta.candles.shooting_star(open_prices, high, low, close),
                'marubozu': ta.candles.marubozu(open_prices, high, low, close)
            })
            
            # Multiple Candlestick Patterns
            patterns.update({
                'engulfing': ta.candles.engulfing(open_prices, high, low, close),
                'morning_star': ta.candles.morning_star(open_prices, high, low, close),
                'evening_star': ta.candles.evening_star(open_prices, high, low, close),
                'three_white_soldiers': ta.candles.three_white_soldiers(open_prices, high, low, close)
            })
            
            return {k: int(v.iloc[-1]) for k, v in patterns.items()}
            
        except Exception as e:
            logger.error(f"Error identifying candlestick patterns: {str(e)}")
            return {}

    def _calculate_fibonacci_levels(self, hist: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement and extension levels"""
        try:
            high = hist['High'].iloc[-20:].max()
            low = hist['Low'].iloc[-20:].min()
            current = hist['Close'].iloc[-1]
            
            # Fibonacci ratios
            ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            # Calculate retracement levels
            range_size = high - low
            retracement_levels = {
                f'fib_ret_{int(ratio*1000)}': round(high - (ratio * range_size), 2)
                for ratio in ratios
            }
            
            # Calculate distance to each level
            level_distances = {
                f'dist_to_fib_{int(ratio*1000)}': round(
                    abs(current - retracement_levels[f'fib_ret_{int(ratio*1000)}']) / current * 100, 2
                )
                for ratio in ratios
            }
            
            return {**retracement_levels, **level_distances}
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {}

    def _calculate_relative_strength_features(self, hist: pd.DataFrame) -> Dict:
        """Calculate relative strength features compared to market and sector"""
        try:
            # Get SPY data for market comparison
            spy_data = self.market_data.get_stock_data("SPY", days=180)
            if not spy_data:
                return {}
                
            stock_returns = hist['Close'].pct_change()
            market_returns = spy_data['history']['Close'].pct_change()
            
            # Calculate relative strength features
            features = {
                'rs_1m': round(
                    (stock_returns.tail(21).mean() / market_returns.tail(21).mean()), 4
                ),
                'rs_3m': round(
                    (stock_returns.tail(63).mean() / market_returns.tail(63).mean()), 4
                ),
                'rs_momentum': round(
                    (stock_returns.tail(5).mean() / market_returns.tail(5).mean()), 4
                ),
                'beta': self._calculate_beta(stock_returns, market_returns),
                'correlation': round(stock_returns.corr(market_returns), 4)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating relative strength features: {str(e)}")
            return {}

    def _calculate_sentiment_price_features(self, hist: pd.DataFrame) -> Dict:
        """Calculate features relating sentiment to price movements"""
        try:
            features = {}
            
            # Get recent sentiment data
            sentiment_trends = self.db.get_sentiment_trends(
                ticker=hist.index.name,
                days=30,
                include_technicals=True
            )
            
            if not sentiment_trends.empty:
                # Calculate sentiment momentum
                sentiment_trends['sentiment_ma5'] = sentiment_trends['comment_sentiment_avg'].rolling(5).mean()
                sentiment_trends['sentiment_ma20'] = sentiment_trends['comment_sentiment_avg'].rolling(20).mean()
                
                # Calculate sentiment-price correlation features
                price_changes = hist['Close'].pct_change()
                sentiment_changes = sentiment_trends['comment_sentiment_avg'].pct_change()
                
                features.update({
                    'sentiment_price_corr': round(
                        sentiment_changes.corr(price_changes), 4
                    ),
                    'sentiment_momentum': round(
                        sentiment_trends['sentiment_ma5'].iloc[-1] 
                        - sentiment_trends['sentiment_ma20'].iloc[-1], 4
                    ),
                    'sentiment_volatility': round(
                        sentiment_changes.std() * np.sqrt(252), 4
                    ),
                    'bullish_ratio_ma5': round(
                        sentiment_trends['bullish_comments_ratio'].rolling(5).mean().iloc[-1], 4
                    ) if 'bullish_comments_ratio' in sentiment_trends.columns else 0
                })
                
            return features
            
        except Exception as e:
            logger.error(f"Error calculating sentiment-price features: {str(e)}")
            return {}

    def analyze_complete_metrics(self, ticker: str) -> Dict:
        """Perform complete analysis including prediction features"""
        try:
            metrics = {}
            
            # Get base metrics
            base_metrics = self.get_stock_metrics(ticker)
            if base_metrics:
                metrics.update(base_metrics)
                
            # Get sentiment metrics
            sentiment_metrics = self._analyze_sentiment_metrics(ticker)
            metrics.update(sentiment_metrics)
            
            # Calculate prediction features
            if 'history' in base_metrics:
                prediction_features = self._calculate_prediction_features(base_metrics['history'])
                metrics.update(prediction_features)
                
            # Calculate composite scores
            metrics.update(self._calculate_composite_scores(metrics))
            
            # Add prediction probabilities
            metrics.update(self._calculate_prediction_probabilities(metrics))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in complete metrics analysis: {str(e)}")
            return {}

    def get_recent_posts_by_ticker(self, ticker: str, days: int = 7, include_comments: bool = False) -> List[Dict]:
        """
        Get recent posts for a specific ticker
        
        Args:
            ticker: Stock symbol
            days: Number of days to look back
            include_comments: Whether to include post comments
            
        Returns:
            List of post dictionaries
        """
        try:
            cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
            
            # First get the posts that mention this ticker
            posts_query = (
                self.db.supabase.table("post_tickers")
                .select("*, reddit_posts(*)")
                .eq("ticker", ticker)
                .gte("mentioned_at", cutoff_date)
                .order("mentioned_at", desc=True)
                .limit(100)
                .execute()
            )
            
            if not posts_query.data:
                return []
                
            posts = []
            for post_ticker in posts_query.data:
                if not post_ticker.get('reddit_posts'):
                    continue
                    
                post_data = {
                    **post_ticker['reddit_posts'],
                    "ticker_mention": {
                        "ticker": post_ticker['ticker'],
                        "mentioned_at": post_ticker['mentioned_at']
                    }
                }
                
                # If comments are requested, get them separately
                if include_comments:
                    try:
                        comments_query = (
                            self.db.supabase.table("post_comments")
                            .select("*")
                            .eq("post_id", post_ticker['reddit_posts']['post_id'])
                            .execute()
                        )
                        post_data["comments"] = comments_query.data
                    except Exception as e:
                        logger.error(f"Error fetching comments for post {post_ticker['reddit_posts']['post_id']}: {str(e)}")
                        post_data["comments"] = []
                
                posts.append(post_data)
                
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving posts for {ticker}: {str(e)}")
            return []
    
    def _analyze_sentiment_metrics(self, ticker: str) -> Dict:
        """
        Calculate sentiment metrics for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary of sentiment metrics
        """
        try:
            metrics = {}
            
            # Get sentiment trends
            sentiment_data = self.db.get_sentiment_trends(
                ticker=ticker,
                days=30,
                include_technicals=True
            )
            
            if not sentiment_data.empty:
                # Calculate basic sentiment metrics
                sentiment_avg = pd.to_numeric(sentiment_data['comment_sentiment_avg'], errors='coerce')
                
                metrics.update({
                    'sentiment_score': float(sentiment_avg.mean()) if not sentiment_avg.isna().all() else 0,
                    'sentiment_std': float(sentiment_avg.std()) if not sentiment_avg.isna().all() else 0,
                    'sentiment_momentum': float(sentiment_avg.diff().mean()) if len(sentiment_avg) > 1 else 0,
                    'sentiment_volatility': float(sentiment_avg.pct_change().std() * np.sqrt(252)) if len(sentiment_avg) > 1 else 0
                })
                
                # Calculate bullish/bearish ratios if available
                if 'bullish_comments_ratio' in sentiment_data.columns:
                    bullish_ratio = pd.to_numeric(sentiment_data['bullish_comments_ratio'], errors='coerce')
                    metrics['avg_bullish_ratio'] = float(bullish_ratio.mean()) if not bullish_ratio.isna().all() else 0.5
                    
                if 'bearish_comments_ratio' in sentiment_data.columns:
                    bearish_ratio = pd.to_numeric(sentiment_data['bearish_comments_ratio'], errors='coerce')
                    metrics['avg_bearish_ratio'] = float(bearish_ratio.mean()) if not bearish_ratio.isna().all() else 0.5
            
            # Get recent posts
            recent_posts = self.get_recent_posts_by_ticker(ticker, days=7, include_comments=True)
            
            if recent_posts:
                recent_sentiments = []
                recent_bull_ratio = []
                recent_bear_ratio = []
                
                for post in recent_posts:
                    # Add post sentiment if available
                    if 'sentiment' in post and isinstance(post['sentiment'], dict):
                        sentiment_value = post['sentiment'].get('compound', 0)
                        if isinstance(sentiment_value, (int, float)):
                            recent_sentiments.append(sentiment_value)
                    
                    # Process comments if available
                    if 'comments' in post and post['comments']:
                        for comment in post['comments']:
                            if isinstance(comment.get('sentiment'), dict):
                                comment_sentiment = comment['sentiment'].get('compound', 0)
                                if isinstance(comment_sentiment, (int, float)):
                                    recent_sentiments.append(comment_sentiment)
                                    
                                    # Classify sentiment
                                    if comment_sentiment > 0.2:
                                        recent_bull_ratio.append(1)
                                        recent_bear_ratio.append(0)
                                    elif comment_sentiment < -0.2:
                                        recent_bull_ratio.append(0)
                                        recent_bear_ratio.append(1)
                                    else:
                                        recent_bull_ratio.append(0)
                                        recent_bear_ratio.append(0)
                
                if recent_sentiments:
                    metrics.update({
                        'recent_sentiment_avg': float(np.mean(recent_sentiments)),
                        'recent_sentiment_std': float(np.std(recent_sentiments)) if len(recent_sentiments) > 1 else 0,
                        'recent_bull_ratio': float(np.mean(recent_bull_ratio)) if recent_bull_ratio else 0.5,
                        'recent_bear_ratio': float(np.mean(recent_bear_ratio)) if recent_bear_ratio else 0.5,
                        'sentiment_confidence': len(recent_sentiments)
                    })
                    
                    # Calculate sentiment strength
                    sentiment_signals = [
                        metrics['recent_sentiment_avg'] > 0,  # Positive recent sentiment
                        metrics['recent_bull_ratio'] > 0.6,  # Strong bullish ratio
                        metrics['sentiment_momentum'] > 0 if 'sentiment_momentum' in metrics else False,
                        metrics['recent_bear_ratio'] < 0.3,  # Low bearish ratio
                    ]
                    
                    metrics['sentiment_strength'] = float(sum(sentiment_signals) / len(sentiment_signals)) * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment metrics for {ticker}: {str(e)}")
            return {}

    def _calculate_prediction_probabilities(self, metrics: Dict) -> Dict:
        """Calculate probabilities for different price movement scenarios"""
        try:
            probs = {}
            
            # Technical signals
            technical_bullish = sum([
                1 if metrics.get('rsi_14', 50) < 30 else 0,  # Oversold
                1 if metrics.get('williams_r', -50) < -80 else 0,  # Oversold
                1 if metrics.get('close_to_low_ratio', 1) < 1.02 else 0,  # Near support
                1 if metrics.get('macd_histogram', 0) > 0 else 0,  # MACD bullish
                1 if metrics.get('volume_price_trend', 0) > 0 else 0  # Volume-price bullish
            ]) / 5
            
            # Sentiment signals
            sentiment_bullish = sum([
                1 if metrics.get('comment_sentiment_avg', 0) > 0.2 else 0,
                1 if metrics.get('bullish_comments_ratio', 0) > 0.6 else 0,
                1 if metrics.get('sentiment_momentum', 0) > 0 else 0,
                1 if metrics.get('sentiment_price_corr', 0) > 0.3 else 0
            ]) / 4
            
            # Pattern signals
            pattern_bullish = sum([
                1 if metrics.get('hammer', 0) == 1 else 0,
                1 if metrics.get('morning_star', 0) == 1 else 0,
                1 if metrics.get('three_white_soldiers', 0) == 1 else 0,
                1 if metrics.get('engulfing', 0) == 1 else 0
            ]) / 4
            
            # Calculate weighted probabilities
            probs['bullish_probability'] = round(
                (technical_bullish * 0.5 + sentiment_bullish * 0.3 + pattern_bullish * 0.2) * 100, 2
            )
            
            # Add trend strength
            probs['trend_strength'] = round(
                (metrics.get('adx_14', 20) / 100) * 100, 2
            )
            
            # Add volatility prediction
            vol_signals = sum([
                1 if metrics.get('atr_14', 0) > metrics.get('atr_14_avg', 0) else 0,
                1 if abs(metrics.get('zscore_20', 0)) > 2 else 0,
                1 if metrics.get('sentiment_volatility', 0) > 0.2 else 0
            ]) / 3
            
            probs['volatility_probability'] = round(vol_signals * 100, 2)
            
            return probs
            
        except Exception as e:
            logger.error(f"Error calculating prediction probabilities: {str(e)}")
            return {}