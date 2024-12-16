import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, Optional, Union, List, Tuple
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
import ta
import re
from utils.logging_config import logger
from connectors.polygon_connector import PolygonConnector
import os

class TechnicalAnalyzer:
    def __init__(self, llm_connector):
        """Initialize technical analyzer with data connector"""
        self.market_data = PolygonConnector(os.getenv("POLYGON_API_KEY", ""))
        self.market_indicators_cache = {}
        self.sector_performance_cache = {}
        self.openai_client = llm_connector
        
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
            stock_data = self.market_data.get_stock_data(ticker, days=60)
            if not stock_data or stock_data['history'].empty:
                logger.warning(f"No data available for {ticker}")
                return None
                
            hist = stock_data['history']
            
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
                momentum_metrics = self._calculate_momentum_indicators(hist)
                metrics.update(momentum_metrics)
                
            # Get market context
            market_metrics = self._get_market_context(ticker)
            
            # fundamental_metrics = self._get_fundamental_metrics(ticker)
            fundamental_metrics = self.market_data.get_financial_ratios(ticker)
            
            # Update all available metrics
            metrics.update(price_metrics)
            metrics.update(volume_metrics)
            metrics.update(market_metrics)
            metrics.update(fundamental_metrics)
            
            # Add fundamental metrics if available
            # fundamental_metrics = {
            #     "market_cap": info.get("market_cap"),
            #     "shares_outstanding": info.get("share_class_shares_outstanding"),
            #     "sector": info.get("sector"),
            #     "industry": info.get("industry"),
            #     "description": info.get("description"),
            #     "exchange": info.get("primary_exchange"),
            #     "type": info.get("type")
            # }
            # metrics.update(fundamental_metrics)
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

    def _calculate_relative_strength(self, sector_etf: str) -> float:
        """
        Calculate relative strength of sector vs market
        
        Args:
            sector_etf: Sector ETF symbol
            
        Returns:
            float: Relative strength value
        """
        try:
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=30)
            
            # Get sector and market data
            sector_data = self.market_data.get_aggregates(
                ticker=sector_etf,
                multiplier=1,
                timespan="day",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            market_data = self.market_data.get_aggregates(
                ticker="SPY",
                multiplier=1,
                timespan="day",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            if sector_data is None or market_data is None:
                return 1.0
                
            # Calculate relative performance
            sector_return = (sector_data['close'].iloc[-1] / sector_data['close'].iloc[0]) - 1
            market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1
            
            if market_return == 0:
                return 1.0
                
            relative_strength = (1 + sector_return) / (1 + market_return)
            
            return round(relative_strength, 3)
            
        except Exception as e:
            logger.error(f"Error calculating relative strength: {str(e)}")
            return 1.0
    
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

    def _get_sector_performance(self, ticker: str) -> Dict:
        """
        Get sector and industry performance metrics using Polygon API
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict containing sector performance metrics
        """
        try:
            # Get stock details from Polygon
            ticker_details = self.market_data.client.get_ticker_details(ticker)
            if not ticker_details:
                return {}
                
            # Get sector info from SIC description and sector data
            sector = ""
            industry = ""
            
            if hasattr(ticker_details, 'sic_description'):
                sector = ticker_details.sic_description or ""
            
            # Alternative field for industry classification
            if hasattr(ticker_details, 'standard_industrial_classification'):
                industry = ticker_details.standard_industrial_classification.get('industry_title', "")
            
            # Map SIC sectors to ETFs
            sector_etfs = {
                "Technology": "XLK",
                "Finance": "XLF",
                "Healthcare": "XLV",
                "Consumer": "XLY",
                "Consumer Products": "XLP",
                "Energy": "XLE",
                "Industrial": "XLI",
                "Materials": "XLB",
                "Real Estate": "XLRE",
                "Utilities": "XLU",
                "Communications": "XLC",
                "Oil": "XLE",  # Add common variations
                "Mining": "XLB",
                "Software": "XLK",
                "Banking": "XLF",
                "Insurance": "XLF",
                "Pharmaceutical": "XLV",
                "Transportation": "XLI",
            }
            
            # Find the best matching sector ETF
            sector_etf = "SPY"  # Default to SPY
            for key, etf in sector_etfs.items():
                if key.lower() in sector.lower() or key.lower() in industry.lower():
                    sector_etf = etf
                    break
            
            # Rest of the function remains the same...
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=30)
            
            etf_aggs = self.market_data.get_aggregates(
                ticker=sector_etf,
                multiplier=1,
                timespan="day",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            if etf_aggs is None or etf_aggs.empty:
                return {
                    "sector": sector,
                    "industry": industry,
                    "sector_performance_1m": 0,
                    "sector_relative_strength": 0,
                    "sector_volatility": 0
                }
            
            # Calculate sector performance metrics
            sector_metrics = {
                "sector": sector,
                "industry": industry,
                "sector_performance_1m": round(
                    ((etf_aggs['close'].iloc[-1] / etf_aggs['close'].iloc[0]) - 1) * 100,
                    2
                ),
                "sector_relative_strength": self._calculate_relative_strength(sector_etf),
                "sector_volatility": round(
                    etf_aggs['close'].pct_change().std() * np.sqrt(252) * 100,
                    2
                )
            }
            
            # Add additional sector metrics
            try:
                # Calculate sector momentum
                sector_metrics["sector_momentum"] = round(
                    etf_aggs['close'].pct_change(5).mean() * 100,  # 5-day momentum
                    2
                )
                
                # Calculate sector volume trend
                avg_volume = etf_aggs['volume'].mean()
                recent_volume = etf_aggs['volume'].tail(5).mean()
                sector_metrics["sector_volume_trend"] = round(
                    (recent_volume / avg_volume - 1) * 100,
                    2
                )
                
                # Calculate sector volatility trend
                recent_volatility = etf_aggs['close'].pct_change().tail(5).std() * np.sqrt(252) * 100
                sector_metrics["sector_volatility_trend"] = round(
                    recent_volatility - sector_metrics["sector_volatility"],
                    2
                )
                
                # Add sector moving averages
                sector_metrics.update({
                    "sector_sma20": round(
                        etf_aggs['close'].rolling(20).mean().iloc[-1],
                        2
                    ),
                    "sector_sma50": round(
                        etf_aggs['close'].rolling(50).mean().iloc[-1],
                        2
                    ) if len(etf_aggs) >= 50 else None
                })
                
                # Calculate sector strength score (0-100)
                strength_factors = [
                    sector_metrics["sector_performance_1m"] > 0,  # Positive performance
                    sector_metrics["sector_momentum"] > 0,  # Positive momentum
                    sector_metrics["sector_volume_trend"] > 0,  # Increasing volume
                    etf_aggs['close'].iloc[-1] > sector_metrics["sector_sma20"],  # Above SMA20
                    sector_metrics["sector_relative_strength"] > 1  # Strong vs market
                ]
                
                sector_metrics["sector_strength_score"] = round(
                    (sum(1 for x in strength_factors if x) / len(strength_factors)) * 100,
                    2
                )
                
            except Exception as e:
                logger.debug(f"Error calculating additional sector metrics: {str(e)}")
            
            return sector_metrics
            
        except Exception as e:
            logger.error(f"Error getting sector performance for {ticker}: {str(e)}")
            return {
                "sector": "",
                "industry": "",
                "sector_performance_1m": 0,
                "sector_relative_strength": 0,
                "sector_volatility": 0
            }
        
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
                "market_sentiment": self._calculate_market_sentiment(implied_volatility)
            }

            # self.market_indicators_cache[cache_key] = market_metrics
            return market_metrics

        except Exception as e:
            logger.error(f"Error getting market indicators: {str(e)}")
            return {}

    def _calculate_market_sentiment(self, volatility: float) -> str:
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
            
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _get_fundamental_metrics(self, ticker: str) -> Dict:
        """
        Get fundamental metrics for a stock using Polygon Financials API
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary containing fundamental metrics
        """
        try:
            # Get financials data using the correct endpoint
            financials = self.market_data.client.vx.list_stock_financials(
                ticker=ticker,
                timeframe="quarterly",
                include_sources=True,
                order="desc",
                limit=1
            )
            # print(financials)
            if not financials:
                logger.warning(f"No financial data available for {ticker}")
                return {}
                
            # latest = financials.results[0]
            
            # Extract financial statement components
            # income_stmt = latest.financials.income_statement
            # balance_sheet = latest.financials.balance_sheet
            # cash_flow = latest.financials.cash_flow_statement
            
            # # Calculate fundamental metrics
            # metrics = {
            #     # Profitability metrics
            #     "gross_profit_margin": self._calculate_ratio(
            #         getattr(income_stmt, 'gross_profit', None),
            #         getattr(income_stmt, 'revenues', None)
            #     ),
            #     "operating_margin": self._calculate_ratio(
            #         getattr(income_stmt, 'operating_income_loss', None),
            #         getattr(income_stmt, 'revenues', None)
            #     ),
            #     "net_profit_margin": self._calculate_ratio(
            #         getattr(income_stmt, 'net_income_loss', None),
            #         getattr(income_stmt, 'revenues', None)
            #     ),
                
            #     # Efficiency metrics
            #     "asset_turnover": self._calculate_ratio(
            #         getattr(income_stmt, 'revenues', None),
            #         getattr(balance_sheet, 'total_assets', None)
            #     ),
            #     "inventory_turnover": self._calculate_ratio(
            #         getattr(income_stmt, 'cost_of_revenue', None),
            #         getattr(balance_sheet, 'inventory', None)
            #     ),
                
            #     # Liquidity metrics
            #     "current_ratio": self._calculate_ratio(
            #         getattr(balance_sheet, 'current_assets', None),
            #         getattr(balance_sheet, 'current_liabilities', None)
            #     ),
            #     "quick_ratio": self._calculate_ratio(
            #         (getattr(balance_sheet, 'current_assets', 0) or 0) - 
            #         (getattr(balance_sheet, 'inventory', 0) or 0),
            #         getattr(balance_sheet, 'current_liabilities', None)
            #     ),
                
            #     # Leverage metrics
            #     "debt_to_equity": self._calculate_ratio(
            #         getattr(balance_sheet, 'total_liabilities', None),
            #         getattr(balance_sheet, 'total_shareholders_equity', None)
            #     ),
            #     "debt_to_assets": self._calculate_ratio(
            #         getattr(balance_sheet, 'total_liabilities', None),
            #         getattr(balance_sheet, 'total_assets', None)
            #     ),
                
            #     # Cash flow metrics
            #     "operating_cash_flow_ratio": self._calculate_ratio(
            #         getattr(cash_flow, 'net_cash_flow_from_operating_activities', None),
            #         getattr(balance_sheet, 'current_liabilities', None)
            #     ),
            #     "free_cash_flow": (
            #         getattr(cash_flow, 'net_cash_flow_from_operating_activities', 0) or 0) - 
            #         (getattr(cash_flow, 'capital_expenditure', 0) or 0),
                
            #     # Growth metrics
            #     "revenue": getattr(income_stmt, 'revenues', None),
            #     "net_income": getattr(income_stmt, 'net_income_loss', None),
            #     "total_assets": getattr(balance_sheet, 'total_assets', None),
            #     "total_liabilities": getattr(balance_sheet, 'total_liabilities', None),
                
            #     # Per share metrics
            #     "book_value_per_share": self._calculate_ratio(
            #         getattr(balance_sheet, 'total_shareholders_equity', None),
            #         getattr(latest, 'shares_outstanding', None)
            #     ),
            #     "earnings_per_share": getattr(income_stmt, 'basic_earnings_per_share', None),
                
            #     # Working capital
            #     "working_capital": (getattr(balance_sheet, 'current_assets', 0) or 0) - 
            #                     (getattr(balance_sheet, 'current_liabilities', 0) or 0),
                
            #     # Filing info
            #     "fiscal_period": latest.fiscal_period,
            #     "fiscal_year": latest.fiscal_year,
            #     "filing_date": latest.filing_date,
            #     "start_date": latest.start_date,
            #     "end_date": latest.end_date
            # }
            
            # # Remove None values and convert to float where needed
            # return {k: float(v) if isinstance(v, (int, float)) else v 
            #         for k, v in metrics.items() 
            #         if v is not None}
            
        except Exception as e:
            logger.error(f"Error getting fundamental metrics for {ticker}: {str(e)}")
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