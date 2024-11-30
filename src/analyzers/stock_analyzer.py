import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional
from utils.logging_config import logger


class StockAnalyzer:
    def __init__(self):
        self.stock_data_cache = {}

    def get_stock_metrics(self, ticker: str) -> Optional[Dict]:
        """Fetch stock metrics with improved error handling"""
        try:
            if ticker in self.stock_data_cache:
                return self.stock_data_cache[ticker]

            stock = yf.Ticker(ticker)

            # Get historical data
            hist = stock.history(period="1mo")

            if hist.empty:
                logger.error(f"{ticker}: No price data found (period=1mo)")
                return None

            if len(hist) < 10:
                logger.error(
                    f"{ticker}: Insufficient price history (only {len(hist)} days available)"
                )
                return None

            try:
                current_price = hist["Close"].iloc[-1]
                price_2w_ago = hist["Close"].iloc[-10]
                avg_volume = hist["Volume"].mean()

                # Verify we have valid numerical data
                if not (
                    pd.notnull(current_price)
                    and pd.notnull(price_2w_ago)
                    and pd.notnull(avg_volume)
                ):
                    logger.error(f"{ticker}: Invalid price or volume data")
                    return None

                metrics = {
                    "current_price": round(current_price, 2),
                    "price_change_2w": round(
                        ((current_price - price_2w_ago) / price_2w_ago) * 100, 2
                    ),
                    "avg_volume": int(avg_volume),
                    "volume_change": round(
                        ((hist["Volume"].iloc[-1] - avg_volume) / avg_volume) * 100, 2
                    ),
                }

                # Add technical indicators with error handling
                try:
                    metrics["sma_20"] = round(
                        hist["Close"].rolling(window=20).mean().iloc[-1], 2
                    )
                    metrics["rsi"] = round(self._calculate_rsi(hist["Close"]), 2)
                    metrics["volatility"] = round(
                        hist["Close"].pct_change().std() * np.sqrt(252) * 100, 2
                    )
                except Exception as e:
                    logger.warning(
                        f"{ticker}: Could not calculate technical indicators: {str(e)}"
                    )
                    metrics.update({"sma_20": None, "rsi": None, "volatility": None})

                # Add additional info with error handling
                try:
                    info = stock.info
                    metrics["market_cap"] = info.get("marketCap")
                    metrics["pe_ratio"] = info.get("forwardPE")
                except Exception as e:
                    logger.warning(
                        f"{ticker}: Could not fetch additional info: {str(e)}"
                    )
                    metrics.update({"market_cap": None, "pe_ratio": None})

                self.stock_data_cache[ticker] = metrics
                return metrics

            except Exception as e:
                logger.error(f"{ticker}: Error calculating metrics: {str(e)}")
                return None

        except Exception as e:
            logger.error(
                f"{ticker}: possibly delisted; no price data found (period=1mo)"
            )
            return None

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))
