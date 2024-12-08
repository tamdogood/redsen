# src/connectors/polygon_connector.py

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient
from utils.logging_config import logger
from utils.cache import CacheManager

class PolygonConnector:
    """Connector for Polygon.io market data API"""
    
    def __init__(self, api_key: str):
        """
        Initialize Polygon connector
        
        Args:
            api_key: Polygon.io API key
        """
        self.client = RESTClient(api_key)
        # self.cache = CacheManager()
        
    def get_stock_data(self, ticker: str, days: int = 180) -> Optional[Dict]:
        """
        Get stock data including price history and details
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data
            
        Returns:
            Dict containing stock data and metadata
        """
        try:
            # cache_key = f"stock_data_{ticker}_{days}"
            # cached_data = self.cache.get(cache_key)
            # if cached_data:
                # return cached_data
                
            # Get stock details
            details = self.client.get_ticker_details(ticker)
            
            # Get price history
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            
            # Convert to DataFrame
            hist_data = pd.DataFrame([{
                'Date': datetime.fromtimestamp(a.timestamp/1000),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume,
                'VWAP': a.vwap,
                'Transactions': a.transactions
            } for a in aggs])
            
            if hist_data.empty:
                return None
                
            hist_data.set_index('Date', inplace=True)
            
            result = {
                'history': hist_data,
                'info': {
                    'name': details.name,
                    'market_cap': details.market_cap,
                    'share_class_shares_outstanding': details.share_class_shares_outstanding,
                    'primary_exchange': details.primary_exchange,
                    'type': details.type,
                    'currency_name': details.currency_name,
                    'description': details.description,
                    'sector': details.sic_description,
                    'market': details.market,
                    'locale': details.locale
                }
            }
            
            # self.cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour
            return result
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            return None
            
    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            status = self.client.get_market_status()
            return {
                'market_status': status.market,
                'server_time': status.server_time,
                'exchanges': status.exchanges
            }
        except Exception as e:
            logger.error(f"Error getting market status: {str(e)}")
            return {}
            
    def get_ticker_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get recent news for a ticker"""
        try:
            news = self.client.get_ticker_news(ticker, limit=limit)
            return [{
                'title': n.title,
                'author': n.author,
                'published_utc': n.published_utc,
                'article_url': n.article_url,
                'tickers': n.tickers,
                'description': n.description
            } for n in news]
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {str(e)}")
            return []
            
    def get_daily_open_close(self, ticker: str, date: str) -> Optional[Dict]:
        """Get daily open/close data for a specific date"""
        try:
            data = self.client.get_daily_open_close(ticker, date)
            return {
                'open': data.open,
                'close': data.close,
                'high': data.high,
                'low': data.low,
                'volume': data.volume,
                'after_hours': data.after_hours,
                'pre_market': data.pre_market
            }
        except Exception as e:
            logger.error(f"Error getting daily data for {ticker}: {str(e)}")
            return None
            
    def get_ticker_types(self, ticker: str) -> Dict:
        """Get supported ticker types and exchanges"""
        try:
            types = self.client.get_ticker_types()
            return {
                'types': types.types,
                'exchanges': types.exchanges
            }
        except Exception as e:
            logger.error(f"Error getting ticker types: {str(e)}")
            return {}
            
    def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get detailed information about a ticker"""
        try:
            details = self.client.get_ticker_details(ticker)
            return {
                'name': details.name,
                'market_cap': details.market_cap,
                'shares_outstanding': details.share_class_shares_outstanding,
                'exchange': details.primary_exchange,
                'type': details.type,
                'currency': details.currency_name,
                'description': details.description,
                'sector': details.sic_description,
                'industry': details.standard_industry_classification,
                'market': details.market,
                'locale': details.locale,
                'primary_exchange': details.primary_exchange,
                'active': details.active
            }
        except Exception as e:
            logger.error(f"Error getting ticker details for {ticker}: {str(e)}")
            return None
            
    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid"""
        try:
            details = self.client.get_ticker_details(ticker)
            return details is not None and details.active
        except Exception:
            return False
            
    def get_previous_close(self, ticker: str) -> Optional[float]:
        """Get previous day's closing price"""
        try:
            prev_close = self.client.get_previous_close(ticker)
            if prev_close and prev_close.close:
                return prev_close.close
            return None
        except Exception as e:
            logger.error(f"Error getting previous close for {ticker}: {str(e)}")
            return None
            
    def get_stock_splits(self, ticker: str) -> List[Dict]:
        """Get historical stock splits"""
        try:
            splits = self.client.get_splits(ticker)
            return [{
                'execution_date': s.execution_date,
                'split_from': s.split_from,
                'split_to': s.split_to
            } for s in splits]
        except Exception as e:
            logger.error(f"Error getting splits for {ticker}: {str(e)}")
            return []

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Get aggregated data for a ticker
        
        Args:
            ticker: Stock symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with aggregated data
        """
        try:
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date
            )
            
            if not aggs:
                return None
                
            df = pd.DataFrame([{
                'timestamp': datetime.fromtimestamp(a.timestamp/1000),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume,
                'vwap': a.vwap,
                'transactions': a.transactions
            } for a in aggs])
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error getting aggregates for {ticker}: {str(e)}")
            return None
        
    def get_pe_ratio(self, ticker: str) -> Optional[float]:
        """
        Calculate P/E ratio using Polygon data
        
        Args:
            ticker: Stock symbol
            
        Returns:
            P/E ratio if available, None otherwise
        """
        try:
            # Get latest stock price
            latest_price = self.client.get_daily_open_close_agg(
                ticker=ticker,
                date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if not latest_price:
                return None
                
            current_price = latest_price.close
            
            # Get latest earnings data
            financials = self.client.get_ticker_financials(
                ticker=ticker,
                limit=1,
                type='Y'  # Annual financials
            )
            
            if not financials:
                return None
                
            # Get EPS from the financials
            latest_financials = financials[0]
            
            # Extract financial metrics
            income_statement = latest_financials.financials.get('income_statement', {})
            basic_earnings_per_share = income_statement.get('basic_earnings_per_share', None)
            
            if not basic_earnings_per_share:
                # Try alternative fields if basic EPS is not available
                net_income = income_statement.get('net_income_loss', 0)
                shares_outstanding = latest_financials.financials.get('shares_outstanding', 0)
                
                if shares_outstanding > 0:
                    eps = net_income / shares_outstanding
                else:
                    return None
            else:
                eps = basic_earnings_per_share
                
            # Calculate P/E ratio
            if eps <= 0:  # Avoid negative or zero P/E ratios
                return None
                
            pe_ratio = current_price / eps
            
            # Return None if P/E ratio is unreasonably high
            if pe_ratio > 1000:
                return None
                
            return round(pe_ratio, 2)
            
        except Exception as e:
            logger.error(f"Error calculating P/E ratio for {ticker}: {str(e)}")
            return None

    def get_financial_ratios(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Get comprehensive financial ratios for a stock
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary of financial ratios
        """
        try:
            ratios = {}
            
            # Get financials
            financials = self.client.get_ticker_financials(
                ticker=ticker,
                limit=1,
                type='Y'
            )
            
            if not financials or not financials[0]:
                return {}
                
            latest = financials[0]
            balance_sheet = latest.financials.get('balance_sheet', {})
            income_statement = latest.financials.get('income_statement', {})
            cash_flow = latest.financials.get('cash_flow_statement', {})
            
            # Calculate P/E Ratio
            ratios['pe_ratio'] = self.get_pe_ratio(ticker)
            
            # Price to Book Ratio
            total_assets = balance_sheet.get('total_assets', 0)
            total_liabilities = balance_sheet.get('total_liabilities', 0)
            book_value = total_assets - total_liabilities
            
            if book_value > 0:
                market_cap = self.get_market_cap(ticker)
                if market_cap:
                    ratios['price_to_book'] = round(market_cap / book_value, 2)
                    
            # Debt to Equity
            total_equity = balance_sheet.get('total_equity', 0)
            if total_equity > 0:
                ratios['debt_to_equity'] = round(total_liabilities / total_equity, 2)
                
            # Current Ratio
            current_assets = balance_sheet.get('current_assets', 0)
            current_liabilities = balance_sheet.get('current_liabilities', 0)
            if current_liabilities > 0:
                ratios['current_ratio'] = round(current_assets / current_liabilities, 2)
                
            # Profit Margin
            revenue = income_statement.get('revenues', 0)
            net_income = income_statement.get('net_income_loss', 0)
            if revenue > 0:
                ratios['profit_margin'] = round(net_income / revenue, 4)
                
            # Return on Equity (ROE)
            if total_equity > 0:
                ratios['roe'] = round(net_income / total_equity, 4)
                
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios for {ticker}: {str(e)}")
            return {}

    def get_market_cap(self, ticker: str) -> Optional[float]:
        """
        Get current market cap for a stock
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Market cap if available, None otherwise
        """
        try:
            # Get shares outstanding from company details
            details = self.client.get_ticker_details(ticker)
            if not details:
                return None
                
            shares_outstanding = details.share_class_shares_outstanding
            
            # Get current price
            latest_price = self.client.get_daily_open_close_agg(
                ticker=ticker,
                date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if not latest_price:
                return None
                
            current_price = latest_price.close
            
            # Calculate market cap
            market_cap = shares_outstanding * current_price
            return market_cap
            
        except Exception as e:
            logger.error(f"Error calculating market cap for {ticker}: {str(e)}")
            return None