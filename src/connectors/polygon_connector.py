
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient
from utils.logging_config import logger
from utils.cache import CacheManager
import time
from functools import wraps
import concurrent.futures

def retry_with_backoff(retries=3, backoff_factor=0.3):
    """Decorator for API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Retry attempt {attempt + 1} of {retries} failed: {str(e)}")
                    if attempt < retries - 1:
                        sleep_time = backoff_factor * (2 ** attempt)
                        time.sleep(sleep_time)
            raise last_exception
        return wrapper
    return decorator

class PolygonConnector:
    """Connector for Polygon.io market data API with improved connection handling"""
    
    def __init__(self, api_key: str, max_workers: int = 5, max_retries: int = 3):
        """
        Initialize Polygon connector
        
        Args:
            api_key: Polygon.io API key
            max_workers: Maximum number of concurrent connections
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.api_key = api_key
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize REST client"""
        self.client = RESTClient(api_key=self.api_key)
        
    def _get_client(self):
        """Get a client instance with connection handling"""
        try:
            if not hasattr(self, 'client') or self.client is None:
                self._initialize_client()
            return self.client
        except Exception as e:
            logger.error(f"Error initializing Polygon client: {str(e)}")
            raise

    @retry_with_backoff()
    def get_stock_data(self, ticker: str, days: int = 180) -> Optional[Dict]:
        """Get stock data including price history and details with improved connection handling"""
        try:
            client = self._get_client()
            
            # Introduce small delay between consecutive API calls
            time.sleep(0.1)
            
            # Get stock details
            details = client.get_ticker_details(ticker)
            
            # Small delay between API calls
            time.sleep(0.1)
            
            # Get price history
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            aggs = client.get_aggs(
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
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            return None

    def __del__(self):
        """Cleanup resources on deletion"""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error cleaning up PolygonConnector: {str(e)}")

    @retry_with_backoff()
    def get_market_status(self) -> Dict:
        """Get current market status with retry logic"""
        try:
            client = self._get_client()
            time.sleep(0.1)  # Small delay to avoid rate limiting
            status = client.get_market_status()
            return {
                'market_status': status.market,
                'server_time': status.server_time,
                'exchanges': status.exchanges
            }
        except Exception as e:
            logger.error(f"Error getting market status: {str(e)}")
            return {}

    @retry_with_backoff()
    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid"""
        try:
            client = self._get_client()
            time.sleep(0.1)  # Small delay to avoid rate limiting
            details = client.get_ticker_details(ticker)
            return details is not None and details.active
        except Exception:
            return False
    
    @retry_with_backoff()
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
            
    @retry_with_backoff()
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
    
    @retry_with_backoff()
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
    
    @retry_with_backoff()
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

    @retry_with_backoff()
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

    @retry_with_backoff()
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