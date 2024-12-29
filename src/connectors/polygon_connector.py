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
                    logger.warning(
                        f"Retry attempt {attempt + 1} of {retries} failed: {str(e)}"
                    )
                    if attempt < retries - 1:
                        sleep_time = backoff_factor * (2**attempt)
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
            if not hasattr(self, "client") or self.client is None:
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
                to=end_date.strftime("%Y-%m-%d"),
            )

            # Convert to DataFrame
            hist_data = pd.DataFrame(
                [
                    {
                        "Date": datetime.fromtimestamp(a.timestamp / 1000),
                        "Open": a.open,
                        "High": a.high,
                        "Low": a.low,
                        "Close": a.close,
                        "Volume": a.volume,
                        "VWAP": a.vwap,
                        "Transactions": a.transactions,
                    }
                    for a in aggs
                ]
            )

            if hist_data.empty:
                return None

            hist_data.set_index("Date", inplace=True)

            result = {
                "history": hist_data,
                "info": {
                    "name": details.name,
                    "market_cap": details.market_cap,
                    "share_class_shares_outstanding": details.share_class_shares_outstanding,
                    "primary_exchange": details.primary_exchange,
                    "type": details.type,
                    "currency_name": details.currency_name,
                    "description": details.description,
                    "sector": details.sic_description,
                    "market": details.market,
                    "locale": details.locale,
                },
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
                "market_status": status.market,
                "server_time": status.server_time,
                "exchanges": status.exchanges,
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
            return [
                {
                    "execution_date": s.execution_date,
                    "split_from": s.split_from,
                    "split_to": s.split_to,
                }
                for s in splits
            ]
        except Exception as e:
            logger.error(f"Error getting splits for {ticker}: {str(e)}")
            return []

    @retry_with_backoff()
    def get_aggregates(
        self, ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str
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
                to=to_date,
            )

            if not aggs:
                return None

            df = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.fromtimestamp(a.timestamp / 1000),
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                        "vwap": a.vwap,
                        "transactions": a.transactions,
                    }
                    for a in aggs
                ]
            )

            df.set_index("timestamp", inplace=True)
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
                ticker=ticker, date=datetime.now().strftime("%Y-%m-%d")
            )

            if not latest_price:
                return None

            current_price = latest_price.close

            # Get latest earnings data
            financials = self.client.get_ticker_financials(
                ticker=ticker, limit=1, type="Y"  # Annual financials
            )

            if not financials:
                return None

            # Get EPS from the financials
            latest_financials = financials[0]

            # Extract financial metrics
            income_statement = latest_financials.financials.get("income_statement", {})
            basic_earnings_per_share = income_statement.get(
                "basic_earnings_per_share", None
            )

            if not basic_earnings_per_share:
                # Try alternative fields if basic EPS is not available
                net_income = income_statement.get("net_income_loss", 0)
                shares_outstanding = latest_financials.financials.get(
                    "shares_outstanding", 0
                )

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

    def _parse_datapoint(self, datapoint) -> Dict:
        """Helper function to parse DataPoint objects"""
        try:
            if datapoint is None:
                return None

            # If it's already a dict, return relevant fields
            if isinstance(datapoint, dict):
                return {
                    "value": datapoint.get("value"),
                    "unit": datapoint.get("unit"),
                    "label": datapoint.get("label"),
                }

            # If it's a named tuple or object, get attributes
            return {
                "value": getattr(datapoint, "value", None),
                "unit": getattr(datapoint, "unit", None),
                "label": getattr(datapoint, "label", None),
            }
        except Exception as e:
            logger.error(f"Error parsing datapoint: {str(e)}")
            return None

    def parse_polygon_financials(self, financial_data) -> Dict:
        """
        Parse Polygon financial data response into a structured dictionary

        Args:
            financial_data: Raw Polygon financial data response

        Returns:
            Dictionary containing parsed financial data
        """
        try:
            logger.debug(f"Financial data type: {type(financial_data)}")
            logger.debug(f"Financial data attributes: {dir(financial_data)}")

            parsed = {
                "metadata": {
                    "cik": getattr(financial_data, "cik", None),
                    "company_name": getattr(financial_data, "company_name", None),
                    "start_date": getattr(financial_data, "start_date", None),
                    "end_date": getattr(financial_data, "end_date", None),
                    "filing_date": getattr(financial_data, "filing_date", None),
                    "fiscal_year": getattr(financial_data, "fiscal_year", None),
                    "fiscal_period": getattr(financial_data, "fiscal_period", None),
                    "source_filing_url": getattr(
                        financial_data, "source_filing_url", None
                    ),
                    "source_filing_file_url": getattr(
                        financial_data, "source_filing_file_url", None
                    ),
                },
                "financials": {
                    "balance_sheet": {},
                    "income_statement": {},
                    "cash_flow_statement": {},
                    "comprehensive_income": {},
                },
            }

            # Get the financials object
            financials = getattr(financial_data, "financials", None)
            if financials:
                logger.debug(f"Financials type: {type(financials)}")
                logger.debug(f"Financials attributes: {dir(financials)}")

                # Parse balance sheet
                balance_sheet = getattr(financials, "balance_sheet", {})
                if balance_sheet:
                    logger.debug(f"Balance sheet type: {type(balance_sheet)}")
                    for key, value in balance_sheet.items():
                        logger.debug(
                            f"Processing balance sheet item: {key}, type: {type(value)}"
                        )
                        parsed["financials"]["balance_sheet"][key] = (
                            self._parse_datapoint(value)
                        )

                # Parse income statement
                income_stmt = getattr(financials, "income_statement", None)
                if income_stmt:
                    logger.debug(f"Income statement type: {type(income_stmt)}")
                    logger.debug(f"Income statement attributes: {dir(income_stmt)}")
                    income_dict = {}

                    for field in [
                        "basic_earnings_per_share",
                        "cost_of_revenue",
                        "gross_profit",
                        "operating_expenses",
                        "revenues",
                    ]:
                        value = getattr(income_stmt, field, None)
                        if value is not None:
                            logger.debug(
                                f"Processing income statement item: {field}, type: {type(value)}"
                            )
                            income_dict[field] = self._parse_datapoint(value)
                    parsed["financials"]["income_statement"] = income_dict

                # Parse cash flow statement
                cash_flow = getattr(financials, "cash_flow_statement", None)
                if cash_flow:
                    logger.debug(f"Cash flow type: {type(cash_flow)}")
                    logger.debug(f"Cash flow attributes: {dir(cash_flow)}")
                    cash_flow_dict = {}

                    for field in [
                        "exchange_gains_losses",
                        "net_cash_flow",
                        "net_cash_flow_from_financing_activities",
                    ]:
                        value = getattr(cash_flow, field, None)
                        if value is not None:
                            logger.debug(
                                f"Processing cash flow item: {field}, type: {type(value)}"
                            )
                            cash_flow_dict[field] = self._parse_datapoint(value)
                    parsed["financials"]["cash_flow_statement"] = cash_flow_dict

                # Parse comprehensive income
                comp_income = getattr(financials, "comprehensive_income", None)
                if comp_income:
                    logger.debug(f"Comprehensive income type: {type(comp_income)}")
                    logger.debug(f"Comprehensive income attributes: {dir(comp_income)}")
                    comp_income_dict = {}

                    for field in [
                        "comprehensive_income_loss",
                        "comprehensive_income_loss_attributable_to_parent",
                        "other_comprehensive_income_loss",
                    ]:
                        value = getattr(comp_income, field, None)
                        if value is not None:
                            logger.debug(
                                f"Processing comprehensive income item: {field}, type: {type(value)}"
                            )
                            comp_income_dict[field] = self._parse_datapoint(value)
                    parsed["financials"]["comprehensive_income"] = comp_income_dict

            return parsed

        except Exception as e:
            logger.error(f"Error parsing financial data: {str(e)}")
            return parsed

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
            # Get financials and handle generator
            financials_gen = self.client.vx.list_stock_financials(
                ticker=ticker, limit=1, filing_date_gte="2024-01-01"
            )

            try:
                latest = next(financials_gen)
            except StopIteration:
                logger.warning(f"No financial data found for {ticker}")
                return {}

            if not latest:
                return {}

            # Parse the polygon response into a clean dictionary
            parsed = self.parse_polygon_financials(latest)
            if not parsed:
                return {}

            # Extract balance sheet items
            balance_sheet = parsed.get("financials", {}).get("balance_sheet", {})
            income_stmt = parsed.get("financials", {}).get("income_statement", {})
            cash_flow = parsed.get("financials", {}).get("cash_flow_statement", {})

            # Helper function to safely get numeric values
            def get_value(data_dict, key):
                if not data_dict or key not in data_dict:
                    return 0
                return float(data_dict[key].get("value", 0) or 0)

            # Calculate key financial metrics
            ratios = {}

            # Balance sheet metrics
            total_assets = get_value(balance_sheet, "assets")
            total_liabilities = get_value(balance_sheet, "liabilities")
            current_assets = get_value(balance_sheet, "current_assets")
            current_liabilities = get_value(balance_sheet, "current_liabilities")
            equity = get_value(balance_sheet, "equity")
            inventory = get_value(balance_sheet, "inventory")

            # Income statement metrics
            revenue = get_value(income_stmt, "revenues")
            operating_expenses = get_value(income_stmt, "operating_expenses")
            eps = get_value(income_stmt, "basic_earnings_per_share")

            # Cash flow metrics
            operating_cash_flow = get_value(cash_flow, "net_cash_flow")
            financing_cash_flow = get_value(
                cash_flow, "net_cash_flow_from_financing_activities"
            )

            # Calculate ratios
            if equity > 0:
                ratios["debt_to_equity"] = round(total_liabilities / equity, 2)
                ratios["roe"] = round(
                    operating_expenses / equity, 4
                )  # Using operating_expenses as proxy for net income

            if current_liabilities > 0:
                ratios["current_ratio"] = round(current_assets / current_liabilities, 2)
                # Quick ratio (acid-test)
                ratios["quick_ratio"] = round(
                    (current_assets - inventory) / current_liabilities, 2
                )

            if revenue > 0:
                ratios["operating_margin"] = round(operating_expenses / revenue, 4)
                ratios["asset_turnover"] = (
                    round(revenue / total_assets, 4) if total_assets > 0 else None
                )

            if total_assets > 0:
                ratios["roa"] = round(operating_expenses / total_assets, 4)

            ratios.update(
                {
                    "total_assets": total_assets,
                    "total_liabilities": total_liabilities,
                    "equity": equity,
                    "revenue": revenue,
                    "eps": eps,
                    "operating_cash_flow": operating_cash_flow,
                    "financing_cash_flow": financing_cash_flow,
                }
            )

            ratios.update(
                {
                    "fiscal_period": parsed["metadata"]["fiscal_period"],
                    "fiscal_year": parsed["metadata"]["fiscal_year"],
                    "filing_date": parsed["metadata"]["filing_date"],
                }
            )

            return {
                k: v
                for k, v in ratios.items()
                if v is not None
                and not isinstance(v, (complex, bool))
                and (not isinstance(v, float) or (not np.isinf(v) and not np.isnan(v)))
            }

        except Exception as e:
            logger.error(f"Error calculating financial ratios for {ticker}: {str(e)}")
            return {}
