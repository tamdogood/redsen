import praw
import nltk
import pandas as pd
import numpy as np
import datetime as dt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from dotenv import load_dotenv
from connectors.supabase_connector import SupabaseConnector
from utils.logging_config import logger
from openai import OpenAI
import json
import os


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
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )

        self.db = SupabaseConnector(
            supabase_url=supabase_url, supabase_key=supabase_key
        )

        self.sia = SentimentIntensityAnalyzer()
        self.stock_data_cache = {}

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
        }

    def is_valid_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker symbol is likely to be valid

        Args:
            ticker (str): Potential stock ticker

        Returns:
            bool: True if likely valid, False otherwise
        """
        if ticker in self.invalid_tickers:
            return False

        try:
            info = yf.Ticker(ticker).history(period="5d", interval="1d")

            return len(info) > 0

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
            if not tickers and False:
                tickers = self._extract_tickers_with_openai(text)

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
            # Initialize OpenAI client (do this in __init__ if used frequently)
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )

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
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst that extracts stock tickers from text. Only respond with a JSON array of tickers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=150,
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
                price_2d_ago = hist["Close"].iloc[-2]
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
                    "price_change_2d": round(
                        ((current_price - price_2d_ago) / price_2d_ago) * 100, 2
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

    def analyze_subreddit_sentiment(
        self, subreddit_name: str, time_filter: str = "week", limit: int = 5
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
                submission.comments.replace_more(limit=0)
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
        Enhanced sentiment analysis for text

        Args:
            text: Text to analyze

        Returns:
            Dict containing sentiment scores and metadata
        """
        # Get base VADER sentiment
        vader_scores = self.sia.polarity_scores(text)

        # Additional sentiment features
        sentiment_features = {
            "has_exclamation": "!" in text,
            "has_dollar_sign": "$" in text,
            "has_rocket": "🚀" in text or "rocket" in text.lower(),
            "has_moon": "🌙" in text.lower() or "moon" in text.lower(),
            "has_buy_terms": any(
                term in text.lower() for term in ["buy", "long", "bull", "calls"]
            ),
            "has_sell_terms": any(
                term in text.lower() for term in ["sell", "short", "bear", "puts"]
            ),
        }

        # Adjust compound score based on additional features
        compound_adjustment = 0
        if sentiment_features["has_exclamation"]:
            compound_adjustment += 0.1
        if sentiment_features["has_dollar_sign"]:
            compound_adjustment += 0.05
        if sentiment_features["has_rocket"] or sentiment_features["has_moon"]:
            compound_adjustment += 0.15
        if sentiment_features["has_buy_terms"]:
            compound_adjustment += 0.1
        if sentiment_features["has_sell_terms"]:
            compound_adjustment -= 0.1

        # Ensure final compound score stays within [-1, 1]
        adjusted_compound = max(
            min(vader_scores["compound"] + compound_adjustment, 1), -1
        )

        return {
            "compound": adjusted_compound,
            "pos": vader_scores["pos"],
            "neg": vader_scores["neg"],
            "neu": vader_scores["neu"],
            "features": sentiment_features,
        }

    def _calculate_comment_weight(self, comment) -> float:
        """
        Calculate weight for a comment based on various factors

        Args:
            comment: Reddit comment object

        Returns:
            float: Weight between 0 and 2
        """
        # Base weight from comment score
        if comment.score <= 0:
            score_weight = 0.5
        else:
            score_weight = min(1 + np.log1p(comment.score) / 10, 1.5)

        # Account for comment awards
        try:
            awards_count = (
                len(comment.all_awardings) if hasattr(comment, "all_awardings") else 0
            )
            awards_weight = min(1 + (awards_count * 0.1), 1.5)
        except:
            awards_weight = 1.0

        # Account for comment length (longer comments might be more substantive)
        length_weight = min(1 + (len(comment.body) / 1000), 1.2)

        # Calculate final weight
        final_weight = score_weight * 0.5 + awards_weight * 0.3 + length_weight * 0.2

        return min(max(final_weight, 0.5), 2.0)  # Ensure weight is between 0.5 and 2

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
        top_gainers = df.nlargest(3, "price_change_2w")
        report.append("\nTop Price Gainers (2 weeks):")
        for _, row in top_gainers.iterrows():
            report.append(
                f"- {row['ticker']}: {row['price_change_2w']}% (Current: ${row['current_price']})"
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
                        "price_change_2w": safe_get(row, "price_change_2w"),
                        "price_change_2d": safe_get(row, "price_change_2d"),
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
                        "price_change": safe_get(r, "price_change_2w", 0.0),
                    }
                    for _, r in df.nlargest(5, "price_change_2w")[
                        ["ticker", "price_change_2w"]
                    ].iterrows()
                ],
                "worst_performing": [
                    {
                        "ticker": str(r["ticker"]),
                        "price_change": safe_get(r, "price_change_2w", 0.0),
                    }
                    for _, r in df.nsmallest(5, "price_change_2w")[
                        ["ticker", "price_change_2w"]
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
