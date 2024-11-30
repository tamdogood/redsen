import praw
import nltk
import pandas as pd
import numpy as np
import datetime as dt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download necessary NLTK resources
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)


class SupabaseConnector:
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase connection"""
        self.supabase: Client = create_client(supabase_url, supabase_key)

    def save_post_data(self, post_data: Dict) -> Dict:
        """
        Save post data to Supabase tables

        Args:
            post_data: Dictionary containing post information
        Returns:
            Dict with operation status
        """
        try:
            # First, save the main post
            post_record = {
                "post_id": post_data["post_id"],
                "title": post_data["title"],
                "content": post_data["content"],
                "url": post_data["url"],
                "author": post_data["author"],
                "score": post_data["score"],
                "num_comments": post_data["num_comments"],
                "upvote_ratio": post_data["upvote_ratio"],
                "created_at": post_data["created_utc"].isoformat(),
                "subreddit": post_data["subreddit"],
                "avg_sentiment": post_data["avg_sentiment"],
            }

            # Insert into posts table
            result = self.supabase.table("reddit_posts").insert(post_record).execute()

            # Save associated tickers
            for ticker in post_data["tickers"]:
                ticker_record = {
                    "post_id": post_data["post_id"],
                    "ticker": ticker,
                    "mentioned_at": post_data["created_utc"].isoformat(),
                }
                self.supabase.table("post_tickers").insert(ticker_record).execute()

            # Save comments
            for comment in post_data["comments"]:
                comment_record = {
                    "post_id": post_data["post_id"],
                    "author": comment["author"],
                    "content": comment["body"],
                    "score": comment["score"],
                    "created_at": comment["created_utc"].isoformat(),
                    "sentiment": comment["sentiment"],
                }
                self.supabase.table("post_comments").insert(comment_record).execute()

            return {"success": True, "post_id": post_data["post_id"]}

        except Exception as e:
            logger.error(f"Error saving post to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_sentiment_analysis(self, sentiment_data: pd.DataFrame) -> Dict:
        """
        Save sentiment analysis results to Supabase with the updated schema

        Args:
            sentiment_data: DataFrame containing sentiment analysis with all metrics
        Returns:
            Dict with operation status
        """
        try:
            # Convert DataFrame to records
            records = sentiment_data.to_dict("records")

            # Prepare records for insertion
            analysis_records = []
            for record in records:
                analysis_record = {
                    "analysis_timestamp": (
                        record["analysis_timestamp"].isoformat()
                        if isinstance(record["analysis_timestamp"], dt.datetime)
                        else record["analysis_timestamp"]
                    ),
                    "data_start_date": (
                        record["data_start_date"].isoformat()
                        if isinstance(record["data_start_date"], dt.datetime)
                        else record["data_start_date"]
                    ),
                    "data_end_date": (
                        record["data_end_date"].isoformat()
                        if isinstance(record["data_end_date"], dt.datetime)
                        else record["data_end_date"]
                    ),
                    "ticker": record["ticker"],
                    "score": (
                        float(record["score"]) if pd.notnull(record["score"]) else None
                    ),
                    "num_comments": (
                        int(record["num_comments"])
                        if pd.notnull(record["num_comments"])
                        else None
                    ),
                    "comment_sentiment_avg": (
                        float(record["comment_sentiment_avg"])
                        if pd.notnull(record["comment_sentiment_avg"])
                        else None
                    ),
                    "bullish_comments_ratio": (
                        float(record["bullish_comments_ratio"])
                        if pd.notnull(record["bullish_comments_ratio"])
                        else None
                    ),
                    "bearish_comments_ratio": (
                        float(record["bearish_comments_ratio"])
                        if pd.notnull(record["bearish_comments_ratio"])
                        else None
                    ),
                    "current_price": (
                        float(record["current_price"])
                        if pd.notnull(record["current_price"])
                        else None
                    ),
                    "price_change_2w": (
                        float(record["price_change_2w"])
                        if pd.notnull(record["price_change_2w"])
                        else None
                    ),
                    "avg_volume": (
                        float(record["avg_volume"])
                        if pd.notnull(record["avg_volume"])
                        else None
                    ),
                    "volume_change": (
                        float(record["volume_change"])
                        if pd.notnull(record["volume_change"])
                        else None
                    ),
                    "sma_20": (
                        float(record["sma_20"])
                        if pd.notnull(record["sma_20"])
                        else None
                    ),
                    "rsi": float(record["rsi"]) if pd.notnull(record["rsi"]) else None,
                    "volatility": (
                        float(record["volatility"])
                        if pd.notnull(record["volatility"])
                        else None
                    ),
                    "market_cap": (
                        float(record["market_cap"])
                        if pd.notnull(record["market_cap"])
                        else None
                    ),
                    "pe_ratio": (
                        float(record["pe_ratio"])
                        if pd.notnull(record["pe_ratio"])
                        else None
                    ),
                    "composite_score": (
                        float(record["composite_score"])
                        if pd.notnull(record["composite_score"])
                        else None
                    ),
                }
                analysis_records.append(analysis_record)

            # Insert records in batches to avoid timeouts
            batch_size = 100
            for i in range(0, len(analysis_records), batch_size):
                batch = analysis_records[i : i + batch_size]

                # Use upsert to update existing records based on ticker and timestamp
                self.supabase.table("sentiment_analysis").upsert(
                    batch, on_conflict="ticker,analysis_timestamp"
                ).execute()

            return {
                "success": True,
                "records_processed": len(analysis_records),
                "message": f"Successfully saved {len(analysis_records)} records to Supabase",
            }

        except Exception as e:
            logger.error(f"Error saving sentiment analysis to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_recent_posts_by_ticker(self, ticker: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent posts for a specific ticker

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of posts to retrieve
        Returns:
            List of post records
        """
        try:
            result = (
                self.supabase.table("post_tickers")
                .select("post_id, reddit_posts(*)")
                .eq("ticker", ticker)
                .order("mentioned_at", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data

        except Exception as e:
            logger.error(f"Error retrieving posts for {ticker}: {str(e)}")
            return []


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
        """Extract and validate stock tickers from text"""
        tickers = re.findall(r"\$?([A-Z]{1,5})\b", text)
        valid_tickers = []

        for ticker in tickers:
            if (
                len(ticker) >= 1
                and len(ticker) <= 5
                and ticker not in self.invalid_tickers
                and self.is_valid_ticker(ticker)
            ):
                valid_tickers.append(ticker)

        return valid_tickers

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
        """Get Reddit sentiment analysis with enhanced post data collection"""
        subreddit = self.reddit.subreddit(subreddit_name)
        sentiment_data = []
        post_data = []  # Store full post data for saving

        try:
            # Analyze top posts in the specified time filter
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                # Extract tickers from submission title and text
                submission_tickers = self.extract_stock_tickers(
                    submission.title + " " + submission.selftext
                )

                if not submission_tickers:
                    continue

                # Analyze comment sentiments
                submission.comments.replace_more(limit=0)  # Flatten comment tree
                comments = submission.comments.list()

                # Collect comment data
                comment_data = []
                sentiment_scores = []
                bullish_count = 0
                bearish_count = 0

                for comment in comments:
                    try:
                        sentiment = self.sia.polarity_scores(comment.body)
                        sentiment_scores.append(sentiment["compound"])

                        if sentiment["compound"] > 0.1:
                            bullish_count += 1
                        elif sentiment["compound"] < -0.1:
                            bearish_count += 1

                        comment_data.append(
                            {
                                "author": str(comment.author),
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": dt.datetime.fromtimestamp(
                                    comment.created_utc
                                ),
                                "sentiment": sentiment["compound"],
                            }
                        )
                    except:
                        continue

                # Calculate averages
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
                total_classified = bullish_count + bearish_count

                # Store full post data
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
                    "avg_sentiment": avg_sentiment,
                    "comments": comment_data,
                    "subreddit": subreddit_name,
                }
                post_data.append(post_info)

                # Add sentiment data for analysis
                for ticker in submission_tickers:
                    sentiment_data.append(
                        {
                            "ticker": ticker,
                            "title": submission.title,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "upvote_ratio": submission.upvote_ratio,
                            "comment_sentiment_avg": avg_sentiment,
                            "bullish_comments_ratio": (
                                bullish_count / total_classified
                                if total_classified > 0
                                else 0
                            ),
                            "bearish_comments_ratio": (
                                bearish_count / total_classified
                                if total_classified > 0
                                else 0
                            ),
                            "timestamp": dt.datetime.fromtimestamp(
                                submission.created_utc
                            ),
                            "post_id": submission.id,  # Add post_id for reference
                        }
                    )

            # Save full post data
            # self.save_detailed_post_data(post_data, subreddit_name)

            # Save to Supabase
            for post in post_data:
                self.db.save_post_data(post)

            # # Save sentiment analysis
            # sentiment_df = pd.DataFrame(sentiment_data)
            # if not sentiment_df.empty:
            #     self.db.save_sentiment_analysis(sentiment_df)

            return pd.DataFrame(sentiment_data)

        except Exception as e:
            logger.error(f"Error analyzing subreddit {subreddit_name}: {str(e)}")
            return pd.DataFrame()

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
        """Save results with additional metrics and timestamps"""
        if not df.empty:
            # Add analysis timestamp to DataFrame
            df["analysis_timestamp"] = dt.datetime.now()
            df["data_start_date"] = dt.datetime.now() - dt.timedelta(days=30)
            df["data_end_date"] = dt.datetime.now()

            # Reorder columns to put timestamps at the beginning
            timestamp_cols = ["analysis_timestamp", "data_start_date", "data_end_date"]
            other_cols = [col for col in df.columns if col not in timestamp_cols]
            df = df[timestamp_cols + other_cols]

            # Save to CSV
            df.to_csv(filename, index=False)

            # Generate and save report
            report = self.generate_analysis_report(df)
            report_filename = filename.replace(".csv", "_report.txt")
            with open(report_filename, "w") as f:
                f.write(report)

            logger.info(f"Results saved to {filename} and {report_filename}")
        else:
            logger.warning("No results to save")


def main():

    analyzer = EnhancedStockAnalyzer(
        os.getenv("CLIENT_ID", ""),
        os.getenv("CLIENT_SECRET", ""),
        os.getenv("USER_AGENT", ""),
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_KEY", ""),
    )

    subreddits_to_analyze = ["wallstreetbets", "stocks", "investing"]
    final_results = []

    for subreddit in subreddits_to_analyze:
        logger.info(f"Analyzing {subreddit}...")
        results = analyzer.analyze_subreddit_sentiment(subreddit)
        if not results.empty:
            final_results.append(results)

    if not final_results:
        logger.error("No data collected from any subreddit")
        return

    combined_results = (
        pd.concat(final_results)
        .groupby("ticker")
        .agg(
            {
                "score": "mean",
                "num_comments": "sum",
                "comment_sentiment_avg": "mean",
                "bullish_comments_ratio": "mean",
                "bearish_comments_ratio": "mean",
                "current_price": "first",
                "price_change_2w": "first",
                "avg_volume": "first",
                "volume_change": "first",
                "sma_20": "first",
                "rsi": "first",
                "volatility": "first",
                "market_cap": "first",
                "pe_ratio": "first",
            }
        )
        .reset_index()
    )

    # Sort by a composite score of sentiment and popularity
    combined_results["composite_score"] = (
        combined_results["num_comments"].rank(pct=True) * 0.4
        + combined_results["comment_sentiment_avg"].rank(pct=True) * 0.3
        + combined_results["volume_change"].rank(pct=True) * 0.3
    )

    top_stocks = combined_results.sort_values("composite_score", ascending=False).head(
        20
    )

    # Save results and generate report
    analyzer.save_results(top_stocks)
    analyzer.db.save_sentiment_analysis(top_stocks)


if __name__ == "__main__":
    main()
