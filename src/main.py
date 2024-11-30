import pandas as pd
import numpy as np
import datetime as dt
import logging
import os
from dotenv import load_dotenv

from analyzers.sentiment_analyzer import EnhancedStockAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    analyzer = EnhancedStockAnalyzer(
        os.getenv("CLIENT_ID", ""),
        os.getenv("CLIENT_SECRET", ""),
        os.getenv("USER_AGENT", ""),
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_KEY", ""),
    )

    subreddits_to_analyze = [
        "wallstreetbets",
        "stocks",
        # "investing",
        # "stockmarket",
        # "BullTrader",
        # "robinhood",
        # "Superstonk",
        # "ValueInvesting",
        # "Wallstreetbetsnew",
        # "stonks",
        # "scottsstocks",
    ]
    final_results = []

    for subreddit in subreddits_to_analyze:
        logger.info(f"Analyzing {subreddit}...")
        results = analyzer.analyze_subreddit_sentiment(subreddit)
        if not results.empty:
            final_results.append(results)

    if not final_results:
        logger.error("No data collected from any subreddit")
        return

    # Add timestamps before aggregation

    for df in final_results:
        df["analysis_timestamp"] = dt.datetime.now()
        df["data_start_date"] = dt.datetime.now() - dt.timedelta(days=30)
        df["data_end_date"] = dt.datetime.now()

    # Combine and aggregate results
    combined_results = (
        pd.concat(final_results)
        .groupby("ticker")
        .agg(
            {
                # Reddit metrics
                "score": "mean",
                "num_comments": "sum",
                # Sentiment metrics
                "comment_sentiment_avg": "mean",
                "base_sentiment": "mean",  # If available
                "submission_sentiment": "mean",  # If available
                "bullish_comments_ratio": "mean",
                "bearish_comments_ratio": "mean",
                # Price metrics
                "current_price": "first",
                "price_change_2w": "first",
                "price_change_2d": "first",
                "avg_volume": "first",
                "volume_change": "first",
                # Technical indicators
                "sma_20": "first",
                "rsi": "first",
                "volatility": "first",
                # Fundamental metrics
                "market_cap": "first",
                "pe_ratio": "first",
                # Timestamps
                "analysis_timestamp": "first",
                "data_start_date": "first",
                "data_end_date": "first",
            }
        )
        .reset_index()
    )

    # Calculate enhanced composite score
    combined_results["composite_score"] = (
        combined_results["num_comments"].rank(pct=True) * 0.3  # Popularity
        + combined_results["comment_sentiment_avg"].rank(pct=True) * 0.2  # Sentiment
        + combined_results["volume_change"].rank(pct=True) * 0.2  # Volume activity
        + combined_results["price_change_2d"].rank(pct=True)
        * 0.15  # Recent price movement
        + combined_results["price_change_2w"].rank(pct=True)
        * 0.15  # Longer-term price movement
    )

    # Get top stocks
    top_stocks = combined_results.sort_values("composite_score", ascending=False).head(
        50
    )

    # Clean up any NaN values before saving
    top_stocks = top_stocks.replace([np.inf, -np.inf], np.nan)
    top_stocks = top_stocks.fillna(
        {
            "score": 0,
            "num_comments": 0,
            "comment_sentiment_avg": 0,
            "bullish_comments_ratio": 0,
            "bearish_comments_ratio": 0,
            "volume_change": 0,
            "composite_score": 0,
        }
    )

    # Save results
    analyzer.save_results(top_stocks)
    analyzer.db.save_sentiment_analysis(top_stocks)

    # Log summary
    logger.info(f"Processed {len(combined_results)} unique tickers")
    logger.info(f"Top ticker by sentiment: {top_stocks.iloc[0]['ticker']}")
    logger.info(
        f"Top ticker composite score: {top_stocks.iloc[0]['composite_score']:.2f}"
    )


if __name__ == "__main__":
    main()
