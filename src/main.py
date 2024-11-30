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
        "investing",
        "stockmarket",
        "BullTrader",
        "robinhood",
        "Superstonk",
        "ValueInvesting",
        "Wallstreetbetsnew",
        "stonks",
        "scottsstocks",
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
                "price_change_2d": "first",
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
        50
    )

    # Save results and generate report
    analyzer.save_results(top_stocks)
    analyzer.db.save_sentiment_analysis(top_stocks)


if __name__ == "__main__":
    main()
