import pandas as pd
import numpy as np
import datetime as dt
import logging
import os
import yfinance as yf
from dotenv import load_dotenv
from scipy import stats

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
        results = analyzer.analyze_subreddit_sentiment(
            subreddit,
            limit=int(os.getenv("REDDIT_TOP_POST_LIMIT", 20)),
        )
        if not results.empty:
            final_results.append(results)

    if not final_results:
        logger.error("No data collected from any subreddit")
        return

    data_start_date = dt.datetime.now() - dt.timedelta(days=30)
    data_end_date = dt.datetime.now()

    # Process and aggregate results with enhanced metrics
    market_data = yf.download(
        "SPY",
        start=data_start_date,
        end=data_end_date,
    )

    for df in final_results:
        df["analysis_timestamp"] = dt.datetime.now()
        df["data_start_date"] = dt.datetime.now() - dt.timedelta(days=30)
        df["data_end_date"] = dt.datetime.now()

        # Add market context with proper scalar conversion
        try:

            def calculate_correlation(ticker):
                if pd.isna(ticker):
                    return 0.0
                try:
                    stock_data = yf.download(
                        ticker,
                        start=df["data_start_date"].iloc[0],
                        end=df["data_end_date"].iloc[0],
                    )

                    # Calculate returns
                    market_returns = market_data["Close"].pct_change().dropna()
                    stock_returns = stock_data["Close"].pct_change().dropna()

                    # Ensure we have matching data points
                    min_len = min(len(market_returns), len(stock_returns))
                    if min_len < 2:  # Need at least 2 points for correlation
                        return 0.0

                    # Calculate correlation and extract the scalar value
                    corr = stats.pearsonr(
                        market_returns[-min_len:], stock_returns[-min_len:]
                    )
                    return (
                        corr[0].item()
                        if isinstance(corr[0], np.ndarray)
                        else float(corr[0])
                    )

                except Exception as e:
                    print(f"Error calculating correlation for {ticker}: {str(e)}")
                    return 0.0

            df["market_correlation"] = df["ticker"].apply(calculate_correlation)

        except Exception as e:
            print(f"Warning: Could not calculate market correlations: {str(e)}")
            df["market_correlation"] = 0.0

    # Combine and aggregate results with available metrics
    combined_results = (
        pd.concat(final_results)
        .groupby("ticker")
        .agg(
            {
                # Reddit and Social Metrics
                "score": "mean",
                "num_comments": "sum",
                # Sentiment Metrics
                "comment_sentiment_avg": "mean",
                "base_sentiment": "mean",
                "submission_sentiment": "mean",
                "bullish_comments_ratio": "mean",
                "bearish_comments_ratio": "mean",
                # Price and Volume Metrics
                "current_price": "first",
                "price_change_2d": "first",
                "price_change_2w": "first",
                "avg_volume": "first",
                "volume_change": "first",
                # Technical Indicators
                "sma_20": "first",
                "rsi": "first",
                "volatility": "first",
                # Fundamental Metrics
                "market_cap": "first",
                "pe_ratio": "first",
                # Market Context
                "market_correlation": "first",
                # Timestamps
                "analysis_timestamp": "first",
                "data_start_date": "first",
                "data_end_date": "first",
            }
        )
        .reset_index()
    )

    # Calculate enhanced composite score with available metrics
    combined_results["composite_score"] = (
        # Social and Sentiment Components (40%)
        combined_results["num_comments"].rank(pct=True) * 0.15  # Social engagement
        + combined_results["comment_sentiment_avg"].rank(pct=True)
        * 0.15  # Combined sentiment
        + combined_results["bullish_comments_ratio"].rank(pct=True)
        * 0.10  # Bullish sentiment
        +
        # Technical Components (40%)
        combined_results["volume_change"].rank(pct=True) * 0.15  # Volume activity
        + combined_results["price_change_2d"].rank(pct=True) * 0.10  # Recent momentum
        + combined_results["price_change_2w"].rank(pct=True)
        * 0.15  # Longer-term momentum
        + combined_results["rsi"]
        .apply(lambda x: 1 - abs(50 - x) / 50 if pd.notnull(x) else 0)
        .rank(pct=True)
        * 0.10  # RSI normalization
        +
        # Market Context (10%)
        combined_results["market_correlation"].rank(pct=True)
        * 0.10  # Market correlation
    )

    # Risk-adjusted score using available volatility metric
    combined_results["risk_adjusted_score"] = combined_results.apply(
        lambda row: row["composite_score"]
        / (
            row["volatility"]
            if pd.notnull(row["volatility"]) and row["volatility"] != 0
            else 1
        ),
        axis=1,
    )

    # Get top stocks with multiple ranking methods - simplified to avoid duplicates issue
    top_stocks = combined_results.nlargest(50, "composite_score")

    # Clean up NaN values
    fill_values = {
        # Social metrics
        "score": 0,
        "num_comments": 0,
        # Sentiment metrics
        "comment_sentiment_avg": 0,
        "base_sentiment": 0,
        "submission_sentiment": 0,
        "bullish_comments_ratio": 0,
        "bearish_comments_ratio": 0,
        # Technical metrics
        "volume_change": 0,
        "rsi": 50,
        # Composite scores
        "composite_score": 0,
        "risk_adjusted_score": 0,
        # Market context
        "market_correlation": 0,
    }

    # Replace infinities and fill NaN values
    top_stocks = top_stocks.replace([np.inf, -np.inf], np.nan)
    top_stocks = top_stocks.fillna(fill_values)

    # Add data quality score based on available metrics
    top_stocks["data_quality_score"] = top_stocks.apply(
        lambda row: sum(pd.notnull(row)) / len(row), axis=1
    )

    # Save results with quality metadata
    analyzer.save_results_to_storage(top_stocks)
    analyzer.db.save_sentiment_analysis(top_stocks)

    # Log summary
    logger.info(f"Processed {len(combined_results)} unique tickers")
    logger.info(f"Top ticker by sentiment: {top_stocks.iloc[0]['ticker']}")
    logger.info(
        f"Top ticker composite score: {top_stocks.iloc[0]['composite_score']:.2f}"
    )


if __name__ == "__main__":
    main()
