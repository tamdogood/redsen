from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import numpy as np
import datetime as dt
import logging
import os
import yfinance as yf
from dotenv import load_dotenv
from scipy import stats
from analyzers.sentiment_analyzer import EnhancedStockAnalyzer
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=6),
)
def initialize_analyzer():
    """Initialize the sentiment analyzer with API credentials"""
    return EnhancedStockAnalyzer(
        os.getenv("CLIENT_ID", ""),
        os.getenv("CLIENT_SECRET", ""),
        os.getenv("USER_AGENT", ""),
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_KEY", ""),
    )


@task(retries=2, retry_delay_seconds=30)
def analyze_subreddit(analyzer, subreddit: str, post_limit: int) -> pd.DataFrame:
    """Analyze sentiment for a specific subreddit"""
    logger.info(f"Analyzing {subreddit}...")
    results = analyzer.analyze_subreddit_sentiment(
        subreddit,
        limit=post_limit,
    )
    return results if not results.empty else pd.DataFrame()


@task
def get_market_data(start_date: dt.datetime, end_date: dt.datetime):
    """Fetch market data for analysis period"""
    return yf.download(
        "SPY",
        start=start_date,
        end=end_date,
    )


@task
def calculate_market_correlation(
    df: pd.DataFrame, market_data, start_date: dt.datetime, end_date: dt.datetime
) -> pd.DataFrame:
    """Calculate market correlation for each ticker"""

    def _calculate_correlation(ticker):
        if pd.isna(ticker):
            return 0.0
        try:
            stock_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
            )

            market_returns = market_data["Close"].pct_change().dropna()
            stock_returns = stock_data["Close"].pct_change().dropna()

            min_len = min(len(market_returns), len(stock_returns))
            if min_len < 2:
                return 0.0

            corr = stats.pearsonr(market_returns[-min_len:], stock_returns[-min_len:])
            return corr[0].item() if isinstance(corr[0], np.ndarray) else float(corr[0])

        except Exception as e:
            logger.warning(f"Error calculating correlation for {ticker}: {str(e)}")
            return 0.0

    df["market_correlation"] = df["ticker"].apply(_calculate_correlation)
    return df


@task
def aggregate_results(final_results: list) -> pd.DataFrame:
    """Aggregate results from multiple subreddits"""
    sample_df = final_results[0] if final_results else pd.DataFrame()
    available_columns = sample_df.columns.tolist()

    agg_dict = {
        "score": "mean",
        "num_comments": "sum",
        "market_correlation": "first",
        "analysis_timestamp": "first",
        "data_start_date": "first",
        "data_end_date": "first",
    }

    optional_columns = {
        "comment_sentiment_avg": "mean",
        "base_sentiment": "mean",
        "submission_sentiment": "mean",
        "bullish_comments_ratio": "mean",
        "bearish_comments_ratio": "mean",
        "current_price": "first",
        "price_change_1d": "first",
        "price_change_1w": "first",
        "volume_sma": "first",
        "volume_ratio": "first",
        "sma_20": "first",
        "rsi": "first",
        "volatility": "first",
        "market_cap": "first",
    }

    for col, agg_func in optional_columns.items():
        if col in available_columns:
            agg_dict[col] = agg_func

    combined_results = pd.concat(final_results).groupby("ticker").agg(agg_dict)

    if "ticker" not in combined_results.columns:
        combined_results = combined_results.reset_index()

    return combined_results


@task
def calculate_composite_scores(combined_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite scores for each stock"""
    score_components = []

    # Add score components based on available columns
    if "num_comments" in combined_results.columns:
        score_components.append(combined_results["num_comments"].rank(pct=True) * 0.15)
    if "comment_sentiment_avg" in combined_results.columns:
        score_components.append(
            combined_results["comment_sentiment_avg"].rank(pct=True) * 0.15
        )
    if "bullish_comments_ratio" in combined_results.columns:
        score_components.append(
            combined_results["bullish_comments_ratio"].rank(pct=True) * 0.10
        )
    if "volume_ratio" in combined_results.columns:
        score_components.append(combined_results["volume_ratio"].rank(pct=True) * 0.15)
    if "price_change_1d" in combined_results.columns:
        score_components.append(
            combined_results["price_change_1d"].rank(pct=True) * 0.10
        )
    if "price_change_1w" in combined_results.columns:
        score_components.append(
            combined_results["price_change_1w"].rank(pct=True) * 0.15
        )
    if "rsi" in combined_results.columns:
        score_components.append(
            combined_results["rsi"]
            .apply(lambda x: 1 - abs(50 - x) / 50 if pd.notnull(x) else 0)
            .rank(pct=True)
            * 0.10
        )
    if "market_correlation" in combined_results.columns:
        score_components.append(
            combined_results["market_correlation"].rank(pct=True) * 0.10
        )

    combined_results["composite_score"] = sum(score_components)
    return combined_results


@task
def save_results(analyzer, top_stocks: pd.DataFrame):
    """Save analysis results to storage and database"""
    # analyzer.save_results_to_storage(top_stocks)
    analyzer.save_results(top_stocks)
    analyzer.db.save_sentiment_analysis(top_stocks)


@flow(name="Reddit Sentiment Analysis")
def reddit_sentiment_flow():
    """Main flow for Reddit sentiment analysis"""
    subreddits_to_analyze = [
        "wallstreetbets",
        "stocks",
        "investing",
        "stockmarket",
        "robinhood",
        "Superstonk",
        "ValueInvesting",
        "Wallstreetbetsnew",
    ]

    # Initialize analyzer
    analyzer = initialize_analyzer()

    # Set analysis timeframe
    data_start_date = dt.datetime.now() - dt.timedelta(days=30)
    data_end_date = dt.datetime.now()

    # Get market data
    market_data = get_market_data(data_start_date, data_end_date)

    # Analyze subreddits
    results = []
    post_limit = int(os.getenv("REDDIT_TOP_POST_LIMIT", 20))

    for subreddit in subreddits_to_analyze:
        result = analyze_subreddit(analyzer, subreddit, post_limit)
        if not result.empty:
            result = calculate_market_correlation(
                result, market_data, data_start_date, data_end_date
            )
            results.append(result)

    if not results:
        logger.error("No data collected from any subreddit")
        return

    # Process results
    combined_results = aggregate_results(results)
    scored_results = calculate_composite_scores(combined_results)

    # Get top stocks
    top_stocks = scored_results.nlargest(50, "composite_score")

    # Add data quality score
    top_stocks["data_quality_score"] = top_stocks.apply(
        lambda row: sum(pd.notnull(row)) / len(row), axis=1
    )

    # Save results
    save_results(analyzer, top_stocks)

    # Log summary
    logger.info(f"Processed {len(combined_results)} unique tickers")
    logger.info(f"Top ticker by sentiment: {top_stocks.iloc[0]['ticker']}")
    logger.info(
        f"Top ticker composite score: {top_stocks.iloc[0]['composite_score']:.2f}"
    )


# Create deployment with schedule
deployment = Deployment.build_from_flow(
    flow=reddit_sentiment_flow,
    name="reddit_sentiment_scheduled",
    schedule=CronSchedule(cron="0 */16 * * *", timezone="UTC"),  # Run every 16 hours
)

if __name__ == "__main__":
    deployment.apply()
    # prefect server start
    # python flows/reddit_sentiment_flow.py
    # prefect worker start -p default-agent-pool
