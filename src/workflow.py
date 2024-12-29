from prefect import flow, task
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


@task
def initialize_analyzer():
    """Initialize the stock analyzer with credentials"""
    return EnhancedStockAnalyzer(
        os.getenv("CLIENT_ID", ""),
        os.getenv("CLIENT_SECRET", ""),
        os.getenv("USER_AGENT", ""),
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_KEY", ""),
    )


@task
def analyze_subreddits(analyzer: EnhancedStockAnalyzer):
    """Analyze sentiment across multiple subreddits"""
    subreddits_to_analyze = [
        "wallstreetbets",
        "stocks",
        "investing",
        "stockmarket",
        "ValueInvesting",
        "Wallstreetbetsnew",
    ]
    final_results = []

    for subreddit in subreddits_to_analyze:
        logger.info("Analyzing %s...", subreddit)
        results = analyzer.analyze_subreddit_sentiment(
            subreddit,
            limit=int(os.getenv("REDDIT_TOP_POST_LIMIT", "20")),
        )
        if not results.empty:
            final_results.append(results)

    if not final_results:
        logger.error("No data collected from any subreddit")
        return None

    return final_results


@task
def process_market_data(final_results):
    """Process market data and calculate correlations"""
    if not final_results:
        return None

    data_start_date = dt.datetime.now() - dt.timedelta(days=30)
    data_end_date = dt.datetime.now()

    market_data = yf.download("SPY", start=data_start_date, end=data_end_date)

    for df in final_results:
        df["analysis_timestamp"] = dt.datetime.now()
        df["data_start_date"] = data_start_date
        df["data_end_date"] = data_end_date

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

                    market_returns = market_data["Close"].pct_change().dropna()
                    stock_returns = stock_data["Close"].pct_change().dropna()

                    min_len = min(len(market_returns), len(stock_returns))
                    if min_len < 2:
                        return 0.0

                    if (
                        len(np.unique(market_returns[-min_len:])) == 1
                        or len(np.unique(stock_returns[-min_len:])) == 1
                    ):
                        logger.warning(
                            "Constant returns detected for %s - correlation undefined",
                            ticker,
                        )
                        return 0.0

                    corr = stats.pearsonr(
                        market_returns[-min_len:], stock_returns[-min_len:]
                    )
                    return (
                        corr[0].item()
                        if isinstance(corr[0], np.ndarray)
                        else float(corr[0])
                    )

                except Exception as e:
                    logger.warning(
                        f"Error calculating correlation for {ticker}: {str(e)}"
                    )
                    return 0.0

            df["market_correlation"] = df["ticker"].apply(calculate_correlation)

        except Exception as e:
            logger.warning(
                f"Warning: Could not calculate market correlations: {str(e)}"
            )
            df["market_correlation"] = 0.0

    return final_results


@task
def aggregate_results(final_results):
    """Aggregate results and calculate composite scores"""
    if not final_results:
        return None

    sample_df = final_results[0]
    available_columns = sample_df.columns.tolist()

    # Define base aggregation dictionary
    agg_dict = {
        "score": "mean",
        "num_comments": "sum",
        "market_correlation": "first",
        "analysis_timestamp": "first",
        "data_start_date": "first",
        "data_end_date": "first",
    }

    # Add optional columns if they exist
    optional_columns = {
        "comment_sentiment_avg": "mean",
        "base_sentiment": "mean",
        "submission_sentiment": "mean",
        "bullish_comments_ratio": "mean",
        "bearish_comments_ratio": "mean",
        "sentiment_confidence": "mean",
        "upvote_ratio": "mean",
        "current_price": "first",
        "price_change_1d": "first",
        "price_change_1w": "first",
        "price_change_1m": "first",
        "price_gaps": "first",
        "price_trend": "first",
        "support_resistance": "first",
        "volume_sma": "first",
        "volume_ratio": "first",
        "avg_volume_10d": "first",
        "avg_volume_30d": "first",
        "volume_price_trend": "first",
        "volume_momentum": "first",
        "volume_trend": "first",
        "relative_volume": "first",
        "sma_20": "first",
        "ema_9": "first",
        "rsi": "first",
        "macd": "first",
        "macd_line": "first",
        "macd_signal": "first",
        "macd_diff": "first",
        "stochastic": "first",
        "stoch_k": "first",
        "stoch_d": "first",
        "adx": "first",
        "cci": "first",
        "money_flow_index": "first",
        "volatility": "first",
        "volatility_daily": "first",
        "volatility_weekly": "first",
        "volatility_monthly": "first",
        "volatility_trend": "first",
        "bollinger_upper": "first",
        "bollinger_lower": "first",
        "bb_upper": "first",
        "bb_lower": "first",
        "bb_middle": "first",
        "keltner_channels": "first",
        "atr": "first",
        "sector_performance": "first",
        "market_indicators": "first",
        "relative_strength": "first",
        "beta": "first",
    }

    for col, agg_func in optional_columns.items():
        if col in available_columns:
            agg_dict[col] = agg_func

    combined_results = pd.concat(final_results).groupby("ticker").agg(agg_dict)

    if "ticker" not in combined_results.columns:
        combined_results = combined_results.reset_index()

    return combined_results


@task
def calculate_scores(combined_results):
    """Calculate final scores and prepare top stocks"""
    if combined_results is None:
        return None

    score_components = []

    # Social and Sentiment Components (40%)
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

    # Technical Components (40%)
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

    # Market Context (10%)
    if "market_correlation" in combined_results.columns:
        score_components.append(
            combined_results["market_correlation"].rank(pct=True) * 0.10
        )

    combined_results["composite_score"] = sum(score_components)
    top_stocks = combined_results.nlargest(150, "composite_score")

    fill_values = {
        "score": 0,
        "num_comments": 0,
        "comment_sentiment_avg": 0,
        "base_sentiment": 0,
        "submission_sentiment": 0,
        "bullish_comments_ratio": 0,
        "bearish_comments_ratio": 0,
        "rsi": 50,
        "volume_ratio": 0,
        "composite_score": 0,
        "market_correlation": 0,
        "price_change_1d": 0,
        "price_change_1w": 0,
        "price_change_1m": 0,
        "volume_sma": 0,
        "relative_volume": 0,
        "volatility": 0,
    }

    top_stocks = top_stocks.replace([np.inf, -np.inf], np.nan)
    top_stocks = top_stocks.fillna(fill_values)

    # Calculate data quality score
    top_stocks["data_quality_score"] = top_stocks.apply(
        lambda row: sum(pd.notnull(row)) / len(row), axis=1
    )

    return top_stocks


@task
def save_results(analyzer: EnhancedStockAnalyzer, top_stocks):
    """Save results to storage and database"""
    if top_stocks is None:
        return

    if os.getenv("SAVE_TO_STORAGE", "0") == "1":
        analyzer.db.save_sentiment_analysis(top_stocks)

    analyzer.save_results(top_stocks)

    logger.info("Processed %d unique tickers", len(top_stocks))
    logger.info("Top ticker by sentiment: %s", top_stocks.iloc[0]["ticker"])
    logger.info(
        "Top ticker composite score: %.2f", top_stocks.iloc[0]["composite_score"]
    )


@flow(name="Stock Analysis Flow")
def stock_analysis_flow():
    """Main flow for stock analysis"""
    analyzer = initialize_analyzer()
    final_results = analyze_subreddits(analyzer)
    processed_results = process_market_data(final_results)
    combined_results = aggregate_results(processed_results)
    top_stocks = calculate_scores(combined_results)
    save_results(analyzer, top_stocks)


if __name__ == "__main__":
    stock_analysis_flow.serve(
        name="stock_analysis_scheduled",
        cron="0 3 * * *",
        tags=["stock-analysis"],
    )
