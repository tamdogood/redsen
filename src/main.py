import pandas as pd
import numpy as np
import datetime as dt
import logging
import os
import yfinance as yf
from dotenv import load_dotenv
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path

from analyzers.sentiment_analyzer import EnhancedStockAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        # "wallstreetbets",
        "stocks",
        # "investing",
        # "stockmarket",
        # "robinhood",
        # "Superstonk",
        # "ValueInvesting",
        # "Wallstreetbetsnew",
        # "stonks",
        # "XGramatikInsights",
        # "scottsstocks",
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
                    logger.warning(f"Error calculating correlation for {ticker}: {str(e)}")
                    return 0.0

            df["market_correlation"] = df["ticker"].apply(calculate_correlation)

        except Exception as e:
            logger.warning(f"Warning: Could not calculate market correlations: {str(e)}")
            df["market_correlation"] = 0.0

    # First, verify which columns exist in the DataFrames
    sample_df = final_results[0] if final_results else pd.DataFrame()
    available_columns = sample_df.columns.tolist()

    # Define aggregation dictionary with only available columns
    agg_dict = {
        # Base columns that should always exist
        "score": "mean",
        "num_comments": "sum",
        "market_correlation": "first",
        "analysis_timestamp": "first",
        "data_start_date": "first",
        "data_end_date": "first"
    }

    # Optional columns - only add if they exist
    optional_columns = {
        # Sentiment Metrics
        "comment_sentiment_avg": "mean",
        "base_sentiment": "mean",
        "submission_sentiment": "mean",
        "bullish_comments_ratio": "mean",
        "bearish_comments_ratio": "mean",
        # Price and Volume Metrics
        "current_price": "first",
        "price_change_1d": "first",
        "price_change_1w": "first",
        "volume_sma": "first",
        "volume_ratio": "first",
        # Technical Indicators
        "sma_20": "first",
        "rsi": "first",
        "volatility": "first",
        # Fundamental Metrics
        "market_cap": "first"
    }

    # Add optional columns only if they exist in the DataFrame
    for col, agg_func in optional_columns.items():
        if col in available_columns:
            agg_dict[col] = agg_func

    # Combine and aggregate results with available metrics
    combined_results = (
        pd.concat(final_results)
        .groupby("ticker")
        .agg(agg_dict)
    )  # Remove reset_index() here as ticker is already the index

    # Reset index only if ticker is not already a column
    if "ticker" not in combined_results.columns:
        combined_results = combined_results.reset_index()

    # Update composite score calculation with only available columns
    score_components = []
    
    # Social and Sentiment Components (40%)
    if "num_comments" in combined_results.columns:
        score_components.append(combined_results["num_comments"].rank(pct=True) * 0.15)
    if "comment_sentiment_avg" in combined_results.columns:
        score_components.append(combined_results["comment_sentiment_avg"].rank(pct=True) * 0.15)
    if "bullish_comments_ratio" in combined_results.columns:
        score_components.append(combined_results["bullish_comments_ratio"].rank(pct=True) * 0.10)
    
    # Technical Components (40%)
    if "volume_ratio" in combined_results.columns:
        score_components.append(combined_results["volume_ratio"].rank(pct=True) * 0.15)
    if "price_change_1d" in combined_results.columns:
        score_components.append(combined_results["price_change_1d"].rank(pct=True) * 0.10)
    if "price_change_1w" in combined_results.columns:
        score_components.append(combined_results["price_change_1w"].rank(pct=True) * 0.15)
    if "rsi" in combined_results.columns:
        score_components.append(
            combined_results["rsi"]
            .apply(lambda x: 1 - abs(50 - x) / 50 if pd.notnull(x) else 0)
            .rank(pct=True) * 0.10
        )
    
    # Market Context (10%)
    if "market_correlation" in combined_results.columns:
        score_components.append(combined_results["market_correlation"].rank(pct=True) * 0.10)

    # Calculate composite score only with available components
    combined_results["composite_score"] = sum(score_components)

    top_stocks = combined_results.nlargest(50, "composite_score")

    # Update fill values dictionary only with available columns
    fill_values = {}
    for col in combined_results.columns:
        if col in ["score", "num_comments"]:
            fill_values[col] = 0
        elif col in ["comment_sentiment_avg", "base_sentiment", "submission_sentiment", 
                    "bullish_comments_ratio", "bearish_comments_ratio"]:
            fill_values[col] = 0
        elif col == "rsi":
            fill_values[col] = 50
        elif col in ["volume_ratio", "composite_score", "market_correlation"]:
            fill_values[col] = 0

    top_stocks = top_stocks.replace([np.inf, -np.inf], np.nan)
    top_stocks = top_stocks.fillna(fill_values)

    # Add data quality score based on available metrics
    top_stocks["data_quality_score"] = top_stocks.apply(
        lambda row: sum(pd.notnull(row)) / len(row), axis=1
    )

    logger.info("Preparing data for model training...")
    
    # Prepare features and targets
    # X, y = prepare_prediction_features(analyzer, combined_results)
    
    # if len(X) > 0:
    #     # Train models
    #     model_results = train_prediction_models(X, y)
        
    #     # Load scaler for predictions
    #     scaler = joblib.load('models/feature_scaler.joblib')
        
    #     # Make predictions for top stocks
    #     top_stocks = predict_top_stocks(model_results, scaler, analyzer, top_stocks)
        
    #     # Save enhanced results
    #     timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    #     results_dir = f'results/{timestamp}'
    #     Path(results_dir).mkdir(parents=True, exist_ok=True)
        
    #     # Save prediction results
    #     prediction_results = {
    #         'timestamp': timestamp,
    #         'model_performance': {
    #             name: {
    #                 'train_score': results['train_score'],
    #                 'test_score': results['test_score'],
    #                 'feature_importance': results['feature_importance'].to_dict('records'),
    #                 'classification_report': results['classification_report'],
    #                 'confusion_matrix': results['confusion_matrix']
    #             }
    #             for name, results in model_results.items()
    #         },
    #         'top_stocks': top_stocks.to_dict('records')
    #     }
        
    #     with open(f'{results_dir}/prediction_results.json', 'w') as f:
    #         json.dump(prediction_results, f, cls=analyzer.db.CustomJSONEncoder)
            
    #     # Save enhanced top stocks DataFrame
    #     top_stocks.to_csv(f'{results_dir}/top_stocks_with_predictions.csv', index=False)
        
    #     # Log model performance
    #     logger.info("\nModel Performance Summary:")
    #     for name, results in model_results.items():
    #         logger.info(f"\n{name.upper()} Model:")
    #         logger.info(f"Train Score: {results['train_score']:.4f}")
    #         logger.info(f"Test Score: {results['test_score']:.4f}")
    #         logger.info("\nTop 5 Important Features:")
    #         logger.info(results['feature_importance'].head().to_string())
              
    #     # Log top predictions
    #     logger.info("\nTop 5 Stocks by Ensemble Probability:")
    #     top_predictions = top_stocks.nlargest(5, 'ensemble_up_probability')
    #     for _, row in top_predictions.iterrows():
    #         logger.info(
    #             f"{row['ticker']}: {row['ensemble_up_probability']:.2%} probability of increase"
    #         )
    # else:
    #     logger.warning("Insufficient data for model training")
    #     # analyzer.save_results_to_storage(top_stocks)
        
    # Save results with quality metadata
    # if os.getenv("SAVE_TO_STORAGE", "0") == "1":
        # analyzer.save_results_to_storage(top_stocks)
        # analyzer.db.save_sentiment_analysis(top_stocks)
    # analyzer.save_results(top_stocks)

    # Log summary
    logger.info(f"Processed {len(combined_results)} unique tickers")
    logger.info(f"Top ticker by sentiment: {top_stocks.iloc[0]['ticker']}")
    logger.info(f"Top ticker composite score: {top_stocks.iloc[0]['composite_score']:.2f}")

if __name__ == "__main__":
    main()