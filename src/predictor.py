import pandas as pd
import numpy as np
import datetime as dt
import logging
import os
from dotenv import load_dotenv
from scipy import stats
from analyzers.sentiment_analyzer import EnhancedStockAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def prepare_training_data(analyzer, ticker_data, prediction_window=5):
    """
    Prepare feature matrix and target vector for model training

    Args:
        analyzer: EnhancedStockAnalyzer instance
        ticker_data: DataFrame with ticker analysis
        prediction_window: Number of days for price prediction
    """
    feature_data = []
    targets = []

    for _, row in ticker_data.iterrows():
        ticker = row["ticker"]
        try:
            # Get complete metrics including new prediction features
            metrics = analyzer.analyze_complete_metrics(ticker)
            if not metrics:
                continue

            # Get future price change for target
            stock_data = analyzer.get_stock_metrics(ticker)
            if not stock_data or "history" not in stock_data:
                continue

            hist = stock_data["history"]
            future_return = (
                hist["Close"].pct_change(prediction_window).shift(-prediction_window)
            )

            # Create target (1 for price increase, 0 for decrease)
            if len(future_return) > prediction_window:
                target = 1 if future_return.iloc[-prediction_window - 1] > 0 else 0

                # Extract features
                features = {
                    # Technical Features
                    "rsi_14": metrics.get("rsi_14", 50),
                    "adx_14": metrics.get("adx_14", 20),
                    "cci_20": metrics.get("cci_20", 0),
                    "mfi_14": metrics.get("mfi_14", 50),
                    "williams_r": metrics.get("williams_r", -50),
                    "ultimate_osc": metrics.get("ultimate_osc", 50),
                    # Volatility Features
                    "atr_14": metrics.get("atr_14", 0),
                    "zscore_20": metrics.get("zscore_20", 0),
                    # Volume Features
                    "volume_price_trend": metrics.get("volume_price_trend", 0),
                    "obv": metrics.get("obv", 0),
                    # Sentiment Features
                    "comment_sentiment_avg": row.get("comment_sentiment_avg", 0),
                    "bullish_comments_ratio": row.get("bullish_comments_ratio", 0.5),
                    "bearish_comments_ratio": row.get("bearish_comments_ratio", 0.5),
                    "sentiment_momentum": metrics.get("sentiment_momentum", 0),
                    "sentiment_volatility": metrics.get("sentiment_volatility", 0),
                    # Market Context
                    "market_correlation": row.get("market_correlation", 0),
                    "beta": metrics.get("beta", 1),
                    "rs_1m": metrics.get("rs_1m", 1),
                    # Pattern Signals
                    "bullish_probability": metrics.get("bullish_probability", 50),
                    "trend_strength": metrics.get("trend_strength", 50),
                    "volatility_probability": metrics.get("volatility_probability", 50),
                }

                feature_data.append(features)
                targets.append(target)

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {str(e)}")
            continue

    return pd.DataFrame(feature_data), np.array(targets)


def train_prediction_models(X, y):
    """Train and evaluate prediction models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Feature importance
        importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        results[name] = {
            "model": model,
            "train_score": train_score,
            "test_score": test_score,
            "feature_importance": importance,
            "classification_report": classification_report(
                y_test, model.predict(X_test)
            ),
        }

        # Save model
        joblib.dump(model, f"models/{name}_stock_predictor.joblib")

    return results


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
        "robinhood",
        "Superstonk",
        "ValueInvesting",
        "Wallstreetbetsnew",
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

    # Process and aggregate results
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
                "market_correlation": "first",
                "composite_score": "mean",
            }
        )
        .reset_index()
    )

    # Prepare training data
    logger.info("Preparing training data...")
    X, y = prepare_training_data(analyzer, combined_results)

    if len(X) > 0:
        # Train prediction models
        logger.info("Training prediction models...")
        model_results = train_prediction_models(X, y)

        # Log model performance
        for name, results in model_results.items():
            logger.info(f"\nModel: {name}")
            logger.info(f"Train Score: {results['train_score']:.4f}")
            logger.info(f"Test Score: {results['test_score']:.4f}")
            logger.info("\nTop 5 Important Features:")
            logger.info(results["feature_importance"].head().to_string())
            logger.info("\nClassification Report:")
            logger.info(results["classification_report"])

        # Make predictions for top stocks
        top_stocks = combined_results.nlargest(50, "composite_score")
        predictions = {}

        for name, results in model_results.items():
            model = results["model"]
            X_pred = prepare_training_data(analyzer, top_stocks)[0]
            if len(X_pred) > 0:
                pred_proba = model.predict_proba(X_pred)
                predictions[name] = pd.DataFrame(
                    {"ticker": top_stocks["ticker"], "up_probability": pred_proba[:, 1]}
                )

        # Save predictions and analysis
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_results = {
            "timestamp": timestamp,
            "predictions": predictions,
            "model_performance": model_results,
            "top_stocks": top_stocks.to_dict("records"),
        }

        # Save to storage
        # analyzer.save_results_to_storage(top_stocks)
        # analyzer.db.save_sentiment_analysis(top_stocks)

        # Save prediction results
        with open(f"predictions/prediction_results_{timestamp}.json", "w") as f:
            json.dump(prediction_results, f, cls=analyzer.db.CustomJSONEncoder)

    else:
        logger.warning("Insufficient data for model training")
        # analyzer.save_results_to_storage(combined_results)
        # analyzer.db.save_sentiment_analysis(combined_results)


if __name__ == "__main__":
    main()
