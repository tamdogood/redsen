import io
from supabase import create_client, Client
from typing import Dict, List, Optional
import pandas as pd
from utils.logging_config import logger
import datetime as dt
import json
import numpy as np
import os


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle timestamps and numpy types"""

    def default(self, obj):
        if isinstance(obj, (dt.datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


class SupabaseConnector:
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase connection"""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.write_controller = os.getenv("DB_WRITE", False)

    def _convert_to_json_serializable(self, obj: any) -> any:
        """Convert values to JSON serializable format"""
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

    def save_sentiment_analysis(self, sentiment_data: pd.DataFrame) -> Dict:
        """Save enhanced sentiment analysis results to Supabase"""
        if self.write_controller == "False":
            return {"success": False, "error": "Write controller is disabled"}
        try:
            if sentiment_data.empty:
                return {"success": False, "error": "Empty DataFrame provided"}

            analysis_records = []
            for _, record in sentiment_data.iterrows():
                analysis_record = {
                    # Timestamps
                    "analysis_timestamp": self._convert_to_json_serializable(
                        record["analysis_timestamp"]
                    ),
                    "data_start_date": self._convert_to_json_serializable(
                        record["data_start_date"]
                    ),
                    "data_end_date": self._convert_to_json_serializable(
                        record["data_end_date"]
                    ),
                    # Basic info
                    "ticker": str(record["ticker"]),
                    # Reddit metrics
                    "score": self._convert_to_json_serializable(record.get("score")),
                    "num_comments": self._convert_to_json_serializable(
                        record.get("num_comments")
                    ),
                    # Sentiment metrics
                    "comment_sentiment_avg": self._convert_to_json_serializable(
                        record.get("comment_sentiment_avg")
                    ),
                    "base_sentiment": self._convert_to_json_serializable(
                        record.get("base_sentiment")
                    ),
                    "submission_sentiment": self._convert_to_json_serializable(
                        record.get("submission_sentiment")
                    ),
                    "bullish_comments_ratio": self._convert_to_json_serializable(
                        record.get("bullish_comments_ratio")
                    ),
                    "bearish_comments_ratio": self._convert_to_json_serializable(
                        record.get("bearish_comments_ratio")
                    ),
                    "sentiment_confidence": self._convert_to_json_serializable(
                        record.get("sentiment_confidence")
                    ),
                    # Price metrics
                    "current_price": self._convert_to_json_serializable(
                        record.get("current_price")
                    ),
                    "price_change_1w": self._convert_to_json_serializable(
                        record.get("price_change_1w")
                    ),
                    "price_change_1d": self._convert_to_json_serializable(
                        record.get("price_change_1d")
                    ),
                    # Volume metrics
                    "avg_volume": self._convert_to_json_serializable(
                        record.get("avg_volume")
                    ),
                    "volume_change": self._convert_to_json_serializable(
                        record.get("volume_change")
                    ),
                    "volume_sma": self._convert_to_json_serializable(
                        record.get("volume_sma")
                    ),
                    "volume_ratio": self._convert_to_json_serializable(
                        record.get("volume_ratio")
                    ),
                    # Technical indicators
                    "sma_20": self._convert_to_json_serializable(record.get("sma_20")),
                    "ema_9": self._convert_to_json_serializable(record.get("ema_9")),
                    "rsi": self._convert_to_json_serializable(record.get("rsi")),
                    "volatility": self._convert_to_json_serializable(
                        record.get("volatility")
                    ),
                    "bollinger_upper": self._convert_to_json_serializable(
                        record.get("bollinger_upper")
                    ),
                    "bollinger_lower": self._convert_to_json_serializable(
                        record.get("bollinger_lower")
                    ),
                    "macd_line": self._convert_to_json_serializable(
                        record.get("macd_line")
                    ),
                    "signal_line": self._convert_to_json_serializable(
                        record.get("signal_line")
                    ),
                    "macd_histogram": self._convert_to_json_serializable(
                        record.get("macd_histogram")
                    ),
                    "stoch_k": self._convert_to_json_serializable(
                        record.get("stoch_k")
                    ),
                    "stoch_d": self._convert_to_json_serializable(
                        record.get("stoch_d")
                    ),
                    # Fundamental metrics
                    "market_cap": self._convert_to_json_serializable(
                        record.get("market_cap")
                    ),
                    "pe_ratio": self._convert_to_json_serializable(
                        record.get("pe_ratio")
                    ),
                    "beta": self._convert_to_json_serializable(record.get("beta")),
                    "dividend_yield": self._convert_to_json_serializable(
                        record.get("dividend_yield")
                    ),
                    "profit_margins": self._convert_to_json_serializable(
                        record.get("profit_margins")
                    ),
                    "revenue_growth": self._convert_to_json_serializable(
                        record.get("revenue_growth")
                    ),
                    # Market metrics
                    "target_price": self._convert_to_json_serializable(
                        record.get("target_price")
                    ),
                    "analyst_count": self._convert_to_json_serializable(
                        record.get("analyst_count")
                    ),
                    "short_ratio": self._convert_to_json_serializable(
                        record.get("short_ratio")
                    ),
                    "relative_volume": self._convert_to_json_serializable(
                        record.get("relative_volume")
                    ),
                    "recommendation": record.get("recommendation"),
                    # Composite scores
                    "composite_score": self._convert_to_json_serializable(
                        record.get("composite_score")
                    ),
                    "technical_score": self._convert_to_json_serializable(
                        record.get("technical_score")
                    ),
                    "sentiment_score": self._convert_to_json_serializable(
                        record.get("sentiment_score")
                    ),
                    "fundamental_score": self._convert_to_json_serializable(
                        record.get("fundamental_score")
                    ),
                }
                analysis_records.append(analysis_record)

            # Insert records in batches with improved error handling
            batch_size = 100
            success_count = 0
            error_records = []

            for i in range(0, len(analysis_records), batch_size):
                batch = analysis_records[i : i + batch_size]
                try:
                    response = (
                        self.supabase.table("sentiment_analysis")
                        .upsert(batch, on_conflict="ticker,analysis_timestamp")
                        .execute()
                    )
                    success_count += len(batch)
                except Exception as e:
                    logger.error(f"Error saving batch {i//batch_size + 1}: {str(e)}")
                    error_records.extend([(r["ticker"], str(e)) for r in batch])

            return {
                "success": True,
                "records_processed": len(analysis_records),
                "records_saved": success_count,
                "failed_records": error_records,
                "timestamp": dt.datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error saving sentiment analysis to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_post_data(self, post_data: Dict) -> Dict:
        """Save post data to Supabase tables with improved error handling"""
        if self.write_controller == "False":
            return {"success": False, "error": "Write controller is disabled"}
        try:
            # Validate required fields
            required_fields = ["post_id", "title", "subreddit"]
            for field in required_fields:
                if not post_data.get(field):
                    raise ValueError(f"Missing required field: {field}")

            # Prepare post record
            post_record = {
                "post_id": str(post_data["post_id"]),
                "title": str(post_data["title"]),
                "content": str(post_data.get("content", "")),
                "url": str(post_data.get("url", "")),
                "author": str(post_data.get("author", "")),
                "score": self._convert_to_json_serializable(post_data.get("score", 0)),
                "num_comments": self._convert_to_json_serializable(
                    post_data.get("num_comments", 0)
                ),
                "upvote_ratio": self._convert_to_json_serializable(
                    post_data.get("upvote_ratio", 0)
                ),
                "created_at": self._convert_to_json_serializable(
                    post_data["created_utc"]
                ),
                "subreddit": str(post_data["subreddit"]),
                "avg_sentiment": self._convert_to_json_serializable(
                    post_data.get("avg_sentiment", {}).get("compound")
                    if isinstance(post_data.get("avg_sentiment"), dict)
                    else post_data.get("avg_sentiment")
                ),
                "submission_sentiment": (
                    post_data.get("submission_sentiment")
                    if isinstance(post_data.get("submission_sentiment"), dict)
                    else None
                ),
                "avg_base_sentiment": self._convert_to_json_serializable(
                    post_data.get("avg_base_sentiment")
                ),
                "avg_weighted_sentiment": self._convert_to_json_serializable(
                    post_data.get("avg_weighted_sentiment")
                ),
            }

            # Upsert post record with error handling
            try:
                self.supabase.table("reddit_posts").upsert(post_record).execute()
            except Exception as e:
                logger.error(f"Error saving post record: {str(e)}")
                return {"success": False, "error": f"Failed to save post: {str(e)}"}

            # Save tickers with error handling
            ticker_errors = []
            if post_data.get("tickers"):
                ticker_records = [
                    {
                        "post_id": post_data["post_id"],
                        "ticker": str(ticker),
                        "mentioned_at": self._convert_to_json_serializable(
                            post_data["created_utc"]
                        ),
                    }
                    for ticker in post_data["tickers"]
                ]
                try:
                    self.supabase.table("post_tickers").upsert(ticker_records).execute()
                except Exception as e:
                    error_msg = f"Error saving ticker records: {str(e)}"
                    logger.error(error_msg)
                    ticker_errors.append(error_msg)

            # Save comments with improved error handling
            comment_errors = []
            if post_data.get("comments"):
                comment_records = []
                for comment in post_data["comments"]:
                    try:
                        # Process sentiment data
                        sentiment_data = comment.get("sentiment", {})
                        if isinstance(sentiment_data, (float, int)):
                            sentiment_json = {
                                "compound": sentiment_data,
                                "pos": None,
                                "neg": None,
                                "neu": None,
                                "features": {},
                            }
                        else:
                            sentiment_json = {
                                "compound": sentiment_data.get("compound"),
                                "pos": sentiment_data.get("pos"),
                                "neg": sentiment_data.get("neg"),
                                "neu": sentiment_data.get("neu"),
                                "features": sentiment_data.get("features", {}),
                            }

                        comment_record = {
                            "post_id": post_data["post_id"],
                            "author": str(comment["author"]),
                            "content": str(comment["body"]),
                            "score": self._convert_to_json_serializable(
                                comment["score"]
                            ),
                            "created_at": self._convert_to_json_serializable(
                                comment["created_utc"]
                            ),
                            "sentiment": sentiment_json,
                        }
                        comment_records.append(comment_record)
                    except Exception as e:
                        error_msg = f"Error processing comment: {str(e)}"
                        logger.error(error_msg)
                        comment_errors.append(error_msg)

                if comment_records:
                    try:
                        # Save comments in batches to prevent timeout
                        batch_size = 50
                        for i in range(0, len(comment_records), batch_size):
                            batch = comment_records[i : i + batch_size]
                            self.supabase.table("post_comments").upsert(batch).execute()
                    except Exception as e:
                        error_msg = f"Error saving comment batch: {str(e)}"
                        logger.error(error_msg)
                        comment_errors.append(error_msg)

            return {
                "success": True,
                "post_id": post_data["post_id"],
                "errors": {
                    "ticker_errors": ticker_errors,
                    "comment_errors": comment_errors,
                },
            }

        except Exception as e:
            logger.error(f"Error saving post to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_sentiment_trends(
        self,
        ticker: str,
        days: int = 30,
        include_technicals: bool = True,
        include_fundamentals: bool = True,
    ) -> pd.DataFrame:
        """Get enhanced sentiment trends for a specific ticker with error handling"""
        try:
            cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()

            # Build select statement based on requirements
            select_columns = [
                "analysis_timestamp",
                "ticker",
                "comment_sentiment_avg",
                "bullish_comments_ratio",
                "bearish_comments_ratio",
                "sentiment_confidence",
                "price_change_1d",
                "price_change_1w",
                "volume_change",
                "composite_score",
                "sentiment_score",
            ]

            if include_technicals:
                select_columns.extend(
                    [
                        "rsi",
                        "macd_line",
                        "signal_line",
                        "macd_histogram",
                        "stoch_k",
                        "stoch_d",
                        "volatility",
                        "technical_score",
                        "bollinger_upper",
                        "bollinger_lower",
                        "sma_20",
                        "ema_9",
                        "volume_sma",
                        "volume_ratio",
                    ]
                )

            if include_fundamentals:
                select_columns.extend(
                    [
                        "market_cap",
                        "pe_ratio",
                        "beta",
                        "dividend_yield",
                        "profit_margins",
                        "revenue_growth",
                        "target_price",
                        "recommendation",
                        "analyst_count",
                        "short_ratio",
                        "relative_volume",
                        "fundamental_score",
                    ]
                )

            result = (
                self.supabase.table("sentiment_analysis")
                .select(",".join(select_columns))
                .eq("ticker", ticker)
                .gte("analysis_timestamp", cutoff_date)
                .order("analysis_timestamp", desc=True)
                .execute()
            )

            df = pd.DataFrame(result.data)
            if df.empty:
                logger.warning(f"No sentiment trends found for ticker {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving sentiment trends for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_recent_posts_by_ticker(
        self,
        ticker: str,
        days: int = 7,
        limit: int = 100,
        include_comments: bool = False,
    ) -> List[Dict]:
        """Get recent posts for a specific ticker with optional comment inclusion"""
        try:
            cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()

            # Build the query
            query = (
                self.supabase.table("post_tickers")
                .select(
                    "*, reddit_posts(*),"
                    + ("post_comments(*)" if include_comments else "")
                )
                .eq("ticker", ticker)
                .gte("mentioned_at", cutoff_date)
                .order("mentioned_at", desc=True)
                .limit(limit)
            )

            result = query.execute()

            # Process the results
            posts = result.data
            if not posts:
                logger.info(f"No recent posts found for ticker {ticker}")
                return []

            # Transform the data structure if needed
            processed_posts = []
            for post in posts:
                processed_post = {
                    **post["reddit_posts"],
                    "ticker_mention": {
                        "ticker": post["ticker"],
                        "mentioned_at": post["mentioned_at"],
                    },
                }
                if include_comments:
                    processed_post["comments"] = post["post_comments"]
                processed_posts.append(processed_post)

            return processed_posts

        except Exception as e:
            logger.error(f"Error retrieving posts for {ticker}: {str(e)}")
            return []

    def get_post_comments(self, post_id: str) -> List[Dict]:
        """Get all comments for a specific post"""
        try:
            result = (
                self.supabase.table("post_comments")
                .select("*")
                .eq("post_id", post_id)
                .order("created_at", desc=True)
                .execute()
            )

            return result.data

        except Exception as e:
            logger.error(f"Error retrieving comments for post {post_id}: {str(e)}")
            return []

    def get_ticker_mentions(self, ticker: str, days: int = 30) -> Dict:
        """Get mention statistics for a specific ticker"""
        try:
            cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()

            result = (
                self.supabase.table("post_tickers")
                .select("ticker, reddit_posts(score, num_comments, avg_sentiment)")
                .eq("ticker", ticker)
                .gte("mentioned_at", cutoff_date)
                .execute()
            )

            if not result.data:
                return {
                    "ticker": ticker,
                    "mention_count": 0,
                    "total_score": 0,
                    "total_comments": 0,
                    "avg_sentiment": 0,
                }

            # Calculate statistics
            mentions = result.data
            total_score = sum(
                m["reddit_posts"]["score"] for m in mentions if m["reddit_posts"]
            )
            total_comments = sum(
                m["reddit_posts"]["num_comments"] for m in mentions if m["reddit_posts"]
            )
            sentiments = [
                m["reddit_posts"]["avg_sentiment"]
                for m in mentions
                if m["reddit_posts"] and m["reddit_posts"]["avg_sentiment"]
            ]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            return {
                "ticker": ticker,
                "mention_count": len(mentions),
                "total_score": total_score,
                "total_comments": total_comments,
                "avg_sentiment": avg_sentiment,
            }

        except Exception as e:
            logger.error(f"Error retrieving mentions for ticker {ticker}: {str(e)}")
            return {"ticker": ticker, "error": str(e)}

    def save_llm_sentiment(self, sentiment_data: Dict) -> Dict:
        """
        Save LLM sentiment analysis results to Supabase

        Args:
            sentiment_data: Dictionary containing sentiment analysis data with fields:
                - text_hash: SHA256 hash of analyzed text
                - original_text: Original text analyzed
                - sentiment_score: Float between -1 and 1
                - confidence_score: Float between 0 and 1
                - features: Dict of identified features
                - terms: List of financial terms
                - raw_response: Complete OpenAI response
                - processing_time_ms: Processing time in milliseconds
                - model_version: Version of the model used

        Returns:
            Dict with operation status
        """
        try:
            if not sentiment_data.get("text_hash"):
                return {"success": False, "error": "Missing text_hash"}

            sentiment_record = {
                "text_hash": sentiment_data["text_hash"],
                "original_text": sentiment_data.get("original_text", ""),
                "analysis_timestamp": dt.datetime.now().isoformat(),
                "sentiment_score": self._convert_to_json_serializable(
                    sentiment_data.get("sentiment_score", 0)
                ),
                "confidence_score": self._convert_to_json_serializable(
                    sentiment_data.get("confidence_score", 0)
                ),
                "features": sentiment_data.get("features", {}),
                "terms": sentiment_data.get("terms", []),
                "raw_response": sentiment_data.get("raw_response", {}),
                "processing_time_ms": self._convert_to_json_serializable(
                    sentiment_data.get("processing_time_ms", 0)
                ),
                "model_version": sentiment_data.get("model_version", "unknown"),
            }

            # Upsert to Supabase
            response = (
                self.supabase.table("llm_sentiment_analysis")
                .upsert(sentiment_record, on_conflict="text_hash")
                .execute()
            )

            return {
                "success": True,
                "text_hash": sentiment_data["text_hash"],
                "timestamp": dt.datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error saving LLM sentiment to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_llm_sentiment(self, text_hash: str) -> Optional[Dict]:
        """
        Retrieve stored LLM sentiment analysis by text hash

        Args:
            text_hash: SHA256 hash of the text

        Returns:
            Dict containing sentiment analysis or None if not found
        """
        try:
            result = (
                self.supabase.table("llm_sentiment_analysis")
                .select("*")
                .eq("text_hash", text_hash)
                .execute()
            )

            if result.data:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error retrieving LLM sentiment: {str(e)}")
            return None

    def save_to_storage(
        self, file_data: Dict, bucket_name: str = "analysis-results"
    ) -> Dict:
        """
        Save data to Supabase storage bucket

        Args:
            file_data: Dictionary containing:
                - content: Bytes or string content to save
                - path: Path within bucket
                - content_type: MIME type of content
            bucket_name: Name of the storage bucket

        Returns:
            Dict with operation status
        """
        if self.write_controller == "False":
            return {"success": False, "error": "Write controller is disabled"}
        try:
            content = file_data.get("content")
            path = file_data.get("path")
            content_type = file_data.get("content_type", "application/octet-stream")

            if not all([content, path]):
                return {
                    "success": False,
                    "error": "Missing required fields: content and path",
                }

            # Convert string content to bytes if necessary
            if isinstance(content, str):
                content = content.encode()

            # Upload to storage
            result = self.supabase.storage.from_(bucket_name).upload(
                path, content, {"content-type": content_type}
            )

            return {
                "success": True,
                "path": path,
                "bucket": bucket_name,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error saving to storage bucket: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_analysis_to_storage(
        self, df: pd.DataFrame, bucket_name: str = "analysis-results"
    ) -> Dict:
        """
        Save analysis results to Supabase storage bucket

        Args:
            df: DataFrame with analysis results
            bucket_name: Name of the storage bucket

        Returns:
            Dict with operation results
        """
        if self.write_controller == "False":
            return {"success": False, "error": "Write controller is disabled"}
        if df.empty:
            return {"success": False, "error": "Empty DataFrame provided"}

        try:
            timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = f"analysis/{timestamp_str}"

            upload_results = {}

            # Save CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_result = self.save_to_storage(
                {
                    "content": csv_buffer.getvalue(),
                    "path": f"{base_path}/analysis.csv",
                    "content_type": "text/csv",
                },
                bucket_name,
            )
            upload_results["csv"] = csv_result

            # Save JSON summary
            summary_data = {
                "analysis_timestamp": dt.datetime.now().isoformat(),
                "metadata": {
                    "total_stocks": len(df),
                    "average_sentiment": (
                        float(df["comment_sentiment_avg"].mean())
                        if "comment_sentiment_avg" in df.columns
                        else 0.0
                    ),
                    "total_comments": (
                        int(df["num_comments"].sum())
                        if "num_comments" in df.columns
                        else 0
                    ),
                },
                "stocks": df.to_dict(orient="records"),
            }

            json_result = self.save_to_storage(
                {
                    "content": json.dumps(summary_data, indent=2),
                    "path": f"{base_path}/summary.json",
                    "content_type": "application/json",
                },
                bucket_name,
            )
            upload_results["json"] = json_result

            # Save metrics summary
            metrics_data = {
                "most_bullish": df.nlargest(5, "comment_sentiment_avg")[
                    ["ticker", "comment_sentiment_avg"]
                ].to_dict(orient="records"),
                "most_bearish": df.nsmallest(5, "comment_sentiment_avg")[
                    ["ticker", "comment_sentiment_avg"]
                ].to_dict(orient="records"),
                "most_active": df.nlargest(5, "num_comments")[
                    ["ticker", "num_comments"]
                ].to_dict(orient="records"),
                "best_performing": df.nlargest(5, "price_change_1w")[
                    ["ticker", "price_change_1w"]
                ].to_dict(orient="records"),
            }

            metrics_result = self.save_to_storage(
                {
                    "content": json.dumps(metrics_data, indent=2),
                    "path": f"{base_path}/metrics.json",
                    "content_type": "application/json",
                },
                bucket_name,
            )
            upload_results["metrics"] = metrics_result

            # Save record of this analysis run
            metadata = {
                "timestamp": dt.datetime.now().isoformat(),
                "base_path": base_path,
                "files_uploaded": [
                    result["path"]
                    for result in upload_results.values()
                    if result["success"]
                ],
                "total_stocks": len(df),
                "bucket_name": bucket_name,
            }

            try:
                self.supabase.table("analysis_runs").insert(metadata).execute()
            except Exception as e:
                logger.error(f"Error saving analysis metadata: {str(e)}")

            return {
                "success": True,
                "base_path": base_path,
                "upload_results": upload_results,
                "timestamp": dt.datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error saving analysis to storage: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_analysis_to_storage(
        self, df: pd.DataFrame, bucket_name: str = "analysis-results"
    ) -> Dict:
        """
        Save analysis results to Supabase storage bucket

        Args:
            df: DataFrame with analysis results
            bucket_name: Name of the storage bucket

        Returns:
            Dict with operation results
        """
        if self.write_controller == "False":
            return {"success": False, "error": "Write controller is disabled"}
        if df.empty:
            return {"success": False, "error": "Empty DataFrame provided"}

        try:
            timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = f"analysis/{timestamp_str}"

            upload_results = {}

            # Save CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_result = self.save_to_storage(
                {
                    "content": csv_buffer.getvalue(),
                    "path": f"{base_path}/analysis.csv",
                    "content_type": "text/csv",
                },
                bucket_name,
            )
            upload_results["csv"] = csv_result

            # Save JSON summary using custom encoder
            summary_data = {
                "analysis_timestamp": dt.datetime.now(),
                "metadata": {
                    "total_stocks": len(df),
                    "average_sentiment": (
                        float(df["comment_sentiment_avg"].mean())
                        if "comment_sentiment_avg" in df.columns
                        else 0.0
                    ),
                    "total_comments": (
                        int(df["num_comments"].sum())
                        if "num_comments" in df.columns
                        else 0
                    ),
                },
                "stocks": df.to_dict(orient="records"),
            }

            json_result = self.save_to_storage(
                {
                    "content": json.dumps(
                        summary_data, indent=2, cls=CustomJSONEncoder
                    ),
                    "path": f"{base_path}/summary.json",
                    "content_type": "application/json",
                },
                bucket_name,
            )
            upload_results["json"] = json_result

            # Save metrics summary
            metrics_data = {
                "most_bullish": df.nlargest(5, "comment_sentiment_avg")[
                    ["ticker", "comment_sentiment_avg"]
                ].to_dict(orient="records"),
                "most_bearish": df.nsmallest(5, "comment_sentiment_avg")[
                    ["ticker", "comment_sentiment_avg"]
                ].to_dict(orient="records"),
                "most_active": df.nlargest(5, "num_comments")[
                    ["ticker", "num_comments"]
                ].to_dict(orient="records"),
                "best_performing": df.nlargest(5, "price_change_1w")[
                    ["ticker", "price_change_1w"]
                ].to_dict(orient="records"),
            }

            metrics_result = self.save_to_storage(
                {
                    "content": json.dumps(
                        metrics_data, indent=2, cls=CustomJSONEncoder
                    ),
                    "path": f"{base_path}/metrics.json",
                    "content_type": "application/json",
                },
                bucket_name,
            )
            upload_results["metrics"] = metrics_result

            # Save metadata
            metadata = {
                "timestamp": dt.datetime.now(),
                "base_path": base_path,
                "files_uploaded": [
                    result["path"]
                    for result in upload_results.values()
                    if result["success"]
                ],
                "total_stocks": len(df),
                "bucket_name": bucket_name,
            }

            try:
                self.supabase.table("analysis_runs").insert(
                    json.loads(json.dumps(metadata, cls=CustomJSONEncoder))
                ).execute()
            except Exception as e:
                logger.error(f"Error saving analysis metadata: {str(e)}")

            return {
                "success": True,
                "base_path": base_path,
                "upload_results": upload_results,
                "timestamp": dt.datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error saving analysis to storage: {str(e)}")
            return {"success": False, "error": str(e)}

    def list_analysis_runs(self, limit: int = 10) -> List[Dict]:
        """
        List recent analysis runs

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of analysis run metadata
        """
        try:
            result = (
                self.supabase.table("analysis_runs")
                .select("*")
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data

        except Exception as e:
            logger.error(f"Error listing analysis runs: {str(e)}")
            return []
