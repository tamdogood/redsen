from supabase import create_client, Client
from typing import Dict, List, Optional, Union
import pandas as pd
from utils.logging_config import logger
import datetime as dt
import json
import numpy as np


class SupabaseConnector:
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase connection"""
        self.supabase: Client = create_client(supabase_url, supabase_key)

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

    def save_post_data(self, post_data: Dict) -> Dict:
        """Save post data to Supabase tables"""
        try:
            # Validate required fields
            required_fields = ["post_id", "title", "subreddit"]
            for field in required_fields:
                if not post_data.get(field):
                    raise ValueError(f"Missing required field: {field}")

            # Prepare post record with correct sentiment handling
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
                # Convert sentiment values correctly
                "avg_sentiment": self._convert_to_json_serializable(
                    post_data.get("avg_sentiment", {}).get("compound")
                    if isinstance(post_data.get("avg_sentiment"), dict)
                    else post_data.get("avg_sentiment")
                ),
                "submission_sentiment": (
                    json.dumps(post_data.get("submission_sentiment", {}))
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

            # Upsert post record
            self.supabase.table("reddit_posts").upsert(post_record).execute()

            # Save tickers
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
                self.supabase.table("post_tickers").upsert(ticker_records).execute()

            # Save comments with correct sentiment handling
            if post_data.get("comments"):
                comment_records = []
                for comment in post_data["comments"]:
                    # Ensure sentiment is properly formatted as JSON
                    sentiment_data = comment.get("sentiment", {})
                    if isinstance(sentiment_data, (float, int)):
                        # If sentiment is just a number, create proper structure
                        sentiment_json = {
                            "compound": sentiment_data,
                            "pos": None,
                            "neg": None,
                            "neu": None,
                            "features": {},
                        }
                    else:
                        # Use existing sentiment structure
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
                        "score": self._convert_to_json_serializable(comment["score"]),
                        "created_at": self._convert_to_json_serializable(
                            comment["created_utc"]
                        ),
                        "sentiment": json.dumps(sentiment_json),
                    }
                    comment_records.append(comment_record)

                if comment_records:
                    self.supabase.table("post_comments").upsert(
                        comment_records
                    ).execute()

            return {"success": True, "post_id": post_data["post_id"]}

        except Exception as e:
            logger.error(f"Error saving post to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_sentiment_analysis(self, sentiment_data: pd.DataFrame) -> Dict:
        """Save sentiment analysis results to Supabase"""
        try:
            if sentiment_data.empty:
                return {"success": False, "error": "Empty DataFrame provided"}

            analysis_records = []
            for _, record in sentiment_data.iterrows():
                analysis_record = {
                    "analysis_timestamp": self._convert_to_json_serializable(
                        record["analysis_timestamp"]
                    ),
                    "data_start_date": self._convert_to_json_serializable(
                        record["data_start_date"]
                    ),
                    "data_end_date": self._convert_to_json_serializable(
                        record["data_end_date"]
                    ),
                    "ticker": str(record["ticker"]),
                    "score": self._convert_to_json_serializable(record.get("score")),
                    "num_comments": self._convert_to_json_serializable(
                        record.get("num_comments")
                    ),
                    "comment_sentiment_avg": self._convert_to_json_serializable(
                        record.get("comment_sentiment_avg")
                    ),
                    "bullish_comments_ratio": self._convert_to_json_serializable(
                        record.get("bullish_comments_ratio")
                    ),
                    "bearish_comments_ratio": self._convert_to_json_serializable(
                        record.get("bearish_comments_ratio")
                    ),
                    "current_price": self._convert_to_json_serializable(
                        record.get("current_price")
                    ),
                    "price_change_2w": self._convert_to_json_serializable(
                        record.get("price_change_2w")
                    ),
                    "price_change_2d": self._convert_to_json_serializable(
                        record.get("price_change_2d")
                    ),
                    "avg_volume": self._convert_to_json_serializable(
                        record.get("avg_volume")
                    ),
                    "volume_change": self._convert_to_json_serializable(
                        record.get("volume_change")
                    ),
                    "sma_20": self._convert_to_json_serializable(record.get("sma_20")),
                    "rsi": self._convert_to_json_serializable(record.get("rsi")),
                    "volatility": self._convert_to_json_serializable(
                        record.get("volatility")
                    ),
                    "market_cap": self._convert_to_json_serializable(
                        record.get("market_cap")
                    ),
                    "pe_ratio": self._convert_to_json_serializable(
                        record.get("pe_ratio")
                    ),
                    "composite_score": self._convert_to_json_serializable(
                        record.get("composite_score")
                    ),
                }
                analysis_records.append(analysis_record)

            # Insert records in batches
            batch_size = 100
            for i in range(0, len(analysis_records), batch_size):
                batch = analysis_records[i : i + batch_size]
                self.supabase.table("sentiment_analysis").upsert(
                    batch, on_conflict="ticker,analysis_timestamp"
                ).execute()

            return {"success": True, "records_processed": len(analysis_records)}

        except Exception as e:
            logger.error(f"Error saving sentiment analysis to Supabase: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_recent_posts_by_ticker(
        self, ticker: str, days: int = 7, limit: int = 100
    ) -> List[Dict]:
        """Get recent posts for a specific ticker"""
        try:
            cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()

            result = (
                self.supabase.table("post_tickers")
                .select("*, reddit_posts(*)")
                .eq("ticker", ticker)
                .gte("mentioned_at", cutoff_date)
                .order("mentioned_at", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data

        except Exception as e:
            logger.error(f"Error retrieving posts for {ticker}: {str(e)}")
            return []

    def get_sentiment_trends(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get sentiment trends for a specific ticker"""
        try:
            cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()

            result = (
                self.supabase.table("sentiment_analysis")
                .select("*")
                .eq("ticker", ticker)
                .gte("analysis_timestamp", cutoff_date)
                .order("analysis_timestamp", desc=True)
                .execute()
            )

            return pd.DataFrame(result.data)

        except Exception as e:
            logger.error(f"Error retrieving sentiment trends for {ticker}: {str(e)}")
            return pd.DataFrame()
