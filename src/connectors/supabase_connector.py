from supabase import create_client, Client
from typing import Dict, List
import pandas as pd
from utils.logging_config import logger
import datetime as dt


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
            Dict with operation dt
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
