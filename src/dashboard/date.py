import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))


def safe_format(value, format_str=".2f", default="N/A"):
    """Safely format a value with proper null handling"""
    try:
        if pd.isna(value) or value is None:
            return default
        return format(float(value), format_str)
    except (ValueError, TypeError):
        return default


def load_sentiment_data(start_date, end_date):
    """Load sentiment analysis data from Supabase"""
    try:
        # Add a day to end_date to include the full current day
        query_end_date = (pd.to_datetime(end_date) + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        response = (
            supabase.table("sentiment_analysis")
            .select("*")
            .gte("analysis_timestamp", start_date)
            .lt("analysis_timestamp", query_end_date)
            .execute()
        )

        if response.data:
            df = pd.DataFrame(response.data)
            # Convert numeric columns and handle NaN values
            numeric_columns = [
                "comment_sentiment_avg",
                "base_sentiment",
                "submission_sentiment",
                "bullish_comments_ratio",
                "bearish_comments_ratio",
                "composite_score",
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading sentiment data: {str(e)}")
        return pd.DataFrame()


def load_post_data(tickers, start_date, end_date):
    """Load related post data for selected tickers"""
    try:
        # Get post IDs for the tickers
        post_ids_response = (
            supabase.table("post_tickers")
            .select("post_id")
            .in_("ticker", tickers)
            .gte("mentioned_at", start_date)
            .lte("mentioned_at", end_date)
            .execute()
        )

        if not post_ids_response.data:
            return pd.DataFrame()

        post_ids = [p["post_id"] for p in post_ids_response.data]

        # Get post details
        posts_response = (
            supabase.table("reddit_posts")
            .select("*")
            .in_("post_id", post_ids)
            .execute()
        )

        df = pd.DataFrame(posts_response.data)

        # Convert numeric columns and handle NaN values
        if not df.empty:
            numeric_cols = [
                "avg_sentiment",
                "avg_base_sentiment",
                "avg_weighted_sentiment",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        st.error(f"Error loading post data: {str(e)}")
        return pd.DataFrame()


def create_sentiment_heatmap(df, metric, title):
    """Create a heatmap visualization for sentiment metrics"""
    try:
        # Clean data for heatmap
        df = df.copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

        # Pivot data for heatmap
        pivot_data = df.pivot_table(
            values=metric,
            index=pd.to_datetime(df["analysis_timestamp"]).dt.date,
            columns="ticker",
            aggfunc="mean",
        ).fillna(
            0
        )  # Fill NaN values with 0 for visualization

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale="RdYlBu",
                colorbar_title=metric,
            )
        )

        fig.update_layout(
            title=title, xaxis_title="Ticker", yaxis_title="Date", height=400
        )

        return fig
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide")

    st.title("Stock Sentiment Analysis Dashboard")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", value=datetime.now() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    # Load data
    df = load_sentiment_data(start_date, end_date)

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Posts"])

    with tab1:
        st.header("Market Overview")

        # Summary metrics with null handling
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Stocks Analyzed", len(df["ticker"].unique()))
        with col2:
            avg_sentiment = df["comment_sentiment_avg"].mean()
            st.metric("Average Sentiment", safe_format(avg_sentiment))
        with col3:
            total_comments = df["num_comments"].sum()
            st.metric(
                "Total Posts Analyzed",
                f"{total_comments:,}" if pd.notnull(total_comments) else "N/A",
            )
        with col4:
            bullish_ratio = df["bullish_comments_ratio"].mean()
            st.metric("Bullish Ratio", safe_format(bullish_ratio, ".1%"))

        # Sentiment heatmap with error handling
        heatmap = create_sentiment_heatmap(
            df, "comment_sentiment_avg", "Sentiment Heatmap"
        )
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

    with tab2:
        st.header("Detailed Stock Analysis")

        # Most Active Stocks
        st.subheader("Most Active Stocks")
        active_stocks = (
            df.groupby("ticker")
            .agg(
                {
                    "num_comments": "sum",
                    "comment_sentiment_avg": "mean",
                    "bullish_comments_ratio": "mean",
                    "bearish_comments_ratio": "mean",
                    "composite_score": "mean",
                }
            )
            .fillna(0)
            .sort_values("num_comments", ascending=False)
        )

        fig_active = px.bar(
            active_stocks.head(10),
            x=active_stocks.head(10).index,
            y="num_comments",
            title="Most Discussed Stocks",
            color="comment_sentiment_avg",
            color_continuous_scale="RdYlBu",
        )
        st.plotly_chart(fig_active, use_container_width=True)

        # Filter for minimum comments
        min_comments = (
            df["num_comments"].mean() if not df["num_comments"].isna().all() else 0
        )
        qualified_stocks = active_stocks[
            active_stocks["num_comments"] > min_comments
        ].copy()

        # Calculate sentiment ratio for better classification
        qualified_stocks["sentiment_ratio"] = qualified_stocks[
            "bullish_comments_ratio"
        ] / (
            qualified_stocks["bearish_comments_ratio"]
            + 0.0001  # Avoid division by zero
        )

        # Identify bullish stocks first
        bullish_stocks = (
            qualified_stocks[
                qualified_stocks["sentiment_ratio"] > 1.5  # Clearly bullish
            ]
            .sort_values("bullish_comments_ratio", ascending=False)
            .head(10)
        )

        # Remove bullish tickers from consideration for bearish
        remaining_stocks = qualified_stocks[
            ~qualified_stocks.index.isin(bullish_stocks.index)
        ]

        # Identify bearish stocks from remaining
        bearish_stocks = (
            remaining_stocks[
                remaining_stocks["sentiment_ratio"] < 0.67  # Clearly bearish (1/1.5)
            ]
            .sort_values("bearish_comments_ratio", ascending=False)
            .head(10)
        )

        # Display Bullish Stocks
        st.subheader("Most Bullish Stocks")
        if not bullish_stocks.empty:
            fig_bullish = px.bar(
                bullish_stocks,
                x=bullish_stocks.index,
                y="bullish_comments_ratio",
                title=f"Most Bullish Stocks (Minimum {min_comments:.0f} comments)",
                color="comment_sentiment_avg",
                color_continuous_scale="RdYlBu",
            )
            fig_bullish.add_trace(
                go.Scatter(
                    x=bullish_stocks.index,
                    y=bullish_stocks["num_comments"]
                    / bullish_stocks["num_comments"].max(),
                    name="Relative Volume",
                    yaxis="y2",
                    line=dict(color="black", dash="dot"),
                )
            )
            fig_bullish.update_layout(
                yaxis2=dict(
                    title="Relative Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                )
            )
            st.plotly_chart(fig_bullish, use_container_width=True)

            # Display additional metrics for bullish stocks
            st.write("Bullish Stocks Details:")
            bullish_details = bullish_stocks[
                [
                    "num_comments",
                    "comment_sentiment_avg",
                    "bullish_comments_ratio",
                    "bearish_comments_ratio",
                ]
            ].round(3)
            st.dataframe(bullish_details, use_container_width=True)
        else:
            st.info("No stocks met the bullish criteria in this period.")

        # Display Bearish Stocks
        st.subheader("Most Bearish Stocks")
        if not bearish_stocks.empty:
            fig_bearish = px.bar(
                bearish_stocks,
                x=bearish_stocks.index,
                y="bearish_comments_ratio",
                title=f"Most Bearish Stocks (Minimum {min_comments:.0f} comments)",
                color="comment_sentiment_avg",
                color_continuous_scale="RdYlBu",
            )
            fig_bearish.add_trace(
                go.Scatter(
                    x=bearish_stocks.index,
                    y=bearish_stocks["num_comments"]
                    / bearish_stocks["num_comments"].max(),
                    name="Relative Volume",
                    yaxis="y2",
                    line=dict(color="black", dash="dot"),
                )
            )
            fig_bearish.update_layout(
                yaxis2=dict(
                    title="Relative Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                )
            )
            st.plotly_chart(fig_bearish, use_container_width=True)

            # Display additional metrics for bearish stocks
            st.write("Bearish Stocks Details:")
            bearish_details = bearish_stocks[
                [
                    "num_comments",
                    "comment_sentiment_avg",
                    "bullish_comments_ratio",
                    "bearish_comments_ratio",
                ]
            ].round(3)
            st.dataframe(bearish_details, use_container_width=True)
        else:
            st.info("No stocks met the bearish criteria in this period.")

    with tab3:
        st.header("Reddit Posts Analysis")

        # Select stocks to view posts for
        selected_tickers = st.multiselect(
            "Select Stocks to View Posts",
            options=df["ticker"].unique(),
            default=active_stocks.index[:3].tolist(),
        )

        if selected_tickers:
            posts_df = load_post_data(selected_tickers, start_date, end_date)

            if not posts_df.empty:
                for _, post in posts_df.iterrows():
                    with st.expander(
                        f"{post.get('title', 'No Title')} ({post.get('subreddit', 'Unknown')})"
                    ):
                        st.write(
                            f"**Score:** {post.get('score', 'N/A')} | **Comments:** {post.get('num_comments', 'N/A')}"
                        )

                        # Safe sentiment display
                        sentiment = safe_format(post.get("avg_sentiment"))
                        st.write(f"**Sentiment:** {sentiment}")

                        # Safe content display
                        content = (
                            post.get("content", "").strip()
                            if post.get("content")
                            else "No content available"
                        )
                        st.write(f"**Content:**\n{content}")

                        # Safe author and date display
                        author = post.get("author", "Unknown")
                        created_at = post.get("created_at", "Unknown date")
                        st.write(f"**Posted by:** u/{author} on {created_at}")
            else:
                st.info("No posts found for the selected stocks in this date range.")
        else:
            st.info("Select stocks to view their Reddit posts.")


if __name__ == "__main__":
    main()
