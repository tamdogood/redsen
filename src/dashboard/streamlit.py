import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables and initialize Supabase client (same as before)
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))


def safe_format(value, format_str=".2f", default="N/A"):
    try:
        if pd.isna(value) or value is None:
            return default
        return format(float(value), format_str)
    except (ValueError, TypeError):
        return default


def load_sentiment_data(start_date, end_date):
    try:
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


def create_sentiment_heatmap(df, metric, title):
    """Create an improved, more readable heatmap visualization for sentiment metrics"""
    try:
        # Clean and prepare data
        df = df.copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

        # Convert timestamp to date and create a clean date column
        df["date"] = pd.to_datetime(df["analysis_timestamp"]).dt.date

        # Get the top N most active stocks for better visibility
        top_stocks = (
            df.groupby("ticker")["num_comments"]
            .sum()
            .sort_values(ascending=False)
            .head(30)  # Limit to top 15 most active stocks
            .index.tolist()
        )

        # Filter for top stocks
        df_filtered = df[df["ticker"].isin(top_stocks)]

        # Pivot data for heatmap
        pivot_data = df_filtered.pivot_table(
            values=metric, index="date", columns="ticker", aggfunc="mean"
        ).round(3)

        # Sort columns by total activity
        column_order = (
            df_filtered.groupby("ticker")["num_comments"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        pivot_data = pivot_data[column_order]

        # Create heatmap with improved styling
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=[d.strftime("%Y-%m-%d") for d in pivot_data.index],
                colorscale=[
                    [0, "rgb(165,0,38)"],  # Strong negative - dark red
                    [0.25, "rgb(215,48,39)"],  # Negative - red
                    [0.5, "rgb(255,255,255)"],  # Neutral - white
                    [0.75, "rgb(34,94,168)"],  # Positive - blue
                    [1, "rgb(0,47,167)"],  # Strong positive - dark blue
                ],
                colorbar=dict(
                    title=dict(text="Sentiment Score", side="right"),
                    thickness=15,
                    len=0.8,
                    tickformat=".2f",
                ),
                hoverongaps=False,
                hovertemplate=(
                    "<b>Stock:</b> %{x}<br>"
                    + "<b>Date:</b> %{y}<br>"
                    + "<b>Sentiment:</b> %{z:.3f}<extra></extra>"
                ),
            )
        )

        # Update layout for better readability
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
            width=None,  # Allow dynamic width
            height=600,  # Increased height
            xaxis=dict(
                title="Ticker Symbol",
                tickangle=45,
                tickfont=dict(size=12),
                side="bottom",
                # gridcolor="lightgrey",
                # showgrid=True,
            ),
            yaxis=dict(
                title="Date",
                tickfont=dict(size=12),
                # gridcolor="lightgrey",
                # showgrid=True,
            ),
            # margin=dict(
            #     l=50,  # left margin
            #     r=50,  # right margin
            #     t=80,  # top margin
            #     b=80,  # bottom margin
            # ),
            paper_bgcolor="black",
            # plot_bgcolor="white",
        )

        # Add a text annotation explaining the color scale
        fig.add_annotation(
            text="Color indicates sentiment: Red (Negative) → White (Neutral) → Blue (Positive)",
            xref="paper",
            yref="paper",
            x=0,
            y=1.06,
            showarrow=False,
            font=dict(size=12),
            align="left",
        )

        return fig
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None
    
def load_post_data(tickers, start_date, end_date):
    try:
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

        posts_response = (
            supabase.table("reddit_posts")
            .select("*")
            .in_("post_id", post_ids)
            .execute()
        )

        df = pd.DataFrame(posts_response.data)

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


def create_stock_card(ticker_data, metrics):
    """Create a card layout for individual stock metrics"""
    st.markdown(f"### {ticker_data.name}")

    cols = st.columns(2)
    with cols[0]:
        st.metric("Comments", f"{int(ticker_data['num_comments']):,}")
        st.metric("Sentiment", safe_format(ticker_data["comment_sentiment_avg"]))
    with cols[1]:
        st.metric("Bullish", safe_format(ticker_data["bullish_comments_ratio"], ".1%"))
        st.metric("Bearish", safe_format(ticker_data["bearish_comments_ratio"], ".1%"))

    # Add sentiment trend if available
    if "sentiment_trend" in metrics and ticker_data.name in metrics["sentiment_trend"]:
        trend = metrics["sentiment_trend"][ticker_data.name]
        st.line_chart(trend)


def calculate_advanced_metrics(df):
    """Calculate additional insights and metrics"""
    metrics = {}

    # Calculate sentiment volatility
    metrics["sentiment_volatility"] = (
        df.groupby("ticker")["comment_sentiment_avg"].std().sort_values(ascending=False)
    )

    # Calculate sentiment momentum (change over time)
    df["date"] = pd.to_datetime(df["analysis_timestamp"]).dt.date
    sentiment_by_date = (
        df.groupby(["ticker", "date"])["comment_sentiment_avg"].mean().unstack()
    )
    metrics["sentiment_momentum"] = (
        sentiment_by_date.iloc[:, -1] - sentiment_by_date.iloc[:, 0]
    ).sort_values()

    # Calculate correlation between sentiment and volume
    metrics["sentiment_volume_corr"] = (
        df.groupby("ticker")
        .apply(lambda x: x["comment_sentiment_avg"].corr(x["num_comments"]))
        .sort_values()
    )

    # Track sentiment trends over time
    metrics["sentiment_trend"] = {
        ticker: group.set_index("date")["comment_sentiment_avg"]
        for ticker, group in df.groupby("ticker")
    }

    return metrics


def create_stock_card(ticker_data, metrics, detailed=False):
    """Create a card layout for individual stock metrics with optional detailed view"""
    st.markdown(f"### {ticker_data.name}")

    if detailed:
        # Detailed view with more metrics and larger charts
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Comments", f"{int(ticker_data['num_comments']):,}")
            st.metric(
                "Average Sentiment", safe_format(ticker_data["comment_sentiment_avg"])
            )
            st.metric("Composite Score", safe_format(ticker_data["composite_score"]))
        with col2:
            st.metric(
                "Bullish Ratio",
                safe_format(ticker_data["bullish_comments_ratio"], ".1%"),
            )
            st.metric(
                "Bearish Ratio",
                safe_format(ticker_data["bearish_comments_ratio"], ".1%"),
            )
            if "sentiment_volume_corr" in metrics:
                corr = metrics["sentiment_volume_corr"].get(ticker_data.name, 0)
                st.metric("Sentiment-Volume Correlation", safe_format(corr))

        # Add sentiment trend if available
        if (
            "sentiment_trend" in metrics
            and ticker_data.name in metrics["sentiment_trend"]
        ):
            st.subheader("Sentiment Trend")
            trend = metrics["sentiment_trend"][ticker_data.name]
            st.line_chart(trend)

            # Calculate and display additional statistics
            if not trend.empty:
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric(
                        "Trend Direction",
                        "↑ Upward" if trend.iloc[-1] > trend.iloc[0] else "↓ Downward",
                        delta=safe_format(trend.iloc[-1] - trend.iloc[0]),
                    )
                with stats_col2:
                    volatility = trend.std()
                    st.metric("Sentiment Volatility", safe_format(volatility))

        # Display any momentum information
        if "sentiment_momentum" in metrics:
            momentum = metrics["sentiment_momentum"].get(ticker_data.name, 0)
            st.metric("Sentiment Momentum", safe_format(momentum, "+.2f"))
    else:
        # Compact view for grid layout
        cols = st.columns(2)
        with cols[0]:
            st.metric("Comments", f"{int(ticker_data['num_comments']):,}")
            st.metric("Sentiment", safe_format(ticker_data["comment_sentiment_avg"]))
        with cols[1]:
            st.metric(
                "Bullish", safe_format(ticker_data["bullish_comments_ratio"], ".1%")
            )
            st.metric(
                "Bearish", safe_format(ticker_data["bearish_comments_ratio"], ".1%")
            )

        if (
            "sentiment_trend" in metrics
            and ticker_data.name in metrics["sentiment_trend"]
        ):
            trend = metrics["sentiment_trend"][ticker_data.name]
            st.line_chart(trend)


def main():
    st.set_page_config(page_title="Enhanced Stock Sentiment Dashboard", layout="wide")

    st.title("Stock Sentiment Analysis Dashboard")

    # Date range selector in a more compact layout
    cols = st.columns([1, 1, 2])
    with cols[0]:
        start_date = st.date_input(
            "Start Date", value=datetime.now() - timedelta(days=7)
        )
    with cols[1]:
        end_date = st.date_input("End Date", value=datetime.now())

    # Load and process data
    df = load_sentiment_data(start_date, end_date)
    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(df)

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Market Overview", "Stock Grid", "Advanced Insights", "Posts"]
    )

    with tab1:
        st.header("Market Overview")

        # Enhanced summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Stocks", len(df["ticker"].unique()))
        with col2:
            st.metric("Avg Sentiment", safe_format(df["comment_sentiment_avg"].mean()))
        with col3:
            st.metric("Total Posts", f"{df['num_comments'].sum():,}")
        with col4:
            st.metric(
                "Bullish Ratio", safe_format(df["bullish_comments_ratio"].mean(), ".1%")
            )
        with col5:
            sentiment_change = advanced_metrics["sentiment_momentum"].mean()
            st.metric("Sentiment Trend", safe_format(sentiment_change, "+.2f"))

        # Sentiment heatmap
        st.subheader("Sentiment Heatmap")
        heatmap = create_sentiment_heatmap(
            df, "comment_sentiment_avg", "Daily Sentiment by Stock"
        )
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

    with tab2:
        st.header("Stock Grid View")

        # Filter stocks by minimum activity
        col1, col2 = st.columns([2, 1])
        with col1:
            min_comments = st.slider(
                "Minimum Comments Filter", 1, int(df["num_comments"].max()), 5
            )

        # Calculate active stocks
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
            .query(f"num_comments >= {min_comments}")
        )

        # Sort options for the stocks
        sort_options = {
            "Most Comments": "num_comments",
            "Highest Sentiment": "comment_sentiment_avg",
            "Most Bullish": "bullish_comments_ratio",
            "Most Bearish": "bearish_comments_ratio",
            "Highest Composite Score": "composite_score",
        }

        with col2:
            sort_by = st.selectbox("Sort Stocks By", options=list(sort_options.keys()))

        # Sort stocks based on selected criteria
        active_stocks = active_stocks.sort_values(
            sort_options[sort_by], ascending=False
        )

        # Stock selector
        selected_stocks = st.multiselect(
            "Select Stocks to Display",
            options=active_stocks.index.tolist(),
            default=active_stocks.index[:3].tolist(),
            help="Choose stocks to view detailed information",
        )

        if selected_stocks:
            # Display selected stocks in a grid
            num_cols = min(3, len(selected_stocks))
            for i in range(0, len(selected_stocks), num_cols):
                cols = st.columns(num_cols)
                for j, col in enumerate(cols):
                    if i + j < len(selected_stocks):
                        with col:
                            stock_data = active_stocks.loc[selected_stocks[i + j]]
                            create_stock_card(
                                stock_data, advanced_metrics, detailed=True
                            )
        else:
            st.info("Please select stocks to view their detailed information.")

        # Optional: Display summary statistics for selected stocks
        if selected_stocks:
            st.subheader("Summary Statistics for Selected Stocks")
            selected_data = active_stocks.loc[selected_stocks]

            # Calculate average metrics
            avg_metrics = pd.DataFrame(
                {
                    "Average Comments": [selected_data["num_comments"].mean()],
                    "Average Sentiment": [
                        selected_data["comment_sentiment_avg"].mean()
                    ],
                    "Average Bullish Ratio": [
                        selected_data["bullish_comments_ratio"].mean()
                    ],
                    "Average Bearish Ratio": [
                        selected_data["bearish_comments_ratio"].mean()
                    ],
                }
            )

            # Display summary metrics
            st.dataframe(avg_metrics.round(3), use_container_width=True)

    with tab3:
        st.header("Advanced Insights")

        # Sentiment Volatility
        st.subheader("Sentiment Volatility")
        fig_vol = px.bar(
            advanced_metrics["sentiment_volatility"].head(10),
            title="Most Volatile Sentiment",
            labels={"value": "Volatility", "ticker": "Stock"},
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Sentiment Momentum
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Positive Momentum")
            st.dataframe(
                advanced_metrics["sentiment_momentum"].tail(5).round(3),
                use_container_width=True,
            )
        with col2:
            st.subheader("Negative Momentum")
            st.dataframe(
                advanced_metrics["sentiment_momentum"].head(5).round(3),
                use_container_width=True,
            )

        # Sentiment-Volume Correlation
        st.subheader("Sentiment-Volume Correlation")
        fig_corr = px.scatter(
            x=active_stocks["num_comments"],
            y=active_stocks["comment_sentiment_avg"],
            text=active_stocks.index,
            title="Sentiment vs. Volume Correlation",
            labels={"x": "Number of Comments", "y": "Average Sentiment"},
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.header("Reddit Posts Analysis")
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
                        cols = st.columns([1, 1, 2])
                        with cols[0]:
                            st.metric("Score", post.get("score", "N/A"))
                        with cols[1]:
                            st.metric("Comments", post.get("num_comments", "N/A"))
                        with cols[2]:
                            st.metric(
                                "Sentiment", safe_format(post.get("avg_sentiment"))
                            )

                        st.markdown("**Content:**")
                        st.write(post.get("content", "No content available").strip())
                        st.caption(
                            f"Posted by u/{post.get('author', 'Unknown')} on {post.get('created_at', 'Unknown date')}"
                        )
            else:
                st.info("No posts found for the selected stocks in this date range.")
        else:
            st.info("Select stocks to view their Reddit posts.")


if __name__ == "__main__":
    main()
