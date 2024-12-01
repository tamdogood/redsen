import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import os


class StockSentimentDashboard:
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize dashboard with Supabase connection"""
        self.supabase = create_client(supabase_url, supabase_key)

    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Stock Sentiment Analysis", page_icon="📈", layout="wide"
        )

        st.title("Stock Sentiment Analysis Dashboard")

    def show_overview_tab(self, df: pd.DataFrame):
        """Show overview metrics"""
        latest = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"${latest['current_price']:.2f}",
                f"{latest['price_change_2w']:.1f}%",
            )

        with col2:
            st.metric(
                "Sentiment Score",
                f"{latest['comment_sentiment_avg']:.2f}",
                f"{latest['bullish_comments_ratio']*100:.1f}% Bullish",
            )

        with col3:
            st.metric(
                "Volume",
                f"{latest['avg_volume']:,.0f}",
                f"{latest['volume_change']:.1f}%",
            )

        with col4:
            st.metric(
                "RSI",
                f"{latest['rsi']:.1f}",
                (
                    "Overbought"
                    if latest["rsi"] > 70
                    else "Oversold" if latest["rsi"] < 30 else "Neutral"
                ),
            )

        # Price and volume chart
        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Scatter(x=df["analysis_timestamp"], y=df["current_price"], name="Price"),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(x=df["analysis_timestamp"], y=df["avg_volume"], name="Volume"),
            row=2,
            col=1,
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    def show_technical_tab(self, df: pd.DataFrame):
        """Show technical analysis"""
        col1, col2 = st.columns(2)

        with col1:
            # RSI Chart
            fig = px.line(df, x="analysis_timestamp", y="rsi", title="RSI")
            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # MACD Chart
            if "macd_line" in df.columns:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["analysis_timestamp"], y=df["macd_line"], name="MACD"
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["analysis_timestamp"], y=df["signal_line"], name="Signal"
                    )
                )
                fig.update_layout(title="MACD")
                st.plotly_chart(fig, use_container_width=True)

    def show_sentiment_tab(self, df: pd.DataFrame, ticker: str):
        """Show sentiment analysis"""
        # Get recent posts
        posts = self.get_recent_posts(ticker)

        # Sentiment trend
        fig = px.line(
            df,
            x="analysis_timestamp",
            y=[
                "comment_sentiment_avg",
                "bullish_comments_ratio",
                "bearish_comments_ratio",
            ],
            title="Sentiment Trends",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recent posts
        st.subheader("Recent Discussions")
        for post in posts:
            # Handle potential missing fields gracefully
            title = post.get("title", "No title")
            score = post.get("score", 0)
            content = post.get("content", "No content")
            sentiment = post.get("avg_sentiment", 0)
            num_comments = post.get("num_comments", 0)

            with st.expander(f"{title} (Score: {score})"):
                st.write(content)
                st.write(f"Sentiment: {sentiment:.2f}")
                st.write(f"Comments: {num_comments}")
                if post.get("url"):
                    st.write(f"[View on Reddit]({post['url']})")

    def get_stock_data(self, ticker: str) -> pd.DataFrame:
        """Get historical data for a stock"""
        try:
            result = (
                self.supabase.table("sentiment_analysis")
                .select("*")
                .eq("ticker", ticker)
                .order("analysis_timestamp")
                .execute()
            )

            return pd.DataFrame(result.data)
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()

    def get_latest_analysis(self) -> pd.DataFrame:
        """Get the most recent sentiment analysis data"""
        try:
            result = (
                self.supabase.table("sentiment_analysis")
                .select("*")
                .order("analysis_timestamp")
                .limit(1000)
                .execute()
            )

            return pd.DataFrame(result.data)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    def get_recent_posts(self, ticker: str) -> list:
        """Get recent Reddit posts for a stock"""
        try:
            result = (
                self.supabase.table("post_tickers")
                .select("*, reddit_posts!fk_post_tickers_post(*)")  # Modified this line
                .eq("ticker", ticker)
                .order("mentioned_at")
                .limit(5)
                .execute()
            )

            return result.data
        except Exception as e:
            st.error(f"Error fetching posts: {str(e)}")
            return []

    def get_sentiment_trends(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get sentiment trends for a specific ticker"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            result = (
                self.supabase.table("sentiment_analysis")
                .select("*")
                .eq("ticker", ticker)
                .gte("analysis_timestamp", cutoff_date)
                .order("analysis_timestamp")
                .execute()
            )

            return pd.DataFrame(result.data)

        except Exception as e:
            st.error(f"Error retrieving sentiment trends for {ticker}: {str(e)}")
            return pd.DataFrame()

    def add_metrics_section(self):
        """Add key metrics summary section"""
        st.header("Market Overview")
        df = self.get_latest_analysis()

        # Calculate market-wide metrics
        metrics = {
            "Total Stocks Tracked": len(df["ticker"].unique()),
            "Avg Market Sentiment": df["comment_sentiment_avg"].mean(),
            "Most Discussed": df.nlargest(1, "num_comments")["ticker"].iloc[0],
            "Most Bullish": df.nlargest(1, "comment_sentiment_avg")["ticker"].iloc[0],
            "Best Performer": df.nlargest(1, "price_change_2w")["ticker"].iloc[0],
        }

        cols = st.columns(len(metrics))
        for col, (metric, value) in zip(cols, metrics.items()):
            col.metric(metric, value)

    def add_correlation_analysis(self, df: pd.DataFrame):
        """Add correlation analysis section"""
        st.header("Correlation Analysis")

        # Select relevant columns for correlation
        correlation_cols = [
            "comment_sentiment_avg",
            "price_change_2w",
            "volume_change",
            "rsi",
            "volatility",
            "num_comments",
        ]

        # Calculate correlation matrix
        corr_matrix = df[correlation_cols].corr()

        # Plot heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=correlation_cols,
                y=correlation_cols,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    def add_trend_analysis(self, df: pd.DataFrame):
        """Add trend analysis section"""
        st.header("Trend Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Sentiment distribution
            fig = px.histogram(
                df,
                x="comment_sentiment_avg",
                title="Sentiment Distribution",
                nbins=30,
                marginal="box",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Volume vs Price Change scatter plot without trendline
            fig = px.scatter(
                df,
                x="volume_change",
                y="price_change_2w",
                color="comment_sentiment_avg",
                title="Volume Change vs Price Change",
                hover_data=["ticker"],
                labels={
                    "volume_change": "Volume Change (%)",
                    "price_change_2w": "Price Change (%)",
                    "comment_sentiment_avg": "Sentiment Score",
                },
            )

            # Add custom layout
            fig.update_layout(
                xaxis_title="Volume Change (%)",
                yaxis_title="Price Change (%)",
                coloraxis_colorbar_title="Sentiment Score",
            )

            st.plotly_chart(fig, use_container_width=True)

        # Add summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Average Sentiment",
                f"{df['comment_sentiment_avg'].mean():.2f}",
                f"{df['comment_sentiment_avg'].std():.2f} σ",
            )

        with col2:
            st.metric(
                "Average Price Change",
                f"{df['price_change_2w'].mean():.1f}%",
                f"{df['price_change_2w'].std():.1f}% σ",
            )

        with col3:
            st.metric(
                "Average Volume Change",
                f"{df['volume_change'].mean():.1f}%",
                f"{df['volume_change'].std():.1f}% σ",
            )

        # Add sentiment trends over time
        st.subheader("Sentiment Trends")
        sentiment_over_time = (
            df.groupby(pd.to_datetime(df["analysis_timestamp"]).dt.date)
            .agg(
                {
                    "comment_sentiment_avg": "mean",
                    "num_comments": "sum",
                    "price_change_2w": "mean",
                }
            )
            .reset_index()
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=sentiment_over_time["analysis_timestamp"],
                y=sentiment_over_time["comment_sentiment_avg"],
                name="Average Sentiment",
                line=dict(color="blue"),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=sentiment_over_time["analysis_timestamp"],
                y=sentiment_over_time["price_change_2w"],
                name="Average Price Change",
                line=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Sentiment and Price Change Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            yaxis2_title="Price Change (%)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add distribution comparison
        st.subheader("Sentiment Distribution by Performance")
        fig = go.Figure()

        # Positive performers
        positive_mask = df["price_change_2w"] > 0
        fig.add_trace(
            go.Histogram(
                x=df[positive_mask]["comment_sentiment_avg"],
                name="Positive Performers",
                nbinsx=30,
                opacity=0.7,
            )
        )

        # Negative performers
        negative_mask = df["price_change_2w"] <= 0
        fig.add_trace(
            go.Histogram(
                x=df[negative_mask]["comment_sentiment_avg"],
                name="Negative Performers",
                nbinsx=30,
                opacity=0.7,
            )
        )

        fig.update_layout(
            title="Sentiment Distribution: Positive vs Negative Performers",
            xaxis_title="Sentiment Score",
            yaxis_title="Count",
            barmode="overlay",
        )

        st.plotly_chart(fig, use_container_width=True)

    def add_sector_analysis(self, df: pd.DataFrame):
        """Add sector-based analysis"""
        st.header("Sector Analysis")

        # Get sector data from yfinance
        sectors = {}
        for ticker in df["ticker"].unique():
            try:
                stock = yf.Ticker(ticker)
                sector = stock.info.get("sector", "Unknown")
                sectors[ticker] = sector
            except:
                sectors[ticker] = "Unknown"

        df["sector"] = df["ticker"].map(sectors)

        # Sector sentiment
        sector_sentiment = (
            df.groupby("sector")
            .agg(
                {
                    "comment_sentiment_avg": "mean",
                    "num_comments": "sum",
                    "price_change_2w": "mean",
                }
            )
            .reset_index()
        )

        fig = px.treemap(
            sector_sentiment,
            path=["sector"],
            values="num_comments",
            color="comment_sentiment_avg",
            title="Sector Sentiment Analysis",
        )
        st.plotly_chart(fig, use_container_width=True)

    def add_technical_signals(self, df: pd.DataFrame):
        """Add technical signals analysis"""
        st.header("Technical Signals")

        # Calculate technical signals
        signals = []
        for _, row in df.iterrows():
            ticker = row["ticker"]
            signals_dict = {"ticker": ticker, "signals": []}

            # RSI signals
            if row["rsi"] > 70:
                signals_dict["signals"].append("Overbought (RSI)")
            elif row["rsi"] < 30:
                signals_dict["signals"].append("Oversold (RSI)")

            # Volatility signals
            if row["volatility"] > df["volatility"].quantile(0.75):
                signals_dict["signals"].append("High Volatility")

            # Volume signals
            if row["volume_change"] > 50:
                signals_dict["signals"].append("Volume Spike")

            # Sentiment signals
            if row["comment_sentiment_avg"] > df["comment_sentiment_avg"].quantile(0.8):
                signals_dict["signals"].append("Strong Bullish Sentiment")
            elif row["comment_sentiment_avg"] < df["comment_sentiment_avg"].quantile(
                0.2
            ):
                signals_dict["signals"].append("Strong Bearish Sentiment")

            if signals_dict["signals"]:
                signals.append(signals_dict)

        # Display signals
        st.subheader("Active Technical Signals")
        for signal in signals:
            st.write(f"**{signal['ticker']}**: {', '.join(signal['signals'])}")

    def add_sentiment_price_analysis(self, df: pd.DataFrame):
        """Add analysis of sentiment vs price relationships"""
        st.header("Sentiment vs Price Analysis")

        # Create bins for sentiment scores
        df["sentiment_category"] = pd.qcut(
            df["comment_sentiment_avg"],
            q=5,
            labels=["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"],
        )

        # Create layout with two tabs for different timeframes
        timeframe_tab1, timeframe_tab2 = st.tabs(["2-Day Analysis", "2-Week Analysis"])

        with timeframe_tab1:
            col1, col2 = st.columns(2)

            with col1:
                # Average 2-day price change by sentiment category
                avg_price_change_2d = (
                    df.groupby("sentiment_category")["price_change_2d"].mean().round(2)
                )
                fig = px.bar(
                    avg_price_change_2d,
                    title="Average 2-Day Price Change by Sentiment Category",
                    labels={
                        "sentiment_category": "Sentiment",
                        "value": "Average Price Change (%)",
                    },
                )
                fig.update_traces(marker_color="lightblue")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Bullish/Bearish ratio impact (2-day)
                fig = px.scatter(
                    df,
                    x="bullish_comments_ratio",
                    y="price_change_2d",
                    color="comment_sentiment_avg",
                    hover_data=["ticker"],
                    title="Bullish Ratio vs 2-Day Price Change",
                    labels={
                        "bullish_comments_ratio": "Bullish Comments Ratio",
                        "price_change_2d": "2-Day Price Change (%)",
                        "comment_sentiment_avg": "Overall Sentiment",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

        with timeframe_tab2:
            col1, col2 = st.columns(2)

            with col1:
                # Average 2-week price change by sentiment category
                avg_price_change_2w = (
                    df.groupby("sentiment_category")["price_change_2w"].mean().round(2)
                )
                fig = px.bar(
                    avg_price_change_2w,
                    title="Average 2-Week Price Change by Sentiment Category",
                    labels={
                        "sentiment_category": "Sentiment",
                        "value": "Average Price Change (%)",
                    },
                )
                fig.update_traces(marker_color="lightgreen")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Bullish/Bearish ratio impact (2-week)
                fig = px.scatter(
                    df,
                    x="bullish_comments_ratio",
                    y="price_change_2w",
                    color="comment_sentiment_avg",
                    hover_data=["ticker"],
                    title="Bullish Ratio vs 2-Week Price Change",
                    labels={
                        "bullish_comments_ratio": "Bullish Comments Ratio",
                        "price_change_2w": "2-Week Price Change (%)",
                        "comment_sentiment_avg": "Overall Sentiment",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

        # Add sentiment effectiveness analysis
        st.subheader("Sentiment Prediction Effectiveness")

        # Calculate prediction accuracy for both timeframes
        df["sentiment_correct_2d"] = (df["comment_sentiment_avg"] > 0) & (
            df["price_change_2d"] > 0
        ) | (df["comment_sentiment_avg"] < 0) & (df["price_change_2d"] < 0)

        df["sentiment_correct_2w"] = (df["comment_sentiment_avg"] > 0) & (
            df["price_change_2w"] > 0
        ) | (df["comment_sentiment_avg"] < 0) & (df["price_change_2w"] < 0)

        accuracy_2d = (df["sentiment_correct_2d"].sum() / len(df)) * 100
        accuracy_2w = (df["sentiment_correct_2w"].sum() / len(df)) * 100

        # Display metrics in tabs
        accuracy_tab1, accuracy_tab2 = st.tabs(["2-Day Accuracy", "2-Week Accuracy"])

        with accuracy_tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("2-Day Prediction Accuracy", f"{accuracy_2d:.1f}%")

            with col2:
                bullish_accuracy_2d = (
                    df[
                        (df["comment_sentiment_avg"] > 0) & (df["price_change_2d"] > 0)
                    ].shape[0]
                    / df[df["comment_sentiment_avg"] > 0].shape[0]
                    * 100
                )
                st.metric("2-Day Bullish Accuracy", f"{bullish_accuracy_2d:.1f}%")

            with col3:
                bearish_accuracy_2d = (
                    df[
                        (df["comment_sentiment_avg"] < 0) & (df["price_change_2d"] < 0)
                    ].shape[0]
                    / df[df["comment_sentiment_avg"] < 0].shape[0]
                    * 100
                )
                st.metric("2-Day Bearish Accuracy", f"{bearish_accuracy_2d:.1f}%")

        with accuracy_tab2:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("2-Week Prediction Accuracy", f"{accuracy_2w:.1f}%")

            with col2:
                bullish_accuracy_2w = (
                    df[
                        (df["comment_sentiment_avg"] > 0) & (df["price_change_2w"] > 0)
                    ].shape[0]
                    / df[df["comment_sentiment_avg"] > 0].shape[0]
                    * 100
                )
                st.metric("2-Week Bullish Accuracy", f"{bullish_accuracy_2w:.1f}%")

            with col3:
                bearish_accuracy_2w = (
                    df[
                        (df["comment_sentiment_avg"] < 0) & (df["price_change_2w"] < 0)
                    ].shape[0]
                    / df[df["comment_sentiment_avg"] < 0].shape[0]
                    * 100
                )
                st.metric("2-Week Bearish Accuracy", f"{bearish_accuracy_2w:.1f}%")

        # Add time lag analysis
        st.subheader("Sentiment Lead/Lag Analysis")

        # Select timeframe for lag analysis
        lag_timeframe = st.radio(
            "Select Timeframe", ["2-Day", "2-Week"], horizontal=True
        )
        price_change_col = (
            "price_change_2d" if lag_timeframe == "2-Day" else "price_change_2w"
        )

        # Calculate lagged correlations
        lags = range(-5, 6)  # -5 to +5 days
        correlations = []

        for lag in lags:
            corr = (
                df.groupby("ticker")
                .apply(
                    lambda x: x["comment_sentiment_avg"]
                    .shift(lag)
                    .corr(x[price_change_col])
                )
                .mean()
            )
            correlations.append({"lag": lag, "correlation": corr})

        lag_df = pd.DataFrame(correlations)

        fig = px.line(
            lag_df,
            x="lag",
            y="correlation",
            title=f"Sentiment-Price Correlation by Time Lag ({lag_timeframe})",
            labels={"lag": "Time Lag (Days)", "correlation": "Correlation Coefficient"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    def show_advanced_metrics_tab(self, df: pd.DataFrame, ticker: str):
        """Enhanced advanced metrics analysis"""
        st.subheader("Advanced Metrics Analysis")

        # Volatility metrics
        volatility_metrics = {
            "Daily Volatility": df["price_change_2d"].std(),
            "Weekly Volatility": df["price_change_2w"].std(),
            "Sentiment Volatility": df["comment_sentiment_avg"].std(),
        }

        cols = st.columns(len(volatility_metrics))
        for col, (metric, value) in zip(cols, volatility_metrics.items()):
            col.metric(metric, f"{value:.2f}")

        # Sentiment-Price correlation analysis
        st.subheader("Sentiment-Price Correlation")

        # Calculate rolling correlation
        window_size = 5  # 5-day rolling window
        df["rolling_correlation"] = (
            df["comment_sentiment_avg"].rolling(window_size).corr(df["price_change_2w"])
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["analysis_timestamp"],
                y=df["rolling_correlation"],
                name=f"{window_size}-day Rolling Correlation",
            )
        )
        fig.update_layout(
            title=f"Rolling Correlation: Sentiment vs Price Change (Window: {window_size} days)",
            yaxis_title="Correlation Coefficient",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        # Add momentum indicators
        st.subheader("Sentiment Momentum")
        df["sentiment_ma5"] = df["comment_sentiment_avg"].rolling(5).mean()
        df["sentiment_ma20"] = df["comment_sentiment_avg"].rolling(20).mean()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add sentiment MAs
        fig.add_trace(
            go.Scatter(
                x=df["analysis_timestamp"], y=df["sentiment_ma5"], name="5-day MA"
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["analysis_timestamp"], y=df["sentiment_ma20"], name="20-day MA"
            ),
            secondary_y=False,
        )

        # Add price
        fig.add_trace(
            go.Scatter(x=df["analysis_timestamp"], y=df["current_price"], name="Price"),
            secondary_y=True,
        )

        fig.update_layout(
            title="Sentiment Moving Averages vs Price",
            yaxis_title="Sentiment Score",
            yaxis2_title="Price ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Enhanced run method with sentiment-price analysis"""
        self.setup_page()

        # Add top-level metrics
        self.add_metrics_section()

        # Get data
        df = self.get_latest_analysis()
        if df.empty:
            st.warning("No data available")
            return

        # Add correlation analysis
        self.add_correlation_analysis(df)

        # Add sentiment-price analysis
        self.add_sentiment_price_analysis(df)

        # Add trend analysis
        self.add_trend_analysis(df)

        # Add technical signals
        self.add_technical_signals(df)

        # Stock details section
        st.header("Stock Details")
        selected_ticker = st.selectbox(
            "Select Stock for Detailed Analysis", options=df["ticker"].unique()
        )

        if selected_ticker:
            self.show_stock_details(selected_ticker)

    def show_stock_correlation_analysis(self, df: pd.DataFrame, ticker: str):
        """Show detailed correlation analysis for a specific stock"""
        st.header(f"Correlation Analysis for {ticker}")

        # Handle NaN values in the data
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Calculate various correlations with error handling
        def safe_correlation(x, y):
            try:
                return x.corr(y)
            except:
                return 0.0

        correlations = {
            "Sentiment vs Price": {
                "Same Day": safe_correlation(
                    df["comment_sentiment_avg"], df["price_change_2w"]
                ),
                "Next Day": safe_correlation(
                    df["comment_sentiment_avg"].shift(1), df["price_change_2w"]
                ),
                "Previous Day": safe_correlation(
                    df["comment_sentiment_avg"].shift(-1), df["price_change_2w"]
                ),
            },
            "Bullish Ratio vs Price": {
                "Same Day": safe_correlation(
                    df["bullish_comments_ratio"], df["price_change_2w"]
                ),
                "Next Day": safe_correlation(
                    df["bullish_comments_ratio"].shift(1), df["price_change_2w"]
                ),
                "Previous Day": safe_correlation(
                    df["bullish_comments_ratio"].shift(-1), df["price_change_2w"]
                ),
            },
            "Volume vs Sentiment": {
                "Same Day": safe_correlation(
                    df["volume_change"], df["comment_sentiment_avg"]
                ),
                "Next Day": safe_correlation(
                    df["volume_change"].shift(1), df["comment_sentiment_avg"]
                ),
                "Previous Day": safe_correlation(
                    df["volume_change"].shift(-1), df["comment_sentiment_avg"]
                ),
            },
        }

        # Display correlation metrics
        col1, col2, col3 = st.columns(3)

        def get_correlation_color(value):
            if pd.isna(value) or value == 0:
                return "normal"
            if abs(value) > 0.7:
                return "strong correlation"
            elif abs(value) > 0.4:
                return "moderate correlation"
            else:
                return "weak correlation"

        for i, (metric, values) in enumerate(correlations.items()):
            with [col1, col2, col3][i]:
                st.subheader(metric)
                for period, value in values.items():
                    if pd.notnull(value):
                        strength = get_correlation_color(value)
                        st.metric(
                            period, f"{value:.3f}", delta=strength, delta_color="normal"
                        )
                    else:
                        st.metric(period, "N/A", delta="insufficient data")

        # Add rolling correlation analysis
        st.subheader("Rolling Correlation Analysis")

        # Only show rolling analysis if we have enough data
        if len(df) >= 5:
            window_size = st.slider(
                "Select Rolling Window Size (days)",
                min_value=5,
                max_value=min(30, len(df)),
                value=min(10, len(df)),
            )

            # Calculate rolling correlations with error handling
            try:
                df["sentiment_price_corr"] = (
                    df["comment_sentiment_avg"]
                    .rolling(window=window_size, min_periods=3)
                    .corr(df["price_change_2w"])
                )

                df["bullish_price_corr"] = (
                    df["bullish_comments_ratio"]
                    .rolling(window=window_size, min_periods=3)
                    .corr(df["price_change_2w"])
                )

                # Plot rolling correlations
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df["analysis_timestamp"],
                        y=df["sentiment_price_corr"],
                        name="Sentiment-Price Correlation",
                        line=dict(color="blue"),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df["analysis_timestamp"],
                        y=df["bullish_price_corr"],
                        name="Bullish Ratio-Price Correlation",
                        line=dict(color="green"),
                    )
                )

                fig.update_layout(
                    title=f"{window_size}-Day Rolling Correlation with Price Change",
                    yaxis_title="Correlation Coefficient",
                    showlegend=True,
                )

                fig.add_hline(
                    y=0.7,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Strong Positive",
                )
                fig.add_hline(
                    y=-0.7,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Strong Negative",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(
                    "Could not calculate rolling correlations due to insufficient data"
                )

            # Add lag analysis
            st.subheader("Correlation Lag Analysis")

            max_lag = min(5, len(df) // 2)
            lags = range(-max_lag, max_lag + 1)
            lag_correlations = []

            for lag in lags:
                sent_corr = safe_correlation(
                    df["comment_sentiment_avg"].shift(lag), df["price_change_2w"]
                )
                bull_corr = safe_correlation(
                    df["bullish_comments_ratio"].shift(lag), df["price_change_2w"]
                )

                lag_correlations.append(
                    {
                        "lag": lag,
                        "Sentiment Correlation": sent_corr,
                        "Bullish Ratio Correlation": bull_corr,
                    }
                )

            lag_df = pd.DataFrame(lag_correlations).fillna(0)

            # Plot lag analysis
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=lag_df["lag"],
                    y=lag_df["Sentiment Correlation"],
                    name="Sentiment",
                    mode="lines+markers",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=lag_df["lag"],
                    y=lag_df["Bullish Ratio Correlation"],
                    name="Bullish Ratio",
                    mode="lines+markers",
                )
            )

            fig.update_layout(
                title="Correlation by Time Lag",
                xaxis_title="Lag (Days)",
                yaxis_title="Correlation Coefficient",
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add insights
            st.subheader("Key Insights")

            # Find strongest correlations safely
            sent_corr_idx = lag_df["Sentiment Correlation"].abs().idxmax()
            bull_corr_idx = lag_df["Bullish Ratio Correlation"].abs().idxmax()

            if sent_corr_idx is not None and bull_corr_idx is not None:
                max_sentiment_corr = lag_df.iloc[sent_corr_idx]
                max_bullish_corr = lag_df.iloc[bull_corr_idx]

                st.write(
                    f"""
                - Strongest sentiment correlation occurs at {int(max_sentiment_corr['lag'])} day(s) 
                lag with correlation of {max_sentiment_corr['Sentiment Correlation']:.3f}
                - Strongest bullish ratio correlation occurs at {int(max_bullish_corr['lag'])} day(s) 
                lag with correlation of {max_bullish_corr['Bullish Ratio Correlation']:.3f}
                """
                )
            else:
                st.write("Insufficient data to determine strongest correlations")

        else:
            st.warning(
                f"Not enough data points for {ticker} to perform correlation analysis. Need at least 5 data points."
            )

    def show_stock_details(self, ticker: str):
        """Enhanced stock details with correlation analysis"""
        stock_data = self.get_stock_data(ticker)
        if stock_data.empty:
            st.warning(f"No data available for {ticker}")
            return

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Overview",
                "Technical Analysis",
                "Social Sentiment",
                "Correlation Analysis",
                "Advanced Metrics",
            ]
        )

        with tab1:
            self.show_overview_tab(stock_data)

        with tab2:
            self.show_technical_tab(stock_data)

        with tab3:
            self.show_sentiment_tab(stock_data, ticker)

        with tab4:
            self.show_stock_correlation_analysis(stock_data, ticker)

        with tab5:
            self.show_advanced_metrics_tab(stock_data, ticker)


if __name__ == "__main__":
    load_dotenv()

    # Initialize and run dashboard
    dashboard = StockSentimentDashboard(
        supabase_url=os.getenv("SUPABASE_URL"), supabase_key=os.getenv("SUPABASE_KEY")
    )
    dashboard.run()
