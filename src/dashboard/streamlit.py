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

    def run(self):
        """Run the Streamlit dashboard"""
        self.setup_page()

        # Add top-level metrics
        self.add_metrics_section()

        # Sidebar filters
        st.sidebar.header("Filters")

        # Get data
        df = self.get_latest_analysis()
        if df.empty:
            st.warning("No data available")
            return

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

    def show_stock_details(self, ticker: str):
        """Enhanced stock details with more analysis"""
        stock_data = self.get_stock_data(ticker)
        if stock_data.empty:
            st.warning(f"No data available for {ticker}")
            return

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Overview", "Technical Analysis", "Social Sentiment", "Advanced Metrics"]
        )

        with tab1:
            self.show_overview_tab(stock_data)

        with tab2:
            self.show_technical_tab(stock_data)

        with tab3:
            self.show_sentiment_tab(stock_data, ticker)

        with tab4:
            self.show_advanced_metrics_tab(stock_data, ticker)

    def show_advanced_metrics_tab(self, df: pd.DataFrame, ticker: str):
        """Show advanced metrics analysis"""
        st.subheader("Advanced Metrics Analysis")

        # Calculate volatility metrics
        volatility_metrics = {
            "Daily Volatility": df["price_change_2d"].std(),
            "Weekly Volatility": df["price_change_2w"].std(),
            "Sentiment Volatility": df["comment_sentiment_avg"].std(),
        }

        # Display metrics
        cols = st.columns(len(volatility_metrics))
        for col, (metric, value) in zip(cols, volatility_metrics.items()):
            col.metric(metric, f"{value:.2f}")

        # Add momentum indicators
        st.subheader("Momentum Analysis")
        df["sentiment_ma5"] = df["comment_sentiment_avg"].rolling(5).mean()
        df["sentiment_ma20"] = df["comment_sentiment_avg"].rolling(20).mean()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["analysis_timestamp"], y=df["sentiment_ma5"], name="5-day MA"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["analysis_timestamp"], y=df["sentiment_ma20"], name="20-day MA"
            )
        )
        fig.update_layout(title="Sentiment Moving Averages")
        st.plotly_chart(fig, use_container_width=True)

        # Add sentiment distribution
        st.subheader("Sentiment Distribution")
        fig = px.histogram(
            df,
            x="comment_sentiment_avg",
            nbins=30,
            title="Sentiment Distribution Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    load_dotenv()

    # Initialize and run dashboard
    dashboard = StockSentimentDashboard(
        supabase_url=os.getenv("SUPABASE_URL"), supabase_key=os.getenv("SUPABASE_KEY")
    )
    dashboard.run()
