import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import statsmodels
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
            # First, get post IDs for the ticker
            post_ids_result = (
                self.supabase.table("post_tickers")
                .select("post_id")
                .eq("ticker", ticker)
                .order("mentioned_at", ascending=False)
                .limit(5)
                .execute()
            )

            if not post_ids_result.data:
                return []

            # Extract post IDs
            post_ids = [item["post_id"] for item in post_ids_result.data]

            # Then get the actual posts
            posts_result = (
                self.supabase.table("reddit_posts")
                .select("*")
                .in_("post_id", post_ids)
                .execute()
            )

            return posts_result.data

        except Exception as e:
            return []

    def show_sentiment_tab(self, df: pd.DataFrame, ticker: str):
        """Show sentiment analysis"""
        try:
            # Get recent posts
            posts = self.get_recent_posts(ticker)

            # Sentiment trend
            if not df.empty:
                fig = px.line(
                    df,
                    x="analysis_timestamp",
                    y=[
                        "comment_sentiment_avg",
                        "bullish_comments_ratio",
                        "bearish_comments_ratio",
                    ],
                    title="Sentiment Trends",
                    labels={
                        "analysis_timestamp": "Date",
                        "value": "Score",
                        "variable": "Metric",
                    },
                )

                # Update line colors and names
                fig.update_traces(
                    line_color="green",
                    name="Overall Sentiment",
                    selector=dict(name="comment_sentiment_avg"),
                )
                fig.update_traces(
                    line_color="blue",
                    name="Bullish Ratio",
                    selector=dict(name="bullish_comments_ratio"),
                )
                fig.update_traces(
                    line_color="red",
                    name="Bearish Ratio",
                    selector=dict(name="bearish_comments_ratio"),
                )

                # Update layout
                fig.update_layout(
                    legend_title="Metrics",
                    hovermode="x unified",
                    plot_bgcolor="white",
                    yaxis=dict(gridcolor="lightgrey"),
                    xaxis=dict(gridcolor="lightgrey"),
                )

                st.plotly_chart(fig, use_container_width=True)

            # Recent posts section
            st.subheader("Recent Discussions")
            if posts:
                for post in posts:
                    with st.expander(
                        f"📝 {post.get('title', 'No title')} (Score: {post.get('score', 0)})"
                    ):
                        # Post content
                        st.markdown(
                            f"**Content:**\n{post.get('content', 'No content')}"
                        )

                        # Metrics in columns
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Sentiment",
                                f"{post.get('avg_sentiment', 0):.2f}",
                                delta=(
                                    "Positive"
                                    if post.get("avg_sentiment", 0) > 0
                                    else "Negative"
                                ),
                            )

                        with col2:
                            st.metric("Comments", post.get("num_comments", 0))

                        with col3:
                            st.metric(
                                "Upvote Ratio", f"{post.get('upvote_ratio', 0):.1%}"
                            )

                        # Additional information
                        st.caption(
                            f"Posted by u/{post.get('author', '[deleted]')} on {post.get('created_at', 'unknown date')}"
                        )

                        if post.get("url"):
                            st.markdown(f"[View on Reddit]({post['url']})")
            else:
                st.info("No recent discussions found for this stock.")

        except Exception as e:
            st.error(f"Error displaying sentiment analysis: {str(e)}")
            logger.error(f"Error in sentiment tab: {str(e)}")

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
        """Add enhanced correlation analysis section"""
        st.header("Correlation Analysis")

        # Define correlation groups
        correlation_groups = {
            "Price & Returns": [
                "current_price",
                "price_change_2w",
                "price_change_2d",
                "volume_change",
            ],
            "Technical Indicators": ["rsi", "volatility", "sma_20", "volume_change"],
            "Sentiment Metrics": [
                "comment_sentiment_avg",
                "bullish_comments_ratio",
                "bearish_comments_ratio",
                "num_comments",
                "score",
            ],
        }

        # Create tabs for different correlation views
        tab1, tab2, tab3 = st.tabs(
            ["🎯 Key Correlations", "📊 Correlation Matrix", "📈 Correlation Insights"]
        )

        with tab1:
            self._display_key_correlations(df, correlation_groups)

        with tab2:
            self._display_correlation_matrix(df, correlation_groups)

        with tab3:
            self._display_correlation_insights(df, correlation_groups)

    def _display_key_correlations(self, df: pd.DataFrame, correlation_groups: dict):
        """Display key correlation pairs with interpretation"""
        st.subheader("Key Correlation Pairs")

        # Calculate important correlations
        key_pairs = [
            {
                "pair": ("comment_sentiment_avg", "price_change_2w"),
                "name": "Sentiment vs Price Change",
                "description": "Relationship between sentiment and 2-week price movement",
            },
            {
                "pair": ("bullish_comments_ratio", "volume_change"),
                "name": "Bullish Sentiment vs Volume",
                "description": "Impact of bullish sentiment on trading volume",
            },
            {
                "pair": ("num_comments", "volatility"),
                "name": "Discussion Activity vs Volatility",
                "description": "Relationship between discussion volume and price volatility",
            },
            {
                "pair": ("rsi", "comment_sentiment_avg"),
                "name": "RSI vs Sentiment",
                "description": "Technical indicator alignment with sentiment",
            },
        ]

        for pair in key_pairs:
            if all(metric in df.columns for metric in pair["pair"]):
                corr = df[pair["pair"][0]].corr(df[pair["pair"][1]])

                with st.expander(f"{pair['name']}: {corr:.2f}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Scatter plot
                        fig = px.scatter(
                            df,
                            x=pair["pair"][0],
                            y=pair["pair"][1],
                            trendline="ols",
                            labels={
                                pair["pair"][0]: pair["pair"][0]
                                .replace("_", " ")
                                .title(),
                                pair["pair"][1]: pair["pair"][1]
                                .replace("_", " ")
                                .title(),
                            },
                            hover_data=["ticker"],
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Correlation strength indicator
                        strength = abs(corr)
                        if strength > 0.7:
                            strength_label = "Strong"
                            color = "red" if corr < 0 else "green"
                        elif strength > 0.4:
                            strength_label = "Moderate"
                            color = "orange" if corr < 0 else "lightgreen"
                        else:
                            strength_label = "Weak"
                            color = "gray"

                        st.metric(
                            "Correlation Strength",
                            f"{strength_label} ({corr:.2f})",
                            delta="Negative" if corr < 0 else "Positive",
                            delta_color="inverse" if corr < 0 else "normal",
                        )

                        st.write(pair["description"])

                        # Add top correlated tickers
                        st.write("**Top Correlated Tickers:**")
                        top_corr = (
                            df.groupby("ticker")
                            .apply(
                                lambda x: x[pair["pair"][0]].corr(x[pair["pair"][1]])
                            )
                            .sort_values(ascending=False)
                            .head(3)
                        )

                        for ticker, corr_val in top_corr.items():
                            st.write(f"- {ticker}: {corr_val:.2f}")

    def _display_correlation_matrix(self, df: pd.DataFrame, correlation_groups: dict):
        """Display enhanced correlation matrix"""
        st.subheader("Correlation Matrix")

        # Let user select correlation groups
        selected_groups = st.multiselect(
            "Select metric groups to compare:",
            list(correlation_groups.keys()),
            default=list(correlation_groups.keys())[:2],
        )

        if selected_groups:
            # Combine selected metrics
            selected_metrics = []
            for group in selected_groups:
                selected_metrics.extend(correlation_groups[group])

            # Remove duplicates while preserving order
            selected_metrics = list(dict.fromkeys(selected_metrics))

            # Filter available columns
            available_metrics = [m for m in selected_metrics if m in df.columns]

            if available_metrics:
                # Calculate correlation matrix
                corr_matrix = df[available_metrics].corr()

                # Create heatmap
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=[col.replace("_", " ").title() for col in available_metrics],
                        y=[col.replace("_", " ").title() for col in available_metrics],
                        colorscale="RdBu",
                        zmin=-1,
                        zmax=1,
                    )
                )

                fig.update_layout(
                    title="Correlation Heatmap",
                    width=800,
                    height=800,
                    xaxis_tickangle=-45,
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No metrics available for selected groups")
        else:
            st.warning("Please select at least one metric group")

    def _display_correlation_insights(self, df: pd.DataFrame, correlation_groups: dict):
        """Display correlation insights and patterns"""
        st.subheader("Correlation Insights")

        # Calculate and display strongest correlations
        all_metrics = [
            metric for group in correlation_groups.values() for metric in group
        ]
        available_metrics = [m for m in all_metrics if m in df.columns]

        if available_metrics:
            corr_matrix = df[available_metrics].corr()

            # Find strongest correlations
            correlations = []
            for i in range(len(available_metrics)):
                for j in range(i + 1, len(available_metrics)):
                    correlations.append(
                        {
                            "metric1": available_metrics[i],
                            "metric2": available_metrics[j],
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )

            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            # Display strongest correlations
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Strongest Positive Correlations**")
                positive_corr = [c for c in correlations if c["correlation"] > 0][:5]
                for corr in positive_corr:
                    st.metric(
                        f"{corr['metric1'].replace('_', ' ').title()} vs {corr['metric2'].replace('_', ' ').title()}",
                        f"{corr['correlation']:.2f}",
                        "Strong" if abs(corr["correlation"]) > 0.7 else "Moderate",
                        delta_color="normal",
                    )

            with col2:
                st.write("**Strongest Negative Correlations**")
                negative_corr = [c for c in correlations if c["correlation"] < 0][:5]
                for corr in negative_corr:
                    st.metric(
                        f"{corr['metric1'].replace('_', ' ').title()} vs {corr['metric2'].replace('_', ' ').title()}",
                        f"{corr['correlation']:.2f}",
                        "Strong" if abs(corr["correlation"]) > 0.7 else "Moderate",
                        delta_color="inverse",
                    )

            # Add pair-wise correlation analysis
            st.subheader("Pair-wise Correlation Analysis")

            # Select metrics to compare
            col1, col2 = st.columns(2)
            with col1:
                metric1 = st.selectbox(
                    "Select first metric", available_metrics, key="metric1"
                )
            with col2:
                metric2 = st.selectbox(
                    "Select second metric", available_metrics, key="metric2"
                )

            if metric1 and metric2:
                # Create scatter plot for selected metrics
                fig = go.Figure()

                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=df[metric1],
                        y=df[metric2],
                        mode="markers",
                        name="Data Points",
                        text=df["ticker"],
                        hovertemplate=f"Ticker: %{{text}}<br>"
                        + f"{metric1}: %{{x}}<br>"
                        + f"{metric2}: %{{y}}<br>",
                    )
                )

                # Update layout
                fig.update_layout(
                    title=f"Correlation between {metric1.replace('_', ' ').title()} and {metric2.replace('_', ' ').title()}",
                    xaxis_title=metric1.replace("_", " ").title(),
                    yaxis_title=metric2.replace("_", " ").title(),
                    showlegend=True,
                )

                # Display correlation coefficient
                corr = df[metric1].corr(df[metric2])
                st.metric(
                    "Correlation Coefficient",
                    f"{corr:.2f}",
                    (
                        "Strong"
                        if abs(corr) > 0.7
                        else "Moderate" if abs(corr) > 0.4 else "Weak"
                    ),
                    delta_color="normal" if corr > 0 else "inverse",
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                # Add top correlated tickers
                st.subheader("Top Correlated Tickers")
                # Group by ticker and calculate correlation
                ticker_correlations = []
                for ticker in df["ticker"].unique():
                    ticker_data = df[df["ticker"] == ticker]
                    if len(ticker_data) > 1:  # Need at least 2 points for correlation
                        ticker_corr = ticker_data[metric1].corr(ticker_data[metric2])
                        if not pd.isna(ticker_corr):  # Check if correlation is valid
                            ticker_correlations.append(
                                {"ticker": ticker, "correlation": ticker_corr}
                            )

                # Sort and display top positive and negative correlations
                if ticker_correlations:
                    ticker_df = pd.DataFrame(ticker_correlations)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Strongest Positive**")
                        top_positive = ticker_df.nlargest(3, "correlation")
                        for _, row in top_positive.iterrows():
                            st.metric(
                                row["ticker"],
                                f"{row['correlation']:.2f}",
                                "Strong Positive",
                            )

                    with col2:
                        st.write("**Strongest Negative**")
                        top_negative = ticker_df.nsmallest(3, "correlation")
                        for _, row in top_negative.iterrows():
                            st.metric(
                                row["ticker"],
                                f"{row['correlation']:.2f}",
                                "Strong Negative",
                            )
        else:
            st.warning("No metrics available for correlation analysis")

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
        """Display enhanced technical analysis signals"""
        st.header("Technical Analysis Signals")

        # Calculate signals with categories
        signal_data = self._calculate_signals(df)

        # Create tabs for different signal categories
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🎯 RSI Signals", "📊 Volatility", "💹 Volume", "💭 Sentiment"]
        )

        with tab1:
            self._display_rsi_signals(signal_data["rsi"])

        with tab2:
            self._display_volatility_signals(signal_data["volatility"])

        with tab3:
            self._display_volume_signals(signal_data["volume"])

        with tab4:
            self._display_sentiment_signals(signal_data["sentiment"])

        # Display summary of all signals
        st.subheader("Signal Summary")
        for ticker, signals in signal_data["all_signals"].items():
            if signals:
                with st.expander(f"🔍 {ticker}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        signal_count = len(signals)
                        st.metric("Total Signals", signal_count)
                    with col2:
                        for signal in signals:
                            self._display_signal_badge(signal)

    def _calculate_signals(self, df: pd.DataFrame) -> dict:
        """Calculate all technical signals"""
        signals = {
            "rsi": {"overbought": [], "oversold": []},
            "volatility": {"high": []},
            "volume": {"spike": []},
            "sentiment": {"bullish": [], "bearish": []},
            "all_signals": {},
        }

        for _, row in df.iterrows():
            ticker = row["ticker"]
            ticker_signals = []

            # RSI Signals
            if row["rsi"] > 70:
                signals["rsi"]["overbought"].append((ticker, row["rsi"]))
                ticker_signals.append(
                    {"type": "RSI", "signal": "Overbought", "value": row["rsi"]}
                )
            elif row["rsi"] < 30:
                signals["rsi"]["oversold"].append((ticker, row["rsi"]))
                ticker_signals.append(
                    {"type": "RSI", "signal": "Oversold", "value": row["rsi"]}
                )

            # Volatility Signals
            if row["volatility"] > df["volatility"].quantile(0.75):
                signals["volatility"]["high"].append((ticker, row["volatility"]))
                ticker_signals.append(
                    {"type": "Volatility", "signal": "High", "value": row["volatility"]}
                )

            # Volume Signals
            if row["volume_change"] > 50:
                signals["volume"]["spike"].append((ticker, row["volume_change"]))
                ticker_signals.append(
                    {"type": "Volume", "signal": "Spike", "value": row["volume_change"]}
                )

            # Sentiment Signals
            if row["comment_sentiment_avg"] > df["comment_sentiment_avg"].quantile(0.8):
                signals["sentiment"]["bullish"].append(
                    (ticker, row["comment_sentiment_avg"])
                )
                ticker_signals.append(
                    {
                        "type": "Sentiment",
                        "signal": "Bullish",
                        "value": row["comment_sentiment_avg"],
                    }
                )
            elif row["comment_sentiment_avg"] < df["comment_sentiment_avg"].quantile(
                0.2
            ):
                signals["sentiment"]["bearish"].append(
                    (ticker, row["comment_sentiment_avg"])
                )
                ticker_signals.append(
                    {
                        "type": "Sentiment",
                        "signal": "Bearish",
                        "value": row["comment_sentiment_avg"],
                    }
                )

            if ticker_signals:
                signals["all_signals"][ticker] = ticker_signals

        return signals

    def _display_rsi_signals(self, rsi_signals: dict):
        """Display RSI signals with visualization"""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Overbought Stocks (RSI > 70)")
            for ticker, rsi in rsi_signals["overbought"]:
                st.metric(
                    ticker, f"RSI: {rsi:.1f}", "⚠️ Overbought", delta_color="inverse"
                )

        with col2:
            st.subheader("Oversold Stocks (RSI < 30)")
            for ticker, rsi in rsi_signals["oversold"]:
                st.metric(
                    ticker, f"RSI: {rsi:.1f}", "💡 Oversold", delta_color="normal"
                )

    def _display_volatility_signals(self, volatility_signals: dict):
        """Display volatility signals"""
        st.subheader("High Volatility Stocks")
        columns = st.columns(3)
        for i, (ticker, vol) in enumerate(volatility_signals["high"]):
            with columns[i % 3]:
                st.metric(ticker, f"Vol: {vol:.1f}%", "📈 High Volatility")

    def _display_volume_signals(self, volume_signals: dict):
        """Display volume signals"""
        st.subheader("Volume Spikes")
        for ticker, change in volume_signals["spike"]:
            st.metric(
                ticker, f"Volume Δ: {change:.1f}%", "🔊 Volume Spike", delta_color="off"
            )

    def _display_sentiment_signals(self, sentiment_signals: dict):
        """Display sentiment signals"""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bullish Sentiment")
            for ticker, sentiment in sentiment_signals["bullish"]:
                st.metric(
                    ticker,
                    f"Score: {sentiment:.2f}",
                    "🚀 Strong Bullish",
                    delta_color="normal",
                )

        with col2:
            st.subheader("Bearish Sentiment")
            for ticker, sentiment in sentiment_signals["bearish"]:
                st.metric(
                    ticker,
                    f"Score: {sentiment:.2f}",
                    "⚠️ Strong Bearish",
                    delta_color="inverse",
                )

    def _display_signal_badge(self, signal: dict):
        """Display a styled signal badge"""
        signal_colors = {
            "RSI": "#FF9800",
            "Volatility": "#2196F3",
            "Volume": "#4CAF50",
            "Sentiment": "#9C27B0",
        }

        signal_icons = {
            "Overbought": "⚠️",
            "Oversold": "💡",
            "High": "📈",
            "Spike": "🔊",
            "Bullish": "🚀",
            "Bearish": "📉",
        }

        st.markdown(
            f"""
            <div style="
                display: inline-block;
                padding: 4px 12px;
                margin: 2px;
                border-radius: 15px;
                background-color: {signal_colors[signal['type']]};
                color: white;
                font-size: 0.9em;
            ">
                {signal_icons.get(signal['signal'], '•')} {signal['type']}: {signal['signal']} ({signal['value']:.1f})
            </div>
            """,
            unsafe_allow_html=True,
        )

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
