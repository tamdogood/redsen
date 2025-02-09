import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
import json
from typing import List, Dict, Any
import re
from openai import OpenAI

# Load environment variables and initialize clients
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add system message
    st.session_state.messages.append(
        {
            "role": "system",
            "content": """You are a stock market sentiment analysis assistant. Analyze sentiment data 
        from Reddit discussions about stocks. Focus on providing clear, actionable insights while
        maintaining a professional tone.""",
        }
    )

if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None


def safe_format(value, format_str=".2f", default="N/A"):
    """Safely format numeric values"""
    try:
        if pd.isna(value) or value is None:
            return default
        return format(float(value), format_str)
    except (ValueError, TypeError):
        return default


def load_sentiment_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Load sentiment data for a specific ticker or all tickers"""
    try:
        query_end_date = (pd.to_datetime(end_date) + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        query = (
            supabase.table("sentiment_analysis")
            .select("*")
            .gte("analysis_timestamp", start_date)
            .lt("analysis_timestamp", query_end_date)
        )

        if ticker != "*":
            query = query.eq("ticker", ticker)

        response = query.execute()

        if response.data:
            df = pd.DataFrame(response.data)
            numeric_columns = [
                "comment_sentiment_avg",
                "base_sentiment",
                "submission_sentiment",
                "bullish_comments_ratio",
                "bearish_comments_ratio",
                "composite_score",
                "price_change_1d",
                "price_change_1w",
                "volume_change",
                "rsi",
                "current_price",
                "market_cap",
                "pe_ratio",
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading sentiment data: {str(e)}")
        return pd.DataFrame()


def load_posts_for_ticker(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Load recent Reddit posts for a ticker"""
    try:
        response = (
            supabase.table("post_tickers")
            .select("*, reddit_posts(*)")
            .eq("ticker", ticker)
            .order("mentioned_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error loading posts: {str(e)}")
        return []


def create_sentiment_heatmap(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Create an improved, more readable heatmap visualization for sentiment metrics"""
    try:
        # Clean and prepare data
        df = df.copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df["date"] = pd.to_datetime(df["analysis_timestamp"]).dt.date

        # Get top N most active stocks
        top_stocks = (
            df.groupby("ticker")["num_comments"]
            .sum()
            .sort_values(ascending=False)
            .head(30)
            .index.tolist()
        )

        # Filter for top stocks
        df_filtered = df[df["ticker"].isin(top_stocks)]

        # Pivot data for heatmap
        pivot_data = df_filtered.pivot_table(
            values=metric, index="date", columns="ticker", aggfunc="mean"
        ).round(3)

        # Sort columns by activity
        column_order = (
            df_filtered.groupby("ticker")["num_comments"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        pivot_data = pivot_data[column_order]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=[d.strftime("%Y-%m-%d") for d in pivot_data.index],
                colorscale=[
                    [0, "rgb(165,0,38)"],  # Strong negative
                    [0.25, "rgb(215,48,39)"],  # Negative
                    [0.5, "rgb(255,255,255)"],  # Neutral
                    [0.75, "rgb(34,94,168)"],  # Positive
                    [1, "rgb(0,47,167)"],  # Strong positive
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

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
            height=600,
            xaxis=dict(
                title="Ticker Symbol",
                tickangle=45,
                tickfont=dict(size=12),
                side="bottom",
            ),
            yaxis=dict(
                title="Date",
                tickfont=dict(size=12),
            ),
            paper_bgcolor="black",
        )

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


def create_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    """Create an interactive sentiment chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(df["analysis_timestamp"]),
            y=df["comment_sentiment_avg"],
            mode="lines+markers",
            name="Sentiment",
            line=dict(color="#17B897"),
            hovertemplate="<b>Date:</b> %{x}<br>"
            + "<b>Sentiment:</b> %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode="x unified",
        template="plotly_dark",
    )

    return fig


def calculate_advanced_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate additional insights and metrics"""
    metrics = {}

    # Calculate sentiment volatility
    metrics["sentiment_volatility"] = (
        df.groupby("ticker")["comment_sentiment_avg"].std().sort_values(ascending=False)
    )

    # Calculate sentiment momentum
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

    # Track sentiment trends
    metrics["sentiment_trend"] = {
        ticker: group.set_index("date")["comment_sentiment_avg"]
        for ticker, group in df.groupby("ticker")
    }

    return metrics


def extract_ticker_and_timeframe(user_input: str) -> Dict[str, Any]:
    """
    Use LLM to extract stock ticker and timeframe from natural language input.
    Returns a dictionary with extracted information and normalized dates.

    Args:
        user_input (str): Natural language query from user

    Returns:
        Dict containing:
            - ticker (str): Extracted stock ticker
            - start_date (datetime): Calculated start date based on timeframe
            - end_date (datetime): Calculated end date (usually current date)
            - timeframe_type (str): Type of timeframe mentioned (day, week, month, year)
            - timeframe_value (int): Number of time units
            - original_query (str): Original user input
    """
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a financial data extraction assistant. Extract the stock ticker 
                and timeframe from the user's query. Return only a JSON object with the following fields:
                - ticker: The stock ticker (uppercase)
                - timeframe_type: One of [day, week, month, year] or null if not specified
                - timeframe_value: Number of time units (integer) or null if not specified
                
                Example 1: "Show me AAPL sentiment for the last 2 weeks"
                {"ticker": "AAPL", "timeframe_type": "week", "timeframe_value": 2}
                
                Example 2: "What's the latest on TSLA?"
                {"ticker": "TSLA", "timeframe_type": null, "timeframe_value": null}""",
            },
            {"role": "user", "content": user_input},
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )

        # Parse LLM response
        extracted = json.loads(response.choices[0].message.content)

        # Calculate dates based on timeframe
        end_date = datetime.now()
        start_date = end_date

        if extracted["timeframe_type"] and extracted["timeframe_value"]:
            time_mapping = {
                "day": lambda x: timedelta(days=x),
                "week": lambda x: timedelta(weeks=x),
                "month": lambda x: timedelta(days=x * 30),
                "year": lambda x: timedelta(days=x * 365),
            }

            delta_func = time_mapping.get(extracted["timeframe_type"])
            if delta_func:
                start_date = end_date - delta_func(extracted["timeframe_value"])
        else:
            # Default to 1 week if no timeframe specified
            start_date = end_date - timedelta(weeks=1)

        return {
            "ticker": extracted["ticker"],
            "start_date": start_date,
            "end_date": end_date,
            "timeframe_type": extracted["timeframe_type"],
            "timeframe_value": extracted["timeframe_value"],
            "original_query": user_input,
        }

    except Exception as e:
        # Fall back to regex-based extraction if LLM fails
        return fallback_extract_ticker(user_input)


def create_sentiment_trend_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create sentiment trend chart with both line and scatter plots"""
    fig = go.Figure()

    # Add line plot
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["comment_sentiment_avg"],
            mode="lines",
            name="Sentiment Trend",
            line=dict(color="blue", width=2),
        )
    )

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["comment_sentiment_avg"],
            mode="markers",
            name="Daily Sentiment",
            marker=dict(size=8, color="blue", symbol="circle"),
        )
    )

    fig.update_layout(
        title=f"{ticker} Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode="x unified",
    )

    return fig


def display_sentiment_analysis(response: Dict[str, Any]) -> None:
    """Display sentiment analysis with charts and metrics"""
    if response["data"] is not None:
        df = response["data"]

        # Create metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Sentiment", f"{df['comment_sentiment_avg'].iloc[0]:.2f}")
        with col2:
            st.metric("Bullish Ratio", f"{df['bullish_comments_ratio'].iloc[0]:.1%}")
        with col3:
            st.metric("Total Comments", f"{int(df['num_comments'].iloc[0])}")

        # Create and display trend chart
        fig = create_sentiment_trend_chart(df, response["ticker"])
        st.plotly_chart(
            fig, use_container_width=True, key=f"sentiment_trend_{response['ticker']}"
        )

        # Display additional charts if needed
        if "price_change" in df.columns:
            price_fig = create_price_chart(df, response["ticker"])
            st.plotly_chart(
                price_fig,
                use_container_width=True,
                key=f"price_trend_{response['ticker']}",
            )

        if "volume" in df.columns:
            volume_fig = create_volume_chart(df, response["ticker"])
            st.plotly_chart(
                volume_fig,
                use_container_width=True,
                key=f"volume_trend_{response['ticker']}",
            )


def create_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create price trend chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["price_change"],
            mode="lines+markers",
            name="Price Change",
            line=dict(color="green", width=2),
        )
    )

    fig.update_layout(
        title=f"{ticker} Price Change",
        xaxis_title="Date",
        yaxis_title="Price Change (%)",
        hovermode="x unified",
    )

    return fig


def create_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create volume trend chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Trading Volume",
            marker_color="rgba(0, 128, 0, 0.5)",
        )
    )

    fig.update_layout(
        title=f"{ticker} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified",
    )

    return fig


def fallback_extract_ticker(user_input: str) -> Dict[str, Any]:
    """Fallback method using regex to extract ticker when LLM fails"""
    patterns = [
        r"\$([A-Z]{1,5})\b",  # Matches tickers with $ prefix
        r"\b([A-Z]{1,5})\b(?!\.[A-Z]{1,2})",  # Matches uppercase words, excludes file extensions
    ]

    tickers = []
    for pattern in patterns:
        tickers.extend(re.findall(pattern, user_input.upper()))

    ticker = tickers[0] if tickers else None
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=1)  # Default to 1 week

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "timeframe_type": None,
        "timeframe_value": None,
        "original_query": user_input,
    }


def get_llm_analysis(data: Dict[str, Any], user_query: str) -> str:
    """Generate LLM analysis of the sentiment data"""
    try:
        context = f"""
        Analyzing {data['ticker']} based on Reddit sentiment data:
        
        Latest metrics:
        - Sentiment score: {data['sentiment_score']}
        - Bullish ratio: {data['bullish_ratio']}
        - Bearish ratio: {data['bearish_ratio']}
        - Total comments: {data['total_comments']}
        - Price change (1w): {data.get('price_change_1w', 'N/A')}
        - Current price: {data.get('current_price', 'N/A')}
        
        Recent posts:
        {data['recent_posts']}
        
        Sentiment trend:
        {data['sentiment_trend']}
        """

        messages = [
            {
                "role": "system",
                "content": """You are a stock market sentiment analyst. Analyze the provided data 
                and generate insights about the stock's sentiment. Focus on key trends, notable 
                discussions, and potential implications. Be concise but thorough.""",
            },
            {
                "role": "user",
                "content": f"Query: {user_query}\n\nData to analyze:\n{context}",
            },
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.7, max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error generating LLM analysis: {str(e)}")
        return "I encountered an error while analyzing the data. Please try again."


def process_user_input(user_input: str) -> Dict[str, Any]:
    """Process user input and generate appropriate response with visualization"""
    # Extract ticker and timeframe
    extracted = extract_ticker_and_timeframe(user_input)

    response = {
        "ticker": extracted["ticker"],
        "type": "text",
        "content": "",
        "data": None,
        "llm_response": None,
    }

    if not extracted["ticker"]:
        response["content"] = (
            "I couldn't identify a stock ticker. Please mention a ticker symbol (e.g., AAPL, TSLA)."
        )
        return response

    # Load data using extracted timeframe
    df = load_sentiment_data(
        extracted["ticker"], extracted["start_date"], extracted["end_date"]
    )
    posts = load_posts_for_ticker(extracted["ticker"])

    if df.empty:
        response["content"] = (
            f"I couldn't find any sentiment data for {extracted['ticker']}."
        )
        return response

    # Prepare data for LLM
    latest = df.iloc[0]
    sentiment_trend = (
        "improving" if df["comment_sentiment_avg"].diff().mean() > 0 else "declining"
    )

    llm_data = {
        "ticker": extracted["ticker"],
        "sentiment_score": safe_format(latest["comment_sentiment_avg"]),
        "bullish_ratio": safe_format(latest["bullish_comments_ratio"], ".1%"),
        "bearish_ratio": safe_format(latest["bearish_comments_ratio"], ".1%"),
        "total_comments": int(latest["num_comments"]),
        "price_change_1w": safe_format(latest.get("price_change_1w", 0), "+.2f") + "%",
        "current_price": safe_format(latest.get("current_price", 0)),
        "recent_posts": "\n".join(
            [f"- {post['reddit_posts']['title']}" for post in posts[:3]]
        ),
        "sentiment_trend": f"The sentiment has been {sentiment_trend} over the past week.",
    }

    # Get LLM analysis
    llm_response = get_llm_analysis(llm_data, user_input)

    # Determine response type and display appropriate visualization
    input_lower = user_input.lower()
    if "sentiment" in input_lower or "analysis" in input_lower:
        response["type"] = "sentiment"
        st.write(llm_response)
        display_sentiment_analysis(
            {"ticker": extracted["ticker"], "data": df, "type": "sentiment"}
        )
    elif (
        "posts" in input_lower or "reddit" in input_lower or "discussion" in input_lower
    ):
        response["type"] = "posts"
        st.write(llm_response)
        # Display recent posts in a nice format
        for post in posts[:5]:
            st.markdown(f"**{post['reddit_posts']['title']}**")
            st.markdown(f"*Score: {post['reddit_posts']['score']}*")
            st.markdown("---")
    elif "trend" in input_lower or "chart" in input_lower:
        response["type"] = "trend"
        st.write(llm_response)
        display_sentiment_analysis(
            {"ticker": extracted["ticker"], "data": df, "type": "trend"}
        )
    else:
        response["type"] = "overview"
        st.write(llm_response)
        display_sentiment_analysis(
            {"ticker": extracted["ticker"], "data": df, "type": "overview"}
        )

    response["content"] = llm_response
    response["data"] = df

    return response


def chat_interface():
    """Render the RAG chat interface"""
    st.header("Stock Sentiment Chat")

    # Sidebar instructions
    with st.sidebar:
        st.markdown(
            """
        ### Chat Instructions
        Ask questions about stocks using their ticker symbols. Examples:
        - "Analyze AAPL sentiment"
        - "What's the trend for TSLA?"
        - "Show me recent discussions about NVDA"
        - "Give me a complete analysis of META"
        """
        )

        st.markdown("---")
        st.caption("Powered by OpenAI GPT-4")

    # Chat container
    chat_container = st.container()

    # Input
    user_input = st.text_input(
        "Ask about a stock",
        key="chat_input",
        placeholder="e.g., 'Analyze AAPL sentiment'",
    )

    if st.button("Send", key="send_button"):
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Process input
            response = process_user_input(user_input)

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] not in ["user", "assistant"]:
                continue

            with st.container():
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown("**Assistant:**")
                    st.markdown(message["content"]["content"])

                    # Display visualizations if
                    if message["content"]["data"] is not None:
                        data = message["content"]["data"]
                        if not data.empty:
                            # Sentiment chart
                            fig = create_sentiment_chart(data)
                            st.plotly_chart(fig, use_container_width=True)

                            # Key metrics
                            cols = st.columns(4)
                            latest = data.iloc[0]
                            cols[0].metric(
                                "Comments", f"{int(latest['num_comments']):,}"
                            )
                            cols[1].metric(
                                "Sentiment",
                                safe_format(latest["comment_sentiment_avg"]),
                            )
                            cols[2].metric(
                                "Bullish Ratio",
                                safe_format(latest["bullish_comments_ratio"], ".1%"),
                            )
                            cols[3].metric(
                                "Bearish Ratio",
                                safe_format(latest["bearish_comments_ratio"], ".1%"),
                            )

                            # Additional metrics if available
                            if (
                                "price_change_1w" in latest
                                and "current_price" in latest
                            ):
                                cols = st.columns(3)
                                cols[0].metric(
                                    "Current Price",
                                    f"${safe_format(latest['current_price'])}",
                                )
                                cols[1].metric(
                                    "Weekly Change",
                                    f"{safe_format(latest['price_change_1w'], '+.2f')}%",
                                )
                                if "rsi" in latest:
                                    cols[2].metric("RSI", safe_format(latest["rsi"]))
                st.markdown("---")


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


def main():
    st.set_page_config(page_title="Enhanced Stock Sentiment Dashboard", layout="wide")
    st.title("Stock Sentiment Analysis Dashboard")

    # Date range selector
    cols = st.columns([1, 1, 2])
    with cols[0]:
        start_date = st.date_input(
            "Start Date", value=datetime.now() - timedelta(days=7)
        )
    with cols[1]:
        end_date = st.date_input("End Date", value=datetime.now())

    # Load data
    df = load_sentiment_data("*", start_date, end_date)  # Load all tickers for overview
    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(df)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Market Overview", "Stock Grid", "Advanced Insights", "Posts", "Chat Analysis"]
    )

    with tab1:
        st.header("Market Overview")

        # Summary metrics
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

        # Sort options
        sort_options = {
            "Most Comments": "num_comments",
            "Highest Sentiment": "comment_sentiment_avg",
            "Most Bullish": "bullish_comments_ratio",
            "Most Bearish": "bearish_comments_ratio",
            "Highest Composite Score": "composite_score",
        }

        with col2:
            sort_by = st.selectbox("Sort Stocks By", options=list(sort_options.keys()))

        # Sort stocks
        active_stocks = active_stocks.sort_values(
            sort_options[sort_by], ascending=False
        )

        # Stock selector
        selected_stocks = st.multiselect(
            "Select Stocks to Display",
            options=active_stocks.index.tolist(),
            default=active_stocks.index[:3].tolist(),
        )

        if selected_stocks:
            # Display selected stocks in grid
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
            st.dataframe(advanced_metrics["sentiment_momentum"].tail(5).round(3))
        with col2:
            st.subheader("Negative Momentum")
            st.dataframe(advanced_metrics["sentiment_momentum"].head(5).round(3))

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
            key="posts_ticker_select",
        )

        if selected_tickers:
            posts_df = pd.DataFrame()
            for ticker in selected_tickers:
                posts = load_posts_for_ticker(ticker)
                if posts:
                    for post in posts:
                        with st.expander(
                            f"{post['reddit_posts']['title']} ({post['reddit_posts'].get('subreddit', 'Unknown')})"
                        ):
                            cols = st.columns([1, 1, 2])
                            with cols[0]:
                                st.metric(
                                    "Score", post["reddit_posts"].get("score", "N/A")
                                )
                            with cols[1]:
                                st.metric(
                                    "Comments",
                                    post["reddit_posts"].get("num_comments", "N/A"),
                                )
                            with cols[2]:
                                st.metric(
                                    "Sentiment",
                                    safe_format(
                                        post["reddit_posts"].get("avg_sentiment")
                                    ),
                                )

                            st.markdown("**Content:**")
                            st.write(
                                post["reddit_posts"]
                                .get("content", "No content available")
                                .strip()
                            )
                            st.caption(
                                f"Posted by u/{post['reddit_posts'].get('author', 'Unknown')} on {post['reddit_posts'].get('created_at', 'Unknown date')}"
                            )

    with tab5:
        chat_interface()


if __name__ == "__main__":
    main()
