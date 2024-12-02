# Stock Sentiment Analysis Dashboard

A comprehensive stock analysis tool that combines technical indicators, sentiment analysis from Reddit, and price predictions using machine learning. The dashboard provides real-time insights into market sentiment, technical analysis, and potential stock movements.

## Features

### Sentiment Analysis
- Reddit post and comment analysis using VADER and advanced NLP
- Bullish/Bearish ratio tracking
- Community engagement metrics
- Discussion volume analysis
- Weighted sentiment scoring based on user engagement

### Technical Analysis
- Multiple timeframe analysis (short, medium, long-term)
- Key technical indicators (RSI, MACD, Bollinger Bands)
- Volume analysis
- Support/Resistance levels
- Trend strength indicators

### Price Prediction
- Combined technical and sentiment-based predictions
- Confidence scoring
- Multiple timeframe forecasts
- Signal strength analysis
- Trend reversals detection

### Dashboard Features
- Interactive charts and visualizations
- Real-time data updates
- Customizable analysis views
- Cross-indicator correlation analysis
- Performance metrics tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tamdogood/redsen.git
cd redsen
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` with your API keys and credentials:
```
SUPABASE_DATABASE_PASSWORD
SUPABASE_URL
SUPABASE_KEY
CLIENT_ID
CLIENT_SECRET
USER_AGENT
OPENAI_API_KEY
REDDIT_TOP_POST_LIMIT=100
REDDIT_COMMENT_LIMIT=60
```

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

2. Access the dashboard in your browser at `http://localhost:8501`

3. Enter a stock ticker to analyze and explore different analysis tabs:
   - Overview
   - Technical Analysis
   - Social Sentiment
   - Price Prediction
   - Advanced Metrics

## Project Structure

```
stock-sentiment-dashboard/
├── analyzers/
│   ├── __init__.py
│   └── sentiment_analyzer.py
├── connectors/
│   ├── __init__.py
│   └── supabase_connector.py
├── utils/
│   ├── __init__.py
│   ├── logging_config.py
│   └── types.py
├── dashboard/
│   ├── __init__.py
│   └──streamlit.py
├── requirements.txt
└── README.md
```

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- yfinance
- praw
- nltk
- supabase
- openai
- python-dotenv

## Database Setup

1. Create required tables in Supabase:
```sql
-- Create table for sentiment analysis results
CREATE TABLE sentiment_analysis (
    id BIGSERIAL PRIMARY KEY,
    analysis_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    data_start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    data_end_date TIMESTAMP WITH TIME ZONE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    
    -- Reddit metrics
    score INTEGER,
    num_comments INTEGER,
    
    -- Sentiment metrics
    comment_sentiment_avg FLOAT,
    base_sentiment FLOAT,
    submission_sentiment FLOAT,
    bullish_comments_ratio FLOAT,
    bearish_comments_ratio FLOAT,
    sentiment_confidence INTEGER,
    
    -- Price metrics
    current_price FLOAT,
    price_change_2w FLOAT,
    price_change_2d FLOAT,
    
    -- Volume metrics
    avg_volume BIGINT,
    volume_change FLOAT,
    volume_sma FLOAT,
    volume_ratio FLOAT,
    
    -- Technical indicators
    sma_20 FLOAT,
    ema_9 FLOAT,
    rsi FLOAT,
    volatility FLOAT,
    bollinger_upper FLOAT,
    bollinger_lower FLOAT,
    macd_line FLOAT,
    signal_line FLOAT,
    macd_histogram FLOAT,
    stoch_k FLOAT,
    stoch_d FLOAT,
    
    -- Fundamental metrics
    market_cap BIGINT,
    pe_ratio FLOAT,
    beta FLOAT,
    dividend_yield FLOAT,
    profit_margins FLOAT,
    revenue_growth FLOAT,
    
    -- Market metrics
    target_price FLOAT,
    analyst_count INTEGER,
    short_ratio FLOAT,
    relative_volume FLOAT,
    recommendation VARCHAR(50),
    
    -- Composite scores
    composite_score FLOAT,
    technical_score FLOAT,
    sentiment_score FLOAT,
    fundamental_score FLOAT,
    
    -- Constraints
    UNIQUE(ticker, analysis_timestamp)
);

-- Create table for Reddit posts
CREATE TABLE reddit_posts (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    author VARCHAR(100),
    score INTEGER,
    num_comments INTEGER,
    upvote_ratio FLOAT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    subreddit VARCHAR(100) NOT NULL,
    avg_sentiment FLOAT,
    submission_sentiment JSONB,
    avg_base_sentiment FLOAT,
    avg_weighted_sentiment FLOAT
);

-- Create table for post comments
CREATE TABLE post_comments (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(50) NOT NULL REFERENCES reddit_posts(post_id),
    author VARCHAR(100),
    content TEXT,
    score INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    sentiment JSONB
);

-- Create table for post tickers
CREATE TABLE post_tickers (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(50) NOT NULL REFERENCES reddit_posts(post_id),
    ticker VARCHAR(10) NOT NULL,
    mentioned_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Unique constraint to prevent duplicates
    UNIQUE(post_id, ticker)
);

-- Create indices for common queries
CREATE INDEX idx_sentiment_ticker ON sentiment_analysis(ticker);
CREATE INDEX idx_sentiment_timestamp ON sentiment_analysis(analysis_timestamp);
CREATE INDEX idx_posts_created ON reddit_posts(created_at);
CREATE INDEX idx_posts_subreddit ON reddit_posts(subreddit);
CREATE INDEX idx_post_comments_post_id ON post_comments(post_id);
CREATE INDEX idx_post_tickers_ticker ON post_tickers(ticker);
CREATE INDEX idx_post_tickers_post_id ON post_tickers(post_id);
```

2. Set up storage bucket in Supabase for analysis results

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- Reddit API for social sentiment data
- VADER Sentiment Analysis
- OpenAI for enhanced sentiment analysis
- Supabase for data storage
- YFinance for stock data

## Contact

Tam - [@tamdogoods](https://twitter.com/tamdogoods)
Project Link: [https://github.com/tamdogood/redsen](https://github.com/tamdogood/redsen)