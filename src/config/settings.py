import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API Settings
REDDIT_CLIENT_ID = os.getenv("CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("USER_AGENT", "")

# Supabase Settings
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Analysis Settings
SUBREDDITS_TO_ANALYZE = ["wallstreetbets", "stocks", "investing"]
INVALID_TICKERS = {
    "CEO",
    "IPO",
    "EPS",
    "GDP",
    "NYSE",
    "SEC",
    "USD",
    "ATH",
    "IMO",
    "PSA",
    "USA",
    "CDC",
    "WHO",
    "ETF",
    "YOLO",
    "FOMO",
    "FUD",
    "DOW",
    "LOL",
    "ANY",
    "ALL",
    "FOR",
    "ARE",
    "THE",
    "NOW",
    "NEW",
    "IRS",
    "FED",
    "NFT",
    "APE",
    "RH",
    "WSB",
    "I",
    "KSP",
    "CUDA",
    "NFT",
    "NFTs",
}
