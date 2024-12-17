import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SentimentAnalysisResult(BaseModel):
    """Schema for sentiment analysis results"""

    score: float = Field(..., ge=-1, le=1)
    features: Dict = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0, le=1)
    terms: List[str] = Field(default_factory=list)


class MarketData(BaseModel):
    """Schema for market data"""

    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime


class StockDetails(BaseModel):
    """Schema for stock details"""

    ticker: str
    name: str
    market_cap: Optional[float]
    sector: Optional[str]
    industry: Optional[str]
    description: Optional[str]
    exchange: str
    type: str


class ErrorResponse(BaseModel):
    """Schema for error responses"""

    success: bool = False
    error: str
    timestamp: datetime
