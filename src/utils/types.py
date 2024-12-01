from typing import Dict, List
from pydantic import BaseModel, Field


class SentimentAnalysisResult(BaseModel):
    score: float = Field(..., description="Sentiment score between -1 and 1")
