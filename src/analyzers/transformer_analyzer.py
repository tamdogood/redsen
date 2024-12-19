from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from typing import Dict, List
import numpy as np
import re


class FinancialSentimentTransformer:
    def __init__(self):
        """Initialize the FinancialSentimentTransformer with multiple models"""
        # Initialize FinBERT model for financial sentiment analysis
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Initialize general sentiment pipeline for backup and validation
        self.general_sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device if self.device == "cuda" else -1,
        )

        # Initialize financial phrase bank for domain-specific analysis
        self.financial_phrases = self._initialize_financial_phrases()

        # Initialize cache for sentiment scores
        self.sentiment_cache = {}

    def _initialize_financial_phrases(self) -> Dict[str, float]:
        """Initialize dictionary of financial phrases and their sentiment scores"""
        return {
            # Positive indicators
            "beat earnings": 0.8,
            "exceeded expectations": 0.7,
            "raised guidance": 0.6,
            "strong growth": 0.6,
            "positive outlook": 0.5,
            "buy rating": 0.5,
            "upgrade": 0.4,
            "outperform": 0.4,
            # Negative indicators
            "missed earnings": -0.8,
            "below expectations": -0.7,
            "lowered guidance": -0.6,
            "weak performance": -0.6,
            "negative outlook": -0.5,
            "sell rating": -0.5,
            "downgrade": -0.4,
            "underperform": -0.4,
            # Market conditions
            "bull market": 0.3,
            "bear market": -0.3,
            "market rally": 0.3,
            "market selloff": -0.3,
            # Technical indicators
            "breakout": 0.4,
            "breakdown": -0.4,
            "golden cross": 0.5,
            "death cross": -0.5,
            "support level": 0.2,
            "resistance level": -0.2,
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove special characters but keep important financial symbols
        text = re.sub(r"[^\w\s$%+-]", " ", text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Convert to lowercase while preserving tickers
        words = text.split()
        processed_words = []
        for word in words:
            if word.startswith("$"):
                processed_words.append(word)
            else:
                processed_words.append(word.lower())

        return " ".join(processed_words)

    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks that respect the model's max length"""
        # Tokenize the full text
        encoded = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)
        
        if len(encoded) <= max_length:
            return [text]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Get token count for this sentence
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=True)
            sentence_length = len(sentence_tokens)
            
            if sentence_length > max_length:
                # If single sentence is too long, split by words
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                    
                # Split long sentence into smaller pieces
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_tokens = self.tokenizer.encode(word, add_special_tokens=True)
                    word_length = len(word_tokens)
                    
                    if temp_length + word_length <= max_length:
                        temp_chunk.append(word)
                        temp_length += word_length
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = word_length
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    
            elif current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def get_sentiment(self, text: str, use_cache: bool = True) -> Dict[str, float]:
        """Get sentiment analysis with proper text chunking"""
        if use_cache and text in self.sentiment_cache:
            return self.sentiment_cache[text]
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Split into chunks if necessary
            chunks = self.chunk_text(processed_text)
            chunk_sentiments = []
            
            # Process each chunk
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.strip():
                    continue
                    
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    scores = softmax(outputs.logits, dim=1)
                    chunk_sentiments.append(scores.cpu().numpy()[0])
            
            # Average the sentiment scores across chunks
            if chunk_sentiments:
                finbert_scores = np.mean(chunk_sentiments, axis=0)
            else:
                finbert_scores = np.array([0.33, 0.33, 0.34])  # Neutral fallback
            
            # Get general sentiment (use first chunk if text is too long)
            first_chunk = chunks[0] if chunks else processed_text
            general_sentiment = self.general_sentiment(first_chunk)[0]
            general_score = 1.0 if general_sentiment['label'] == 'POSITIVE' else -1.0
            general_confidence = general_sentiment['score']
            
            # Get phrase-based sentiment
            phrase_sentiment = self._analyze_financial_phrases(processed_text)
            
            # Combine scores with weights
            combined_score = self._combine_sentiment_scores(
                finbert_positive=float(finbert_scores[0]),
                finbert_negative=float(finbert_scores[1]),
                finbert_neutral=float(finbert_scores[2]),
                general_score=general_score,
                general_confidence=general_confidence,
                phrase_sentiment=phrase_sentiment
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                finbert_neutral=float(finbert_scores[2]),
                general_confidence=general_confidence,
                text_length=len(processed_text)
            )
            
            result = {
                "compound": combined_score,
                "positive": float(finbert_scores[0]),
                "negative": float(finbert_scores[1]),
                "neutral": float(finbert_scores[2]),
                "confidence": confidence,
                "phrase_sentiment": phrase_sentiment,
                "general_sentiment": {
                    "score": general_score,
                    "confidence": general_confidence
                }
            }
            
            if use_cache:
                self.sentiment_cache[text] = result
            
            return result
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            # Fallback to general sentiment
            try:
                general_sentiment = self.general_sentiment(processed_text)[0]
                return {
                    "compound": 1.0 if general_sentiment['label'] == 'POSITIVE' else -1.0,
                    "positive": float(general_sentiment['score'] if general_sentiment['label'] == 'POSITIVE' else 0),
                    "negative": float(general_sentiment['score'] if general_sentiment['label'] == 'NEGATIVE' else 0),
                    "neutral": 0.0,
                    "confidence": float(general_sentiment['score']),
                    "phrase_sentiment": 0.0,
                    "general_sentiment": {
                        "score": 1.0 if general_sentiment['label'] == 'POSITIVE' else -1.0,
                        "confidence": float(general_sentiment['score'])
                    }
                }
            except:
                return {
                    "compound": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0,
                    "confidence": 0.0,
                    "phrase_sentiment": 0.0,
                    "general_sentiment": {
                        "score": 0.0,
                        "confidence": 0.0
                    }
                }

    def _analyze_financial_phrases(self, text: str) -> float:
        """Analyze text for financial phrases and their sentiment impact"""
        total_sentiment = 0.0
        matches = 0

        for phrase, score in self.financial_phrases.items():
            if phrase in text:
                total_sentiment += score
                matches += 1

        # Normalize sentiment
        return total_sentiment / max(matches, 1)

    def _combine_sentiment_scores(
        self,
        finbert_positive: float,
        finbert_negative: float,
        finbert_neutral: float,
        general_score: float,
        general_confidence: float,
        phrase_sentiment: float,
    ) -> float:
        """Combine different sentiment scores with weighted approach"""
        # Calculate FinBERT compound
        finbert_compound = finbert_positive - finbert_negative

        # Weights for different components
        weights = {"finbert": 0.6, "general": 0.2, "phrase": 0.2}

        # Combine scores
        combined_score = (
            finbert_compound * weights["finbert"]
            + general_score * weights["general"] * general_confidence
            + phrase_sentiment * weights["phrase"]
        )

        # Normalize to [-1, 1]
        return max(min(combined_score, 1.0), -1.0)

    def _calculate_confidence(
        self, finbert_neutral: float, general_confidence: float, text_length: int
    ) -> float:
        """Calculate confidence score for sentiment analysis"""
        # Base confidence from FinBERT (inversely proportional to neutral score)
        finbert_confidence = 1.0 - finbert_neutral

        # Length-based confidence (caps at 1000 characters)
        length_confidence = min(text_length / 1000.0, 1.0)

        # Combine confidence scores
        confidence = (
            finbert_confidence * 0.6
            + general_confidence * 0.3
            + length_confidence * 0.1
        )

        return max(min(confidence, 1.0), 0.0)

    def analyze_batch(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict[str, float]]:
        """Analyze sentiment for a batch of texts efficiently"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = [self.get_sentiment(text) for text in batch]
            results.extend(batch_results)
        return results
