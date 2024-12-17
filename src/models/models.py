import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class TimeSeriesDataset(Dataset):
    """Custom dataset for financial time series data"""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.X[idx : idx + self.sequence_length]
        target = self.y[idx + self.sequence_length - 1]
        return features, target


class SentimentAttention(nn.Module):
    """Custom attention mechanism for sentiment features"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_features = torch.sum(x * attention_weights, dim=1)
        return attended_features, attention_weights


class MultiModalTransformer(pl.LightningModule):
    """Advanced transformer model combining sentiment and technical data"""

    def __init__(
        self,
        input_dim: int,
        sentiment_dim: int,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Separate embeddings for technical and sentiment features
        self.technical_embedding = nn.Linear(input_dim - sentiment_dim, 32)
        self.sentiment_embedding = nn.Linear(sentiment_dim, 32)

        # Transformer components
        encoder_layer = TransformerEncoderLayer(
            d_model=64,  # Combined embedding dimension
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Attention mechanisms
        self.sentiment_attention = SentimentAttention(32)
        self.technical_attention = SentimentAttention(32)

        # Output layers
        self.projection = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Split features
        technical_features = x[:, :, : -self.hparams.sentiment_dim]
        sentiment_features = x[:, :, -self.hparams.sentiment_dim :]

        # Process technical features
        technical_embedded = self.technical_embedding(technical_features)
        technical_attended, tech_weights = self.technical_attention(technical_embedded)

        # Process sentiment features
        sentiment_embedded = self.sentiment_embedding(sentiment_features)
        sentiment_attended, sent_weights = self.sentiment_attention(sentiment_embedded)

        # Combine features
        combined = torch.cat((technical_attended, sentiment_attended), dim=-1)
        combined = combined.unsqueeze(1).repeat(1, seq_len, 1)

        # Transformer processing
        transformer_out = self.transformer(combined)

        # Final prediction
        prediction = self.projection(transformer_out[:, -1, :])

        attention_weights = {"technical": tech_weights, "sentiment": sent_weights}

        return prediction, attention_weights

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat, _ = self(x)
        loss = nn.BCELoss()(y_hat.squeeze(), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat, _ = self(x)
        loss = nn.BCELoss()(y_hat.squeeze(), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class DeepTradingModel:
    """Main trading model class with PyTorch components"""

    def __init__(
        self,
        sequence_length: int = 10,
        batch_size: int = 32,
        num_layers: int = 3,
        learning_rate: float = 1e-4,
    ):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()

    def prepare_data(
        self, data: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        # Split features
        sentiment_columns = [
            "comment_sentiment_avg",
            "bullish_comments_ratio",
            "bearish_comments_ratio",
            "sentiment_momentum",
            "sentiment_volatility",
            "sentiment_strength",
        ]

        technical_columns = [
            col
            for col in data.columns
            if col not in sentiment_columns + ["future_return"]
        ]

        # Scale features
        X_technical = self.scaler.fit_transform(data[technical_columns])
        X_sentiment = self.scaler.fit_transform(data[sentiment_columns])

        # Combine features
        X = np.concatenate([X_technical, X_sentiment], axis=1)
        y = (data["future_return"] > 0).astype(int).values

        # Create time series splits
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size : train_size + val_size]
        y_val = y[train_size : train_size + val_size]
        X_test = X[train_size + val_size :]
        y_test = y[train_size + val_size :]

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, self.sequence_length)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        return train_loader, val_loader, test_loader

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
        sentiment_dim: int,
        max_epochs: int = 100,
    ) -> MultiModalTransformer:
        """Train the model with early stopping"""
        model = MultiModalTransformer(
            input_dim=input_dim,
            sentiment_dim=sentiment_dim,
            num_layers=self.num_layers,
            learning_rate=self.learning_rate,
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="best_model",
            save_top_k=1,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            accelerator="auto",
            devices=1,
        )

        trainer.fit(model, train_loader, val_loader)

        return model

    def predict(
        self, model: MultiModalTransformer, data_loader: DataLoader
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate predictions and attention weights"""
        model.eval()
        predictions = []
        tech_attention = []
        sent_attention = []

        with torch.no_grad():
            for batch in data_loader:
                x, _ = batch
                pred, attention = model(x)
                predictions.append(pred.cpu().numpy())
                tech_attention.append(attention["technical"].cpu().numpy())
                sent_attention.append(attention["sentiment"].cpu().numpy())

        return (
            np.concatenate(predictions),
            {
                "technical": np.concatenate(tech_attention),
                "sentiment": np.concatenate(sent_attention),
            },
        )

    def generate_trading_signals(
        self, predictions: np.ndarray, threshold: float = 0.5
    ) -> pd.Series:
        """Generate trading signals from model predictions"""
        signals = pd.Series(0, index=range(len(predictions)))
        signals[predictions.squeeze() > threshold] = 1
        signals[predictions.squeeze() < (1 - threshold)] = -1
        return signals


# Example usage
if __name__ == "__main__":
    # Initialize model
    trading_model = DeepTradingModel(sequence_length=10, batch_size=32, num_layers=3)

    # Prepare your data here
    data = pd.DataFrame(...)  # Your preprocessed data

    # Prepare data loaders
    train_loader, val_loader, test_loader = trading_model.prepare_data(data)

    # Train model
    input_dim = next(iter(train_loader))[0].shape[-1]  # Total feature dimension
    sentiment_dim = 6  # Number of sentiment features

    model = trading_model.train_model(
        train_loader, val_loader, input_dim=input_dim, sentiment_dim=sentiment_dim
    )

    # Generate predictions
    predictions, attention_weights = trading_model.predict(model, test_loader)

    # Generate trading signals
    signals = trading_model.generate_trading_signals(predictions, threshold=0.6)
