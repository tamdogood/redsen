import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class MarkovStateDataset(Dataset):
    """Dataset for Markov state transitions"""

    def __init__(
        self, data: pd.DataFrame, num_states: int = 5, sequence_length: int = 10
    ):
        self.sequence_length = sequence_length
        self.num_states = num_states

        # Discretize states
        self.states = self._discretize_states(data)

        # Create sequences
        self.sequences = []
        self.next_states = []

        for i in range(len(self.states) - sequence_length):
            self.sequences.append(self.states[i : i + sequence_length])
            self.next_states.append(self.states[i + sequence_length])

    def _discretize_states(self, data: pd.DataFrame) -> np.ndarray:
        """Convert continuous data to discrete states"""
        states = np.zeros(len(data))

        # Price movement
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=20).std()
        sentiment = data["comment_sentiment_avg"]

        # Create combined state based on returns, volatility, and sentiment
        for i in range(len(data)):
            if i < 20:
                states[i] = 2  # Neutral state for initial points
                continue

            ret = returns.iloc[i]
            vol = volatility.iloc[i]
            sent = sentiment.iloc[i]

            if ret > 0.02 and sent > 0.5:
                states[i] = 4  # Strong bullish
            elif ret > 0 and sent > 0:
                states[i] = 3  # Moderate bullish
            elif ret < -0.02 and sent < -0.5:
                states[i] = 0  # Strong bearish
            elif ret < 0 and sent < 0:
                states[i] = 1  # Moderate bearish
            else:
                states[i] = 2  # Neutral

        return states

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.LongTensor(self.sequences[idx])
        next_state = torch.LongTensor([self.next_states[idx]])
        return sequence, next_state


class MarkovTransitionMatrix(nn.Module):
    """Neural Markov Chain transition matrix"""

    def __init__(self, num_states: int, hidden_dim: int = 64):
        super().__init__()
        self.num_states = num_states

        # Embedding for state sequences
        self.state_embedding = nn.Embedding(num_states, hidden_dim)

        # LSTM to process state sequences
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # Output layer for transition probabilities
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed states
        embedded = self.state_embedding(x)  # [batch, seq_len, hidden_dim]

        # Process with LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim]

        # Calculate attention weights
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_dim]

        # Calculate transition probabilities
        transition_probs = self.transition(context)  # [batch, num_states]

        return transition_probs, attention_weights


class MarkovChainModel(pl.LightningModule):
    """PyTorch Lightning module for Markov Chain model"""

    def __init__(self, num_states: int = 5, hidden_dim: int = 64, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = MarkovTransitionMatrix(num_states, hidden_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        probs, _ = self(x)
        loss = self.criterion(probs, y.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        probs, _ = self(x)
        loss = self.criterion(probs, y.squeeze())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class MarkovTrader:
    """Trading system using Markov Chain predictions"""

    def __init__(
        self, num_states: int = 5, sequence_length: int = 10, hidden_dim: int = 64
    ):
        self.num_states = num_states
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.model = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        dataset = MarkovStateDataset(
            data, num_states=self.num_states, sequence_length=self.sequence_length
        )

        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )

        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4
        )

        return train_loader, val_loader

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> MarkovChainModel:
        """Train the Markov Chain model"""
        model = MarkovChainModel(num_states=self.num_states, hidden_dim=self.hidden_dim)

        trainer = pl.Trainer(
            max_epochs=100,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
            ],
            accelerator="auto",
            devices=1,
        )

        trainer.fit(model, train_loader, val_loader)
        self.model = model

        return model

    def predict_next_state(self, state_sequence: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict next state and probabilities"""
        if self.model is None:
            raise ValueError("Model must be trained first")

        self.model.eval()
        with torch.no_grad():
            x = torch.LongTensor(state_sequence).unsqueeze(0)
            probs, attention = self.model(x)

            # Get most likely next state
            next_state = torch.argmax(probs).item()

            return next_state, probs.numpy()[0]

    def generate_trading_signals(
        self, predictions: np.ndarray, threshold: float = 0.6
    ) -> pd.Series:
        """Generate trading signals based on state predictions"""
        signals = pd.Series(0, index=range(len(predictions)))

        # Strong bullish states (3, 4) generate buy signals
        signals[predictions >= 3] = 1

        # Strong bearish states (0, 1) generate sell signals
        signals[predictions <= 1] = -1

        return signals

    def calculate_state_metrics(
        self, true_states: np.ndarray, pred_states: np.ndarray
    ) -> Dict:
        """Calculate performance metrics"""
        accuracy = (true_states == pred_states).mean()

        # Calculate transition matrix
        transition_matrix = np.zeros((self.num_states, self.num_states))
        for i in range(len(true_states) - 1):
            current_state = true_states[i]
            next_state = true_states[i + 1]
            transition_matrix[int(current_state), int(next_state)] += 1

        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

        return {"accuracy": accuracy, "transition_matrix": transition_matrix}


def add_markov_predictions(
    data: pd.DataFrame, markov_trader: MarkovTrader
) -> pd.DataFrame:
    """Add Markov Chain predictions to the dataset"""
    # Create state sequences
    sequences = []
    for i in range(len(data) - markov_trader.sequence_length):
        seq = data.iloc[i : i + markov_trader.sequence_length]
        next_state, probs = markov_trader.predict_next_state(seq)
        sequences.append(
            {
                "predicted_state": next_state,
                "bull_probability": probs[3]
                + probs[4],  # Probability of bullish states
                "bear_probability": probs[0]
                + probs[1],  # Probability of bearish states
            }
        )

    predictions_df = pd.DataFrame(sequences)

    # Add predictions to original data
    data = data.copy()
    data.loc[markov_trader.sequence_length :, "predicted_state"] = predictions_df[
        "predicted_state"
    ].values
    data.loc[markov_trader.sequence_length :, "bull_probability"] = predictions_df[
        "bull_probability"
    ].values
    data.loc[markov_trader.sequence_length :, "bear_probability"] = predictions_df[
        "bear_probability"
    ].values

    return data
