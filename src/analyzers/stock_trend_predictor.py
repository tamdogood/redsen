import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StockTrendPredictor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = None

    def preprocess_data(self):
        """Preprocess the data for model training"""
        # Fill missing values or drop them
        self.data.fillna(method="ffill", inplace=True)

        # Feature selection
        self.features = self.data[
            [
                "current_price",
                "price_change_2w",
                "price_change_2d",
                "avg_volume",
                "sma_20",
                "rsi",
                "volatility",
                "market_cap",
                "pe_ratio",
                "dividend_yield",
                "beta",
                "profit_margins",
                "revenue_growth",
                "target_price",
                "short_ratio",
            ]
        ]
        self.labels = self.data["price_change_2w"].apply(
            lambda x: 1 if x > 0 else 0
        )  # Example label

    def train_model(self):
        """Train a machine learning model to predict stock movements"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Model accuracy: {accuracy}")

    def predict_stock_movement(self, new_data: pd.DataFrame):
        """Predict stock movement using the trained model"""
        if self.model is None:
            raise Exception("Model is not trained yet.")
        return self.model.predict(new_data)

    def analyze_correlation(self):
        """Analyze the correlation between stock metrics and price movement"""
        correlation_matrix = self.data.corr()
        logger.info("Correlation matrix:\n" + str(correlation_matrix))
        return correlation_matrix

    def evaluate_model(self):
        """Evaluate the model with additional metrics"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logger.info(f"Model Mean Squared Error: {mse}")


# Example usage
if __name__ == "__main__":
    pass
    # Load your data into a DataFrame
    # df = pd.read_csv("your_data.csv")

    # Initialize the predictor
    # predictor = StockTrendPredictor(df)

    # Preprocess data
    # predictor.preprocess_data()

    # Train the model
    # predictor.train_model()

    # Analyze correlation
    # predictor.analyze_correlation()

    # Evaluate the model
    # predictor.evaluate_model()

    # Predict new data
    # new_data = pd.DataFrame(...)  # New data for prediction
    # predictions = predictor.predict_stock_movement(new_data)
    # print(predictions)
