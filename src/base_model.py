"""
base_model.py - Abstract base class for all model implementations.

Every algorithm in models/<algorithm>/model.py should subclass BaseModel
and implement the required methods. This ensures all models are interchangeable
and can be run through run_experiment.py.
"""

from abc import ABC, abstractmethod
import numpy as np
import os
import pickle

from src.core.evaluate import compute_rmse
from src.core.submission import generate_submission


class BaseModel(ABC):
    """Abstract base class for NCAA seed prediction models."""

    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.config = config or {}
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Generate predictions for the given features."""
        pass

    def evaluate(self, X_val, y_val) -> float:
        """Evaluate the model and return RMSE."""
        predictions = self.predict(X_val)
        rmse = compute_rmse(y_val, predictions)
        print(f"[{self.name}] RMSE: {rmse:.4f}")
        return rmse

    def create_submission(self, X_test, record_ids, output_path=None) -> None:
        """Generate a submission CSV from test predictions."""
        predictions = self.predict(X_test)
        generate_submission(record_ids, predictions, output_path)

    def save(self, path: str = None) -> None:
        """Save the trained model to disk."""
        if path is None:
            path = os.path.join('models', self.name, 'results', 'model.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[{self.name}] Model saved to {path}")

    def load(self, path: str = None) -> None:
        """Load a trained model from disk."""
        if path is None:
            path = os.path.join('models', self.name, 'results', 'model.pkl')
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"[{self.name}] Model loaded from {path}")

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
