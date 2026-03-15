"""
Random Forest model for NCAA seed prediction.
"""

import numpy as np
import yaml
import os
from sklearn.ensemble import RandomForestRegressor
from src.base_model import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}

        super().__init__(name='random_forest', config=config)

    def train(self, X_train, y_train):
        self.model = RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 200),
            max_depth=self.config.get('max_depth', None),
            min_samples_split=self.config.get('min_samples_split', 5),
            min_samples_leaf=self.config.get('min_samples_leaf', 2),
            random_state=self.config.get('random_state', 42),
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        print(f"[{self.name}] Trained on {X_train.shape[0]} samples, {X_train.shape[1]} features")

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)
