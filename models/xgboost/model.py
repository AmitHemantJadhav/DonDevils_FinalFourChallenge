"""
XGBoost model for NCAA seed prediction.
"""

import numpy as np
import yaml
import os
from xgboost import XGBRegressor
from src.base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}

        super().__init__(name='xgboost', config=config)

    def train(self, X_train, y_train):
        self.model = XGBRegressor(
            n_estimators=self.config.get('n_estimators', 300),
            max_depth=self.config.get('max_depth', 6),
            learning_rate=self.config.get('learning_rate', 0.1),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            random_state=self.config.get('random_state', 42),
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X_train, y_train)
        print(f"[{self.name}] Trained on {X_train.shape[0]} samples, {X_train.shape[1]} features")

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)
