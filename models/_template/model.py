"""
Model template - Copy this folder to create a new algorithm.

Usage:
    1. Copy models/_template/ to models/your_algo_name/
    2. Rename this class and implement train() and predict()
    3. Update config.yaml with your hyperparameters
    4. Run: python run_experiment.py --model your_algo_name
"""

import numpy as np
import yaml
import os
from src.base_model import BaseModel


class TemplateModel(BaseModel):
    """
    Template model - replace with your algorithm.
    """

    def __init__(self, config_path: str = None):
        # Load config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}

        super().__init__(name='template', config=config)

    def train(self, X_train, y_train):
        """Train your model here."""
        # TODO: Implement training logic
        # self.model = YourAlgorithm(**self.config)
        # self.model.fit(X_train, y_train)
        raise NotImplementedError("Implement train() in your model subclass")

    def predict(self, X) -> np.ndarray:
        """Generate predictions here."""
        # TODO: Implement prediction logic
        # return self.model.predict(X)
        raise NotImplementedError("Implement predict() in your model subclass")
