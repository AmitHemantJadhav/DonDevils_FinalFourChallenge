"""
Stacking ensemble: GBR + XGBoost + LightGBM base learners with a Ridge meta-learner.

The meta-learner is trained on out-of-fold predictions from the base models,
which prevents the stacker from overfitting to in-sample predictions.
"""

import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from src.base_model import BaseModel


# Optuna-tuned hyperparameters for base learners (12-feature set, 150 trials each)
GBR_PARAMS = {
    'n_estimators': 103, 'max_depth': 5, 'learning_rate': 0.0549697,
    'subsample': 0.877023, 'min_samples_leaf': 3, 'min_samples_split': 22,
    'max_features': 0.310804, 'random_state': 42,
}
XGB_PARAMS = {
    'n_estimators': 479, 'max_depth': 5, 'learning_rate': 0.0258471,
    'subsample': 0.767562, 'colsample_bytree': 0.524293, 'min_child_weight': 1,
    'reg_alpha': 0.514173, 'reg_lambda': 0.00016857,
    'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
}
LGBM_PARAMS = {
    'n_estimators': 232, 'max_depth': 5, 'learning_rate': 0.0151731,
    'subsample': 0.879468, 'colsample_bytree': 0.341859, 'num_leaves': 22,
    'min_child_samples': 15, 'reg_alpha': 4.37117e-07, 'reg_lambda': 7.06562e-08,
    'random_state': 42, 'n_jobs': -1, 'verbose': -1,
}


class EnsembleModel(BaseModel):

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}

        super().__init__(name='ensemble', config=config)
        self.sub_models = []
        self.meta_model = None

    def _build_base_models(self):
        """Build fresh tuned base models."""
        from sklearn.ensemble import GradientBoostingRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor

        return [
            ('gbr', GradientBoostingRegressor(**GBR_PARAMS)),
            ('xgb', XGBRegressor(**XGB_PARAMS)),
            ('lgbm', LGBMRegressor(**LGBM_PARAMS)),
        ]

    def train(self, X_train, y_train):
        X = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        base_models = self._build_base_models()
        n_models = len(base_models)
        n_samples = len(X)

        # Generate out-of-fold predictions for the meta-learner
        oof_preds = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            for model_idx, (name, model_template) in enumerate(base_models):
                # Clone model for this fold
                model = model_template.__class__(**model_template.get_params())
                model.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, model_idx] = model.predict(X[val_idx])

        # Train meta-learner on OOF predictions
        self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self.meta_model.fit(oof_preds, y)

        meta_weights = self.meta_model.coef_
        print(f"[{self.name}] Meta-learner weights: "
              f"GBR={meta_weights[0]:.3f}, XGB={meta_weights[1]:.3f}, "
              f"LGBM={meta_weights[2]:.3f}, intercept={self.meta_model.intercept_:.3f}, "
              f"alpha={self.meta_model.alpha_:.2f}")

        # Retrain base models on full data for final predictions
        self.sub_models = []
        for name, model_template in base_models:
            model = model_template.__class__(**model_template.get_params())
            model.fit(X, y)
            self.sub_models.append((name, model))

        self.model = {
            'sub_models': self.sub_models,
            'meta_model': self.meta_model,
        }
        print(f"[{self.name}] Trained stacking ensemble of {n_models} models")

    def predict(self, X) -> np.ndarray:
        if not self.sub_models and self.model:
            self.sub_models = self.model['sub_models']
            self.meta_model = self.model['meta_model']

        X_arr = np.array(X) if not isinstance(X, np.ndarray) else X

        # Get base model predictions
        base_preds = np.column_stack([
            model.predict(X_arr) for _, model in self.sub_models
        ])

        # Meta-learner combines them
        return self.meta_model.predict(base_preds)
