"""
tuning.py - Hyperparameter tuning with Optuna.
"""

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sklearn.model_selection import cross_val_score, KFold
from src.core.evaluate import compute_rmse, temporal_cross_validate


def _xgboost_objective(trial, X, y, seasons=None):
    from xgboost import XGBRegressor
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    if seasons is not None:
        result = temporal_cross_validate(XGBRegressor, X, y, seasons, **params)
    else:
        model = XGBRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        result = {'mean_rmse': rmse_scores.mean()}

    return result['mean_rmse']


def _lightgbm_objective(trial, X, y, seasons=None):
    from lightgbm import LGBMRegressor
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    if seasons is not None:
        result = temporal_cross_validate(LGBMRegressor, X, y, seasons, **params)
    else:
        model = LGBMRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        result = {'mean_rmse': rmse_scores.mean()}

    return result['mean_rmse']


def _random_forest_objective(trial, X, y, seasons=None):
    from sklearn.ensemble import RandomForestRegressor
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': 42,
        'n_jobs': -1,
    }

    if seasons is not None:
        result = temporal_cross_validate(RandomForestRegressor, X, y, seasons, **params)
    else:
        model = RandomForestRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        result = {'mean_rmse': rmse_scores.mean()}

    return result['mean_rmse']


_OBJECTIVE_MAP = {
    'xgboost': _xgboost_objective,
    'lightgbm': _lightgbm_objective,
    'random_forest': _random_forest_objective,
}


def tune_model(model_name: str, X, y, seasons=None, n_trials: int = 50):
    """Run Optuna hyperparameter tuning for a model.

    Args:
        model_name: One of 'xgboost', 'lightgbm', 'random_forest'
        X: Feature matrix
        y: Target vector
        seasons: Optional season labels for temporal CV
        n_trials: Number of Optuna trials

    Returns:
        (best_params, best_rmse) tuple
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    if model_name not in _OBJECTIVE_MAP:
        raise ValueError(f"Tuning not supported for '{model_name}'. "
                         f"Available: {list(_OBJECTIVE_MAP.keys())}")

    objective_fn = _OBJECTIVE_MAP[model_name]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_fn(trial, X, y, seasons), n_trials=n_trials)

    print(f"\n[Tuning] Best trial:")
    print(f"  RMSE: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    return study.best_trial.params, study.best_trial.value
