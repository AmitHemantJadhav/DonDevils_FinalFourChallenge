"""
evaluate.py - RMSE scoring and cross-validation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold


def compute_rmse(y_true, y_pred) -> float:
    """Compute Root Mean Squared Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cross_validate_model(model, X, y, n_splits: int = 5, random_state: int = 42) -> dict:
    """Run K-fold cross-validation and return RMSE statistics.

    Args:
        model: A scikit-learn compatible estimator
        X: Feature matrix
        y: Target vector
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        Dict with 'mean_rmse', 'std_rmse', 'fold_scores'
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(
        model, X, y,
        cv=kf,
        scoring='neg_mean_squared_error'
    )

    rmse_scores = np.sqrt(-scores)

    return {
        'mean_rmse': float(rmse_scores.mean()),
        'std_rmse': float(rmse_scores.std()),
        'fold_scores': rmse_scores.tolist(),
    }


def temporal_cross_validate(model_class, X, y, seasons, **model_kwargs) -> dict:
    """Leave-one-season-out cross-validation to prevent temporal leakage.

    Args:
        model_class: A class with fit() and predict() methods (sklearn estimator)
        X: Feature matrix (DataFrame or array)
        y: Target vector
        seasons: Series/array of season labels aligned with X
        **model_kwargs: Keyword args passed to model_class()

    Returns:
        Dict with 'mean_rmse', 'std_rmse', 'fold_scores', 'fold_seasons'
    """
    unique_seasons = sorted(seasons.unique())
    fold_scores = []
    fold_seasons = []

    for held_out in unique_seasons:
        train_mask = seasons != held_out
        val_mask = seasons == held_out

        if val_mask.sum() == 0:
            continue

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = compute_rmse(y_val, preds)

        fold_scores.append(rmse)
        fold_seasons.append(held_out)
        print(f"  Season {held_out}: RMSE = {rmse:.4f} (n={val_mask.sum()})")

    rmse_arr = np.array(fold_scores)
    return {
        'mean_rmse': float(rmse_arr.mean()),
        'std_rmse': float(rmse_arr.std()),
        'fold_scores': fold_scores,
        'fold_seasons': fold_seasons,
    }
