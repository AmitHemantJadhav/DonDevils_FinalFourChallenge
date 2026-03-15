#!/usr/bin/env python3
"""
run_experiment.py - CLI to train, evaluate, and compare models.

Usage:
    python run_experiment.py --model random_forest
    python run_experiment.py --model xgboost --cv 10
    python run_experiment.py --model xgboost --temporal-cv
    python run_experiment.py --compare
    python run_experiment.py --model xgboost --submit
    python run_experiment.py --model xgboost --tune
"""

import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd

from src.core.data_loader import load_training_data, load_test_data
from src.core.preprocessing import clean_data, handle_missing, encode_categoricals
from src.core.feature_engineering import create_features, select_features
from src.core.evaluate import cross_validate_model, temporal_cross_validate


# Registry of available models
MODEL_REGISTRY = {
    'random_forest': ('models.random_forest.model', 'RandomForestModel'),
    'xgboost': ('models.xgboost.model', 'XGBoostModel'),
    'lightgbm': ('models.lightgbm.model', 'LightGBMModel'),
    'ensemble': ('models.ensemble.model', 'EnsembleModel'),
}

# Columns to drop before training (non-feature columns)
DROP_COLS = ['recordid', 'team']

# Target column (after column name standardization)
TARGET_COL = 'overall_seed'


def load_model_class(model_name: str):
    """Dynamically import and return a model class."""
    if model_name not in MODEL_REGISTRY:
        module_path = f"models.{model_name}.model"
        try:
            module = importlib.import_module(module_path)
            from src.base_model import BaseModel
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseModel) and
                    attr is not BaseModel):
                    return attr
            raise ImportError(f"No BaseModel subclass found in {module_path}")
        except ImportError as e:
            print(f"Error: Model '{model_name}' not found. {e}")
            print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    module_path, class_name = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def prepare_data(include_non_tournament: bool = False):
    """Load and prepare training data.

    Returns:
        X: Feature matrix (only tournament teams by default)
        y: Target vector
        record_ids: RecordID series
        seasons: Season series (for temporal CV)
        training_columns: List of column names after encoding (for test alignment)
    """
    print("Loading training data...")
    df = load_training_data()
    df = clean_data(df)
    df = create_features(df)

    # Try external data merge
    try:
        from src.core.external_data import merge_external_features
        df = merge_external_features(df)
    except (ImportError, FileNotFoundError) as e:
        print(f"  (Skipping external data: {e})")

    # Extract metadata before encoding
    record_ids = df['recordid'] if 'recordid' in df.columns else None
    seasons = df['season'] if 'season' in df.columns else None

    # Filter to tournament teams only (those with seeds)
    if not include_non_tournament:
        has_seed = df[TARGET_COL].notna()
        print(f"  Filtering to tournament teams: {has_seed.sum()} of {len(df)} rows")
        df = df[has_seed].reset_index(drop=True)
        if record_ids is not None:
            record_ids = record_ids[has_seed].reset_index(drop=True)
        if seasons is not None:
            seasons = seasons[has_seed].reset_index(drop=True)

    y = df[TARGET_COL]

    # Drop target + non-feature columns
    cols_to_drop = [c for c in [TARGET_COL] + DROP_COLS if c in df.columns]
    # Also drop 'season' — it's metadata, not a feature (used for temporal CV separately)
    if 'season' in df.columns:
        cols_to_drop.append('season')
    # Drop 'conference' after feature engineering has used it (conf_avg_net created)
    if 'conference' in df.columns:
        cols_to_drop.append('conference')

    X = df.drop(columns=cols_to_drop)

    X = handle_missing(X)
    X = encode_categoricals(X)
    X = select_features(X)

    training_columns = X.columns.tolist()

    print(f"  Data ready: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, record_ids, seasons, training_columns


def prepare_test_data(training_columns: list):
    """Load and prepare test data, aligning columns with training.

    Returns:
        X_test: Feature matrix aligned with training columns
        record_ids: RecordID series for submission
        test_df: Full test DataFrame (for constrained assignment context)
    """
    print("Loading test data...")
    df = load_test_data()
    df = clean_data(df)

    # Keep context columns before feature engineering drops them (e.g. bid_type)
    context_cols = ['season', 'team', 'bid_type']
    test_df = df[[c for c in context_cols if c in df.columns]].copy()

    df = create_features(df)

    try:
        from src.core.external_data import merge_external_features
        df = merge_external_features(df)
    except (ImportError, FileNotFoundError) as e:
        print(f"  (Skipping external data: {e})")

    record_ids = df['recordid'] if 'recordid' in df.columns else df.iloc[:, 0]

    # Drop same columns as training (target won't exist in test)
    cols_to_drop = [c for c in DROP_COLS + ['season', 'conference', TARGET_COL]
                    if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    X = handle_missing(X)
    X = encode_categoricals(X, training_columns=training_columns)
    X = select_features(X)

    print(f"  Test data ready: {X.shape[0]} samples, {X.shape[1]} features")
    return X, record_ids, test_df


def run_single(model_name: str, cv_folds: int = 5, use_temporal_cv: bool = False):
    """Train and evaluate a single model."""
    ModelClass = load_model_class(model_name)
    model = ModelClass()

    X, y, _, seasons, training_columns = prepare_data()

    print(f"\n{'='*50}")
    print(f"Running: {model.name}")
    print(f"{'='*50}")

    # Build a fresh sklearn estimator for CV by doing a dummy train
    temp_model = ModelClass()
    temp_model.train(X.iloc[:5], y.iloc[:5])
    sklearn_estimator = temp_model.model

    # Check if this is a sklearn-compatible estimator (not ensemble's dict)
    is_sklearn = hasattr(sklearn_estimator, 'get_params')

    # Cross-validate
    if use_temporal_cv and seasons is not None:
        print(f"\nTemporal CV (leave-one-season-out):")
        if is_sklearn:
            results = temporal_cross_validate(
                type(sklearn_estimator), X, y, seasons
            )
        else:
            # Manual temporal CV for non-sklearn models (e.g. ensemble)
            from src.core.evaluate import compute_rmse
            unique_seasons = sorted(seasons.unique())
            fold_scores, fold_seasons = [], []
            for held_out in unique_seasons:
                train_mask = seasons != held_out
                val_mask = seasons == held_out
                m = ModelClass()
                m.train(X[train_mask], y[train_mask])
                preds = m.predict(X[val_mask])
                score = compute_rmse(y[val_mask], preds)
                fold_scores.append(score)
                fold_seasons.append(held_out)
                print(f"  Season {held_out}: RMSE = {score:.4f} (n={val_mask.sum()})")
            results = {
                'mean_rmse': float(np.mean(fold_scores)),
                'std_rmse': float(np.std(fold_scores)),
                'fold_scores': fold_scores,
                'fold_seasons': fold_seasons,
            }
    else:
        if is_sklearn:
            fresh_estimator = type(sklearn_estimator)(**sklearn_estimator.get_params())
            results = cross_validate_model(
                fresh_estimator, X, y, n_splits=cv_folds
            )
        else:
            # Manual k-fold CV for non-sklearn models
            from sklearn.model_selection import KFold as KF
            from src.core.evaluate import compute_rmse
            kf = KF(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_scores = []
            for train_idx, val_idx in kf.split(X):
                m = ModelClass()
                m.train(X.iloc[train_idx], y.iloc[train_idx])
                preds = m.predict(X.iloc[val_idx])
                fold_scores.append(compute_rmse(y.iloc[val_idx], preds))
            results = {
                'mean_rmse': float(np.mean(fold_scores)),
                'std_rmse': float(np.std(fold_scores)),
                'fold_scores': fold_scores,
            }

    # Full train + holdout eval
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model.train(X_train, y_train)
    rmse = model.evaluate(X_val, y_val)

    cv_type = "Temporal" if use_temporal_cv else f"{cv_folds}-fold"
    print(f"\nCross-validation ({cv_type}):")
    print(f"  Mean RMSE: {results['mean_rmse']:.4f} +/- {results['std_rmse']:.4f}")
    if 'fold_seasons' in results:
        for season, score in zip(results['fold_seasons'], results['fold_scores']):
            print(f"    {season}: {score:.4f}")
    else:
        print(f"  Fold scores: {[f'{s:.4f}' for s in results['fold_scores']]}")
    print(f"\nHoldout RMSE: {rmse:.4f}")

    # Retrain on full training data for submission
    model.train(X, y)
    model.save()

    return model, rmse, training_columns, X, y


def run_compare(cv_folds: int = 5):
    """Compare all registered models."""
    X, y, _, _, _ = prepare_data()

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    for model_name in MODEL_REGISTRY:
        try:
            ModelClass = load_model_class(model_name)
            model = ModelClass()
            model.train(X_train, y_train)
            rmse = model.evaluate(X_val, y_val)
            results.append((model_name, rmse))
        except Exception as e:
            print(f"[{model_name}] Error: {e}")
            results.append((model_name, float('inf')))

    results.sort(key=lambda x: x[1])
    print(f"\n{'='*50}")
    print(f"{'Model Comparison':^50}")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'RMSE':>10}")
    print(f"{'-'*30}")
    for name, rmse in results:
        marker = " *" if rmse == results[0][1] else ""
        print(f"{name:<20} {rmse:>10.4f}{marker}")


def run_tune(model_name: str, n_trials: int = 50):
    """Run hyperparameter tuning for a model."""
    from src.core.tuning import tune_model
    X, y, _, seasons, _ = prepare_data()
    best_params, best_rmse = tune_model(model_name, X, y, seasons, n_trials=n_trials)
    print(f"\nBest RMSE: {best_rmse:.4f}")
    print(f"Best params: {best_params}")


def main():
    parser = argparse.ArgumentParser(description='FFAC 26 - Model Experiment Runner')
    parser.add_argument('--model', type=str, help='Model name to run')
    parser.add_argument('--compare', action='store_true', help='Compare all registered models')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--temporal-cv', action='store_true',
                        help='Use leave-one-season-out CV')
    parser.add_argument('--submit', action='store_true', help='Generate submission CSV')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=50, help='Tuning trials')

    args = parser.parse_args()

    if args.compare:
        run_compare(cv_folds=args.cv)
    elif args.model:
        if args.tune:
            run_tune(args.model, n_trials=args.n_trials)
        else:
            model, rmse, training_columns, X, y = run_single(
                args.model, cv_folds=args.cv, use_temporal_cv=args.temporal_cv
            )
            if args.submit:
                print("\nGenerating submission...")
                X_test, test_ids, test_df = prepare_test_data(training_columns)

                # Multi-seed ensemble: train 5 models with different random
                # seeds and average predictions to reduce variance
                N_SEEDS = 5
                print(f"  Training {N_SEEDS} ensemble seeds for averaging...")
                all_preds = []
                from models.ensemble.model import (
                    EnsembleModel, GBR_PARAMS, XGB_PARAMS, LGBM_PARAMS,
                )
                for seed_idx in range(N_SEEDS):
                    seed_val = 42 + seed_idx * 7
                    # Override base model random states for diversity
                    GBR_PARAMS['random_state'] = seed_val
                    XGB_PARAMS['random_state'] = seed_val
                    LGBM_PARAMS['random_state'] = seed_val
                    m = EnsembleModel()
                    m.train(X, y)
                    preds = m.predict(X_test)
                    all_preds.append(preds)
                    print(f"    Seed {seed_val}: mean pred = {preds.mean():.2f}")
                # Restore original random states
                GBR_PARAMS['random_state'] = 42
                XGB_PARAMS['random_state'] = 42
                LGBM_PARAMS['random_state'] = 42

                raw_predictions = np.mean(all_preds, axis=0)
                print(f"  Averaged predictions: mean = {raw_predictions.mean():.2f}, "
                      f"std = {raw_predictions.std():.2f}")

                # Apply constrained seed assignment
                try:
                    from src.core.external_data import (
                        get_traditional_seeds, get_barttorvik_stats,
                    )
                    from src.core.seed_assignment import constrained_seed_assignment

                    # Load clean training data for occupied slot lookup
                    train_raw = load_training_data()
                    train_clean = clean_data(train_raw)

                    # Get traditional seeds and barttorvik stats
                    combined = pd.concat([train_clean, test_df], ignore_index=True)
                    trad_seeds = get_traditional_seeds(combined)
                    bart_stats = get_barttorvik_stats(combined)

                    predictions = constrained_seed_assignment(
                        test_df, train_clean, raw_predictions, trad_seeds,
                        barttorvik_stats=bart_stats,
                    )
                    print(f"  Constrained predictions: "
                          f"{(predictions == 0).sum()} zeros, "
                          f"{(predictions > 0).sum()} tournament")
                except Exception as e:
                    print(f"  Constrained assignment failed ({e}), using raw predictions")
                    import traceback
                    traceback.print_exc()
                    predictions = raw_predictions

                from src.core.submission import generate_submission
                generate_submission(test_ids.tolist(), predictions)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
