"""
feature_engineering.py - Domain-specific feature creation for NCAA seed prediction.
"""

import pandas as pd
import numpy as np

# Top features ranked by permutation importance (temporal CV with GBR).
# Determined empirically: 8-12 features optimal for 249 training samples.
SELECTED_FEATURES = [
    'wab_pct_season', 'net_pct_season', 'conf_avg_net', 'barthag_pct_season', 'net_rank', 'prevnet',
    'netsos', 'q1_win_pct', 'is_at_large', 'total_q1q2_wins',
    'total_q3q4_losses', 'conf_wins',
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features for NCAA tournament seed prediction.

    Expects columns already parsed by preprocessing.fix_wl_columns
    (total_wins, total_losses, q1_wins, etc.) and standardized to lowercase.

    Args:
        df: Cleaned DataFrame with parsed W-L columns

    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()

    # --- Bid type features ---
    if 'bid_type' in df.columns:
        df['is_at_large'] = (df['bid_type'] == 'AL').astype(float)
        df['is_auto_qualifier'] = (df['bid_type'] == 'AQ').astype(float)
        df = df.drop(columns=['bid_type'])

    # --- NET rank improvement ---
    if 'prevnet' in df.columns and 'net_rank' in df.columns:
        df['net_rank_change'] = df['prevnet'] - df['net_rank']

    # --- Strength of schedule gap ---
    if 'netsos' in df.columns and 'netnonconfsos' in df.columns:
        df['sos_diff'] = df['netsos'] - df['netnonconfsos']

    # --- Quality win composites ---
    if 'q1_wins' in df.columns and 'q2_wins' in df.columns:
        df['total_q1q2_wins'] = df['q1_wins'].fillna(0) + df['q2_wins'].fillna(0)

    if 'q1_losses' in df.columns and 'q2_losses' in df.columns:
        df['total_q1q2_losses'] = df['q1_losses'].fillna(0) + df['q2_losses'].fillna(0)

    # --- Bad loss composite ---
    if 'q3_losses' in df.columns and 'q4_losses' in df.columns:
        df['total_q3q4_losses'] = df['q3_losses'].fillna(0) + df['q4_losses'].fillna(0)

    # --- Q1+Q2 combined win percentage ---
    q1q2_cols = ['q1_wins', 'q2_wins', 'q1_losses', 'q2_losses']
    if all(c in df.columns for c in q1q2_cols):
        q1q2_w = df['q1_wins'].fillna(0) + df['q2_wins'].fillna(0)
        q1q2_l = df['q1_losses'].fillna(0) + df['q2_losses'].fillna(0)
        q1q2_total = q1q2_w + q1q2_l
        df['q1q2_win_pct'] = np.where(q1q2_total > 0, q1q2_w / q1q2_total, 0.0)

    # --- Conference strength (mean NET rank of conference peers) ---
    if 'conference' in df.columns and 'net_rank' in df.columns:
        conf_strength = df.groupby('conference')['net_rank'].transform('mean')
        df['conf_avg_net'] = conf_strength

    # --- Road win ratio relative to total wins ---
    if 'road_wins' in df.columns and 'total_wins' in df.columns:
        df['road_win_ratio'] = np.where(
            df['total_wins'] > 0,
            df['road_wins'].fillna(0) / df['total_wins'],
            0.0,
        )

    # --- Non-conference win pct vs conference win pct gap ---
    if 'nonconf_win_pct' in df.columns and 'conf_win_pct' in df.columns:
        df['conf_nonconf_gap'] = df['conf_win_pct'] - df['nonconf_win_pct']

    # --- Average opponent NET relative to own NET ---
    if 'avgoppnetrank' in df.columns and 'net_rank' in df.columns:
        df['net_vs_opp'] = df['net_rank'] - df['avgoppnetrank']

    # --- Within-season percentile features (powerful for ranking) ---
    if 'season' in df.columns and 'net_rank' in df.columns:
        df['net_pct_season'] = df.groupby('season')['net_rank'].rank(pct=True)

    if 'season' in df.columns and 'wab' in df.columns:
        df['wab_pct_season'] = df.groupby('season')['wab'].rank(pct=True)
    elif 'season' in df.columns and 'wn_wab_rank' in df.columns:
        season_max = df.groupby('season')['wn_wab_rank'].transform('max')
        df['wab_pct_season'] = np.where(
            season_max > 0,
            (season_max - df['wn_wab_rank'] + 1) / season_max,
            np.nan,
        )

    if 'season' in df.columns and 'barthag' in df.columns:
        df['barthag_pct_season'] = df.groupby('season')['barthag'].rank(
            pct=True, ascending=False)
    elif 'season' in df.columns and 'wn_t_rank' in df.columns:
        season_max = df.groupby('season')['wn_t_rank'].transform('max')
        df['barthag_pct_season'] = np.where(
            season_max > 0,
            (season_max - df['wn_t_rank'] + 1) / season_max,
            np.nan,
        )

    # --- Efficiency margin ---
    if 'adjoe' in df.columns and 'adjde' in df.columns:
        df['adjoe_minus_adjde'] = df['adjoe'].fillna(0) - df['adjde'].fillna(0)
        if 'season' in df.columns:
            df['efficiency_margin_pct'] = df.groupby('season')[
                'adjoe_minus_adjde'].rank(pct=True, ascending=False)

    # --- WAB x BARTHAG interaction ---
    if 'wab' in df.columns and 'barthag' in df.columns:
        df['wab_x_barthag'] = df['wab'].fillna(0) * df['barthag'].fillna(0)

    return df


def select_features(X: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """Select only the top features to reduce overfitting.

    Args:
        X: Feature matrix (after encoding)
        features: List of feature names. Defaults to SELECTED_FEATURES.

    Returns:
        Filtered DataFrame with only selected features
    """
    if features is None:
        features = SELECTED_FEATURES
    available = [f for f in features if f in X.columns]
    return X[available]
