"""
preprocessing.py - Data cleaning, encoding, and imputation.
"""

import pandas as pd
import numpy as np
import re

# Month abbreviations Excel uses when it corrupts "W-L" strings into dates
_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

# Columns that contain W-L records (subject to Excel date corruption)
WL_COLUMNS = [
    'WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL',
    'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4',
]


def _parse_wl_value(val: str) -> tuple:
    """Parse a W-L string that may be Excel-corrupted into (wins, losses).

    Examples:
        "8-Sep"  -> (8, 9)    # Excel turned "8-9" into a date
        "19-7"   -> (19, 7)   # normal
        "0-0"    -> (0, 0)
        "1-Jan"  -> (1, 1)
        NaN      -> (NaN, NaN)
    """
    if pd.isna(val):
        return np.nan, np.nan

    val = str(val).strip()
    if not val or val.lower() == 'nan':
        return np.nan, np.nan

    parts = val.split('-')
    if len(parts) != 2:
        return np.nan, np.nan

    left, right = parts[0].strip(), parts[1].strip()

    # Parse left (always numeric = wins)
    try:
        wins = int(left)
    except ValueError:
        # Left side could also be a month name in rare cases
        wins = _MONTH_MAP.get(left.lower())
        if wins is None:
            return np.nan, np.nan

    # Parse right (could be numeric or a month name)
    try:
        losses = int(right)
    except ValueError:
        losses = _MONTH_MAP.get(right.lower())
        if losses is None:
            return np.nan, np.nan

    return wins, losses


def fix_wl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix Excel date-corrupted W-L columns and extract wins/losses/win_pct.

    For each W-L column, creates three new columns:
        <prefix>_wins, <prefix>_losses, <prefix>_win_pct
    Then drops the original W-L column.
    """
    df = df.copy()

    col_prefix_map = {
        'WL': 'total',
        'Conf.Record': 'conf',
        'Non-ConferenceRecord': 'nonconf',
        'RoadWL': 'road',
        'Quadrant1': 'q1',
        'Quadrant2': 'q2',
        'Quadrant3': 'q3',
        'Quadrant4': 'q4',
    }

    for col, prefix in col_prefix_map.items():
        if col not in df.columns:
            continue

        parsed = df[col].apply(_parse_wl_value)
        df[f'{prefix}_wins'] = parsed.apply(lambda x: x[0]).astype(float)
        df[f'{prefix}_losses'] = parsed.apply(lambda x: x[1]).astype(float)

        total = df[f'{prefix}_wins'] + df[f'{prefix}_losses']
        df[f'{prefix}_win_pct'] = np.where(total > 0, df[f'{prefix}_wins'] / total, 0.0)

        df = df.drop(columns=[col])

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning pipeline."""
    df = df.copy()

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Fix W-L date corruption (before standardizing column names so we match original names)
    df = fix_wl_columns(df)

    # Standardize column names: lowercase, replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    return df


def handle_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Handle missing values in numeric columns.

    Args:
        df: Input DataFrame
        strategy: 'median', 'mean', or 'drop'
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    if strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def encode_categoricals(df: pd.DataFrame, columns: list = None,
                        training_columns: list = None) -> pd.DataFrame:
    """One-hot encode categorical columns.

    Args:
        df: Input DataFrame
        columns: List of columns to encode. If None, encode all object columns.
        training_columns: If provided, reindex to match training columns (for test data).
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include='object').columns.tolist()

    df = pd.get_dummies(df, columns=columns, drop_first=True)

    if training_columns is not None:
        df = df.reindex(columns=training_columns, fill_value=0)

    return df
