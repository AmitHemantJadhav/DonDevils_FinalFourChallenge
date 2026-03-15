"""
external_data.py - Merge external basketball statistics (barttorvik/Kaggle).

Source: Andrew Sundberg's College Basketball Dataset
Download: kaggle datasets download -d andrewsundberg/college-basketball-dataset -p data/external/ --unzip
"""

import os
import pandas as pd
import numpy as np

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'external')

# Season label -> barttorvik file year
_SEASON_FILE_MAP = {
    '2020-21': 21,
    '2021-22': 22,
    '2022-23': 23,
    '2023-24': 24,
    '2024-25': 25,
}

# Team name normalization: competition name -> barttorvik name
_NAME_MAP = {
    # --- Abbreviations / nicknames ---
    'UConn': 'Connecticut',
    'Uconn': 'Connecticut',
    'N.C. State': 'North Carolina St.',
    'NC State': 'North Carolina St.',
    'UNC': 'North Carolina',
    'Ole Miss': 'Mississippi',
    'Pitt': 'Pittsburgh',
    'SMU': 'SMU',
    'UCF': 'UCF',
    'USC': 'USC',
    'LSU': 'LSU',
    'VCU': 'VCU',
    'BYU': 'BYU',
    'UAB': 'UAB',
    'FDU': 'Fairleigh Dickinson',
    'FAU': 'Florida Atlantic',
    'Southern California': 'USC',
    'Miami (FL)': 'Miami FL',
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Peter's": "Saint Peter's",
    "St. John's (NY)": "St. John's",
    "Mount St. Mary's": "Mount St. Mary's",
    'UC San Diego': 'UC San Diego',
    'Loyola Chicago': 'Loyola Chicago',

    # --- "St." abbreviations (already match) ---
    'Michigan St.': 'Michigan St.',
    'Ohio St.': 'Ohio St.',
    'Penn St.': 'Penn St.',
    'Iowa St.': 'Iowa St.',
    'Kansas St.': 'Kansas St.',
    'Colorado St.': 'Colorado St.',
    'Oregon St.': 'Oregon St.',
    'San Diego St.': 'San Diego St.',
    'Boise St.': 'Boise St.',
    'Wichita St.': 'Wichita St.',
    'Mississippi St.': 'Mississippi St.',
    'Arizona St.': 'Arizona St.',
    'Fresno St.': 'Fresno St.',
    'Utah St.': 'Utah St.',

    # --- Abbreviated names -> full barttorvik names ---
    'App State': 'Appalachian St.',
    'A&M-Corpus Christi': 'Texas A&M Corpus Chris',
    'Alcorn': 'Alcorn St.',
    'Ark.-Pine Bluff': 'Arkansas Pine Bluff',
    'Army West Point': 'Army',
    'Boston U.': 'Boston University',
    'CSU Bakersfield': 'Cal St. Bakersfield',
    'CSUN': 'Cal St. Northridge',
    'California Baptist': 'Cal Baptist',
    'Central Ark.': 'Central Arkansas',
    'Central Conn. St.': 'Central Connecticut',
    'Central Mich.': 'Central Michigan',
    'Charleston So.': 'Charleston Southern',
    'Col. of Charleston': 'College of Charleston',
    'Detroit Mercy': 'Detroit',
    'ETSU': 'East Tennessee St.',
    'East Texas A&M': 'Texas A&M Commerce',
    'Eastern Ill.': 'Eastern Illinois',
    'Eastern Ky.': 'Eastern Kentucky',
    'Eastern Mich.': 'Eastern Michigan',
    'Eastern Wash.': 'Eastern Washington',
    'FGCU': 'Florida Gulf Coast',
    'Fla. Atlantic': 'Florida Atlantic',
    'Ga. Southern': 'Georgia Southern',
    'Gardner-Webb': 'Gardner Webb',
    'Grambling': 'Grambling St.',
    'Grambling St.': 'Grambling St.',
    'Houston Christian': 'Houston Christian',
    'IU Indy': 'IU Indy',
    'Kansas City': 'UMKC',
    'LIU': 'LIU',
    'LMU (CA)': 'Loyola Marymount',
    'Lamar University': 'Lamar',
    'Louisiana': 'Louisiana Lafayette',
    'Loyola Maryland': 'Loyola MD',
    'McNeese': 'McNeese St.',
    'McNeese St.': 'McNeese St.',
    'Miami (OH)': 'Miami OH',
    'Middle Tenn.': 'Middle Tennessee',
    'Mississippi Val.': 'Mississippi Valley St.',
    'N.C. A&T': 'North Carolina A&T',
    'N.C. Central': 'North Carolina Central',
    'NIU': 'Northern Illinois',
    'Nicholls': 'Nicholls St.',
    'North Ala.': 'North Alabama',
    'Northern Ariz.': 'Northern Arizona',
    'Northern Colo.': 'Northern Colorado',
    'Northern Ky.': 'Northern Kentucky',
    'Omaha': 'Nebraska Omaha',
    'Penn': 'Penn',
    'Prairie View': 'Prairie View A&M',
    'Purdue Fort Wayne': 'Purdue Fort Wayne',
    'Queens (NC)': 'Queens',
    'SFA': 'Stephen F. Austin',
    'SIUE': 'SIU Edwardsville',
    'Saint Francis': 'St. Francis PA',
    'Sam Houston': 'Sam Houston St.',
    'Seattle U': 'Seattle',
    'South Fla.': 'South Florida',
    'Southeast Mo. St.': 'Southeast Missouri St.',
    'Southeastern La.': 'Southeastern Louisiana',
    'Southern Ill.': 'Southern Illinois',
    'Southern Ind.': 'Southern Indiana',
    'Southern U.': 'Southern',
    'St. Francis Brooklyn': 'St. Francis NY',
    'St. Thomas (MN)': 'St. Thomas',
    'UAlbany': 'Albany',
    'UIC': 'UIC',
    'UIW': 'Incarnate Word',
    'ULM': 'Louisiana Monroe',
    'UMES': 'Maryland Eastern Shore',
    'UNI': 'Northern Iowa',
    'UNCW': 'UNC Wilmington',
    'UT Martin': 'Tennessee Martin',
    'UTRGV': 'UT Rio Grande Valley',
    'Utah Tech': 'Utah Tech',
    'West Ga.': 'West Georgia',
    'Western Caro.': 'Western Carolina',
    'Western Ill.': 'Western Illinois',
    'Western Ky.': 'Western Kentucky',
    'Western Mich.': 'Western Michigan',

    # --- Ivy League ---
    'Harvard': 'Harvard',
    'Yale': 'Yale',
    'Princeton': 'Princeton',
    'Brown': 'Brown',
    'Columbia': 'Columbia',
    'Cornell': 'Cornell',
    'Dartmouth': 'Dartmouth',

    # --- Bethune-Cookman ---
    'Bethune-Cookman': 'Bethune Cookman',
}

# Year-specific overrides (barttorvik name changes across years)
_YEAR_NAME_OVERRIDES = {
    21: {
        'LIU': 'LIU Brooklyn',
        'IU Indy': 'IUPUI',
        'Utah Tech': 'Dixie St.',
        'Detroit Mercy': 'Detroit',
        'Col. of Charleston': 'College of Charleston',
        'Houston Christian': 'Houston Baptist',
    },
    22: {
        'LIU': 'LIU Brooklyn',
        'IU Indy': 'IUPUI',
        'Detroit Mercy': 'Detroit',
        'Col. of Charleston': 'College of Charleston',
    },
    23: {
        'LIU': 'LIU Brooklyn',
        'IU Indy': 'IUPUI',
        'Detroit Mercy': 'Detroit',
        'Col. of Charleston': 'College of Charleston',
    },
    24: {
        'LIU': 'LIU Brooklyn',
        'IU Indy': 'IUPUI',
        'Detroit Mercy': 'Detroit',
        'Col. of Charleston': 'College of Charleston',
    },
    25: {
        'Col. of Charleston': 'Charleston',
        'Detroit Mercy': 'Detroit Mercy',
    },
}

# Columns to extract from external data
MERGE_COLS = [
    'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
    'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD',
    '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB',
]


def _normalize_name(name: str, year_suffix: int = None) -> str:
    """Normalize a team name for matching, with optional year-specific overrides."""
    if year_suffix and year_suffix in _YEAR_NAME_OVERRIDES:
        override = _YEAR_NAME_OVERRIDES[year_suffix].get(name)
        if override:
            return override
    return _NAME_MAP.get(name, name)


def _load_year_file(year_suffix: int) -> pd.DataFrame:
    """Load a single barttorvik year file (e.g. cbb21.csv)."""
    path = os.path.join(os.path.abspath(EXTERNAL_DIR), f'cbb{year_suffix}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"External file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names to uppercase
    df.columns = [c.upper() for c in df.columns]

    # Fix inconsistent column names across years
    renames = {}
    if 'EFGD_D' in df.columns:
        renames['EFGD_D'] = 'EFG_D'
    if 'EFG%' in df.columns:
        renames['EFG%'] = 'EFG_O'
    if 'EFGD%' in df.columns:
        renames['EFGD%'] = 'EFG_D'
    if renames:
        df = df.rename(columns=renames)

    df['YEAR_SUFFIX'] = year_suffix
    return df


def get_traditional_seeds(df: pd.DataFrame) -> dict:
    """Extract traditional NCAA seeds (1-16) from barttorvik for all teams.

    Args:
        df: Competition DataFrame with 'season' and 'team' columns (lowercase).

    Returns:
        Dict of {(season, team): traditional_seed} for teams with a barttorvik SEED.
    """
    if 'season' not in df.columns or 'team' not in df.columns:
        return {}

    result = {}
    for season in df['season'].unique():
        year_suffix = _SEASON_FILE_MAP.get(season)
        if year_suffix is None:
            continue
        try:
            ext = _load_year_file(year_suffix)
        except FileNotFoundError:
            continue

        if 'SEED' not in ext.columns:
            continue

        # Build barttorvik_name -> traditional seed lookup
        seed_lookup = {}
        for _, row in ext.iterrows():
            team_name = row['TEAM'] if 'TEAM' in ext.columns else str(row.iloc[0])
            if pd.notna(row['SEED']):
                seed_lookup[team_name] = int(row['SEED'])

        # Match competition team names to barttorvik names
        season_teams = df[df['season'] == season]['team'].unique()
        for comp_name in season_teams:
            norm_name = _normalize_name(comp_name, year_suffix)

            trad_seed = seed_lookup.get(comp_name) or seed_lookup.get(norm_name)
            if trad_seed is None:
                # Fuzzy match
                clean = comp_name.replace('.', '').replace("'", "\u2019")
                for ext_name, seed_val in seed_lookup.items():
                    ext_clean = ext_name.replace('.', '').replace("'", "\u2019")
                    if clean == ext_clean or clean.lower() == ext_clean.lower():
                        trad_seed = seed_val
                        break

            if trad_seed is not None:
                result[(season, comp_name)] = trad_seed

    return result


def get_barttorvik_stats(df: pd.DataFrame) -> dict:
    """Extract ADJOE and ADJDE from barttorvik for all teams.

    Args:
        df: Competition DataFrame with 'season' and 'team' columns.

    Returns:
        Dict of {(season, team): {'ADJOE': x, 'ADJDE': y}} for matched teams.
    """
    if 'season' not in df.columns or 'team' not in df.columns:
        return {}

    result = {}
    for season in df['season'].unique():
        year_suffix = _SEASON_FILE_MAP.get(season)
        if year_suffix is None:
            continue
        try:
            ext = _load_year_file(year_suffix)
        except FileNotFoundError:
            continue

        # Build barttorvik_name -> stats lookup
        stats_lookup = {}
        for _, row in ext.iterrows():
            team_name = row['TEAM'] if 'TEAM' in ext.columns else str(row.iloc[0])
            stats = {}
            if 'ADJOE' in ext.columns and pd.notna(row['ADJOE']):
                stats['ADJOE'] = float(row['ADJOE'])
            if 'ADJDE' in ext.columns and pd.notna(row['ADJDE']):
                stats['ADJDE'] = float(row['ADJDE'])
            if stats:
                stats_lookup[team_name] = stats

        # Match competition team names to barttorvik names
        season_teams = df[df['season'] == season]['team'].unique()
        for comp_name in season_teams:
            norm_name = _normalize_name(comp_name, year_suffix)

            stats = stats_lookup.get(comp_name) or stats_lookup.get(norm_name)
            if stats is None:
                # Fuzzy match
                clean = comp_name.replace('.', '').replace("'", "\u2019")
                for ext_name, ext_stats in stats_lookup.items():
                    ext_clean = ext_name.replace('.', '').replace("'", "\u2019")
                    if clean == ext_clean or clean.lower() == ext_clean.lower():
                        stats = ext_stats
                        break

            if stats:
                result[(season, comp_name)] = stats

    return result


def merge_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge external barttorvik stats into the competition DataFrame.

    Args:
        df: Competition DataFrame (already cleaned, with lowercase column names)

    Returns:
        DataFrame with additional advanced stat columns
    """
    if 'season' not in df.columns:
        print("  Warning: No season column, skipping external merge")
        return df

    team_col = 'team' if 'team' in df.columns else None
    if team_col is None:
        print("  Warning: No team column, skipping external merge")
        return df

    df = df.copy()
    total_matched = 0
    total_rows = 0

    # Process each season separately
    result_dfs = []
    for season in df['season'].unique():
        year_suffix = _SEASON_FILE_MAP.get(season)
        if year_suffix is None:
            result_dfs.append(df[df['season'] == season])
            continue

        season_df = df[df['season'] == season].copy()
        total_rows += len(season_df)

        try:
            ext = _load_year_file(year_suffix)
        except FileNotFoundError:
            result_dfs.append(season_df)
            continue

        # Build lookup dict from external data
        available_cols = [c for c in MERGE_COLS if c in ext.columns]
        ext_lookup = {}
        for _, row in ext.iterrows():
            team_name = row['TEAM'] if 'TEAM' in ext.columns else str(row.iloc[0])
            ext_lookup[team_name] = {c: row[c] for c in available_cols if pd.notna(row[c])}

        # Match each competition team
        for col in available_cols:
            season_df[col.lower()] = np.nan

        for idx in season_df.index:
            comp_name = season_df.loc[idx, team_col]
            norm_name = _normalize_name(comp_name, year_suffix)

            # Try exact match first, then normalized
            ext_row = ext_lookup.get(comp_name) or ext_lookup.get(norm_name)

            if ext_row is None:
                # Fuzzy: try removing periods and common suffixes
                clean = comp_name.replace('.', '').replace("'", "'")
                for ext_name in ext_lookup:
                    ext_clean = ext_name.replace('.', '').replace("'", "'")
                    if clean == ext_clean or clean.lower() == ext_clean.lower():
                        ext_row = ext_lookup[ext_name]
                        break

            if ext_row:
                total_matched += 1
                for col_name, val in ext_row.items():
                    season_df.loc[idx, col_name.lower()] = val

        result_dfs.append(season_df)

    merged = pd.concat(result_dfs, ignore_index=True)
    match_pct = total_matched / total_rows * 100 if total_rows > 0 else 0
    print(f"  External data: matched {total_matched}/{total_rows} teams ({match_pct:.1f}%)")

    return merged
