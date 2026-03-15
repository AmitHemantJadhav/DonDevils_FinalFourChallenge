"""
data_loader.py - Load and validate competition CSV files.
"""

import os
import pandas as pd


DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

EXPECTED_FILES = {
    'train': 'NCAA_Seed_Training_Set2.0.csv',
    'test': 'NCAA_Seed_Test_Set2.0.csv',
    'template': 'submission_template2.0.csv',
    'dictionary': 'FFAC Data Dictionary.xlsx',
}


def get_data_path(key: str, raw: bool = True) -> str:
    """Get absolute path to a data file."""
    base_dir = DATA_RAW_DIR if raw else DATA_PROCESSED_DIR
    filename = EXPECTED_FILES.get(key, key)
    return os.path.abspath(os.path.join(base_dir, filename))


def load_training_data() -> pd.DataFrame:
    """Load the training dataset."""
    path = get_data_path('train')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Download from BARK portal and place in data/raw/"
        )
    return pd.read_csv(path)


def load_test_data() -> pd.DataFrame:
    """Load the test dataset."""
    path = get_data_path('test')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test data not found at {path}. "
            "Download from BARK portal and place in data/raw/"
        )
    return pd.read_csv(path)


def load_submission_template() -> pd.DataFrame:
    """Load the submission template."""
    path = get_data_path('template')
    return pd.read_csv(path)


def load_data_dictionary() -> pd.DataFrame:
    """Load the data dictionary."""
    path = get_data_path('dictionary')
    return pd.read_excel(path)


def validate_data(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Return a summary dict of dataset health."""
    return {
        'name': name,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
    }
