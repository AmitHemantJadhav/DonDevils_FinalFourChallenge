"""
submission.py - Generate submission CSV in the competition's required format.
"""

import os
import pandas as pd
import numpy as np


def generate_submission(
    record_ids: list,
    predictions: np.ndarray,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate a submission file in the required format.

    Expected format:
        RecordID, Overall Seed
        2020-21-Baylor, 3
        2020-21-Arkansas, 6

    Args:
        record_ids: List of RecordID values from the test set
        predictions: Array of predicted seeds
        output_path: Path to save CSV. If None, saves to submissions/ dir.

    Returns:
        Submission DataFrame
    """
    # Round predictions to nearest integer and clip to valid seed range
    # 0 is valid for non-tournament teams
    predictions = np.clip(np.round(predictions), 0, 68).astype(int)

    submission = pd.DataFrame({
        'RecordID': record_ids,
        'Overall Seed': predictions,
    })

    if output_path is None:
        submissions_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'submissions'
        )
        os.makedirs(submissions_dir, exist_ok=True)
        output_path = os.path.join(submissions_dir, 'submission.csv')

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} predictions)")

    return submission
