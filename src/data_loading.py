"""
Data Loading Module for Cognifyz Internship Level 2
Handles loading and basic validation of the restaurant dataset.
"""

import os
from typing import Optional
import pandas as pd


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the restaurant dataset from the specified path or default location.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the dataset CSV file. If None, tries default local paths.
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    default_paths = [
        file_path,
        os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv"),
        os.path.join("data", "dataset.csv"),
        r"C:\Users\varun\-Data-Exploration-and-Preprocessing\data\dataset.csv"
    ]
    
    valid_paths = [p for p in default_paths if p is not None and os.path.exists(p)]
    
    if not valid_paths:
        raise FileNotFoundError(
            "Could not locate dataset.csv. Please ensure it exists in the 'data/' directory."
        )
    
    target_path = valid_paths[0]
    print(f"[INFO] Loading dataset from: {os.path.abspath(target_path)}")
    
    # Try utf-8 first, fallback to latin-1 for special characters in restaurant names
    try:
        df = pd.read_csv(target_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("[WARN] UTF-8 decoding failed, falling back to 'latin-1' encoding.")
        df = pd.read_csv(target_path, encoding='latin-1')
        
    print(f"[INFO] Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return basic summary statistics of the dataset.
    """
    summary = {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values_total": int(df.isnull().sum().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    return summary


if __name__ == "__main__":
    df = load_dataset()
    summary = get_dataset_summary(df)
    print("\nDataset Summary:")
    for k, v in summary.items():
        if k != "columns":
            print(f" - {k}: {v}")
    print(f" - Columns ({len(summary['columns'])}): {', '.join(summary['columns'][:5])} ...")
