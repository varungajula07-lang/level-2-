"""
Data Preprocessing Module for Cognifyz Internship Level 2
Handles data cleaning, missing value treatment, and column name validation.
"""

from typing import Tuple, Dict, Any
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and pre-processes the dataset:
    - Strips whitespace from column names and string values
    - Handles missing values safely
    - Verifies numerical types for ratings and price range
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    cleaned_df : pd.DataFrame
        Cleaned dataset copy
    stats : dict
        Dictionary of preprocessing metrics
    """
    cleaned_df = df.copy()
    
    # 1. Standardize column names (strip leading/trailing whitespace)
    cleaned_df.columns = [col.strip() for col in cleaned_df.columns]
    
    # 2. Count missing values before cleaning
    initial_nulls = cleaned_df.isnull().sum().to_dict()
    
    # 3. Clean string columns
    str_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in str_cols:
        cleaned_df[col] = cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
    # Handle Cuisines missing values if any
    if 'Cuisines' in cleaned_df.columns and cleaned_df['Cuisines'].isnull().sum() > 0:
        cleaned_df['Cuisines'] = cleaned_df['Cuisines'].fillna('Unknown')
        
    # 4. Validate and convert key numerical columns
    num_cols = ['Aggregate rating', 'Price range', 'Votes', 'Average Cost for two']
    for col in num_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
    final_nulls = cleaned_df.isnull().sum().to_dict()
    
    stats = {
        "initial_rows": len(df),
        "cleaned_rows": len(cleaned_df),
        "columns": list(cleaned_df.columns),
        "initial_nulls": {k: v for k, v in initial_nulls.items() if v > 0},
        "final_nulls": {k: v for k, v in final_nulls.items() if v > 0}
    }
    
    print(f"[INFO] Preprocessing complete. Rows: {len(cleaned_df)}, Columns: {len(cleaned_df.columns)}")
    return cleaned_df, stats


if __name__ == "__main__":
    from data_loading import load_dataset
    raw_df = load_dataset()
    clean_df, stats = preprocess_data(raw_df)
    print("Preprocessing Stats:", stats)
