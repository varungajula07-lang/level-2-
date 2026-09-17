"""
Task 3: Feature Engineering
Cognifyz Data Science Internship - Level 2

Extracts and engineers meaningful features:
1. Restaurant_Name_Length: Character length of Restaurant Name.
2. Address_Length: Character length of Address.
3. Has_Table_Booking: Binary encoded (1 for Yes, 0 for No).
4. Has_Online_Delivery: Binary encoded (1 for Yes, 0 for No).
5. Additional justified domain features:
   - Cuisines_Count: Count of cuisines offered.
   - Is_Delivering_Now: Binary encoded (1 for Yes, 0 for No).
   - Is_High_Rated: Binary flag (1 if rating >= 4.0, else 0).
6. Comprehensive validation, missing value checks, and saving to:
   outputs/feature_engineered_dataset.csv
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _find_column(df: pd.DataFrame, candidates: list) -> str:
    """Find column matching any candidate name case-insensitively."""
    cols_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def engineer_features(
    df: pd.DataFrame,
    output_dir: str = "outputs",
    viz_dir: str = "visualizations"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform complete feature engineering, validation, and export.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    fe_df = df.copy()
    
    # Resolve source columns
    col_name = _find_column(fe_df, ['Restaurant Name', 'Restaurant name'])
    col_addr = _find_column(fe_df, ['Address', 'Restaurant Address'])
    col_tb = _find_column(fe_df, ['Has Table booking', 'Has Table Booking', 'Table booking'])
    col_od = _find_column(fe_df, ['Has Online delivery', 'Has Online Delivery', 'Online delivery'])
    col_cuisines = _find_column(fe_df, ['Cuisines', 'Cuisine'])
    col_rating = _find_column(fe_df, ['Aggregate rating', 'Aggregate Rating', 'Rating'])
    
    # Optional yes/no columns
    col_delivering_now = None
    for cand in ['Is delivering now', 'Is Delivering Now']:
        if any(cand.lower() == c.lower() for c in fe_df.columns):
            col_delivering_now = _find_column(fe_df, [cand])
            break
            
    # ==========================================================
    # 1. Length-based Features
    # ==========================================================
    fe_df['Restaurant_Name_Length'] = fe_df[col_name].astype(str).str.strip().str.len()
    fe_df['Address_Length'] = fe_df[col_addr].astype(str).str.strip().str.len()
    
    # ==========================================================
    # 2. Encoded Boolean / Binary Features (Yes -> 1, No -> 0)
    # ==========================================================
    def encode_binary(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        return 1 if s in ['yes', 'y', 'true', '1'] else 0

    fe_df['Has_Table_Booking'] = fe_df[col_tb].apply(encode_binary)
    fe_df['Has_Online_Delivery'] = fe_df[col_od].apply(encode_binary)
    
    if col_delivering_now:
        fe_df['Is_Delivering_Now'] = fe_df[col_delivering_now].apply(encode_binary)
        
    # ==========================================================
    # 3. Additional Domain-Justified Features
    # ==========================================
    # Cuisines Count: Number of distinct cuisines offered by the restaurant
    fe_df['Cuisines_Count'] = fe_df[col_cuisines].astype(str).apply(
        lambda x: len([c.strip() for c in x.split(',') if c.strip()]) if x != 'nan' and x != 'Unknown' else 0
    )
    
    # Is_High_Rated: Indicator for premium rated restaurants (rating >= 4.0)
    fe_df['Is_High_Rated'] = (fe_df[col_rating] >= 4.0).astype(int)
    
    new_features = [
        'Restaurant_Name_Length',
        'Address_Length',
        'Has_Table_Booking',
        'Has_Online_Delivery',
        'Cuisines_Count',
        'Is_High_Rated'
    ]
    if col_delivering_now:
        new_features.append('Is_Delivering_Now')
        
    # ==========================================================
    # 4. Check for Errors & Missing Values
    # ==========================================================
    null_counts_new = fe_df[new_features].isnull().sum().to_dict()
    dtypes_new = {col: str(fe_df[col].dtype) for col in new_features}
    stats_summary = fe_df[new_features].describe().round(2).to_dict()
    
    # Assert integrity: newly created features must not have any NaN values
    for col in new_features:
        assert fe_df[col].isnull().sum() == 0, f"Error: Feature '{col}' contains NaN values."
        
    # ==========================================================
    # 5. Save Feature-Engineered Dataset
    # ==========================================================
    output_csv_path = os.path.join(output_dir, "feature_engineered_dataset.csv")
    fe_df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Saved feature-engineered dataset to: {output_csv_path}")
    print(f"       Total rows: {fe_df.shape[0]}, Total columns: {fe_df.shape[1]}")
    
    # ==========================================================
    # 6. Feature Visualizations
    # ==========================================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    # 1. Restaurant Name Length Distribution
    sns.histplot(fe_df['Restaurant_Name_Length'], bins=25, kde=True, color='#3498db', ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Restaurant Name Length (chars)", fontsize=12, weight='bold')
    axes[0, 0].set_xlabel("Name Length (characters)", fontsize=10)
    axes[0, 0].set_ylabel("Count", fontsize=10)
    axes[0, 0].axvline(fe_df['Restaurant_Name_Length'].mean(), color='red', linestyle='--', label=f"Mean: {fe_df['Restaurant_Name_Length'].mean():.1f}")
    axes[0, 0].legend()
    
    # 2. Address Length Distribution
    sns.histplot(fe_df['Address_Length'], bins=25, kde=True, color='#2ecc71', ax=axes[0, 1])
    axes[0, 1].set_title("Distribution of Address Length (chars)", fontsize=12, weight='bold')
    axes[0, 1].set_xlabel("Address Length (characters)", fontsize=10)
    axes[0, 1].set_ylabel("Count", fontsize=10)
    axes[0, 1].axvline(fe_df['Address_Length'].mean(), color='red', linestyle='--', label=f"Mean: {fe_df['Address_Length'].mean():.1f}")
    axes[0, 1].legend()
    
    # 3. Cuisines Count Distribution
    sns.countplot(data=fe_df, x='Cuisines_Count', hue='Cuisines_Count', legend=False, palette='crest', ax=axes[1, 0])
    axes[1, 0].set_title("Distribution of Number of Cuisines Offered", fontsize=12, weight='bold')
    axes[1, 0].set_xlabel("Number of Cuisines", fontsize=10)
    axes[1, 0].set_ylabel("Count", fontsize=10)
    for p in axes[1, 0].patches:
        h = p.get_height()
        if h > 0:
            axes[1, 0].annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
                                
    # 4. Binary Features Proportions
    binary_means = fe_df[['Has_Table_Booking', 'Has_Online_Delivery', 'Is_High_Rated']].mean() * 100
    bars = axes[1, 1].bar(
        ['Table Booking', 'Online Delivery', 'High Rated (>=4.0)'],
        binary_means.values,
        color=['#e67e22', '#9b59b6', '#1abc9c'],
        width=0.5
    )
    axes[1, 1].set_title("Adoption Rate of Binary Engineered Features (%)", fontsize=12, weight='bold')
    axes[1, 1].set_ylabel("Percentage (%)", fontsize=10)
    axes[1, 1].set_ylim(0, 100)
    for b in bars:
        h = b.get_height()
        axes[1, 1].annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=10, weight='bold', xytext=(0, 4), textcoords='offset points')
                            
    plt.tight_layout()
    viz_path = os.path.join(viz_dir, "task3_feature_distributions.png")
    fig.savefig(viz_path, dpi=300)
    plt.close(fig)
    
    results = {
        "new_features_created": new_features,
        "new_features_nulls": null_counts_new,
        "new_features_dtypes": dtypes_new,
        "new_features_stats": stats_summary,
        "output_dataset_path": output_csv_path,
        "rows": len(fe_df),
        "columns_total": len(fe_df.columns),
        "visualization_path": viz_path
    }
    
    print("[INFO] Task 3 completed successfully.")
    print(f"       Features created: {', '.join(new_features)}")
    print(f"       Engineered dataset exported: {fe_df.shape[0]} rows, {fe_df.shape[1]} columns.")
    return fe_df, results


if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess_data
    df = load_dataset()
    df_clean, _ = preprocess_data(df)
    fe_df, results = engineer_features(df_clean)
