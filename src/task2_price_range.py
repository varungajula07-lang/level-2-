"""
Task 2: Price Range Analysis
Cognifyz Data Science Internship - Level 2

Analyzes:
1. Most common price range.
2. Average Aggregate rating for each price range.
3. Identifies the price range with the highest average rating.
4. Visualizations: Price range distribution, Average rating by price range, Rating color heatmap.
5. Rating color relationship and color representing the highest average rating.
6. Saves results to CSV.
"""

import os
from typing import Dict, Any
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


def analyze_price_range(
    df: pd.DataFrame,
    output_dir: str = "outputs",
    viz_dir: str = "visualizations"
) -> Dict[str, Any]:
    """
    Perform complete Task 2 analysis and export CSVs and visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    col_price = _find_column(df, ['Price range', 'Price Range'])
    col_rating = _find_column(df, ['Aggregate rating', 'Aggregate Rating', 'Rating'])
    col_color = _find_column(df, ['Rating color', 'Rating Color']) if any('rating color' in c.lower() for c in df.columns) else None
    col_text = _find_column(df, ['Rating text', 'Rating Text']) if any('rating text' in c.lower() for c in df.columns) else None
    
    total_count = len(df)
    
    # ==========================================
    # 1. Most Common Price Range
    # ==========================================
    pr_counts = df[col_price].value_counts().sort_index()
    pr_pct = (pr_counts / total_count * 100).round(2)
    most_common_pr = int(df[col_price].mode()[0])
    most_common_count = int(pr_counts.loc[most_common_pr])
    most_common_pct = float(pr_pct.loc[most_common_pr])
    
    # ==========================================
    # 2. Average Rating for EACH Price Range
    # ==========================================
    pr_rating_stats = df.groupby(col_price)[col_rating].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(3)
    
    # Non-zero rating stats (for deep data science insight on unrated zero-skew)
    pr_rating_nonzero = df[df[col_rating] > 0].groupby(col_price)[col_rating].agg([
        'count', 'mean', 'median'
    ]).round(3)
    
    # ==========================================
    # 3. Price Range with Highest Average Rating
    # ==========================================
    highest_avg_rating_pr = int(pr_rating_stats['mean'].idxmax())
    highest_avg_rating_val = float(pr_rating_stats.loc[highest_avg_rating_pr, 'mean'])
    
    # Price Range Summary DataFrame
    pr_summary = pd.DataFrame({
        'Price Range': pr_rating_stats.index,
        'Restaurant Count': pr_rating_stats['count'],
        'Share of Total (%)': pr_pct.values,
        'Average Rating (Overall)': pr_rating_stats['mean'],
        'Median Rating': pr_rating_stats['median'],
        'Std Dev': pr_rating_stats['std'],
        'Min Rating': pr_rating_stats['min'],
        'Max Rating': pr_rating_stats['max'],
        'Rated Count (>0)': pr_rating_nonzero['count'],
        'Average Rating (Rated Only)': pr_rating_nonzero['mean']
    })
    
    pr_csv_path = os.path.join(output_dir, "task2_price_range_analysis.csv")
    pr_summary.to_csv(pr_csv_path, index=False)
    
    # ==========================================
    # 4. Rating Color Analysis
    # ==========================================
    color_summary = None
    color_crosstab = None
    highest_rating_color = None
    highest_rating_color_val = None
    
    if col_color:
        color_stats = df.groupby(col_color)[col_rating].agg([
            'count', 'mean', 'median', 'min', 'max'
        ]).round(3).sort_values(by='mean', ascending=False)
        
        highest_rating_color = str(color_stats['mean'].idxmax())
        highest_rating_color_val = float(color_stats.loc[highest_rating_color, 'mean'])
        
        color_crosstab = pd.crosstab(
            df[col_price],
            df[col_color],
            margins=True,
            margins_name='Total'
        )
        
        color_crosstab_pct = (pd.crosstab(
            df[col_price],
            df[col_color],
            normalize='index'
        ) * 100).round(2)
        
        color_summary = color_stats.reset_index().rename(columns={col_color: 'Rating Color'})
        color_csv_path = os.path.join(output_dir, "task2_rating_color_analysis.csv")
        color_summary.to_csv(color_csv_path, index=False)
    
    # ==========================================
    # 5. Visualizations
    # ==========================================
    sns.set_theme(style="whitegrid")
    
    # 1. Price Range Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    palette_pr = ['#e74c3c' if x == most_common_pr else '#3498db' for x in pr_counts.index]
    bars = ax.bar(
        [f"Tier {x}" for x in pr_counts.index],
        pr_counts.values,
        color=palette_pr,
        edgecolor='black',
        alpha=0.85
    )
    ax.set_title(f"Distribution of Restaurants by Price Range\n(Most Common: Tier {most_common_pr} with {most_common_pct}%)", fontsize=13, weight='bold', pad=15)
    ax.set_xlabel("Price Range Tier", fontsize=11, weight='bold')
    ax.set_ylabel("Number of Restaurants", fontsize=11, weight='bold')
    
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_count) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2., h),
            ha='center', va='bottom', fontsize=10, weight='bold',
            xytext=(0, 4), textcoords='offset points'
        )
    ax.set_ylim(0, max(pr_counts.values) * 1.18)
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "task2_price_range_distribution.png"), dpi=300)
    plt.close(fig)
    
    # 2. Average Rating by Price Range
    fig, ax = plt.subplots(figsize=(8, 5))
    means = pr_rating_stats['mean'].values
    tiers = [f"Tier {x}" for x in pr_rating_stats.index]
    
    line = ax.plot(tiers, means, marker='o', markersize=10, linewidth=3, color='#2c3e50', label='Average Rating Trend')
    bars = ax.bar(tiers, means, color=['#bdc3c7', '#95a5a6', '#5dade2', '#27ae60'], alpha=0.7, width=0.45)
    
    ax.set_title("Average Aggregate Rating by Price Range Tier", fontsize=13, weight='bold', pad=15)
    ax.set_xlabel("Price Range Tier", fontsize=11, weight='bold')
    ax.set_ylabel("Average Rating (0 to 5)", fontsize=11, weight='bold')
    ax.set_ylim(0, 5.0)
    
    for bar, mean_val in zip(bars, means):
        ax.annotate(
            f"{mean_val:.2f}",
            (bar.get_x() + bar.get_width() / 2., mean_val),
            ha='center', va='bottom', fontsize=11, weight='bold',
            xytext=(0, 6), textcoords='offset points'
        )
    
    ax.annotate(
        f"Highest: Tier {highest_avg_rating_pr} ({highest_avg_rating_val:.2f})",
        xy=(3, highest_avg_rating_val),
        xytext=(2.2, 4.4),
        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
        fontsize=11, weight='bold', color='#1e8449',
        bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60", lw=1.5)
    )
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "task2_rating_by_price_range.png"), dpi=300)
    plt.close(fig)
    
    # 3. Rating Color Heatmap and Color Stats
    if col_color and color_crosstab is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Left: Heatmap of Price Range vs Rating Color
        ct_clean = pd.crosstab(df[col_price], df[col_color])
        sns.heatmap(
            ct_clean,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            ax=axes[0],
            cbar_kws={'label': 'Restaurant Count'}
        )
        axes[0].set_title("Restaurant Count: Price Range vs Rating Color", fontsize=12, weight='bold')
        axes[0].set_xlabel("Rating Color", fontsize=11, weight='bold')
        axes[0].set_ylabel("Price Range Tier", fontsize=11, weight='bold')
        
        # Right: Average rating by color
        color_order = color_stats.index.tolist()
        color_map = {
            'Dark Green': '#006400',
            'Green': '#2ecc71',
            'Yellow': '#f1c40f',
            'Orange': '#e67e22',
            'Red': '#e74c3c',
            'White': '#bdc3c7'
        }
        bar_colors = [color_map.get(c, '#3498db') for c in color_order]
        
        b = axes[1].barh(color_order, color_stats['mean'], color=bar_colors, edgecolor='black', alpha=0.85)
        axes[1].set_title(f"Mean Rating by Official Rating Color\n(Highest Color: {highest_rating_color} at {highest_rating_color_val})", fontsize=12, weight='bold')
        axes[1].set_xlabel("Average Rating (0 to 5)", fontsize=11, weight='bold')
        axes[1].set_xlim(0, 5.2)
        
        for p in b:
            w = p.get_width()
            axes[1].annotate(
                f"{w:.2f}",
                (w, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=10, weight='bold',
                xytext=(5, 0), textcoords='offset points'
            )
            
        plt.tight_layout()
        fig.savefig(os.path.join(viz_dir, "task2_rating_color_heatmap.png"), dpi=300)
        plt.close(fig)
        
    results = {
        "most_common_price_range": most_common_pr,
        "most_common_count": most_common_count,
        "most_common_pct": most_common_pct,
        "average_rating_by_price_range": pr_rating_stats['mean'].to_dict(),
        "highest_avg_rating_price_range": highest_avg_rating_pr,
        "highest_avg_rating_val": highest_avg_rating_val,
        "highest_rating_color": highest_rating_color,
        "highest_rating_color_val": highest_rating_color_val,
        "price_range_summary": pr_summary.to_dict(orient="records"),
        "files_saved": [
            pr_csv_path,
            os.path.join(viz_dir, "task2_price_range_distribution.png"),
            os.path.join(viz_dir, "task2_rating_by_price_range.png"),
            os.path.join(viz_dir, "task2_rating_color_heatmap.png")
        ]
    }
    
    print("[INFO] Task 2 completed successfully.")
    print(f"       Most common Price Range: Tier {most_common_pr} ({most_common_count:,} restaurants, {most_common_pct}%)")
    print(f"       Highest Avg Rating Price Range: Tier {highest_avg_rating_pr} (Rating: {highest_avg_rating_val:.2f})")
    if highest_rating_color:
        print(f"       Color representing Highest Rating: '{highest_rating_color}' (Mean Rating: {highest_rating_color_val:.2f})")
    return results


if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess_data
    df = load_dataset()
    df_clean, _ = preprocess_data(df)
    results = analyze_price_range(df_clean)
