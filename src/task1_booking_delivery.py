"""
Task 1: Table Booking & Online Delivery Analysis
Cognifyz Data Science Internship - Level 2

Analyzes:
1. Percentage of restaurants offering table booking vs not.
2. Percentage of restaurants offering online delivery vs not.
3. Average Aggregate rating comparison between table booking availability.
4. Online delivery availability across different Price Ranges.
5. High-resolution visualizations and CSV summaries.
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


def analyze_table_booking_and_delivery(
    df: pd.DataFrame,
    output_dir: str = "outputs",
    viz_dir: str = "visualizations"
) -> Dict[str, Any]:
    """
    Perform complete Task 1 analysis and generate charts and CSV outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Resolve columns safely
    col_tb = _find_column(df, ['Has Table booking', 'Has Table Booking', 'Table booking'])
    col_od = _find_column(df, ['Has Online delivery', 'Has Online Delivery', 'Online delivery'])
    col_rating = _find_column(df, ['Aggregate rating', 'Aggregate Rating', 'Rating'])
    col_price = _find_column(df, ['Price range', 'Price Range'])
    
    total_count = len(df)
    
    # ==========================================
    # 1. Table Booking Percentages
    # ==========================================
    tb_counts = df[col_tb].value_counts()
    tb_pct = (tb_counts / total_count * 100).round(2)
    
    has_tb_pct = float(tb_pct.get('Yes', 0.0))
    no_tb_pct = float(tb_pct.get('No', 0.0))
    has_tb_count = int(tb_counts.get('Yes', 0))
    no_tb_count = int(tb_counts.get('No', 0))
    
    # ==========================================
    # 2. Online Delivery Percentages
    # ==========================================
    od_counts = df[col_od].value_counts()
    od_pct = (od_counts / total_count * 100).round(2)
    
    has_od_pct = float(od_pct.get('Yes', 0.0))
    no_od_pct = float(od_pct.get('No', 0.0))
    has_od_count = int(od_counts.get('Yes', 0))
    no_od_count = int(od_counts.get('No', 0))
    
    # ==========================================
    # 3. Rating Comparison: Table Booking vs None
    # ==========================================
    rating_by_tb = df.groupby(col_tb)[col_rating].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(3)
    
    avg_rating_tb_yes = float(rating_by_tb.loc['Yes', 'mean']) if 'Yes' in rating_by_tb.index else 0.0
    avg_rating_tb_no = float(rating_by_tb.loc['No', 'mean']) if 'No' in rating_by_tb.index else 0.0
    rating_diff = round(avg_rating_tb_yes - avg_rating_tb_no, 3)
    
    # Also evaluate non-zero ratings to provide deeper data science context
    non_zero_df = df[df[col_rating] > 0]
    rating_by_tb_nonzero = non_zero_df.groupby(col_tb)[col_rating].agg([
        'count', 'mean', 'median', 'std'
    ]).round(3)
    
    # ==========================================
    # 4. Online Delivery across Price Ranges
    # ==========================================
    delivery_by_price_counts = pd.crosstab(
        df[col_price],
        df[col_od],
        margins=True,
        margins_name="Total"
    )
    
    delivery_by_price_pct = pd.crosstab(
        df[col_price],
        df[col_od],
        normalize='index'
    ) * 100
    delivery_by_price_pct = delivery_by_price_pct.round(2)
    
    # Combine counts and percentages into a neat summary table
    price_delivery_summary = pd.DataFrame({
        'Price Range': delivery_by_price_pct.index,
        'No Delivery (Count)': [delivery_by_price_counts.loc[idx, 'No'] for idx in delivery_by_price_pct.index],
        'Online Delivery (Count)': [delivery_by_price_counts.loc[idx, 'Yes'] for idx in delivery_by_price_pct.index],
        'Total Restaurants': [delivery_by_price_counts.loc[idx, 'Total'] for idx in delivery_by_price_pct.index],
        'Online Delivery (%)': delivery_by_price_pct['Yes'].values,
        'No Delivery (%)': delivery_by_price_pct['No'].values
    })
    
    # ==========================================
    # Save CSV Outputs
    # ==========================================
    summary_overview = pd.DataFrame([
        {"Service": "Table Booking", "Category": "Yes", "Count": has_tb_count, "Percentage (%)": has_tb_pct, "Avg Rating": avg_rating_tb_yes},
        {"Service": "Table Booking", "Category": "No", "Count": no_tb_count, "Percentage (%)": no_tb_pct, "Avg Rating": avg_rating_tb_no},
        {"Service": "Online Delivery", "Category": "Yes", "Count": has_od_count, "Percentage (%)": has_od_pct, "Avg Rating": round(df[df[col_od] == 'Yes'][col_rating].mean(), 3)},
        {"Service": "Online Delivery", "Category": "No", "Count": no_od_count, "Percentage (%)": no_od_pct, "Avg Rating": round(df[df[col_od] == 'No'][col_rating].mean(), 3)}
    ])
    
    summary_csv_path = os.path.join(output_dir, "task1_booking_delivery_summary.csv")
    summary_overview.to_csv(summary_csv_path, index=False)
    
    price_delivery_csv_path = os.path.join(output_dir, "task1_online_delivery_by_price_range.csv")
    price_delivery_summary.to_csv(price_delivery_csv_path, index=False)
    
    # ==========================================
    # Visualizations (Aesthetic, Polished, Clear)
    # ==========================================
    sns.set_theme(style="whitegrid", palette="muted")
    colors_binary = ['#2b5c8f', '#e67e22']
    
    # 1. Table Booking Donut Chart
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        [has_tb_count, no_tb_count],
        labels=[f'Has Table Booking\n({has_tb_count:,})', f'No Table Booking\n({no_tb_count:,})'],
        autopct='%1.2f%%',
        startangle=140,
        colors=['#27ae60', '#e74c3c'],
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        textprops=dict(fontsize=12)
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax.set_title("Table Booking Availability\n(Cognifyz Dataset: 9,551 Restaurants)", fontsize=14, pad=15, weight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "task1_table_booking_availability.png"), dpi=300)
    plt.close(fig)
    
    # 2. Online Delivery Donut Chart
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        [has_od_count, no_od_count],
        labels=[f'Offers Online Delivery\n({has_od_count:,})', f'No Online Delivery\n({no_od_count:,})'],
        autopct='%1.2f%%',
        startangle=140,
        colors=['#2980b9', '#95a5a6'],
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        textprops=dict(fontsize=12)
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax.set_title("Online Delivery Availability\n(Cognifyz Dataset: 9,551 Restaurants)", fontsize=14, pad=15, weight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "task1_online_delivery_availability.png"), dpi=300)
    plt.close(fig)
    
    # 3. Rating Comparison: Table Booking vs No Table Booking (Dual Subplot: Mean + Boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Mean rating comparison bar chart
    bar_data = pd.DataFrame({
        'Table Booking': ['With Table Booking', 'Without Table Booking'],
        'Mean Aggregate Rating': [avg_rating_tb_yes, avg_rating_tb_no]
    })
    bars = sns.barplot(
        data=bar_data,
        x='Table Booking',
        y='Mean Aggregate Rating',
        hue='Table Booking',
        legend=False,
        ax=axes[0],
        palette=['#27ae60', '#e74c3c']
    )
    axes[0].set_ylim(0, 5.0)
    axes[0].set_title("Average Aggregate Rating by Table Booking", fontsize=13, weight='bold')
    axes[0].set_ylabel("Average Rating (0 to 5)", fontsize=11)
    axes[0].set_xlabel("")
    for p in bars.patches:
        axes[0].annotate(
            f"{p.get_height():.2f} / 5.0",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=12, weight='bold',
            xytext=(0, 5), textcoords='offset points'
        )
        
    # Right: Rating distribution boxplot
    sns.boxplot(
        data=df,
        x=col_tb,
        y=col_rating,
        hue=col_tb,
        legend=False,
        ax=axes[1],
        palette=['#e74c3c', '#27ae60'],
        order=['No', 'Yes'],
        width=0.4,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "7"}
    )
    axes[1].set_title("Rating Distribution Spread (Diamond = Mean)", fontsize=13, weight='bold')
    axes[1].set_xlabel("Has Table Booking", fontsize=11)
    axes[1].set_ylabel("Aggregate Rating", fontsize=11)
    
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "task1_rating_booking_comparison.png"), dpi=300)
    plt.close(fig)
    
    # 4. Online Delivery Availability by Price Range (Grouped & Stacked)
    fig, ax = plt.subplots(figsize=(10, 6))
    price_tiers = price_delivery_summary['Price Range'].tolist()
    pct_yes = price_delivery_summary['Online Delivery (%)'].tolist()
    pct_no = price_delivery_summary['No Delivery (%)'].tolist()
    
    x = np.arange(len(price_tiers))
    width = 0.55
    
    p1 = ax.bar(x, pct_yes, width, label='Offers Online Delivery (%)', color='#2980b9', edgecolor='white')
    p2 = ax.bar(x, pct_no, width, bottom=pct_yes, label='No Online Delivery (%)', color='#bdc3c7', edgecolor='white')
    
    ax.set_ylabel('Percentage of Restaurants (%)', fontsize=12, weight='bold')
    ax.set_xlabel('Price Range Tier (1 = Lowest, 4 = Highest)', fontsize=12, weight='bold')
    ax.set_title('Online Delivery Availability Across Price Ranges (%)', fontsize=14, pad=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Price Range {pr}" for pr in price_tiers], fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True)
    
    # Annotate percentage on bars
    for i in range(len(price_tiers)):
        ax.text(x[i], pct_yes[i] / 2, f"{pct_yes[i]:.1f}%", ha='center', va='center', color='white', weight='bold', fontsize=11)
        ax.text(x[i], pct_yes[i] + pct_no[i] / 2, f"{pct_no[i]:.1f}%", ha='center', va='center', color='#2c3e50', weight='bold', fontsize=11)
        
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "task1_delivery_by_price_range.png"), dpi=300)
    plt.close(fig)
    
    results = {
        "has_table_booking_pct": has_tb_pct,
        "no_table_booking_pct": no_tb_pct,
        "has_table_booking_count": has_tb_count,
        "no_table_booking_count": no_tb_count,
        "has_online_delivery_pct": has_od_pct,
        "no_online_delivery_pct": no_od_pct,
        "has_online_delivery_count": has_od_count,
        "no_online_delivery_count": no_od_count,
        "avg_rating_tb_yes": avg_rating_tb_yes,
        "avg_rating_tb_no": avg_rating_tb_no,
        "rating_diff": rating_diff,
        "rating_by_tb_stats": rating_by_tb.to_dict(),
        "rating_by_tb_nonzero_stats": rating_by_tb_nonzero.to_dict(),
        "price_delivery_summary": price_delivery_summary.to_dict(orient="records"),
        "files_saved": [
            summary_csv_path,
            price_delivery_csv_path,
            os.path.join(viz_dir, "task1_table_booking_availability.png"),
            os.path.join(viz_dir, "task1_online_delivery_availability.png"),
            os.path.join(viz_dir, "task1_rating_booking_comparison.png"),
            os.path.join(viz_dir, "task1_delivery_by_price_range.png")
        ]
    }
    
    print("[INFO] Task 1 completed successfully.")
    print(f"       Table Booking: {has_tb_pct}% Yes vs {no_tb_pct}% No")
    print(f"       Online Delivery: {has_od_pct}% Yes vs {no_od_pct}% No")
    print(f"       Avg Rating: With Booking = {avg_rating_tb_yes} vs Without = {avg_rating_tb_no} (Diff: +{rating_diff})")
    return results


if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess_data
    df = load_dataset()
    df_clean, _ = preprocess_data(df)
    results = analyze_table_booking_and_delivery(df_clean)
