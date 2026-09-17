# Cognifyz Data Science Internship — Level 2 Project
## Restaurant Analytics: Service Offerings, Price Tiers & Feature Engineering

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-orange.svg)](https://seaborn.pydata.org/)
[![Status](https://img.shields.io/badge/Level%202-Complete%20%26%20Validated-brightgreen.svg)]()

---

## 📌 Internship Objective
This project fulfills **Level 2** of the **Cognifyz Technologies Data Science Internship**. The objective is to perform in-depth analysis on a comprehensive real-world restaurant dataset containing 9,551 records across multiple dimensions:
1. **Task 1: Table Booking & Online Delivery Analysis** — Quantify adoption of table booking and online delivery, measure their impact on customer aggregate ratings, and examine delivery availability across price tiers.
2. **Task 2: Price Range Analysis** — Identify the most common price range tier, evaluate mean ratings per price bracket, identify the highest-rated price tier, and investigate official rating colors.
3. **Task 3: Feature Engineering** — Extract and create new analytical and domain-justified features, encode categorical yes/no attributes, validate dataset integrity, and export the enriched dataset.

---

## 📊 Dataset Description
The analysis uses the official real-world restaurant dataset (`data/dataset.csv`):
- **Total Records:** 9,551 restaurants
- **Initial Attributes:** 21 columns
- **Engineered Attributes:** 28 columns (post-Feature Engineering)
- **Key Columns Analyzed:** `Has Table booking`, `Has Online delivery`, `Price range`, `Aggregate rating`, `Rating color`, `Rating text`, `Cuisines`, `Address`, `Restaurant Name`, `Votes`.

---

## 🚀 Tasks Overview & Empirical Findings

### Task 1: Table Booking & Online Delivery Analysis
- **Table Booking Availability:**
  - **12.12%** of restaurants offer table booking (1,158 venues).
  - **87.88%** do not offer table booking (8,393 venues).
- **Online Delivery Availability:**
  - **25.66%** of restaurants offer online delivery (2,451 venues).
  - **74.34%** do not offer online delivery (7,100 venues).
- **Rating Impact:**
  - Restaurants **with** table booking achieve an average aggregate rating of **3.44 / 5.0**.
  - Restaurants **without** table booking achieve an average aggregate rating of **2.56 / 5.0**.
  - **Rating Premium:** Restaurants offering table booking boast a **+0.88** rating advantage over those without. Isolating active, non-zero rated venues still shows a distinct advantage (**3.59** vs **3.41**).
- **Online Delivery across Price Ranges:**
  - **Price Range 1 (Budget):** 15.77% offer online delivery (701 / 4,444).
  - **Price Range 2 (Mid-Range):** **41.31%** offer online delivery (1,286 / 3,113) — **Highest delivery adoption**.
  - **Price Range 3 (Upscale):** 29.19% offer online delivery (411 / 1,408).
  - **Price Range 4 (Luxury / Fine Dining):** **9.04%** offer online delivery (53 / 586) — **Lowest delivery adoption**.

### Task 2: Price Range Analysis
- **Most Common Price Range:**
  - **Price Range 1 (Budget)** is the most common, encompassing **4,444 restaurants (46.53%)**, followed by Tier 2 (3,113, 32.59%), Tier 3 (1,408, 14.74%), and Tier 4 (586, 6.14%).
- **Average Rating by Price Range:**
  - **Tier 1:** 2.00 / 5.0 (All) | 3.14 / 5.0 (Non-zero rated)
  - **Tier 2:** 2.94 / 5.0 (All) | 3.38 / 5.0 (Non-zero rated)
  - **Tier 3:** 3.68 / 5.0 (All) | 3.78 / 5.0 (Non-zero rated)
  - **Tier 4:** **3.82 / 5.0 (All)** | **3.89 / 5.0 (Non-zero rated)** — **Highest Average Rating**.
- **Rating Color Analysis:**
  - **Highest Rating Color:** **'Dark Green'** has the highest average rating at **4.66 / 5.0** (rating range: 4.5 – 4.9, Excellent).
  - **Color Hierarchy:** Dark Green (4.66) > Green (4.17) > Yellow (3.68) > Orange (3.05) > Red (2.30) > White (0.00 / unrated).
  - **Price Tier Correlation:** Higher price tiers directly correlate with superior rating colors. In Tier 4, **45.73%** of restaurants fall into Green/Dark Green, whereas in Tier 1, only **3.96%** reach Green/Dark Green while **38.25%** are White (unrated).

### Task 3: Feature Engineering
1. **`Restaurant_Name_Length`:** Character count of restaurant name (Mean: 15.16 chars, Range: 2 to 54).
2. **`Address_Length`:** Character count of restaurant address (Mean: 63.88 chars, Range: 8 to 226).
3. **`Has_Table_Booking`:** Binary integer encoding (`1` for Yes, `0` for No).
4. **`Has_Online_Delivery`:** Binary integer encoding (`1` for Yes, `0` for No).
5. **`Cuisines_Count`:** Number of distinct cuisines offered (Mean: 1.86, Range: 0 to 8).
6. **`Is_High_Rated`:** Binary indicator (`1` if Aggregate rating >= 4.0, else `0`).
7. **`Is_Delivering_Now`:** Binary integer encoding (`1` for Yes, `0` for No).
- **Data Integrity & Validation:** Zero missing values across all engineered features; saved to `outputs/feature_engineered_dataset.csv` (9,551 rows, 28 columns).

---

## 🛠️ Technology Stack
- **Language:** Python 3.9+
- **Data Processing:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Interactive Notebooks:** Jupyter, Nbformat, Nbclient

---

## 📁 Project Structure

```text
level-2-/
├── data/
│   └── dataset.csv                               # Real restaurant dataset (9,551 records)
├── notebooks/
│   └── level2_restaurant_analysis.ipynb          # Fully executed Jupyter notebook with all outputs
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── data_loading.py                           # Robust dataset loading & summary utilities
│   ├── preprocessing.py                          # Data cleaning & validation
│   ├── task1_booking_delivery.py                 # Task 1 analytics & plots
│   ├── task2_price_range.py                      # Task 2 analytics & plots
│   ├── task3_feature_engineering.py              # Task 3 feature extraction & integrity checks
│   └── main.py                                   # Orchestration pipeline
├── outputs/
│   ├── task1_booking_delivery_summary.csv        # Service availability summary & rating comparison
│   ├── task1_online_delivery_by_price_range.csv  # Delivery availability across price tiers
│   ├── task2_price_range_analysis.csv            # Price range statistics & rating metrics
│   ├── task2_rating_color_analysis.csv           # Rating color statistics & hierarchy
│   ├── feature_engineered_dataset.csv            # Final 28-column dataset with engineered features
│   └── insights.txt                              # Comprehensive data science insights report
├── visualizations/
│   ├── task1_table_booking_availability.png      # Donut chart of table booking availability
│   ├── task1_online_delivery_availability.png    # Donut chart of online delivery availability
│   ├── task1_rating_booking_comparison.png       # Bar chart & box plot of ratings by booking
│   ├── task1_delivery_by_price_range.png         # 100% stacked bar chart of delivery by price tier
│   ├── task2_price_range_distribution.png        # Bar chart showing distribution of price ranges
│   ├── task2_rating_by_price_range.png           # Rating trend across price range tiers
│   ├── task2_rating_color_heatmap.png            # Heatmap of price ranges vs rating colors
│   └── task3_feature_distributions.png           # Histograms and adoption rates of new features
├── build_level2_notebook.py                      # Script to programmatically build & execute notebook
├── requirements.txt                              # Project dependencies
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone or Navigate to the Repository:**
   ```bash
   cd level-2-
   ```

2. **(Optional) Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 How to Run

### Run the Full Pipeline:
To execute data loading, preprocessing, Tasks 1–3, and generate all CSVs, charts, and insights:
```bash
python src/main.py
```

### Rebuild and Execute the Jupyter Notebook:
To recreate and execute `notebooks/level2_restaurant_analysis.ipynb` with fresh outputs:
```bash
python build_level2_notebook.py
```

### Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/level2_restaurant_analysis.ipynb
```

---

## 📈 Visualizations Showcase

| Visualization | Description |
| :--- | :--- |
| `visualizations/task1_table_booking_availability.png` | Donut chart displaying the 12.12% vs 87.88% booking split |
| `visualizations/task1_online_delivery_availability.png` | Donut chart displaying the 25.66% vs 74.34% delivery split |
| `visualizations/task1_rating_booking_comparison.png` | Dual subplot: Mean rating bar chart and boxplot spread |
| `visualizations/task1_delivery_by_price_range.png` | 100% Stacked bar chart showing delivery adoption by price tier |
| `visualizations/task2_price_range_distribution.png` | Distribution of price range tiers with count & percentage annotations |
| `visualizations/task2_rating_by_price_range.png` | Rating trend highlighting Tier 4 as the highest-rated (3.82) |
| `visualizations/task2_rating_color_heatmap.png` | Heatmap linking price range tiers with official rating colors |
| `visualizations/task3_feature_distributions.png` | Feature distributions for lengths, cuisine counts, and service flags |

---

## 💡 Key Business & Data Science Insights

1. **Table Booking Drives Rating Premiums:**
   Restaurants offering table booking achieve significantly higher average ratings (+0.88 points). Table booking venues tend to be formal dine-in establishments with greater investments in service quality.
2. **Online Delivery Inverted-U Phenomenon:**
   Online delivery peaks in Price Range 2 (41.31%) before sharply declining in Price Range 4 (9.04%). Premium fine-dining venues intentionally avoid food delivery to protect plating quality and ambiance.
3. **Price Directly Signals Quality:**
   Average customer ratings monotonically increase from Price Range 1 (2.00) to Price Range 4 (3.82). Higher price brackets correspond with greater proportions of 'Dark Green' and 'Green' rating classifications.
4. **Machine Learning Feature Ready:**
   The 7 newly engineered features provide strong numeric, categorical, and behavioral indicators that can be directly passed into rating prediction or classification models.

---

## ✅ Final Validation Checklist

- [x] Real restaurant dataset used (`9,551` rows, `21` columns)
- [x] Task 1: Table booking percentages calculated (`12.12%` Yes, `87.88%` No)
- [x] Task 1: Online delivery percentages calculated (`25.66%` Yes, `74.34%` No)
- [x] Task 1: Booking vs rating comparison completed (`3.44` vs `2.56`, diff `+0.88`)
- [x] Task 1: Delivery vs price range analysis completed (Tier 2 highest at `41.31%`, Tier 4 lowest at `9.04%`)
- [x] Task 2: Most common price range identified (Tier `1` with `4,444` restaurants, `46.53%`)
- [x] Task 2: Average rating for every price range calculated (`2.00`, `2.94`, `3.68`, `3.82`)
- [x] Task 2: Highest average rating identified (Tier `4` at `3.82`)
- [x] Task 2: Official rating color hierarchy analyzed (`Dark Green` highest at `4.66`)
- [x] Task 3: `Restaurant_Name_Length` and `Address_Length` created
- [x] Task 3: `Has_Table_Booking` and `Has_Online_Delivery` binary encoded
- [x] Task 3: Additional features created (`Cuisines_Count`, `Is_High_Rated`, `Is_Delivering_Now`)
- [x] Task 3: Feature-engineered dataset saved to `outputs/feature_engineered_dataset.csv`
- [x] All 8 publication-quality visualizations generated in `visualizations/`
- [x] Interactive Jupyter Notebook created, fully executed, and saved in `notebooks/`
- [x] Summary report saved to `outputs/insights.txt`
- [x] Clean modular code in `src/` with zero errors and warnings