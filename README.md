# **Walmart Sales Intelligence Hub**

**Version:** 2.0 (Full Stack, API-Driven)
**Date:** 2025-11-07

## **1. Project Overview**

The Walmart Sales Intelligence Hub is a full-stack data science application designed to provide executives, analysts, and store managers with a suite of tools for sales forecasting, promotional planning, and strategic insight discovery.

This project moves beyond a simple predictive model by deploying it as a robust, scalable backend service, which is consumed by an interactive and user-friendly frontend dashboard. The application is built on a foundation of meticulously prepared, time-series-aware historical data, ensuring that all forecasts and insights are both accurate and contextually relevant.

**Core Features:**
*   **Executive KPI Dashboard:** High-level overview of business health and AI-driven insights.
*   **Live Forecast Deep Dive:** On-demand, multi-week sales forecasts for any store-department combination, with model explainability.
*   **Interactive Promotion Simulator:** A "what-if" tool to calculate the financial ROI of promotional strategies *before* execution, preventing costly mistakes.
*   **Strategic Insights Center:** Dynamic analysis of promotional effectiveness (ROI) and operational risks (volatility hotspots).

---

## **2. Project Structure**

```
/walmart_intelligence_hub/
|
|-- api_backend/
|   |-- api.py                      # Flask API server (the "brain")
|   |-- predictor.py                # Core ML logic and feature engineering
|   |-- walmart_sales_model_.../    # Directory containing the trained model & artifacts|
|-- data/
|   |-- full_historical_data.csv    # The single source of truth for the backend
|   |-- data_metadata.json          # Auto-generated description of the historical data
|   |-- store_locations.csv         # Mock geo-data for map visualizations
|   |-- (app_data.csv, test_sample.csv - Deprecated, can be deleted)
|
|-- app.py                          # Streamlit frontend application (the "face")
|-- requirements.txt                # Python dependencies 
|-- README.md                       # This file
```

---

## **3. The Data Foundation (`/data` directory)**

This project's accuracy is built upon a single, unified historical dataset.

### `full_historical_data.csv`

This is the **most important file** in the project. It is the sole data source for the backend API and contains the complete, chronologically sorted sales history for all stores and departments.

**How it was created:**
1.  **Loading:** The raw `train.csv`, `test.csv`, `stores.csv`, and `features.csv` were loaded.
2.  **Merging:** Store metadata (`Type`, `Size`) and external features (`CPI`, `Unemployment`, etc.) were merged into both the training and testing dataframes.
3.  **Schema Unification (Critical Step):** Since `test.csv` lacks a `Weekly_Sales` column, a placeholder column was added and filled with `NaN`. A new boolean column, `Is_Prediction_Target`, was also added to distinguish historical ground truth (`False`) from future periods that need forecasting (`True`).
4.  **Concatenation:** The unified train and test dataframes were concatenated into a single, continuous timeline.
5.  **Sorting:** The resulting dataframe was sorted by `Store`, `Dept`, and `Date` to ensure correct chronological order, which is essential for calculating time-series features like lags and rolling averages.
6.  **Imputation:** Missing values were handled with a robust strategy (e.g., filling markdowns with 0, forward-filling CPI within store groups) to create a clean, analysis-ready dataset.

<details>
<summary>The code for saving full historical data</summary>

```
# ==============================================================================
# PRODUCTION-GRADE DATA PREPARATION FOR FLASK BACKEND
# This creates a SINGLE, TIME-ORDERED dataset with ALL historical sales
# Crucial for accurate lag feature computation in the API
# ==============================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("="*80)
print("WALMART SALES FORECASTING - BACKEND DATA PREPARATION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# STEP 1: LOAD RAW DATA
# ==============================================================================
print("[1/7] Loading raw data files...")

try:
    # Load datasets
    train_df = pd.read_csv('/kaggle/input/walmart-sales-forecast/train.csv', parse_dates=['Date'])
    test_df = pd.read_csv('/kaggle/input/walmart-sales-forecast/test.csv', parse_dates=['Date'])
    stores_df = pd.read_csv('/kaggle/input/walmart-sales-forecast/stores.csv')
    features_df = pd.read_csv('/kaggle/input/walmart-sales-forecast/features.csv', parse_dates=['Date'])
    
    print(f"   ✓ Train data:    {len(train_df):,} rows")
    print(f"   ✓ Test data:     {len(test_df):,} rows")
    print(f"   ✓ Stores:        {len(stores_df)} stores")
    print(f"   ✓ Features:      {len(features_df):,} rows")
    
except FileNotFoundError as e:
    print(f"   ❌ ERROR: Missing data file - {e}")
    raise

# ==============================================================================
# STEP 2: DATA QUALITY CHECKS
# ==============================================================================
print("\n[2/7] Running data quality checks...")

# Check for duplicates
train_dupes = train_df.duplicated(subset=['Store', 'Dept', 'Date']).sum()
test_dupes = test_df.duplicated(subset=['Store', 'Dept', 'Date']).sum()

if train_dupes > 0 or test_dupes > 0:
    print(f"   ⚠️  WARNING: Found {train_dupes} train + {test_dupes} test duplicates")
    train_df = train_df.drop_duplicates(subset=['Store', 'Dept', 'Date'], keep='first')
    test_df = test_df.drop_duplicates(subset=['Store', 'Dept', 'Date'], keep='first')
    print(f"   ✓ Removed duplicates")
else:
    print(f"   ✓ No duplicates found")

# Date range validation
print(f"   ✓ Train date range: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
print(f"   ✓ Test date range:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")

# ==============================================================================
# STEP 3: MERGE STORE & FEATURE DATA
# ==============================================================================
print("\n[3/7] Merging store metadata and external features...")

# Deduplicate features (some dates might have duplicates)
features_clean = features_df.drop_duplicates(subset=['Store', 'Date'])

# Merge stores
train_merged = pd.merge(train_df, stores_df, on='Store', how='left', validate='m:1')
test_merged = pd.merge(test_df, stores_df, on='Store', how='left', validate='m:1')

# Merge features
# Handle IsHoliday conflict: keep the one from features.csv (more reliable)
if 'IsHoliday' in train_merged.columns:
    train_merged = train_merged.drop('IsHoliday', axis=1)
if 'IsHoliday' in test_merged.columns:
    test_merged = test_merged.drop('IsHoliday', axis=1)

train_full = pd.merge(
    train_merged, 
    features_clean, 
    on=['Store', 'Date'], 
    how='left'
)

test_full = pd.merge(
    test_merged, 
    features_clean, 
    on=['Store', 'Date'], 
    how='left'
)

print(f"   ✓ Train merged: {len(train_full):,} rows, {len(train_full.columns)} columns")
print(f"   ✓ Test merged:  {len(test_full):,} rows, {len(test_full.columns)} columns")

# ==============================================================================
# STEP 4: CRITICAL FIX - HANDLE MISSING WEEKLY_SALES IN TEST
# ==============================================================================
print("\n[4/7] Handling test set Weekly_Sales column...")

# OPTION A: If test.csv has Weekly_Sales (competition test set with actuals)
if 'Weekly_Sales' in test_df.columns:
    print("   ✓ Test set already contains Weekly_Sales (ground truth)")
    # No action needed
    
# OPTION B: If test.csv has NO Weekly_Sales (production scenario)
else:
    print("   ⚠️  Test set has NO Weekly_Sales (production mode)")
    print("   → Adding placeholder column (will be filled by predictions)")
    
    # Add placeholder column to maintain schema consistency
    test_full['Weekly_Sales'] = np.nan
    
    # CRITICAL: Mark these rows as "prediction target"
    test_full['Is_Prediction_Target'] = True
    train_full['Is_Prediction_Target'] = False
    
    print("   ✓ Added 'Is_Prediction_Target' flag for API filtering")

# ==============================================================================
# STEP 5: CREATE UNIFIED TIMELINE
# ==============================================================================
print("\n[5/7] Creating unified historical timeline...")

# Concatenate train and test
full_data = pd.concat([train_full, test_full], ignore_index=True)

# Sort chronologically (CRITICAL for lag calculations)
full_data = full_data.sort_values(
    by=['Store', 'Dept', 'Date'], 
    ascending=[True, True, True]
).reset_index(drop=True)

print(f"   ✓ Combined dataset: {len(full_data):,} rows")
print(f"   ✓ Date range: {full_data['Date'].min().date()} to {full_data['Date'].max().date()}")
print(f"   ✓ Unique stores: {full_data['Store'].nunique()}")
print(f"   ✓ Unique departments: {full_data['Dept'].nunique()}")

# ==============================================================================
# STEP 6: INTELLIGENT MISSING VALUE HANDLING
# ==============================================================================
print("\n[6/7] Handling missing values...")

# Markdowns: Fill with 0 (absence is meaningful)
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    if col in full_data.columns:
        missing_before = full_data[col].isna().sum()
        full_data[col] = full_data[col].fillna(0)
        print(f"   ✓ {col}: Filled {missing_before:,} missing values with 0")

# CPI & Unemployment: Forward-fill within store groups
for col in ['CPI', 'Unemployment']:
    if col in full_data.columns:
        missing_before = full_data[col].isna().sum()
        
        # Forward fill within each store
        full_data[col] = full_data.groupby('Store')[col].transform(
            lambda x: x.ffill().bfill()
        )
        
        # Global fill for any remaining NaNs
        if full_data[col].isna().any():
            global_median = full_data[col].median()
            full_data[col] = full_data[col].fillna(global_median)
        
        missing_after = full_data[col].isna().sum()
        print(f"   ✓ {col}: {missing_before:,} → {missing_after} missing values")

# Temperature & Fuel_Price: Interpolate
for col in ['Temperature', 'Fuel_Price']:
    if col in full_data.columns:
        missing_before = full_data[col].isna().sum()
        
        # Interpolate within store groups
        full_data[col] = full_data.groupby('Store')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
        missing_after = full_data[col].isna().sum()
        if missing_after > 0:
            print(f"   ⚠️  {col}: Still has {missing_after} NaNs after interpolation")

# ==============================================================================
# STEP 7: VALIDATION & EXPORT
# ==============================================================================
print("\n[7/7] Final validation and export...")

# Data quality report
print("\n" + "="*80)
print("DATA QUALITY REPORT")
print("="*80)

missing_summary = full_data.isnull().sum()
if missing_summary.sum() > 0:
    print("\n⚠️  Columns with remaining missing values:")
    for col, count in missing_summary[missing_summary > 0].items():
        pct = (count / len(full_data)) * 100
        print(f"   - {col}: {count:,} ({pct:.2f}%)")
else:
    print("✅ No missing values in critical columns!")

# Store-Dept coverage
store_dept_combinations = full_data.groupby(['Store', 'Dept']).size()
print(f"\n✓ Total Store-Dept combinations: {len(store_dept_combinations):,}")
print(f"✓ Avg weeks per combination: {store_dept_combinations.mean():.1f}")
print(f"✓ Min weeks per combination: {store_dept_combinations.min()}")
print(f"✓ Max weeks per combination: {store_dept_combinations.max()}")

# Date continuity check
date_range = pd.date_range(
    start=full_data['Date'].min(), 
    end=full_data['Date'].max(), 
    freq='W-FRI'
)
expected_weeks = len(date_range)
unique_weeks = full_data['Date'].nunique()

print(f"\n✓ Date continuity: {unique_weeks}/{expected_weeks} weeks present")
if unique_weeks < expected_weeks:
    print(f"   ⚠️  Missing {expected_weeks - unique_weeks} weeks")

# Save to CSV
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'full_historical_data.csv')
full_data.to_csv(output_path, index=False)

print("\n" + "="*80)
print("✅ SUCCESS! PRODUCTION DATA FILE CREATED")
print("="*80)
print(f"File location: {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
print(f"Total rows: {len(full_data):,}")
print(f"Total columns: {len(full_data.columns)}")
print(f"Date range: {full_data['Date'].min().date()} to {full_data['Date'].max().date()}")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ==============================================================================
# BONUS: GENERATE METADATA FILE FOR API
# ==============================================================================
print("\n[BONUS] Generating metadata file for API reference...")

metadata = {
    'created_at': datetime.now().isoformat(),
    'total_rows': int(len(full_data)),
    'total_columns': int(len(full_data.columns)),
    'date_range': {
        'start': str(full_data['Date'].min().date()),
        'end': str(full_data['Date'].max().date())
    },
    'stores': {
        'count': int(full_data['Store'].nunique()),
        'ids': sorted(full_data['Store'].unique().tolist())
    },
    'departments': {
        'count': int(full_data['Dept'].nunique()),
        'ids': sorted(full_data['Dept'].unique().tolist())
    },
    'store_dept_combinations': int(len(store_dept_combinations)),
    'columns': full_data.columns.tolist(),
    'missing_values': missing_summary[missing_summary > 0].to_dict(),
    'data_quality': {
        'has_duplicates': False,
        'has_critical_nulls': bool(full_data[['Store', 'Dept', 'Date']].isnull().any().any()),
        'date_continuity': f"{unique_weeks}/{expected_weeks} weeks"
    }
}

import json
metadata_path = os.path.join(output_dir, 'data_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✓ Metadata saved to: {metadata_path}")
print("\n🎉 All done! Your backend data is production-ready.")
```

</details>

### `data_metadata.json`

This file is an auto-generated report on the contents and quality of `full_historical_data.csv`. It provides a quick reference for developers on the data's scope, including date ranges, column names, and missing value counts.

### `store_locations.csv`

This is a **mock data file** created solely for the frontend's map visualization. It contains dummy latitude and longitude coordinates for each of the 45 stores.

---

## **4. The Backend (`/api_backend` directory)**

The backend is a Flask application that serves the machine learning model and all associated business logic as a REST API. It is designed to be the "brain" of the operation.

### **How to Run the Backend**

1.  Navigate to the `api_backend` directory: `cd api_backend`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the Flask server: `python api.py`

The API will now be running at `http://127.0.0.1:5000`.

### **API Endpoints**

The backend exposes several endpoints that the frontend (or any other client) can call.

#### `GET /dashboard_summary`
*   **Purpose:** Provides all dynamically calculated KPIs and text for the main Executive Dashboard.
*   **Frontend Usage:** Called once when the dashboard page loads to populate all the main metrics.

#### `POST /forecast`
*   **Purpose:** Generates a live, multi-week forecast for a specific store and department.
*   **Request Body:** `{"store": <int>, "dept": <int>, "hist_weeks": <int>, "forecast_weeks": <int>}`
*   **Frontend Usage:** Called by the "Forecast Deep Dive" tab when the user clicks "Generate Live Forecast." The backend performs an iterative prediction, using the output of week 1 as an input for week 2, and so on.

#### `POST /simulate`
*   **Purpose:** Runs a "what-if" simulation for a promotional strategy. This is a decoupled endpoint.
*   **Request Body:** `{"store": <int>, "dept": <int>, "markdowns": {"MarkDown1": <float>, ...}}`
*   **Frontend Usage:** This is the core of the "Promotion Simulator" tab. The frontend only needs to send the user's inputs. The backend is responsible for fetching all other ~40 required features (historical context, CPI, etc.), running the model, and returning the final financial analysis (ROI, Net Profit).

#### `GET /insights/roi`
*   **Purpose:** Returns the pre-calculated analysis of promotional effectiveness for all departments.
*   **Frontend Usage:** Called by the "Strategic Insights" tab to populate the "Markdown ROI Explorer" chart and table.

#### `GET /insights/hotspots`
*   **Purpose:** Returns the pre-calculated analysis of the most volatile and hard-to-predict store-department combinations.
*   **Frontend Usage:** Called by the "Strategic Insights" tab to populate the "Operational Watchlist."

---

## **5. The Frontend (`app.py`)**

The frontend is a Streamlit application designed to be a "dumb" but beautiful and interactive client. Its sole responsibility is to manage the user interface, send requests to the backend API, and display the results.

**Key Architectural Principles:**
*   **No Local Data Processing:** The app does not load the `full_historical_data.csv`. All complex calculations are offloaded to the backend API. This keeps the frontend fast and simple.
*   **API-Driven:** Every dynamic chart and number in the application is the result of an API call to the backend.
*   **User-Centric Design:** The interface is designed for non-technical users (executives, managers) with a focus on clear visualizations and actionable insights.
*   **Loading States:** The app uses spinners (`st.spinner`) during API calls to provide a smooth user experience.

### **How to Run the Frontend**

1.  Navigate to the project's root directory.
2.  Install dependencies: `pip install -r requirements_app.txt`
3.  Run the Streamlit app: `streamlit run app.py`

Your web browser will open with the application. **Note: The backend API must be running for the frontend to function correctly.**

---

## **6. Full-Stack Workflow Example (Promotion Simulator)**

To understand how the pieces fit together, here is the end-to-end flow when a user clicks "Calculate ROI":

1.  **[Frontend]** The user selects Store `17`, Department `25`, and adjusts the markdown sliders. They click the "Calculate ROI" button.
2.  **[Frontend]** The Streamlit app constructs a simple JSON payload:
    ```json
    {"store": 17, "dept": 25, "markdowns": {"MarkDown1": 8000, ...}}
    ```
3.  **[Frontend]** It displays a loading spinner (`st.spinner`) and sends a `POST` request to `http://127.0.0.1:5000/simulate` with this payload.
4.  **[Backend]** The Flask server receives the request. The `run_simulation` function in `predictor.py` is called.
5.  **[Backend]** The function looks into its in-memory `full_historical_data` to fetch all the necessary context for Store 17, Dept 25 (e.g., `Size`, `Type`, `CPI`, `Lag_1`, `Lag_52`, `Rolling_Avg_4`, etc.).
6.  **[Backend]** It combines this fetched context with the user's markdown values to build a complete feature vector.
7.  **[Backend]** This vector is passed through the preprocessor and then the trained LightGBM model to get a log-scale prediction.
8.  **[Backend]** The prediction is inverse-transformed (`np.expm1`) to get the final dollar value.
9.  **[Backend]** It calculates the baseline sales, sales lift, ROI, and net profit impact.
10. **[Backend]** It sends back a complete JSON response:
    ```json
    {"predicted_sales": 12528.0, "sales_lift": 7260.0, "roi": 0.12, ...}
    ```
11. **[Frontend]** The Streamlit app receives this JSON. The spinner disappears.
12. **[Frontend]** It uses the data in the response to populate the bar chart, the ROI gauge, the financial analysis metrics, and the final recommendation card. The user sees the result.

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)

![alt text](image-7.png)

![alt text](image-8.png)

![alt text](image-9.png)

![alt text](image-10.png)

![alt text](image-11.png)

![alt text](image-12.png)

![alt text](image-13.png)

![alt text](image-14.png)

![alt text](image-15.png)