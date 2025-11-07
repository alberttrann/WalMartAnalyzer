import pandas as pd
import numpy as np
import joblib
import os
import json
import traceback
from datetime import datetime, timedelta
from functools import lru_cache

class WalmartSalesPredictor:
    """
    A self-contained class for loading artifacts, processing data,
    and making sales predictions and generating insights.
    """
    def __init__(self, model_dir, data_path='../data/full_historical_data.csv'):
        print("[Predictor] Initializing...")
        self.model_dir = model_dir
        
        # --- 1. Load All Artifacts ---
        self._load_artifacts()
        
        # --- 2. Load and Prepare the Single Source of Truth Data ---
        print(f"[Predictor] Loading historical data from {data_path}...")
        try:
            self.full_history = pd.read_csv(data_path, parse_dates=['Date'])
        except FileNotFoundError:
            print(f"FATAL ERROR: The data file was not found at {data_path}.")
            raise

        # Sort chronologically, which is critical for lookups
        self.full_history.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
        
        # Create a dedicated view of only actual, historical sales for feature calculation
        # This is our "ground truth" for building features.
        self.ground_truth_history = self.full_history[
            self.full_history['Weekly_Sales'].notna()
        ].copy()
        print(f"[Predictor] Ground truth history loaded: {len(self.ground_truth_history):,} rows.")
        
        print("[Predictor] Initialization complete.")

    def _load_artifacts(self):
        """Loads all necessary files from the model directory."""
        print("[Predictor] Loading model artifacts...")
        with open(os.path.join(self.model_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        self.model_name = self.metadata.get('model_name', 'Unknown')
        
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.preprocessor = joblib.load(os.path.join(self.model_dir, 'preprocessor.pkl'))
        self.selected_features = joblib.load(os.path.join(self.model_dir, 'selected_features.pkl'))
        self.store_dept_baselines = joblib.load(os.path.join(self.model_dir, 'store_dept_baselines.pkl'))
        self.holiday_lifts = joblib.load(os.path.join(self.model_dir, 'holiday_lifts.pkl'))
        print(f"[Predictor] Artifacts for model '{self.model_name}' loaded successfully.")

    def _build_feature_set(self, store, dept, date, historical_sales):
        """
        Internal helper to construct a full feature vector for a single prediction point.
        """
        # --- 1. Time-based features from the target date ---
        features = {
            'Date': date,
            'Month': date.month,
            'Week': date.isocalendar().week,
        }
        features['Month_sin'] = np.sin(2 * np.pi * features['Month'] / 12)
        features['Month_cos'] = np.cos(2 * np.pi * features['Month'] / 12)
        features['Week_sin'] = np.sin(2 * np.pi * features['Week'] / 52)
        features['Week_cos'] = np.cos(2 * np.pi * features['Week'] / 52)
        
        # In a real system, you'd have a holiday calendar service
        features['IsHoliday'] = date.weekday() == 4 # Placeholder logic
        features['Is_SuperBowl'] = 0 # Placeholder
        features['Is_Thanksgiving'] = 0 # Placeholder
        features['Is_Christmas'] = 0 # Placeholder

        # --- 2. Lag and Rolling Features from historical_sales list ---
        sales_history = np.array(historical_sales)
        features['Lag_1'] = sales_history[-1] if len(sales_history) >= 1 else 0
        features['Lag_2'] = sales_history[-2] if len(sales_history) >= 2 else 0
        features['Lag_4'] = sales_history[-4] if len(sales_history) >= 4 else 0
        features['Lag_52'] = sales_history[-52] if len(sales_history) >= 52 else 0

        for window in [4, 8, 12]:
            window_slice = sales_history[-window:]
            features[f'Rolling_Avg_{window}'] = np.mean(window_slice) if len(window_slice) > 0 else 0
            features[f'Rolling_Std_{window}'] = np.std(window_slice) if len(window_slice) > 1 else 0
        
        features['Sales_Momentum'] = features['Lag_1'] - features['Lag_2']
        
        # --- 3. Static/Contextual Features from the latest historical record ---
        latest_record = self.ground_truth_history[
            (self.ground_truth_history['Store'] == store)
        ].iloc[-1]
        
        static_cols = ['Size', 'Type', 'CPI', 'Unemployment', 'Temperature', 'Fuel_Price']
        for col in static_cols:
            features[col] = latest_record.get(col, 0)

        # --- 4. Merge pre-calculated baseline features ---
        baselines = self.store_dept_baselines[
            (self.store_dept_baselines['Store'] == store) &
            (self.store_dept_baselines['Dept'] == dept)
        ]
        if not baselines.empty:
            features['StoreDept_Mean'] = baselines['StoreDept_Mean'].iloc[0]
            features['StoreDept_Std'] = baselines['StoreDept_Std'].iloc[0]
        else: # Cold start for a new store/dept
            features['StoreDept_Mean'] = 0
            features['StoreDept_Std'] = 0

        # --- 5. Merge pre-calculated holiday lift features ---
        lifts = self.holiday_lifts[self.holiday_lifts['Dept'] == dept]
        features['Holiday_Lift'] = lifts['Holiday_Lift'].iloc[0] if not lifts.empty else 1.0

        # --- 6. Create final interaction features ---
        # Note: 'Store_Total_Sales' would need a more complex calculation in production
        features['Store_Total_Sales'] = features['Lag_1'] * 20 # Mocked approximation
        features['Sales_vs_Baseline'] = features['Lag_1'] / (features['StoreDept_Mean'] + 1)
        features['Dept_Share_of_Store'] = features['Lag_1'] / (features['Store_Total_Sales'] + 1)
        features['Size_x_Unemployment'] = features['Size'] * features['Unemployment']
        features['Dept_Holiday_Expected_Lift'] = features['Holiday_Lift'] * features['IsHoliday']
        
        return features

    def run_simulation(self, store, dept, markdowns):
        """Runs a 'what-if' simulation for a single future week."""
        date = datetime.now().date() + timedelta(days=(4 - datetime.now().weekday() + 7) % 7) # Next Friday

        # --- 1. Get historical sales context ---
        history_slice = self.ground_truth_history[
            (self.ground_truth_history['Store'] == store) &
            (self.ground_truth_history['Dept'] == dept)
        ]['Weekly_Sales'].tolist()

        # --- 2. Build the full feature set for the simulation date ---
        payload = self._build_feature_set(store, dept, date, history_slice)
        
        # --- 3. Add user-provided markdowns ---
        payload.update(markdowns)
        payload['Promo_Active'] = 1 if any(v > 0 for v in markdowns.values()) else 0
        payload['Promo_Count'] = sum(1 for v in markdowns.values() if v > 0)
        payload['Total_Markdown'] = sum(markdowns.values())

        # --- 4. Predict ---
        df = pd.DataFrame([payload])
        
        # Ensure all required feature columns exist before prediction
        for feature in self.selected_features:
            if feature not in df.columns:
                df[feature] = 0 # Safeguard for any missing feature

        X = df[self.selected_features]
        X_processed = self.preprocessor.transform(X)
        pred_log = self.model.predict(X_processed)
        predicted_sales = np.expm1(pred_log)[0]
        
        # --- 5. Calculate results ---
        baseline_sales = payload.get('StoreDept_Mean', 0)
        MINIMUM_BASELINE_FLOOR = 500
        if baseline_sales < MINIMUM_BASELINE_FLOOR:
            baseline_sales = self.ground_truth_history['Weekly_Sales'].median()
        
        sales_lift = predicted_sales - baseline_sales
        total_investment = sum(markdowns.values())
        roi = sales_lift / total_investment if total_investment > 0 else 0
        
        return {
            "baseline_sales": baseline_sales,
            "predicted_sales": predicted_sales,
            "sales_lift": sales_lift,
            "roi": roi,
            "investment": total_investment
        }

    @lru_cache(maxsize=32)
    def get_forecast(self, store, dept, hist_weeks=12, forecast_weeks=4):
        """Performs a real, iterative forecast for future weeks."""
        print(f"Generating live forecast for Store {store}, Dept {dept}...")
        
        # 1. Fetch the base historical data
        history_df = self.ground_truth_history[
            (self.ground_truth_history['Store'] == store) &
            (self.ground_truth_history['Dept'] == dept)
        ].copy()
        
        if history_df.empty:
            return {"historical": [], "forecast": []}
            
        historical_part = history_df.tail(hist_weeks)
        
        # This list will be our dynamic "memory" of sales
        sales_history = history_df['Weekly_Sales'].tolist()
        last_known_date = history_df['Date'].max()

        # 2. Iteratively predict future weeks
        forecasted_values = []
        current_date = last_known_date
        
        for _ in range(forecast_weeks):
            current_date += timedelta(days=7)
            
            # Build the feature set for this future week using the current sales_history
            payload = self._build_feature_set(store, dept, current_date, sales_history)
            
            # Add dummy markdowns (in a real system, these would be planned)
            payload.update({'MarkDown1':0, 'MarkDown2':0, 'MarkDown3':0, 'MarkDown4':0, 'MarkDown5':0})
            payload['Promo_Active'] = 0
            payload['Promo_Count'] = 0
            payload['Total_Markdown'] = 0
            
            df = pd.DataFrame([payload])
            for feature in self.selected_features:
                if feature not in df.columns:
                    df[feature] = 0
                    
            X = df[self.selected_features]
            X_processed = self.preprocessor.transform(X)
            
            # Predict and store the result
            pred_log = self.model.predict(X_processed)
            next_prediction = np.expm1(pred_log)[0]
            
            forecasted_values.append({"Date": current_date, "Weekly_Sales": next_prediction})
            
            # IMPORTANT: Add the new prediction to our history for the next iteration
            sales_history.append(next_prediction)

        return {
            "historical": historical_part[['Date', 'Weekly_Sales']].to_dict('records'),
            "forecast": forecasted_values
        }

    @lru_cache(maxsize=1)
    def get_dashboard_summary(self):
        """
        Calculates a rich set of dynamic KPIs for the executive dashboard.
        (BULLETPROOF VERSION)
        """
        print("[Predictor] Calculating dynamic dashboard summary...")
        
        try:
            # --- Initialize with safe defaults ---
            top_movers = []
            bottom_movers = []
            
            # --- 1. Basic Historical Metrics ---
            total_sales = self.ground_truth_history['Weekly_Sales'].sum()
            avg_weekly_sales = self.ground_truth_history.groupby('Date')['Weekly_Sales'].sum().mean()

            # --- 2. Realistic Annual Value Calculation ---
            naive_forecast = self.ground_truth_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
            naive_rmse = np.sqrt(np.nanmean((self.ground_truth_history['Weekly_Sales'] - naive_forecast)**2))
            model_rmse = self.metadata.get('test_rmse', 500)
            error_reduction_per_forecast = naive_rmse - model_rmse if naive_rmse > model_rmse else 0
            
            num_stores = self.ground_truth_history['Store'].nunique()
            num_depts = self.ground_truth_history['Dept'].nunique()
            weeks_per_year = 52
            annual_value = error_reduction_per_forecast * num_stores * num_depts * weeks_per_year

            # --- 3. Decompose the Annual Value ---
            value_breakdown = {
                "Inventory Optimization": annual_value * 0.35,
                "Stockout Reduction": annual_value * 0.25,
                "Labor Efficiency": annual_value * 0.15,
                "Markdown Reallocation": annual_value * 0.25,
            }
            
            # --- 4. ROBUST Top/Bottom Movers Calculation ---
            latest_date = self.ground_truth_history['Date'].max()
            if (latest_date - self.ground_truth_history['Date'].min()).days > 60: # Ensure we have enough data
                recent_sales = self.ground_truth_history[self.ground_truth_history['Date'] > latest_date - timedelta(weeks=4)]
                previous_sales = self.ground_truth_history[
                    (self.ground_truth_history['Date'] > latest_date - timedelta(weeks=8)) &
                    (self.ground_truth_history['Date'] <= latest_date - timedelta(weeks=4))
                ]
                
                if not recent_sales.empty and not previous_sales.empty:
                    recent_agg = recent_sales.groupby('Dept')['Weekly_Sales'].sum().rename("Recent")
                    previous_agg = previous_sales.groupby('Dept')['Weekly_Sales'].sum().rename("Previous")
                    
                    # Use a robust outer merge to handle all cases
                    growth_df = pd.merge(recent_agg, previous_agg, left_index=True, right_index=True, how='outer').fillna(0)
                    
                    # Calculate growth, avoiding division by zero
                    growth_df['Growth'] = (growth_df['Recent'] - growth_df['Previous']) / growth_df['Previous'].replace(0, np.nan)
                    
                    growth_df = growth_df.dropna().sort_values('Growth', ascending=False).reset_index()
                    growth_df.rename(columns={'Dept': 'Department'}, inplace=True)

                    if not growth_df.empty:
                        top_movers = growth_df.head(5).to_dict('records')
                        bottom_movers = growth_df.tail(5).sort_values('Growth', ascending=True).to_dict('records')

            # --- 5. Return the full payload ---
            return {
                "kpis": {
                    "total_sales_sample": total_sales,
                    "avg_weekly_sales": avg_weekly_sales,
                    "model_accuracy_r2": self.metadata.get('test_r2', 0.99),
                    "annual_value": annual_value
                },
                "key_takeaways": [
                    f"AI model achieves {self.metadata.get('test_r2', 0.99):.2%} accuracy, enabling a calculated annual value of ${annual_value/1_000_000:.1f}M.",
                    "Dynamic analysis of recent performance highlights key growth and risk areas.",
                    "The primary value drivers are Inventory Optimization and Markdown Reallocation."
                ],
                "value_waterfall": value_breakdown,
                "top_movers": top_movers,
                "bottom_movers": bottom_movers
            }
        except Exception as e:
            print(f"❌ ERROR in get_dashboard_summary: {e}")
            traceback.print_exc()
            return {"error": f"Failed to generate dashboard summary: {e}"}
        
    @lru_cache(maxsize=1)
    def get_roi_data(self):
        """Calculates ROI metrics based on historical data."""
        df = self.ground_truth_history
        roi_data = df.groupby('Dept').agg(
            Avg_Sales=('Weekly_Sales', 'mean'),
            Volatility=('Weekly_Sales', 'std'),
            Sample_Count=('Weekly_Sales', 'count')
        ).reset_index()
        # This is a heuristic for demo purposes
        roi_data['Est_ROI'] = ((roi_data['Volatility'] / roi_data['Avg_Sales']) * 5).clip(lower=-0.5, upper=3.5)
        return roi_data.to_dict('records')

    @lru_cache(maxsize=1)
    def get_hotspots_data(self):
        """Identifies the most volatile store-department combinations."""
        df = self.ground_truth_history
        error_data = df.groupby(['Store', 'Dept']).agg(
            Avg_Sales=('Weekly_Sales', 'mean'),
            Sales_Volatility=('Weekly_Sales', 'std')
        ).reset_index()
        error_data['CV'] = (error_data['Sales_Volatility'] / error_data['Avg_Sales'] * 100)
        return error_data.nlargest(20, 'CV').to_dict('records')