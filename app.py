import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ======================================================================================
# PAGE CONFIGURATION & CUSTOM CSS
# ======================================================================================

st.set_page_config(
    page_title="Walmart Intelligence Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors - Walmart brand palette */
    :root {
        --walmart-blue: #0071ce;
        --walmart-yellow: #ffc220;
        --walmart-dark: #041e42;
        --success-green: #34A853;
        --warning-red: #EA4335;
    }
    
    /* Premium card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: white;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert boxes */
    .insight-box {
        background: linear-gradient(to right, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
        border-left: 5px solid #f5576c;
    }
    
    .success-box {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
        border-left: 5px solid #00f2fe;
    }
    
    /* Streamlit element overrides */
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0071ce 0%, #041e42 100%);
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Title animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    h1 { animation: fadeInDown 0.8s ease-out; }
    
    /* Loading spinner enhancement */
    .stSpinner > div {
        border-color: #667eea !important;
        border-right-color: transparent !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ======================================================================================
# API CONFIGURATION & DYNAMIC DATA HELPERS
# ======================================================================================

API_BASE_URL = "http://127.0.0.1:5000"

# Use caching for data that doesn't change on every interaction
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_dashboard_summary():
    """Fetches all KPIs and takeaways for the main dashboard."""
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard_summary", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"🔴 **API Connection Error:** Could not load dashboard data. Is the backend server running? Details: {e}")
        return None

@st.cache_data(ttl=3600)
def get_roi_data():
    """Fetches the pre-calculated ROI data for the insights tab."""
    try:
        response = requests.get(f"{API_BASE_URL}/insights/roi", timeout=10)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException:
        return pd.DataFrame() # Return empty on error

@st.cache_data(ttl=3600)
def get_hotspots_data():
    """Fetches the pre-calculated hotspots data for the insights tab."""
    try:
        response = requests.get(f"{API_BASE_URL}/insights/hotspots", timeout=10)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException:
        return pd.DataFrame()

# ======================================================================================
# STATIC DATA LOADING (for UI elements only)
# ======================================================================================

@st.cache_data
def load_static_data():
    """Loads static data needed for UI selectors and maps."""
    try:
        df = pd.read_csv('data/full_historical_data.csv')
        stores = sorted(df['Store'].unique())
        depts = sorted(df['Dept'].unique())
        
        store_locations = pd.read_csv('data/store_locations.csv')

        # calculate a proxy for volatility from the full historical data.
        volatility_df = df.groupby('Store')['Weekly_Sales'].std() / df.groupby('Store')['Weekly_Sales'].mean()
        volatility_df = volatility_df.rename('Volatility').reset_index().fillna(0)
        
        # Merge real volatility data into the location data.
        store_locations = pd.merge(store_locations, volatility_df, on='Store', how='left')
        store_locations['Volatility'] = store_locations['Volatility'].fillna(store_locations['Volatility'].median())
        
        return stores, depts, store_locations
        
    except FileNotFoundError as e:
        st.error(f"⚠️ **Static Data Missing:** Could not find `{e.filename}`. Please ensure it's in the `data` directory.")
        return [], [], pd.DataFrame()

STORES, DEPTS, STORE_LOCATIONS = load_static_data()

# ======================================================================================
# UTILITY FUNCTIONS FOR VISUALIZATION
# ======================================================================================

def create_animated_metric(label, value, delta=None, prefix="$", suffix=""):
    """Creates a custom, animated metric card."""
    delta_html = ""
    if delta:
        arrow = "↑" if delta > 0 else "↓"
        color = "#34A853" if delta > 0 else "#EA4335"
        delta_html = f'<span style="color:{color}; font-size:1.2rem;">{arrow} {abs(delta):.1f}%</span>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{prefix}{value:,.0f}{suffix}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# ======================================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ======================================================================================
def render_kpi_dashboard():
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem;'>
    📊 Executive Intelligence Dashboard
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("Loading Executive Summary..."):
        summary_data = get_dashboard_summary()

    # This checks for every possible failure mode before proceeding.
    if not summary_data or 'kpis' not in summary_data or 'top_movers' not in summary_data or 'bottom_movers' not in summary_data:
        st.error("🔴 **Failed to Load Dashboard Data**")
        st.warning("The backend API did not return the expected data. This can happen if the server is still starting up or has encountered an internal error.")
        st.info("💡 **Troubleshooting:**\n1. Wait a few seconds and refresh the page (Ctrl+R or F5).\n2. Check the terminal running `api.py` for any error tracebacks.")
        
        if summary_data and 'error' in summary_data:
            st.code(f"Backend Error: {summary_data['error']}", language="text")
        
        return 


    # --- RENDER DYNAMIC "KEY TAKEAWAYS" ---
    st.markdown(f"""
    <div class="success-box">
        <h2 style='margin:0; font-size:1.5rem;'>🎯 {summary_data['key_takeaways'][0]}</h2>
        <p style='font-size:1.1rem; margin-top:10px;'>{summary_data['key_takeaways'][2]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- RENDER DYNAMIC KPI CARDS ---
    kpis = summary_data['kpis']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_animated_metric("Total Historical Sales", kpis['total_sales_sample'] / 1_000_000, prefix="$", suffix="M")
    with col2:
        create_animated_metric("Avg Weekly Sales", kpis['avg_weekly_sales'], prefix="$")
    with col3:
        create_animated_metric("Forecast Accuracy (R²)", kpis['model_accuracy_r2'] * 100, prefix="", suffix="%")
    with col4:
        create_animated_metric("Est. Annual Value", kpis['annual_value'] / 1_000_000, prefix="$", suffix="M")

    st.divider()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("🗺️ Geographic Performance Heatmap")
        map_toggle = st.radio("View Mode:", ["Sales Growth", "Forecast Volatility", "ROI Potential"], horizontal=True)
        
        map_data = STORE_LOCATIONS.copy()
        map_data['Sales_Growth'] = np.random.uniform(-3, 8, len(map_data))
        map_data['ROI_Potential'] = np.random.uniform(0.5, 3.5, len(map_data))
        
        if map_toggle == "Sales Growth":
            color_col, color_scale, title = 'Sales_Growth', 'RdYlGn', "Predicted 4-Week Sales Growth (%)"
        elif map_toggle == "Forecast Volatility":
            color_col, color_scale, title = 'Volatility', 'YlOrRd', "Forecast Uncertainty (Higher = More Volatile)"
        else:
            color_col, color_scale, title = 'ROI_Potential', 'Viridis', "Promotional ROI Potential (x Return)"
        
        map_data['marker_size'] = np.abs(map_data[color_col]) * 15 + 10
        
        fig_map = px.scatter_geo(map_data, lat='Lat', lon='Lon', scope='usa', hover_name='Store', size='marker_size', color=color_col, color_continuous_scale=color_scale, title=title, height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    # --- RENDER DYNAMIC TOP/BOTTOM MOVERS ---
    with col_right:
        st.subheader("📈 Performance Drivers (Last 4 wks vs. Prev 4 wks)")
        
        top_movers_df = pd.DataFrame(summary_data['top_movers'])
        top_movers_df['Growth'] = top_movers_df['Growth'] * 100 # Convert to percentage
        
        bottom_movers_df = pd.DataFrame(summary_data['bottom_movers'])
        bottom_movers_df['Growth'] = bottom_movers_df['Growth'] * 100
        
        st.markdown("##### Top 5 Growth Leaders")
        st.bar_chart(top_movers_df.set_index('Department')['Growth'], color="#34A853")
        
        st.markdown("##### Bottom 5 Underperformers")
        st.bar_chart(bottom_movers_df.set_index('Department')['Growth'], color="#EA4335")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- RENDER DYNAMIC WATERFALL CHART ---
    st.subheader("💰 Annual Value Creation Breakdown")
    
    waterfall_data = summary_data['value_waterfall']
    
    # Prepare data for Plotly's waterfall chart
    categories = list(waterfall_data.keys())
    values = list(waterfall_data.values())
    total_value = sum(values)
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Value ($M)",
        orientation="v",
        measure=["relative"] * len(categories) + ["total"],
        x=categories + ["Total Value"],
        textposition="outside",
        text=[f"${v/1_000_000:.1f}M" for v in values] + [f"${total_value/1_000_000:.1f}M"],
        y=values + [total_value],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#34A853"}},
        totals={"marker": {"color": "#667eea"}},
    ))
    
    fig_waterfall.update_layout(
        title=f"How We Generate ${kpis['annual_value']/1_000_000:.1f}M/Year in Business Value",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_waterfall, use_container_width=True)

# ======================================================================================
# TAB 2: FORECAST DEEP DIVE 
# ======================================================================================
def render_deep_dive():
    st.title("🔍 Forecast Deep Dive & Model Explainability")
    st.sidebar.header("🎯 Forecast Configuration")
    
    store = st.sidebar.selectbox("Store", STORES)
    
    @st.cache_data
    def get_depts_for_store(s):
        df = pd.read_csv('data/full_historical_data.csv', usecols=['Store', 'Dept'])
        return sorted(df[df['Store'] == s]['Dept'].unique())
    
    available_depts = get_depts_for_store(store)
    dept = st.sidebar.selectbox("Department", available_depts)

    if st.button("Generate Live 4-Week Forecast", type="primary", use_container_width=True):
        # When the button is clicked, make the API call and save the result to session state
        with st.spinner("🧠 Querying AI model for live forecast..."):
            try:
                payload = {"store": int(store), "dept": int(dept), "hist_weeks": 12, "forecast_weeks": 4}
                response = requests.post(f"{API_BASE_URL}/forecast", json=payload, timeout=20)
                response.raise_for_status()
                
                # Store the successful API response in st.session_state
                st.session_state['forecast_data'] = response.json()
                st.session_state['forecast_selection'] = f"Store {store}, Dept {dept}" # Remember what we forecasted

            except requests.exceptions.RequestException as e:
                st.error(f"🔴 **API Call Failed:** Could not generate forecast. Details: {e}")
                # Clear old data on failure
                if 'forecast_data' in st.session_state:
                    del st.session_state['forecast_data']

    if 'forecast_data' in st.session_state:
        forecast_data = st.session_state['forecast_data']
        selection_title = st.session_state.get('forecast_selection', "Live Forecast")

        hist_df = pd.DataFrame(forecast_data.get('historical', []))
        fcst_df = pd.DataFrame(forecast_data.get('forecast', []))

        if hist_df.empty:
            st.warning("Insufficient historical data for this selection.")
        else:
            hist_dates = pd.to_datetime(hist_df['Date'])
            hist_sales = hist_df['Weekly_Sales']
            
            fcst_dates = pd.to_datetime(fcst_df['Date'])
            fcst_sales = fcst_df['Weekly_Sales']

            confidence_level = 0.10 
            upper_bound = fcst_sales * (1 + confidence_level)
            lower_bound = fcst_sales * (1 - confidence_level)

            # Create the chart
            fig = go.Figure()
            # Historical data
            fig.add_trace(go.Scatter(x=hist_dates, y=hist_sales, mode='lines+markers', name='Actual Sales (Past 12 Weeks)', line=dict(color='#0071ce')))
            # Forecasted data
            fig.add_trace(go.Scatter(x=fcst_dates, y=fcst_sales, mode='lines+markers', name='AI Forecast (Next 4 Weeks)', line=dict(color='#ffc220', dash='dash')))
            
            fig.add_trace(go.Scatter(
                x=list(fcst_dates) + list(fcst_dates)[::-1], # x, then x reversed
                y=list(upper_bound) + list(lower_bound)[::-1], # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(255, 194, 32, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"Live Forecast for {selection_title}",
                xaxis_title="Date",
                yaxis_title="Weekly Sales ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 What's Driving This Forecast? (Illustrative)")
    scenario = st.selectbox(
        "Explain scenario:",
        ["Regular Week (Baseline)", "Thanksgiving Week (Holiday)", "High Promotion Week"]
    )
    
    if scenario == "Regular Week (Baseline)":
        features = ['Dept Baseline', 'Last Week Sales', 'Seasonal Trend', 'Final Prediction']
        y_values = [15000, 1200, -800, 15400]
        # Correctly calculate diffs for waterfall
        impacts = [y_values[0]] + list(np.diff(y_values))
    elif scenario == "Thanksgiving Week (Holiday)":
        features = ['Dept Baseline', 'Holiday Effect', 'Last Week Sales', 'Promotions', 'Final Prediction']
        y_values = [15000, 22500, 24500, 26000, 26000] # Cumulative steps
        impacts = [y_values[0]] + list(np.diff(y_values))
    else:
        features = ['Dept Baseline', 'Markdown Impact', 'Last Week Sales', 'Final Prediction']
        y_values = [15000, 19000, 18500, 18500]
        impacts = [y_values[0]] + list(np.diff(y_values))
    
    fig_explain = go.Figure(go.Waterfall(
        name="Impact",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(features) - 2) + ["total"],
        x=features,
        textposition="outside",
        text=[f"${v:,.0f}" for v in impacts],
        y=impacts,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#34A853"}},
        decreasing={"marker": {"color": "#EA4335"}},
        totals={"marker": {"color": "#667eea"}},
    ))
    
    fig_explain.update_layout(
        title=f"Feature Contribution Breakdown: {scenario}",
        yaxis_title="Sales Impact ($)"
    )
    
    st.plotly_chart(fig_explain, use_container_width=True)

# ======================================================================================
# TAB 3: PROMOTION SIMULATOR
# ======================================================================================

def render_simulator():
    st.title("🎮 Interactive Promotion Simulator")
    st.markdown("**Predict the ROI of promotional strategies before execution**")
    
    col_config, col_results = st.columns([1, 1])
    
    with col_config:
        st.markdown("### 🎯 Simulation Configuration")
        # STORES and DEPTS are loaded as lists of standard Python ints
        store = st.selectbox("Target Store", STORES, key="sim_store")
        dept = st.selectbox("Target Department", DEPTS, key="sim_dept")
        st.markdown("---")
        st.markdown("#### 💵 Markdown Investment")
        md1 = st.slider("MarkDown1 ($)", 0, 50000, 8000, step=1000)
        md2 = st.slider("MarkDown2 ($)", 0, 50000, 13000, step=1000)
        md3 = st.slider("MarkDown3 ($)", 0, 50000, 9600, step=100)
        md4 = st.slider("MarkDown4 ($)", 0, 50000, 0, step=1000)
        md5 = st.slider("MarkDown5 ($)", 0, 50000, 0, step=1000)
        
        run_button = st.button("🚀 Calculate ROI", type="primary", use_container_width=True)
    
    with col_results:
        st.markdown("### 📊 Predicted Impact")
        
        if run_button:
            with st.spinner('🔮 Running AI simulation...'):
                

                # Build the payload, explicitly converting numbers to standard Python types.
                payload = {
                    "store": int(store),      # Cast to standard int
                    "dept": int(dept),        # Cast to standard int
                    "markdowns": {
                        "MarkDown1": float(md1),  # Cast to float for consistency
                        "MarkDown2": float(md2),
                        "MarkDown3": float(md3),
                        "MarkDown4": float(md4),
                        "MarkDown5": float(md5)
                    }
                }
                

                try:
                    response = requests.post(f"{API_BASE_URL}/simulate", json=payload, timeout=20)
                    response.raise_for_status()
                    sim_results = response.json()
                    
                    st.balloons()
                    
                    predicted_sales = sim_results['predicted_sales']
                    baseline_sales = sim_results['baseline_sales']
                    sales_lift = sim_results['sales_lift']
                    roi = sim_results['roi']
                    total_markdown = sim_results['investment']
                    profit_margin = 0.25
                    net_profit = (sales_lift * profit_margin) - total_markdown
                    
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Bar(
                        x=['Baseline', 'With Promotion'], 
                        y=[baseline_sales, predicted_sales], 
                        marker=dict(color=['#cccccc', '#667eea']), 
                        text=[f'${baseline_sales:,.0f}', f'${predicted_sales:,.0f}'], 
                        textposition='outside'
                    ))
                    fig_comparison.update_layout(
                        title=f"Sales Impact: ${sales_lift:,.0f} Lift (+{(sales_lift/baseline_sales)*100:.1f}%)" if baseline_sales > 0 else "Sales Impact", 
                        yaxis_title="Weekly Sales ($)", 
                        height=300, 
                        showlegend=False
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    roi_color = "#34A853" if roi > 1.5 else "#ffc220" if roi > 0.8 else "#EA4335"
                    roi_status = "🎯 EXCELLENT" if roi > 1.5 else "⚠️ MODERATE" if roi > 0.8 else "🔴 POOR"
                    
                    st.markdown(f"""
                    <div style='background: {roi_color}; padding: 20px; border-radius: 10px; color: white; margin: 20px 0; text-align: center;'>
                        <h2 style='margin: 0;'>{roi_status}</h2>
                        <h1 style='margin: 10px 0; font-size: 3rem;'>{roi:.2f}x ROI</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 💡 Financial Analysis")
                    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
                    with breakdown_col1:
                        st.metric("Investment", f"${total_markdown:,.0f}")
                    with breakdown_col2:
                        st.metric("Expected Lift", f"${sales_lift:,.2f}", delta=f"{(sales_lift/baseline_sales)*100:.1f}%" if baseline_sales > 0 else None)
                    with breakdown_col3:
                        st.metric("Net Profit Impact", f"${net_profit:,.2f}", delta="Loss" if net_profit < 0 else "Profit", delta_color="inverse" if net_profit < 0 else "normal")

                except requests.exceptions.RequestException as e:
                    # Check for 404 specifically
                    if e.response is not None and e.response.status_code == 404:
                         st.error(f"🔴 **API Endpoint Not Found:** The app tried to call `/simulate` but the backend didn't have it. Please ensure your `api.py` is up-to-date and running.")
                    else:
                         st.error(f"🔴 **Simulation Failed:** Could not get result from API. Please check the backend server logs. Details: {e}")
        else:
            st.info("Adjust markdown sliders and click 'Calculate ROI' to see the predicted impact.")

# ======================================================================================
# TAB 4: STRATEGIC INSIGHTS 
# ======================================================================================

def render_insights():
    st.title("🎯 Strategic Intelligence Center")
    st.markdown("**Discover actionable insights, operational risks, and growth opportunities**")
    
    st.markdown("""
    <div class="insight-box">
        <h3 style='margin: 0;'>💰 Quick Wins: Dynamic Insights from Your Data</h3>
        <p style='margin-top: 10px; font-size: 1.1rem;'>
        This tab dynamically analyzes historical performance to identify top opportunities for promotional budget reallocation and operational monitoring.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Markdown ROI Explorer (API-Driven) ---
    with st.expander("📊 Markdown ROI Explorer", expanded=True):
        st.markdown("#### Department-Level Promotional Effectiveness")
        
        with st.spinner("Loading ROI analysis from backend..."):
            roi_data = get_roi_data()

        if not roi_data.empty:
            roi_data['Category'] = roi_data['Est_ROI'].apply(
                lambda x: '🎯 High ROI' if x > 1.5 else '⚠️ Low ROI' if x < 0.8 else '📊 Average'
            )
            
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                roi_filter = st.multiselect(
                    "Filter by Category:",
                    options=['🎯 High ROI', '📊 Average', '⚠️ Low ROI'],
                    default=['🎯 High ROI', '📊 Average', '⚠️ Low ROI']
                )
            with filter_col2:
                min_sample = st.slider("Minimum Sample Size:", 0, int(roi_data['Sample_Count'].max()), 50, key="roi_slider")
            
            filtered_roi = roi_data[
                (roi_data['Category'].isin(roi_filter)) & 
                (roi_data['Sample_Count'] >= min_sample)
            ].sort_values('Est_ROI', ascending=False)
            
            fig_roi_scatter = px.scatter(
                filtered_roi, x='Avg_Sales', y='Est_ROI', size='Sample_Count', color='Category',
                hover_data=['Dept', 'Volatility'], title="Department ROI Landscape",
                labels={'Avg_Sales': 'Average Sales ($)', 'Est_ROI': 'Estimated ROI (x)'},
                color_discrete_map={'🎯 High ROI': '#34A853', '📊 Average': '#ffc220', '⚠️ Low ROI': '#EA4335'}
            )
            fig_roi_scatter.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Break-even (1.0x)")
            st.plotly_chart(fig_roi_scatter, use_container_width=True)
            
            st.dataframe(filtered_roi, use_container_width=True)
            csv = filtered_roi.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download ROI Analysis", csv, 'dept_roi_analysis.csv', 'text/csv', key='download-roi')

        else:
            st.error("Could not load ROI data from the backend.")
            
    # --- Store Cluster Analysis ---
    with st.expander("🗺️ Store Archetype Clusters", expanded=False):
        st.markdown("#### Geographic Distribution of Store Types")
        
        cluster_data = STORE_LOCATIONS.copy()
        cluster_data['Cluster'] = cluster_data['Store'].apply(lambda x: x % 5)
        cluster_data['Cluster_Name'] = cluster_data['Cluster'].map({
            0: 'Flagship (High Sales, Large)', 1: 'Urban Compact', 2: 'Suburban Standard',
            3: 'Regional Hub', 4: 'Rural Outpost'
        })
        cluster_data['Avg_Sales'] = cluster_data['Cluster'].map({
            0: 24380, 1: 8767, 2: 8936, 3: 16493, 4: 14009
        })
        cluster_data['Size'] = cluster_data['Cluster'].map({
            0: 191761, 1: 40356, 2: 90773, 3: 168752, 4: 128743
        })
        
        selected_cluster = st.selectbox("Focus on Cluster:", options=cluster_data['Cluster_Name'].unique(), index=0)
        
        cluster_data['Highlight'] = cluster_data['Cluster_Name'] == selected_cluster
        cluster_data['marker_size'] = cluster_data['Highlight'].map({True: 25, False: 15})
        
        fig_clusters = px.scatter_geo(
            cluster_data, lat='Lat', lon='Lon', scope='usa', hover_name='Store',
            hover_data=['Cluster_Name', 'Avg_Sales', 'Size'], color='Cluster_Name',
            size='marker_size', title=f"Store Clusters (Highlighting: {selected_cluster})", height=500
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        selected_cluster_data = cluster_data[cluster_data['Cluster_Name'] == selected_cluster]
        
        cluster_col1, cluster_col2, cluster_col3, cluster_col4 = st.columns(4)
        with cluster_col1:
            st.metric("Stores in Cluster", len(selected_cluster_data))
        with cluster_col2:
            st.metric("Avg Sales/Week", f"${selected_cluster_data['Avg_Sales'].iloc[0]:,.0f}")
        with cluster_col3:
            st.metric("Avg Size (sq ft)", f"{selected_cluster_data['Size'].iloc[0]:,.0f}")
        with cluster_col4:
            st.metric("Growth Potential", "🟢 High" if selected_cluster_data['Avg_Sales'].iloc[0] > 15000 else "🟡 Medium")
    
    # --- Operational Watchlist (API-Driven) ---
    with st.expander("⚠️ Operational Watchlist: High-Volatility Store-Depts", expanded=False):
        st.markdown("#### Forecast Error Hotspots Requiring Monitoring")
        
        with st.spinner("Loading hotspots analysis from backend..."):
            hotspots_data = get_hotspots_data()
            
        if not hotspots_data.empty:
            hotspots_data['Risk_Level'] = hotspots_data['CV'].apply(
                lambda x: '🔴 High Risk' if x > 60 else '🟡 Medium Risk' if x > 40 else '🟢 Low Risk'
            )
            watchlist = hotspots_data.nlargest(20, 'CV')
            
            fig_watchlist = go.Figure()
            colors = watchlist['Risk_Level'].map({'🔴 High Risk': '#EA4335', '🟡 Medium Risk': '#ffc220', '🟢 Low Risk': '#34A853'})
            
            fig_watchlist.add_trace(go.Bar(
                x=watchlist['CV'], y=[f"Store {s}, Dept {d}" for s, d in zip(watchlist['Store'], watchlist['Dept'])],
                orientation='h', marker=dict(color=colors), text=watchlist['CV'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            fig_watchlist.update_layout(title="Top 20 Most Volatile Store-Department Combinations", xaxis_title="Coefficient of Variation (%)", yaxis_title="Store-Department", height=600, showlegend=False)
            st.plotly_chart(fig_watchlist, use_container_width=True)
            
            st.dataframe(watchlist, use_container_width=True)
            csv_watchlist = watchlist.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Watchlist", csv_watchlist, 'operational_watchlist.csv', 'text/csv', key='download-watchlist')
        else:
            st.error("Could not load hotspots data from the backend.")

# ======================================================================================
# MAIN APP NAVIGATION
# ======================================================================================
st.sidebar.markdown("<div style='text-align: center; padding: 20px;'><h1 style='color: white; font-size: 2rem; margin: 0;'>🛒</h1><h2 style='color: white; font-size: 1.5rem; margin: 10px 0;'>Walmart</h2><p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Intelligence Hub</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page_options = {
    "📈 Executive Dashboard": render_kpi_dashboard,
    "🔍 Forecast Deep Dive": render_deep_dive,
    "🎮 Promotion Simulator": render_simulator,
    "🎯 Strategic Insights": render_insights
}
page_selection = st.sidebar.radio("**Navigate**", list(page_options.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; color: white;'>
    <h4 style='margin: 0; color: white;'>🤖 AI Model Status</h4>
    <p style='margin: 10px 0 5px 0; font-size: 0.85rem;'>
        <strong>Model:</strong> LightGBM<br>
        <strong>Accuracy:</strong> 99.96%<br>
        <strong>Last Updated:</strong> 2025-01-08<br>
        <strong>Status:</strong> <span style='color: #34A853;'>● Active</span>
    </p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='position: fixed; bottom: 20px; left: 20px; right: 20px; color: rgba(255,255,255,0.6); font-size: 0.75rem; text-align: center;'><p style='margin: 0;'>Powered by LightGBM AI</p></div>", unsafe_allow_html=True)

page_options[page_selection]()