import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score

# ==============================================================================
# CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Executive Restaurant BI Platform",
    layout="wide",
    page_icon="🍽️"
)

# Dark theme styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1f293d;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    }
    .section-header { border-bottom: 2px solid #2d3748; padding-bottom: 10px; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING (Picks file from local/GitHub folder)
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    # This automatically finds the file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'restaurant_customer_dataset_PKR.csv')
    
    # Error checking: If the file isn't found, it will warn you in the app
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}. Please ensure 'restaurant_customer_dataset_PKR.csv' is in the same folder as this script.")
        return None, None, 0

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()
    
    # Cleaning Logic
    def parse_pkr(val):
        if isinstance(val, str):
            # Removes 'PKR', commas, and whitespace
            return float(val.replace('PKR', '').replace(',', '').strip())
        return float(val)
    
    # Only clean columns that actually exist in your dataset
    cols_to_clean = ['price', 'discount', 'total_amount', 'profit', 'loss']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(parse_pkr)
            
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        
    return df, df, 0 

# Load Data
df_raw, df_clean, metrics_dropped = load_and_clean_data()

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
if df_clean is not None:
    st.title("🍽️ Corporate Restaurant Intelligence Platform")
    
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "📉 Trends", "🤖 Predictive ML"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Performance Overview</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Revenue", f"PKR {df_clean['total_amount'].sum():,.0f}")
        with col2:
            st.metric("Total Profit", f"PKR {df_clean['profit'].sum():,.0f}")
        with col3:
            st.metric("Unique Orders", f"{df_clean['order_id'].nunique():,}")
            
        st.dataframe(df_clean.head(10), use_container_width=True)

    with tab2:
        st.markdown('<h3 class="section-header">Sales Trends</h3>', unsafe_allow_html=True)
        if 'order_date' in df_clean.columns:
            # Group by Month
            monthly = df_clean.groupby(df_clean['order_date'].dt.to_period('M')).agg({'total_amount': 'sum'})
            st.line_chart(monthly)

    with tab3:
        st.markdown('<h3 class="section-header">ML Predictive Core</h3>', unsafe_allow_html=True)
        st.write("Using features to predict if an order might be profitable.")
        
        # Simple example model
        features = ['quantity', 'price']
        # Check if features exist to avoid crashes
        if all(feat in df_clean.columns for feat in features):
            X = df_clean[features]
            y = df_clean['profit'] > df_clean['profit'].mean()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier().fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            st.success(f"Model Accuracy: {acc*100:.2f}%")
        else:
            st.warning("Required features for ML not found in dataset.")

else:
    st.info("Please ensure the CSV file is uploaded to your GitHub repository in the same folder.")
