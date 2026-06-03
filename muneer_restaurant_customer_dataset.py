import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(page_title="Corporate Restaurant Intelligence", layout="wide", page_icon="🍽️")

# Injection of custom CSS for a premium dark, brand-focused interface
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-box { background-color: #1c2128; padding: 20px; border-radius: 10px; border-left: 5px solid #00d4ff; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    h1, h2, h3 { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA PIPELINE (Reads directly from GitHub folder)
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    file_path = 'restaurant_customer_dataset_PKR.csv'
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()
    
    # Financial Parsing: Removing 'PKR' and commas
    financial_cols = ['price', 'total_amount', 'profit', 'loss']
    for col in financial_cols:
        if col in df.columns:
            df[col] = df[col].replace({'PKR': '', ',': ''}, regex=True).astype(float)
    
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

df = load_and_clean_data()

# ==============================================================================
# DASHBOARD INTERFACE
# ==============================================================================
if df is not None:
    st.title("🍽️ Corporate Restaurant Intelligence Platform")
    
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "📈 Operational Trends", "🤖 Predictive ML Core"])
    
    # TAB 1: EXECUTIVE SUMMARY
    with tab1:
        st.subheader("Performance Overview")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Revenue", f"PKR {df['total_amount'].sum():,.0f}")
        with col2: st.metric("Total Profit", f"PKR {df['profit'].sum():,.0f}")
        with col3: st.metric("Avg Customer Rating", f"{df['rating'].mean():.1f} / 5.0")
        
        st.write("### Data Snapshot")
        st.dataframe(df.head(15), use_container_width=True)

    # TAB 2: OPERATIONAL TRENDS
    with tab2:
        st.subheader("Financial Trajectory")
        fig = px.line(df.sort_values('order_date'), x='order_date', y='total_amount', 
                      template="plotly_dark", title="Revenue Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = px.bar(df.groupby('city')['profit'].sum().reset_index(), x='city', y='profit', 
                          template="plotly_dark", title="Profit by City")
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            fig3 = px.pie(df, names='dish', values='quantity', template="plotly_dark", title="Dish Popularity")
            st.plotly_chart(fig3, use_container_width=True)

    # TAB 3: PREDICTIVE ML
    with tab3:
        st.subheader("Predictive Churn Model")
        st.write("Using features such as Frequency, Last Order Days, and Rating to predict potential Churn (1 = At Risk).")
        
        features = ['frequency', 'last_order_days', 'rating', 'quantity']
        X = df[features]
        y = df['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        st.success(f"Model Training Complete. Accuracy on test set: {acc*100:.2f}%")
        
        st.write("### Feature Importance")
        importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        st.bar_chart(importance.set_index('Feature'))

else:
    st.error("Dataset not found. Please upload 'restaurant_customer_dataset_PKR.csv' to the project root.")
