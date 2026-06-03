import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score

# ==============================================================================
# 1. UI CONTEXT & PREMIUM DARK THEME CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Executive Restaurant BI Platform (PKR)",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

# Dark dashboard styling accents with corporate blue/emerald touches
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .metric-card {
        background-color: #1f293d;
        padding: 22px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    .metric-card h5 { color: #9ca3af; margin: 0 0 8px 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
    .metric-card h3 { color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: 700; }
    .section-header {
        border-bottom: 2px solid #2d3748;
        padding-bottom: 8px;
        margin-top: 30px;
        margin-bottom: 20px;
        color: #f3f4f6;
    }
    </style>
""", unsafe_allow_html=True)

sns.set_style("darkgrid")
plt.rcParams.update({
    'text.color': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    'figure.facecolor': '#1f293d',
    'axes.facecolor': '#1f293d',
    'grid.color': '#2d3748'
})

# ==============================================================================
# 2. DATA LOADING & PIPELINE CLEANING ENGINE
# ==============================================================================
@st.cache_data
def load_and_clean_data(file_path):
    # Ingest from CSV ledger
    raw_df = pd.read_csv(file_path)
    
    # Standardize column headers
    raw_df.columns = raw_df.columns.str.lower().str.strip()
    cleaned_df = raw_df.copy()
    
    # Vectorized string parser for cleaning currency strings (e.g., 'PKR 385')
    def parse_pkr_currency(val):
        if isinstance(val, str):
            return float(val.replace('PKR', '').replace(',', '').strip())
        return float(val)
    
    pkr_columns = ['price', 'discount', 'total_amount', 'profit', 'loss']
    for col in pkr_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(parse_pkr_currency)
            
    # Process Datetime
    if 'order_date' in cleaned_df.columns:
        cleaned_df['order_date'] = pd.to_datetime(cleaned_df['order_date'])
        
    # Standardize Profit Margin metric
    if 'profit' in cleaned_df.columns and 'total_amount' in cleaned_df.columns:
        cleaned_df['profit_margin'] = np.where(
            cleaned_df['total_amount'] > 0,
            (cleaned_df['profit'] / cleaned_df['total_amount']) * 100,
            0.0
        )
        
    # Statistical Anomaly Pruning (Z-score filter on key financial vectors)
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = cleaned_df[num_cols].apply(lambda x: np.abs((x - x.mean()) / (x.std() + 1e-9)))
    clean_mask = (z_scores < 3).all(axis=1)
    outliers_removed = len(cleaned_df) - clean_mask.sum()
    cleaned_df = cleaned_df[clean_mask].copy()
    
    return raw_df, cleaned_df, outliers_removed

# File Discovery Protocol
target_csv = 'restaurant_customer_dataset_PKR.csv'

if os.path.exists(target_csv):
    df_raw, df_clean, metrics_dropped = load_and_clean_data(target_csv)
    data_source_msg = f"✅ Running successfully on auto-discovered dataset: `{target_csv}`"
else:
    st.sidebar.warning(f"⚠️ `{target_csv}` not auto-detected.")
    uploaded_file = st.sidebar.file_uploader("Upload Restaurant CSV Ledger", type=["csv"])
    if uploaded_file is not None:
        df_raw, df_clean, metrics_dropped = load_and_clean_data(uploaded_file)
        data_source_msg = "✅ Uploaded dataset parsed successfully."
    else:
        st.error(f"Please place `{target_csv}` in your project workspace directory or use the sidebar uploader.")
        st.stop()

numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Sidebar Metadata Panels
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")
st.sidebar.success(data_source_msg)

st.sidebar.markdown("### 📈 Global Pipeline Metrics")
st.sidebar.markdown(f"""
<div class="metric-card" style="border-left-color: #10b981;">
    <h5>Ingested Dimension Rows</h5>
    <h3>{df_raw.shape[0]:,} x {df_raw.shape[1]}</h3>
</div>
<div class="metric-card" style="border-left-color: #f59e0b;">
    <h5>Anomalies Pruned</h5>
    <h3>{metrics_dropped:,} records</h3>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. INTERFACE TABS
# ==============================================================================
st.title("🍽️ Corporate Restaurant Intelligence Platform (PKR)")
st.markdown("Automated performance tracking dashboards, customer behavioral segmentation models, and corporate ML engines.")

tab_summary, tab_audit, tab_eda, tab_cluster, tab_predict = st.tabs([
    "📊 Executive Summary", "📋 Data Audit & Quality", "📉 Trend Visualization", "🧬 Behavioral Cohorts", "🤖 Predictive ML Core"
])

# --------------------------------------------------------------------------
# TAB 1: SUMMARY LEDGER
# --------------------------------------------------------------------------
with tab_summary:
    st.markdown('<h3 class="section-header">👑 Executive Performance Ledger</h3>', unsafe_allow_html=True)
    
    total_orders = df_clean['order_id'].nunique() if 'order_id' in df_clean.columns else len(df_clean)
    total_customers = df_clean['customer_id'].nunique() if 'customer_id' in df_clean.columns else 0
    total_restaurants = df_clean['restaurant_id'].nunique() if 'restaurant_id' in df_clean.columns else 0
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(f'<div class="metric-card"><h5>Unique Orders Processed</h5><h3>{total_orders:,}</h3></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Active Customer Base</h5><h3>{total_customers}</h3></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #8b5cf6;"><h5>Active Service Locations</h5><h3>{total_restaurants}</h3></div>', unsafe_allow_html=True)

    rev_val = df_clean['total_amount'].sum() if 'total_amount' in df_clean.columns else 0
    prof_val = df_clean['profit'].sum() if 'profit' in df_clean.columns else 0
    loss_val = df_clean['loss'].sum() if 'loss' in df_clean.columns else 0
    net_profit = prof_val - loss_val
    
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Gross Revenue</h5><h3>PKR {rev_val:,.2f}</h3></div>', unsafe_allow_html=True)
    with f_col2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #3b82f6;"><h5>Gross Profit</h5><h3>PKR {prof_val:,.2f}</h3></div>', unsafe_allow_html=True)
    with f_col3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #ef4444;"><h5>Logged Net Losses</h5><h3>PKR {loss_val:,.2f}</h3></div>', unsafe_allow_html=True)
    with f_col4:
        net_color = "#10b981" if net_profit >= 0 else "#ef4444"
        st.markdown(f'<div class="metric-card" style="border-left-color: {net_color};"><h5>Net Retained Margin</h5><h3>PKR {net_profit:,.2f}</h3></div>', unsafe_allow_html=True)

    st.markdown("#### Operational Efficiency Matrix")
    avg_col1, avg_col2, avg_col3, avg_col4 = st.columns(4)
    with avg_col1:
        st.metric("Average Order Value (AOV)", f"PKR {(rev_val / total_orders):,.2f}" if total_orders > 0 else "0.0")
    with avg_col2:
        st.metric("Mean Customer Rating", f"{df_clean['rating'].mean():.2f} ⭐" if 'rating' in df_clean.columns else "N/A")
    with avg_col3:
        avg_disc = df_clean['discount'].mean()
        if avg_disc <= 1.0 and avg_disc > 0:
            avg_disc *= 100
        st.metric("Average Order Discount", f"{avg_disc:.1f}%" if 'discount' in df_clean.columns else "N/A")
    with avg_col4:
        st.metric("Average Items Per Ticket", f"{df_clean['quantity'].mean():.1f}" if 'quantity' in df_clean.columns else "N/A")

# --------------------------------------------------------------------------
# TAB 2: AUDIT RUNS
# --------------------------------------------------------------------------
with tab_audit:
    st.markdown('<h3 class="section-header">🔍 Verification Stream Profiles</h3>', unsafe_allow_html=True)
    st.dataframe(df_clean.head(15), width="stretch")
    
    c_aud1, c_aud2 = st.columns(2)
    with c_aud1:
        st.markdown("#### Dataset Schema Layout")
        buffer = io.StringIO()
        df_clean.info(buf=buffer)
        st.text(buffer.getvalue())
    with c_aud2:
        st.markdown("#### Missing Entry Assertions")
        null_summary = pd.DataFrame({'Null Count': df_clean.isnull().sum(), 'Null Ratio (%)': (df_clean.isnull().sum() / len(df_clean)) * 100})
        st.dataframe(null_summary.round(2), width="stretch")

# --------------------------------------------------------------------------
# TAB 3: TREND VISUALIZATION
# --------------------------------------------------------------------------
with tab_eda:
    st.markdown('<h3 class="section-header">📉 Distribution Profiles & Feature Intersections</h3>', unsafe_allow_html=True)
    
    if 'order_date' in df_clean.columns:
        st.markdown("#### Rolling Performance Curves (Revenue vs Gross Profit)")
        monthly_sales = df_clean.groupby(df_clean['order_date'].dt.to_period('M')).agg({'total_amount': 'sum', 'profit': 'sum'})
        
        fig_time, ax_time = plt.subplots(figsize=(14, 4))
        ax_time.plot(monthly_sales.index.strftime('%b %Y'), monthly_sales['total_amount'], label='Monthly Revenue (PKR)', marker='o', linewidth=2, color='#3b82f6')
        ax_time.plot(monthly_sales.index.strftime('%b %Y'), monthly_sales['profit'], label='Monthly Profit (PKR)', marker='o', linewidth=2, color='#10b981')
        ax_time.set_ylabel('Currency in PKR')
        ax_time.legend(facecolor='#1f293d', edgecolor='none')
        st.pyplot(fig_time)
        plt.close(fig_time)

    c_vis1, c_vis2 = st.columns(2)
    with c_vis1:
        if 'rating' in df_clean.columns:
            st.markdown("#### Experience Distribution Profiles")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            sns.countplot(data=df_clean, x='rating', palette='Blues_r', ax=ax1)
            ax1.set_title("Order Volumetrics by Rating Score")
            st.pyplot(fig1)
            plt.close(fig1)
    with c_vis2:
        st.markdown("#### Multi-Feature Linear Correlations")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        valid_corr_cols = [c for c in ['quantity', 'price', 'discount', 'total_amount', 'profit', 'loss', 'frequency', 'last_order_days', 'rating'] if c in df_clean.columns]
        sns.heatmap(df_clean[valid_corr_cols].corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax2, cbar=False)
        st.pyplot(fig2)
        plt.close(fig2)

# --------------------------------------------------------------------------
# TAB 4: SEGMENTATION
# --------------------------------------------------------------------------
with tab_cluster:
    st.markdown('<h3 class="section-header">🧬 Unsupervised Behavioral Customer Segmentation</h3>', unsafe_allow_html=True)
    
    valid_features = [c for c in ['quantity', 'total_amount', 'frequency', 'last_order_days', 'rating', 'profit'] if c in df_clean.columns]
    
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        target_features = st.multiselect("Isolate Segmentation Vectors:", options=valid_features, default=valid_features[:2])
        cluster_count = st.slider("Target Cohorts Count (K):", min_value=2, max_value=6, value=3)
    
    with cc2:
        if len(target_features) == 2:
            X_scaled = StandardScaler().fit_transform(df_clean[target_features])
            km = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            df_clean['cluster_id'] = km.fit_predict(X_scaled)
            
            fig3, ax3 = plt.subplots(figsize=(7, 3.8))
            sns.scatterplot(data=df_clean, x=target_features[0], y=target_features[1], hue='cluster_id', palette='Set1', ax=ax3)
            ax3.set_title("Behavioral Footprint Space Scatter Matrix")
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.warning("Please isolate exactly two variables to render the visual cluster layout map.")

# --------------------------------------------------------------------------
# TAB 5: PREDICTIVE MACHINE LEARNING ENGINE
# --------------------------------------------------------------------------
with tab_predict:
    st.markdown('<h3 class="section-header">🤖 Supervised Production Pipeline Blocks</h3>', unsafe_allow_html=True)
    
    options_y = [c for c in ['churn', 'rating', 'total_amount'] if c in df_clean.columns]
    if options_y:
        target_variable = st.selectbox("Designate Target Optimization Objective (Y):", options=options_y)
        chosen_task = "Regression" if target_variable == 'total_amount' else "Classification"
        
        st.info(f"⚙️ Running automated validation layers optimized for **{chosen_task}** predictions.")
        
        features = [c for c in ['quantity', 'price', 'discount', 'frequency', 'last_order_days'] if c in df_clean.columns]
        X = df_clean[features]
        y = df_clean[target_variable]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if chosen_task == "Classification":
            model = RandomForestClassifier(random_state=42).fit(X_train, y_train.astype(str))
            acc = accuracy_score(y_test.astype(str), model.predict(X_test))
            st.markdown(f'<div class="metric-card" style="border-left-color: #8b5cf6;"><h5>Random Forest Accuracy Metric</h5><h3>{acc * 100:.2f}%</h3></div>', unsafe_allow_html=True)
            st.text("Classification Matrix Report:")
            st.text(classification_report(y_test.astype(str), model.predict(X_test)))
        else:
            model = LinearRegression().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Model Adjusted R² Score</h5><h3>{r2_score(y_test, preds):.4f}</h3></div>', unsafe_allow_html=True)
    else:
        st.error("No target features found for predictive modeling.")
