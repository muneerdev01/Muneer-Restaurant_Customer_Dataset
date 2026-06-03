import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ==============================================================================
# 1. UI CONTEXT & PREMIUM DARK THEME CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Executive Restaurant BI Platform",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

# Dark dashboard styling accents
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
# 2. EMBEDDED DATASET GENERATOR & ENGINE 
# ==============================================================================
@st.cache_data
def load_and_clean_embedded_data():
    np.random.seed(42)
    n_samples = 1200
    
    dates = pd.date_range(start="2025-01-01", periods=180, freq="D")
    chosen_dates = np.random.choice(dates, n_samples)
    
    data = {
        'order_id': [f"ORD-{1000 + i}" for i in range(n_samples)],
        'customer_id': [f"CUST-{np.random.randint(1, 350)}" for _ in range(n_samples)],
        'restaurant_id': [f"REST-{np.random.randint(1, 8)}" for _ in range(n_samples)],
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], n_samples),
        'order_date': chosen_dates,
        'dish': np.random.choice(['Truffle Burger', 'Salmon Fillet', 'Caesar Salad', 'Ribeye Steak', 'Pasta Carbonara'], n_samples),
        'quantity': np.random.randint(1, 6, n_samples),
        'price': np.random.choice([12.50, 24.00, 10.00, 35.00, 18.50], n_samples),
        'discount': np.random.choice([0, 5, 10, 15, 20], n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.08, 0.17, 0.35, 0.35]),
        'churn': np.random.choice([0, 1], n_samples, p=[0.78, 0.22])
    }
    
    raw_df = pd.DataFrame(data)
    
    raw_df['total_amount'] = (raw_df['quantity'] * raw_df['price']) * (1 - raw_df['discount'] / 100)
    raw_df['profit'] = raw_df['total_amount'] * np.random.uniform(0.15, 0.45, n_samples)
    raw_df['loss'] = np.where(raw_df['rating'] <= 2, raw_df['total_amount'] * 0.1, 0)
    raw_df['calculated_total_amount'] = raw_df['total_amount']
    raw_df['profit_margin'] = (raw_df['profit'] / raw_df['total_amount']) * 100
    
    raw_df.columns = raw_df.columns.str.lower().str.strip()
    cleaned_df = raw_df.copy()
    
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = cleaned_df[num_cols].apply(lambda x: np.abs((x - x.mean()) / (x.std() + 1e-9)))
    clean_mask = (z_scores < 3).all(axis=1)
    outliers_removed = len(cleaned_df) - clean_mask.sum()
    cleaned_df = cleaned_df[clean_mask].copy()
    
    return raw_df, cleaned_df, outliers_removed

df_raw, df_clean, metrics_dropped = load_and_clean_embedded_data()
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

col_mapping = {
    'order': 'order_id', 'customer': 'customer_id', 'restaurant': 'restaurant_id',
    'revenue': 'total_amount', 'profit': 'profit', 'loss': 'loss',
    'rating': 'rating', 'discount': 'discount', 'quantity': 'quantity', 'date': 'order_date'
}

st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")
st.sidebar.success("⚡ Data Engine: Running on Integrated Embedded CSV Database.")

st.sidebar.markdown("### 📈 Global Pipeline Metrics")
st.sidebar.markdown(f"""
<div class="metric-card" style="border-left-color: #10b981;">
    <h5>Ingested Dimensions</h5>
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
st.title("🍽️ Corporate Restaurant Intelligence Platform")
st.markdown("Automated performance tracking dashboards, customer behavioral segmentation models, and production ML engines.")

tab_summary, tab_audit, tab_eda, tab_cluster, tab_predict = st.tabs([
    "📊 Executive Summary", "📋 Data Audit & Quality", "📉 Trend Visualization", "🧬 Behavioral Cohorts", "🤖 Predictive ML Core"
])

# --------------------------------------------------------------------------
# TAB 1: SUMMARY LEDGER
# --------------------------------------------------------------------------
with tab_summary:
    st.markdown('<h3 class="section-header">👑 Executive Performance Ledger</h3>', unsafe_allow_html=True)
    
    total_orders = df_clean[col_mapping['order']].nunique()
    total_customers = df_clean[col_mapping['customer']].nunique()
    total_restaurants = df_clean[col_mapping['restaurant']].nunique()
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(f'<div class="metric-card"><h5>Unique Orders Processed</h5><h3>{total_orders:,}</h3></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Active Customer Base</h5><h3>{total_customers}</h3></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #8b5cf6;"><h5>Active Service Locations</h5><h3>{total_restaurants}</h3></div>', unsafe_allow_html=True)

    rev_val = df_clean[col_mapping['revenue']].sum()
    prof_val = df_clean[col_mapping['profit']].sum()
    loss_val = df_clean[col_mapping['loss']].sum()
    net_profit = prof_val - loss_val
    
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Gross Revenue</h5><h3>${rev_val:,.2f}</h3></div>', unsafe_allow_html=True)
    with f_col2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #3b82f6;"><h5>Gross Profit Margin</h5><h3>${prof_val:,.2f}</h3></div>', unsafe_allow_html=True)
    with f_col3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #ef4444;"><h5>Logged Net Losses</h5><h3>${loss_val:,.2f}</h3></div>', unsafe_allow_html=True)
    with f_col4:
        net_color = "#10b981" if net_profit >= 0 else "#ef4444"
        st.markdown(f'<div class="metric-card" style="border-left-color: {net_color};"><h5>Net Retained Margin</h5><h3>${net_profit:,.2f}</h3></div>', unsafe_allow_html=True)

    st.markdown("#### Operational Efficiency Matrix")
    avg_col1, avg_col2, avg_col3, avg_col4 = st.columns(4)
    with avg_col1:
        st.metric("Average Order Value (AOV)", f"${(rev_val / total_orders):.2f}")
    with avg_col2:
        st.metric("Mean Customer Rating", f"{df_clean[col_mapping['rating']].mean():.2f} ⭐")
    with avg_col3:
        st.metric("Average Order Discount", f"{df_clean[col_mapping['discount']].mean():.1f}%")
    with avg_col4:
        st.metric("Average Items Per Ticket", f"{df_clean[col_mapping['quantity']].mean():.1f}")

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
# TAB 3: TREND VISUALIZATION (FIXED PIPELINE VIA PERIODS)
# --------------------------------------------------------------------------
with tab_eda:
    st.markdown('<h3 class="section-header">📉 Distribution Profiles & Feature Intersections</h3>', unsafe_allow_html=True)
    
    st.markdown("#### Rolling Performance Curves (Revenue vs Gross Profit)")
    
    # FIX: Explicitly convert to datetime and perform grouping via PeriodIndex to avoid pd.Grouper version crashes
    df_clean['order_date'] = pd.to_datetime(df_clean['order_date'])
    monthly_sales = df_clean.groupby(df_clean['order_date'].dt.to_period('M')).agg({'total_amount': 'sum', 'profit': 'sum'})
    
    fig_time, ax_time = plt.subplots(figsize=(14, 4))
    ax_time.plot(monthly_sales.index.strftime('%b %Y'), monthly_sales['total_amount'], label='Monthly Revenue', marker='o', linewidth=2)
    ax_time.plot(monthly_sales.index.strftime('%b %Y'), monthly_sales['profit'], label='Monthly Profit', marker='o', linewidth=2)
    ax_time.legend(facecolor='#1f293d', edgecolor='none')
    st.pyplot(fig_time)
    plt.close(fig_time)

    c_vis1, c_vis2 = st.columns(2)
    with c_vis1:
        st.markdown("#### Experience Distribution Profiles")
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        sns.countplot(data=df_clean, x='rating', palette='Blues_r', ax=ax1)
        st.pyplot(fig1)
        plt.close(fig1)
    with c_vis2:
        st.markdown("#### Multi-Feature Linear Correlations")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(df_clean[numeric_columns].corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax2, cbar=False)
        st.pyplot(fig2)
        plt.close(fig2)

# --------------------------------------------------------------------------
# TAB 4: SEGMENTATION
# --------------------------------------------------------------------------
with tab_cluster:
    st.markdown('<h3 class="section-header">🧬 Unsupervised Behavioral Customer Segmentation</h3>', unsafe_allow_html=True)
    
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        target_features = st.multiselect("Isolate Segmentation Vectors:", options=numeric_columns, default=['quantity', 'total_amount'])
        cluster_count = st.slider("Target Cohorts Count (K):", min_value=2, max_value=6, value=3)
    
    with cc2:
        if len(target_features) == 2:
            X_scaled = StandardScaler().fit_transform(df_clean[target_features])
            km = KMeans(n_clusters=cluster_count, random_state=42)
            df_clean['cluster_id'] = km.fit_predict(X_scaled)
            
            fig3, ax3 = plt.subplots(figsize=(7, 3.8))
            sns.scatterplot(data=df_clean, x=target_features[0], y=target_features[1], hue='cluster_id', palette='Set1', ax=ax3)
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.warning("Please isolate exactly two variables to render the visual layout map.")

# --------------------------------------------------------------------------
# TAB 5: PREDICTIVE MACHINE LEARNING ENGINE
# --------------------------------------------------------------------------
with tab_predict:
    st.markdown('<h3 class="section-header">🤖 Supervised Production Pipeline Blocks</h3>', unsafe_allow_html=True)
    
    target_variable = st.selectbox("Designate Target Optimization Objective (Y):", options=['churn', 'rating', 'total_amount'])
    chosen_task = "Regression" if target_variable == 'total_amount' else "Classification"
    
    st.info(f"⚙️ Running automated validation layers optimized for **{chosen_task}** predictions.")
    
    features = ['quantity', 'price', 'discount', 'profit_margin']
    X = df_clean[features]
    y = df_clean[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if chosen_task == "Classification":
        model = RandomForestClassifier(random_state=42).fit(X_train, y_train.astype(str))
        acc = accuracy_score(y_test.astype(str), model.predict(X_test))
        st.markdown(f'<div class="metric-card" style="border-left-color: #8b5cf6;"><h5>Random Forest Accuracy Metric</h5><h3>{acc * 100:.2f}%</h3></div>', unsafe_allow_html=True)
        st.text(classification_report(y_test.astype(str), model.predict(X_test)))
    else:
        model = LinearRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
        st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Model Adjusted R² Score</h5><h3>{r2_score(y_test, preds):.4f}</h3></div>', unsafe_allow_html=True)
