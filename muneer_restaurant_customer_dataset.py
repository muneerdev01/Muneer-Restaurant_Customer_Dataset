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

# Premium dark dashboard design with clean container styling and left-border accents
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

# Apply global dark styles to matplotlib plots
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
# 2. AUTOMATED DATA INGESTION & PIPELINE ENGINE
# ==============================================================================
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Target file name definition
DEFAULT_CSV_PATH = "restaurant_customer_dataset.csv"

# Global data containers
raw_bytes = None
load_method = None

# Step A: Check if the CSV exists locally to perform automatic upload
if os.path.exists(DEFAULT_CSV_PATH):
    try:
        with open(DEFAULT_CSV_PATH, "rb") as f:
            raw_bytes = f.read()
        load_method = "auto"
    except Exception as e:
        st.sidebar.error(f"Error auto-reading `{DEFAULT_CSV_PATH}`: {e}")

# Step B: Fallback to manual web file uploader if auto-load isn't available
if raw_bytes is None:
    uploaded_file = st.sidebar.file_uploader("📥 Ingest Restaurant CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        load_method = "manual"

# Welcome / Initialization State if both ingestion routes are empty
if raw_bytes is None:
    st.title("📊 Executive Restaurant Analytics Matrix & Predictive Engine")
    st.markdown("---")
    st.info(f"👋 **System Initialization Ready.** Could not find `{DEFAULT_CSV_PATH}` in the current folder path.")
    st.markdown(f"""
    ### 🛠️ Next Steps to Initialize:
    1. **Option A (Auto-load):** Place your dataset file named exactly `{DEFAULT_CSV_PATH}` into the same directory folder as this script, then refresh.
    2. **Option B (Manual load):** Click the **Browse files** uploader button inside the sidebar control panel to feed the operational logs directly.
    """)
    
    with st.expander("💡 Expected Dataset Schema Requirements", expanded=False):
        st.markdown("""
        The matrix engine automatically parses data arrays but functions best when the following vectors are provided:
        * **Transactional Keys:** `order_id`, `customer_id`, `restaurant_id`
        * **Financial Metrics:** `total_amount` (Revenue), `profit`, `loss`, `discount`
        * **Operational Data:** `rating`, `quantity`, `order_date`, `city`, `churn`
        """)
    st.stop()

# Step C: Stable cached pipeline for dataset manipulation and cleaning
@st.cache_data
def process_and_clean_data(data_stream):
    raw_df = pd.read_csv(io.BytesIO(data_stream))
    raw_df.columns = raw_df.columns.str.lower().str.strip()
    
    cleaned_df = raw_df.copy()
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if num_cols:
        # Impute missing values with median to prevent model crashes
        for col in num_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Robust Pandas-backed Z-Score outlier removal (ignores Scipy NaN pitfalls)
        z_scores = cleaned_df[num_cols].apply(lambda x: np.abs((x - x.mean()) / (x.std() + 1e-9)))
        clean_mask = (z_scores < 3).all(axis=1)
        outliers_removed = len(cleaned_df) - clean_mask.sum()
        cleaned_df = cleaned_df[clean_mask].copy()
    else:
        outliers_removed = 0
        
    return raw_df, cleaned_df, outliers_removed

# Populate execution dataframes
df_raw, df_clean, metrics_dropped = process_and_clean_data(raw_bytes)
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Dynamic metric mapping layer to support flexible column name variants
col_mapping = {
    'order': 'order_id' if 'order_id' in df_clean.columns else (df_clean.columns[0] if 'id' in df_clean.columns[0] else None),
    'customer': 'customer_id' if 'customer_id' in df_clean.columns else None,
    'restaurant': 'restaurant_id' if 'restaurant_id' in df_clean.columns else None,
    'revenue': 'total_amount' if 'total_amount' in df_clean.columns else ('revenue' if 'revenue' in df_clean.columns else None),
    'profit': 'profit' if 'profit' in df_clean.columns else None,
    'loss': 'loss' if 'loss' in df_clean.columns else None,
    'rating': 'rating' if 'rating' in df_clean.columns else ('customer_rating' if 'customer_rating' in df_clean.columns else None),
    'discount': 'discount' if 'discount' in df_clean.columns else None,
    'quantity': 'quantity' if 'quantity' in df_clean.columns else None,
    'date': 'order_date' if 'order_date' in df_clean.columns else ('date' if 'date' in df_clean.columns else None)
}

# Sidebar Stream Metrics
if load_method == "auto":
    st.sidebar.success(f"⚡ Auto-loaded: `{DEFAULT_CSV_PATH}`")
else:
    st.sidebar.success("✅ Log stream ingested successfully.")

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
# 3. WORKFLOW MANAGER (MODULAR INTERFACE TABS)
# ==============================================================================
st.title("🍽️ Corporate Restaurant Intelligence Platform")
st.markdown("Automated performance tracking dashboards, customer behavioral segmentation models, and production ML engines.")

tab_summary, tab_audit, tab_eda, tab_cluster, tab_predict = st.tabs([
    "📊 Executive Summary",
    "📋 Data Audit & Quality", 
    "📉 Trend Visualization", 
    "🧬 Behavioral Cohorts", 
    "🤖 Predictive ML Core"
])

# --------------------------------------------------------------------------
# WORKFLOW 1: EXECUTIVE PERFORMANCE LEDGER
# --------------------------------------------------------------------------
with tab_summary:
    st.markdown('<h3 class="section-header">👑 Executive Performance Ledger</h3>', unsafe_allow_html=True)
    
    # Calculate dimensional counts
    total_orders = df_clean[col_mapping['order']].nunique() if col_mapping['order'] else len(df_clean)
    total_customers = df_clean[col_mapping['customer']].nunique() if col_mapping['customer'] else "N/A"
    total_restaurants = df_clean[col_mapping['restaurant']].nunique() if col_mapping['restaurant'] else "N/A"
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(f'<div class="metric-card"><h5>Unique Orders Processed</h5><h3>{total_orders:,}</h3></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Active Customer Base</h5><h3>{total_customers}</h3></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #8b5cf6;"><h5>Active Service Locations</h5><h3>{total_restaurants}</h3></div>', unsafe_allow_html=True)

    # Calculate financial performance aggregates
    rev_val = df_clean[col_mapping['revenue']].sum() if col_mapping['revenue'] else 0
    prof_val = df_clean[col_mapping['profit']].sum() if col_mapping['profit'] else 0
    loss_val = df_clean[col_mapping['loss']].sum() if col_mapping['loss'] else 0
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

    # Unit Averages
    st.markdown("#### Operational Efficiency Matrix")
    avg_col1, avg_col2, avg_col3, avg_col4 = st.columns(4)
    
    with avg_col1:
        aov = (rev_val / total_orders) if total_orders > 0 else 0
        st.metric("Average Order Value (AOV)", f"${aov:.2f}")
    with avg_col2:
        avg_rat = df_clean[col_mapping['rating']].mean() if col_mapping['rating'] else 0
        st.metric("Mean Customer Rating", f"{avg_rat:.2f} ⭐")
    with avg_col3:
        avg_disc = df_clean[col_mapping['discount']].mean() if col_mapping['discount'] else 0
        st.metric("Average Order Discount", f"{avg_disc:.1f}%")
    with avg_col4:
        avg_qty = df_clean[col_mapping['quantity']].mean() if col_mapping['quantity'] else 0
        st.metric("Average Items Per Ticket", f"{avg_qty:.1f}")

# --------------------------------------------------------------------------
# WORKFLOW 2: DATA AUDIT & QUALITY CHECK
# --------------------------------------------------------------------------
with tab_audit:
    st.markdown('<h3 class="section-header">🔍 Verification Stream Profiles</h3>', unsafe_allow_html=True)
    st.markdown("#### Operational Ledger Snapshot (First 15 Rows)")
    st.dataframe(df_clean.head(15), width="stretch")
    
    c_aud1, c_aud2 = st.columns(2)
    with c_aud1:
        st.markdown("#### Dataset Structural Schema Information")
        buffer = io.StringIO()
        df_clean.info(buf=buffer)
        st.text(buffer.getvalue())
        
    with c_aud2:
        st.markdown("#### Null Variance Accumulations Matrix")
        null_summary = pd.DataFrame({
            'Missing Fields (Count)': df_clean.isnull().sum(),
            'Null Ratio (%)': (df_clean.isnull().sum() / len(df_clean)) * 100
        })
        st.dataframe(null_summary.round(2), width="stretch")

# --------------------------------------------------------------------------
# WORKFLOW 3: TREND VISUALIZATION
# --------------------------------------------------------------------------
with tab_eda:
    st.markdown('<h3 class="section-header">📉 Distribution Profiles & Feature Intersections</h3>', unsafe_allow_html=True)
    
    # Render time trends if date elements are available
    if col_mapping['date'] and col_mapping['revenue'] and col_mapping['profit']:
        try:
            st.markdown("#### Rolling Performance Curves (Revenue vs Gross Profit)")
            df_clean['parsed_date'] = pd.to_datetime(df_clean[col_mapping['date']])
            monthly_sales = df_clean.set_index('parsed_date').groupby(pd.Grouper(freq='M')).agg({
                col_mapping['revenue']: 'sum',
                col_mapping['profit']: 'sum'
            })
            
            fig_time, ax_time = plt.subplots(figsize=(14, 4.5))
            ax_time.plot(monthly_sales.index, monthly_sales[col_mapping['revenue']], label='Monthly Gross Revenue', marker='o', linewidth=2.5)
            ax_time.plot(monthly_sales.index, monthly_sales[col_mapping['profit']], label='Monthly Gross Profit', marker='o', linewidth=2.5)
            ax_time.set_title("Aggregated Financial Rolling Trends", color="#ffffff", pad=15)
            ax_time.set_ylabel("Value in USD ($)", color="#ffffff")
            ax_time.legend(facecolor='#1f293d', edgecolor='none')
            st.pyplot(fig_time)
            plt.close(fig_time)
        except Exception:
            st.info("💡 Chronological parsing bypassed. Ensure your `order_date` fields utilize recognizable standard datetime string sequences.")

    c_vis1, c_vis2 = st.columns(2)
    with c_vis1:
        if col_mapping['rating']:
            st.markdown("#### Experience Distribution Profiles")
            fig1, ax1 = plt.subplots(figsize=(6, 3.8))
            sns.countplot(data=df_clean, x=col_mapping['rating'], palette='Blues_r', ax=ax1)
            ax1.set_title("Total Rating Ingestion Volumes", color="#ffffff")
            st.pyplot(fig1)
            plt.close(fig1)
            
    with c_vis2:
        if len(numeric_columns) > 1:
            st.markdown("#### Multi-Feature Linear Correlations")
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            sns.heatmap(df_clean[numeric_columns].corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax2, cbar=False)
            ax2.set_title("Correlation Coefficient Matrix", color="#ffffff")
            st.pyplot(fig2)
            plt.close(fig2)

# --------------------------------------------------------------------------
# WORKFLOW 4: BEHAVIORAL COHORT CLUSTERING
# --------------------------------------------------------------------------
with tab_cluster:
    st.markdown('<h3 class="section-header">🧬 Unsupervised Behavioral Customer Segmentation</h3>', unsafe_allow_html=True)
    
    if len(numeric_columns) >= 2:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            target_features = st.multiselect(
                "Isolate Segmentation Vectors:",
                options=numeric_columns,
                default=numeric_columns[:2],
                key="cluster_select"
            )
            cluster_count = st.slider("Target Cohorts Count (K):", min_value=2, max_value=6, value=3)
        
        with cc2:
            if len(target_features) == 2:
                X_scaled = StandardScaler().fit_transform(df_clean[target_features])
                km = KMeans(n_clusters=cluster_count, random_state=42)
                df_clean['cluster_id'] = km.fit_predict(X_scaled)
                
                fig3, ax3 = plt.subplots(figsize=(7, 4))
                sns.scatterplot(
                    data=df_clean, x=target_features[0], y=target_features[1],
                    hue='cluster_id', palette='Set1', alpha=0.9, ax=ax3
                )
                ax3.set_title(f"Customer Cluster Segments Space (K={cluster_count})", color="#ffffff")
                st.pyplot(fig3)
                plt.close(fig3)
                
                st.markdown("##### Cluster Center Feature Means Matrix")
                st.dataframe(df_clean.groupby('cluster_id')[target_features].mean(), width="stretch")
            else:
                st.warning("⚠️ High dimensional rendering limited. Select exactly two numerical variables to generate visualization charts.")
    else:
        st.error("❌ Segmentation pipelines require at least two separate numerical columns inside the source log file.")

# --------------------------------------------------------------------------
# WORKFLOW 5: PREDICTIVE MACHINE LEARNING ENGINE
# --------------------------------------------------------------------------
with tab_predict:
    st.markdown('<h3 class="section-header">🤖 Supervised Production Pipeline Blocks</h3>', unsafe_allow_html=True)
    
    target_variable = st.selectbox("Designate Target Optimization Objective (Y):", options=df_clean.columns, key="ml_target")
    
    if target_variable:
        is_numeric = pd.api.types.is_numeric_dtype(df_clean[target_variable])
        distinct_values = df_clean[target_variable].nunique()
        
        # Smart Default Suggestion Routing
        suggested_type = "Regression" if (is_numeric and distinct_values >= 10) else "Classification"
        
        task_type = st.radio(
            "Algorithmic Model Selection Mode:",
            options=["Auto-Detect", "Classification", "Regression"],
            help="Manually steer modeling frameworks depending on target properties."
        )
        
        chosen_task = suggested_type if task_type == "Auto-Detect" else task_type
        st.info(f"⚙️ Supervised execution path active via **{chosen_task}** algorithm cascades.")
        
        # Filter and segregate explanatory feature sets
        all_features = [col for col in df_clean.columns if col != target_variable and col != 'cluster_id' and col != 'parsed_date']
        numeric_features = [col for col in all_features if pd.api.types.is_numeric_dtype(df_clean[col])]
        categorical_features = [col for col in all_features if not pd.api.types.is_numeric_dtype(df_clean[col])]
        
        st.markdown("##### Feature Vector Parameter Inclusion Matrix")
        selected_num_feats = st.multiselect("Include Continuous Features:", options=numeric_features, default=numeric_features)
        selected_cat_feats = st.multiselect("Include Categorical Features (Auto-Encoded):", options=categorical_features, default=[])
        
        chosen_features = selected_num_feats + selected_cat_feats
        
        if not chosen_features:
            st.warning("⚠️ Isolate feature factors to trigger structural weight model training.")
        else:
            # Clean record sets to prevent model training execution failures
            df_model = df_clean[[target_variable] + chosen_features].dropna()
            
            if len(df_model) < 20:
                st.error("❌ Training array index floor limits violated. Provide cleaner/larger structural data layers.")
            else:
                X = df_model[chosen_features]
                if selected_cat_feats:
                    X = pd.get_dummies(X, columns=selected_cat_feats, drop_first=True)
                
                y = df_model[target_variable]
                
                # Intercept regression errors before feeding arrays into scikit-learn
                if chosen_task == "Regression":
                    try:
                        y = y.astype(float)
                    except ValueError:
                        st.error(f"❌ **Data Type Defect:** Target `{target_variable}` contains values that cannot be cast to floats. Switch to Classification mode above.")
                        st.stop()
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # ROUTE A: CATEGORICAL CLASSIFICATION RUNS
                if chosen_task == "Classification":
                    y_train = y_train.astype(str)
                    y_test = y_test.astype(str)
                    
                    m_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                    m_rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                    
                    acc_lr = accuracy_score(y_test, m_lr.predict(X_test))
                    acc_rf = accuracy_score(y_test, m_rf.predict(X_test))
                    
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown(f'<div class="metric-card" style="border-left-color: #3b82f6;"><h5>Logistic Regression Accuracy</h5><h3>{acc_lr * 100:.2f}%</h3></div>', unsafe_allow_html=True)
                    with mc2:
                        st.markdown(f'<div class="metric-card" style="border-left-color: #8b5cf6;"><h5>Random Forest Accuracy</h5><h3>{acc_rf * 100:.2f}%</h3></div>', unsafe_allow_html=True)
                        
                    st.markdown("#### Random Forest Model Categorical Metrics Matrix Report")
                    st.text(classification_report(y_test, m_rf.predict(X_test)))
                
                # ROUTE B: CONTINUOUS REGRESSION RUNS
                else:
                    m_lin = LinearRegression().fit(X_train, y_train)
                    preds = m_lin.predict(X_test)
                    
                    r2 = r2_score(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Adjusted R² Variance Score</h5><h3>{r2:.4f}</h3></div>', unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f'<div class="metric-card" style="border-left-color: #ef4444;"><h5>Root Mean Squared Error (RMSE)</h5><h3>{rmse:.2f}</h3></div>', unsafe_allow_html=True)
