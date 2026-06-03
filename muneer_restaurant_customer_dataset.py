import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# 1. PAGE ARCHITECTURE & DARK MODE CANVAS SETUP
st.set_page_config(
    page_title="Executive Restaurant BI Platform",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished typography, padding distributions, and dark background cards
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

# 2. CONTROL PANEL & DATA INGESTION MATRIX
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📥 Ingest Restaurant CSV Dataset", type=["csv"])

# Welcome State when no file is present
if uploaded_file is None:
    st.title("📊 Executive Restaurant Analytics Matrix & Predictive Engine")
    st.markdown("---")
    st.info("👋 **System Initialization Ready.** Please upload an operational restaurant transaction log (`.csv`) in the sidebar control panel to populate the analytics framework.")
    
    with st.expander("💡 Expected Dataset Schema Requirements", expanded=True):
        st.markdown("""
        The engine dynamically parses tabular data but provides maximum feature engineering utility when the following vectors are present:
        * **Target Metrics:** `customer_rating` (Categorical/Discrete Ordinal)
        * **Operational Logs:** Revenue, Check Size, Table Size, Visit Frequency (Numerical)
        * **Demographics:** Gender, Age, Customer Segments (Categorical/Numerical)
        """)
else:
    # 3. DATA PERSISTENCE & PIPELINE UNIFICATION
    @st.cache_data
    def process_and_clean_data(file):
        raw_df = pd.read_csv(file)
        raw_df.columns = raw_df.columns.str.lower().str.strip()
        
        # Isolate numeric metrics for Z-score validation
        num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            z_scores = np.abs(zscore(raw_df[num_cols]))
            # Maintain structural indices with absolute z-scores < 3 standard deviations
            clean_mask = (z_scores < 3).all(axis=1)
            cleaned_df = raw_df[clean_mask].copy()
            outliers_removed = len(raw_df) - len(cleaned_df)
        else:
            cleaned_df = raw_df.copy()
            outliers_removed = 0
            
        return raw_df, cleaned_df, outliers_removed

    df_raw, df_clean, metrics_dropped = process_and_clean_data(uploaded_file)
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    # 4. SIDEBAR GLOBAL SUMMARY MATRIX
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
    
    # 5. WORKFLOW MANAGER (MAIN INTERFACE TABS)
    st.title("🍽️ Corporate Restaurant Intelligence Matrix")
    st.markdown("Real-time data mining, cohort customer segmentation, and machine learning deployment cascades.")
    
    tab_audit, tab_eda, tab_cluster, tab_predict = st.tabs([
        "📋 Data Quality & Audit", 
        "📉 Trend Visualization", 
        "🧬 Behavioral Segmentation", 
        "🤖 Predictive ML Core"
    ])

    # 🔬 WORKFLOW 1: DATA QUALITY & AUDIT PIPELINE
    with tab_audit:
        st.markdown('<h3 class="section-header">🔍 Ingested Stream Verifications</h3>', unsafe_allow_html=True)
        
        st.markdown("#### Granular Ledger Snapshot (First 10 Rows)")
        st.dataframe(df_clean.head(10), width="stretch")
        
        c_aud1, c_aud2 = st.columns(2)
        with c_aud1:
            st.markdown("#### File Metadata Schema")
            buffer = io.StringIO()
            df_clean.info(buf=buffer)
            st.text(buffer.getvalue())
            
        with c_aud2:
            st.markdown("#### Null Variance Accumulations")
            null_summary = pd.DataFrame({
                'Missing Values (Count)': df_clean.isnull().sum(),
                'Null Ratio (%)': (df_clean.isnull().sum() / len(df_clean)) * 100
            })
            st.dataframe(null_summary.round(2), width="stretch")

    # 📊 WORKFLOW 2: EXPLORATORY DATA ANALYSIS
    with tab_eda:
        st.markdown('<h3 class="section-header">📉 Distribution Profiles & Feature Intersections</h3>', unsafe_allow_html=True)
        
        c_vis1, c_vis2 = st.columns(2)
        
        with c_vis1:
            if 'customer_rating' in df_clean.columns:
                st.markdown("#### Metric Spread: Customer Feedback Ratings")
                fig1, ax1 = plt.subplots(figsize=(6, 3.5))
                # Explicit dark palette elements
                sns.countplot(data=df_clean, x='customer_rating', palette='Blues_r', ax=ax1)
                ax1.set_title("Rating Volume Ingestions", color="#ffffff")
                fig1.patch.set_facecolor('#1f293d')
                ax1.set_facecolor('#1f293d')
                ax1.tick_params(colors='#ffffff')
                ax1.xaxis.label.set_color('#ffffff')
                ax1.yaxis.label.set_color('#ffffff')
                st.pyplot(fig1)
                plt.close(fig1)
            else:
                st.info("ℹ️ `customer_rating` feature column absent from source CSV layout.")
                
        with c_vis2:
            if len(numeric_columns) > 1:
                st.markdown("#### Linear Multivariable Feature Correlations")
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                sns.heatmap(df_clean[numeric_columns].corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax2, cbar=False)
                ax2.set_title("Correlation Coefficient Matrix", color="#ffffff")
                fig2.patch.set_facecolor('#1f293d')
                ax2.tick_params(colors='#ffffff')
                st.pyplot(fig2)
                plt.close(fig2)

    # 🧬 WORKFLOW 3: BEHAVIORAL COHORT CLUSTERING
    with tab_cluster:
        st.markdown('<h3 class="section-header">🧬 Unsupervised Demographic Segment Generation</h3>', unsafe_allow_html=True)
        st.markdown("Applies standard feature scaling before fitting an unsupervised K-Means algorithm to isolate core target behaviors.")
        
        if len(numeric_columns) >= 2:
            cc1, cc2 = st.columns([1, 2])
            
            with cc1:
                target_features = st.multiselect(
                    "Isolate Target Vectors:",
                    options=numeric_columns,
                    default=numeric_columns[:2],
                    key="cluster_select"
                )
                cluster_count = st.slider("Target Clusters Target (K):", min_value=2, max_value=6, value=3)
            
            with cc2:
                if len(target_features) == 2:
                    X_scaled = StandardScaler().fit_transform(df_clean[target_features])
                    km = KMeans(n_clusters=cluster_count, random_state=42)
                    df_clean['cluster_id'] = km.fit_predict(X_scaled)
                    
                    fig3, ax3 = plt.subplots(figsize=(7, 3.8))
                    sns.scatterplot(
                        data=df_clean, x=target_features[0], y=target_features[1],
                        hue='cluster_id', palette='Set1', alpha=0.9, ax=ax3
                    )
                    ax3.set_title(f"Spatial Domain Matrix: K={cluster_count}", color="#ffffff")
                    fig3.patch.set_facecolor('#1f293d')
                    ax3.set_facecolor('#1f293d')
                    ax3.tick_params(colors='#ffffff')
                    ax3.xaxis.label.set_color('#ffffff')
                    ax3.yaxis.label.set_color('#ffffff')
                    st.pyplot(fig3)
                    plt.close(fig3)
                    
                    st.markdown("#### Cluster Centroid Mean Profiles")
                    st.dataframe(df_clean.groupby('cluster_id')[target_features].mean(), width="stretch")
                else:
                    st.warning("⚠️ High dimensional segmentation mapping constrained. Please explicitly select exactly 2 numerical features.")
        else:
            st.error("❌ Deep cluster operations require at least 2 distinct numeric columns inside the source file.")

    # 🤖 WORKFLOW 4: SUPERVISED MACHINE LEARNING CORES
    with tab_predict:
        st.markdown('<h3 class="section-header">🔮 Supervised Model Training Cascades</h3>', unsafe_allow_html=True)
        
        target_variable = st.selectbox("Assign Predictive Dependent Optimization Target (Y):", options=df_clean.columns, key="ml_target")
        
        if target_variable:
            # Construct independent matrix from numeric vectors minus target variable
            feature_pool = [col for col in numeric_columns if col != target_variable]
            
            if feature_pool:
                X = df_clean[feature_pool]
                y = df_clean[target_variable]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # BRANCH A: Discrete/Categorical Modeling Routes (Classification)
                if df_clean[target_variable].dtype == 'object' or df_clean[target_variable].nunique() < 10:
                    st.markdown(f"### 🎯 Classifier Pipeline Active: `{target_variable}` Estimation")
                    
                    m_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                    m_rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                    
                    acc_lr = accuracy_score(y_test, m_lr.predict(X_test))
                    acc_rf = accuracy_score(y_test, m_rf.predict(X_test))
                    
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #3b82f6;">
                            <h5>Logistic Regression Accuracy</h5>
                            <h3>{acc_lr * 100:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with mc2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #8b5cf6;">
                            <h5>Random Forest Accuracy</h5>
                            <h3>{acc_rf * 100:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("#### Ensemble Model Exhaustive Matrix Output")
                    st.text(classification_report(y_test, m_rf.predict(X_test)))
                
                # BRANCH B: Continuous Modeling Routes (Regression)
                else:
                    st.markdown(f"### 📈 Continuous Regressor Active: `{target_variable}` Estimation")
                    
                    m_lin = LinearRegression().fit(X_train, y_train)
                    preds = m_lin.predict(X_test)
                    
                    r2 = r2_score(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #10b981;">
                            <h5>Explained Variance Variance Metric (R²)</h5>
                            <h3>{r2:.4f}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #ef4444;">
                            <h5>Root Mean Squared Error (RMSE)</h5>
                            <h3>{rmse:.2f}</h3>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("❌ Model compilation aborted: Insufficient auxiliary numeric feature columns to establish independent variables.")
