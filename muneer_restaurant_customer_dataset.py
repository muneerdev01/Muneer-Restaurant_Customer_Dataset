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

# Custom dark-theme container styling
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
# 2. FILE INGESTION & DATA ENGINE (WITH ROBUST CLEANING)
# ==============================================================================
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📥 Ingest Restaurant CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.title("📊 Executive Restaurant Analytics Matrix & Predictive Engine")
    st.markdown("---")
    st.info("👋 **System Initialization Ready.** Please upload an operational restaurant transaction log (`.csv`) in the sidebar control panel to populate the analytics framework.")
    
    with st.expander("💡 Expected Dataset Schema Requirements", expanded=True):
        st.markdown("""
        The engine dynamically parses tabular data but provides maximum utility when the following vectors are present:
        * **Target Metrics:** `customer_rating` or `rating` (Discrete Ordinal/Numerical)
        * **Operational Logs:** `total_amount`, `profit`, `loss`, `discount`, `quantity` (Numerical)
        * **Identifications:** `order_id`, `customer_id`, `restaurant_id`
        """)
else:
    @st.cache_data
    def process_and_clean_data(file):
        raw_df = pd.read_csv(file)
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

    df_raw, df_clean, metrics_dropped = process_and_clean_data(uploaded_file)
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    # Dynamic column mapping helper to support varying dataset variants
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
        'date': 'date' if 'date' in df_clean.columns else ('order_date' if 'order_date' in df_clean.columns else None)
    }

    # ==============================================================================
    # 3. GLOBAL EXECUTIVE KPIS (SIDEBAR)
    # ==============================================================================
    st.sidebar.markdown("### 📈 Global Pipeline Metrics")
    st.sidebar.markdown(f"""
    <div class="metric-card" style="border-left-color: #10b981;">
        <h5>Ingested Volume</h5>
        <h3>{df_raw.shape[0]:,} x {df_raw.shape[1]}</h3>
    </div>
    <div class="metric-card" style="border-left-color: #f59e0b;">
        <h5>Anomalies Pruned</h5>
        <h3>{metrics_dropped:,} records</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # ==============================================================================
    # 4. TABBED WORKFLOW APPLICATION MANAGEMENT
    # ==============================================================================
    st.title("🍽️ Corporate Restaurant Intelligence Dashboard")
    st.markdown("Automated operational reporting, dynamic cohort clustering, and machine learning inference engines.")
    
    tab_summary, tab_audit, tab_eda, tab_cluster, tab_predict = st.tabs([
        "📊 Executive Summary",
        "📋 Data Audit", 
        "📉 Trend Visualization", 
        "🧬 Customer Segmentation", 
        "🤖 Predictive ML Engine"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: EXECUTIVE KPI SUMMARY SUMMARY (Merged from analysis script formulas)
    # --------------------------------------------------------------------------
    with tab_summary:
        st.markdown('<h3 class="section-header">👑 Executive Performance Ledger</h3>', unsafe_allow_html=True)
        
        # Calculate primary transactional counts
        total_orders = df_clean[col_mapping['order']].nunique() if col_mapping['order'] else len(df_clean)
        total_customers = df_clean[col_mapping['customer']].nunique() if col_mapping['customer'] else "N/A"
        total_restaurants = df_clean[col_mapping['restaurant']].nunique() if col_mapping['restaurant'] else "N/A"
        
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.markdown(f'<div class="metric-card"><h5>Unique Orders Processed</h5><h3>{total_orders:,}</h3></div>', unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Active Customer Base</h5><h3>{total_customers}</h3></div>', unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f'<div class="metric-card" style="border-left-color: #8b5cf6;"><h5>Active Restaurant Locations</h5><h3>{total_restaurants}</h3></div>', unsafe_allow_html=True)

        # Calculate fiscal metrics safely
        rev_val = df_clean[col_mapping['revenue']].sum() if col_mapping['revenue'] else 0
        prof_val = df_clean[col_mapping['profit']].sum() if col_mapping['profit'] else 0
        loss_val = df_clean[col_mapping['loss']].sum() if col_mapping['loss'] else 0
        net_profit = prof_val - loss_val
        
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Gross Revenue</h5><h3>${rev_val:,.2f}</h3></div>', unsafe_allow_html=True)
        with f_col2:
            st.markdown(f'<div class="metric-card" style="border-left-color: #3b82f6;"><h5>Gross Profit</h5><h3>${prof_val:,.2f}</h3></div>', unsafe_allow_html=True)
        with f_col3:
            st.markdown(f'<div class="metric-card" style="border-left-color: #ef4444;"><h5>Logged Material Losses</h5><h3>${loss_val:,.2f}</h3></div>', unsafe_allow_html=True)
        with f_col4:
            net_color = "#10b981" if net_profit >= 0 else "#ef4444"
            st.markdown(f'<div class="metric-card" style="border-left-color: {net_color};"><h5>Net Retained Margin</h5><h3>${net_profit:,.2f}</h3></div>', unsafe_allow_html=True)

        # Operational Performance Averages
        st.markdown("#### Operational Unit Metrics")
        avg_col1, avg_col2, avg_col3, avg_col4 = st.columns(4)
        
        with avg_col1:
            aov = (rev_val / total_orders) if total_orders > 0 else 0
            st.metric("Average Order Value (AOV)", f"${aov:.2f}")
        with avg_col2:
            avg_rat = df_clean[col_mapping['rating']].mean() if col_mapping['rating'] else 0
            st.metric("Mean Experience Rating", f"{avg_rat:.2f} ⭐")
        with avg_col3:
            avg_disc = df_clean[col_mapping['discount']].mean() if col_mapping['discount'] else 0
            st.metric("Average Applied Discount", f"{avg_disc:.1f}%")
        with avg_col4:
            avg_qty = df_clean[col_mapping['quantity']].mean() if col_mapping['quantity'] else 0
            st.metric("Average Items Per Order", f"{avg_qty:.1f}")

    # --------------------------------------------------------------------------
    # TAB 2: DATA AUDIT PIPELINE
    # --------------------------------------------------------------------------
    with tab_audit:
        st.markdown('<h3 class="section-header">🔍 Verification Stream Profiles</h3>', unsafe_allow_html=True)
        st.markdown("#### Dynamic Tabular Record Browser")
        st.dataframe(df_clean.head(15), width="stretch")
        
        c_aud1, c_aud2 = st.columns(2)
        with c_aud1:
            st.markdown("#### System Schema Structural Summary")
            buffer = io.StringIO()
            df_clean.info(buf=buffer)
            st.text(buffer.getvalue())
            
        with c_aud2:
            st.markdown("#### Column Variable Null Densities")
            null_summary = pd.DataFrame({
                'Null Fields (Count)': df_clean.isnull().sum(),
                'Null Proportion (%)': (df_clean.isnull().sum() / len(df_clean)) * 100
            })
            st.dataframe(null_summary.round(2), width="stretch")

    # --------------------------------------------------------------------------
    # TAB 3: TREND VISUALIZATION (With automated native script conversions)
    # --------------------------------------------------------------------------
    with tab_eda:
        st.markdown('<h3 class="section-header">📉 Interactive Distribution & Historical Charts</h3>', unsafe_allow_html=True)
        
        # Plot Time-Series Trends dynamically if Date is mapped
        if col_mapping['date'] and col_mapping['revenue'] and col_mapping['profit']:
            try:
                st.markdown("#### Operational Growth Over Time (Revenue vs Profit)")
                df_clean['parsed_date'] = pd.to_datetime(df_clean[col_mapping['date']])
                monthly_sales = df_clean.set_index('parsed_date').groupby(pd.Grouper(freq='M')).agg({
                    col_mapping['revenue']: 'sum',
                    col_mapping['profit']: 'sum'
                })
                
                fig_time, ax_time = plt.subplots(figsize=(14, 5))
                ax_time.plot(monthly_sales.index, monthly_sales[col_mapping['revenue']], label='Monthly Total Revenue', marker='o', linewidth=2)
                ax_time.plot(monthly_sales.index, monthly_sales[col_mapping['profit']], label='Monthly Total Profit', marker='o', linewidth=2)
                ax_time.set_title("Aggregated Financial Rolling Trends", color="#ffffff", pad=15)
                ax_time.legend(facecolor='#1f293d', edgecolor='none')
                st.pyplot(fig_time)
                plt.close(fig_time)
            except Exception:
                st.info("💡 Date column parsing bypassed. To unlock time series forecasting charts, formatting requirements must match ISO datetime parameters.")

        c_vis1, c_vis2 = st.columns(2)
        with c_vis1:
            if col_mapping['rating']:
                st.markdown("#### Volume Allocations Across Rating Profiles")
                fig1, ax1 = plt.subplots(figsize=(6, 3.8))
                sns.countplot(data=df_clean, x=col_mapping['rating'], palette='Blues_r', ax=ax1)
                ax1.set_title("Total Rating Ingestion Volumetrics", color="#ffffff")
                st.pyplot(fig1)
                plt.close(fig1)
                
        with c_vis2:
            if len(numeric_columns) > 1:
                st.markdown("#### Multi-Feature Linear Correlation Mapping")
                fig2, ax2 = plt.subplots(figsize=(6, 3.8))
                sns.heatmap(df_clean[numeric_columns].corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax2, cbar=False)
                ax2.set_title("Feature Correlation Coefficients Matrix", color="#ffffff")
                st.pyplot(fig2)
                plt.close(fig2)

    # --------------------------------------------------------------------------
    # TAB 4: BEHAVIORAL CUSTOMER SEGMENTATION
    # --------------------------------------------------------------------------
    with tab_cluster:
        st.markdown('<h3 class="section-header">🧬 Unsupervised K-Means Cluster Models</h3>', unsafe_allow_html=True)
        
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
            st.error("❌ High segmentation analytics require at least two available continuous numeric data columns.")

    # --------------------------------------------------------------------------
    # TAB 5: PREDICTIVE ML ENGINE (FAIL-SAFE PIPELINE)
    # --------------------------------------------------------------------------
    with tab_predict:
        st.markdown('<h3 class="section-header">🤖 Supervised Production Pipeline Blocks</h3>', unsafe_allow_html=True)
        
        target_variable = st.selectbox("Designate Target Prediction Vector (Y):", options=df_clean.columns, key="ml_target")
        
        if target_variable:
            is_numeric = pd.api.types.is_numeric_dtype(df_clean[target_variable])
            distinct_values = df_clean[target_variable].nunique()
            
            # Smart Default Suggestion Routing
            suggested_type = "Regression" if (is_numeric and distinct_values >= 10) else "Classification"
            
            task_type = st.radio(
                "Algorithmic Task Routing Mode:",
                options=["Auto-Detect", "Classification", "Regression"],
                help="Manually override modeling workflows depending on target characteristics."
            )
            
            chosen_task = suggested_type if task_type == "Auto-Detect" else task_type
            st.info(f"⚙️ Supervised Core executing via **{chosen_task}** pipeline structures.")
            
            # Isolate Feature Sets
            all_features = [col for col in df_clean.columns if col != target_variable and col != 'cluster_id' and col != 'parsed_date']
            numeric_features = [col for col in all_features if pd.api.types.is_numeric_dtype(df_clean[col])]
            categorical_features = [col for col in all_features if not pd.api.types.is_numeric_dtype(df_clean[col])]
            
            st.markdown("##### Model Feature Inclusions")
            selected_num_feats = st.multiselect("Select Continuous Covariates:", options=numeric_features, default=numeric_features)
            selected_cat_feats = st.multiselect("Select Discrete Factors (Auto One-Hot Encoded):", options=categorical_features, default=[])
            
            chosen_features = selected_num_feats + selected_cat_feats
            
            if not chosen_features:
                st.warning("⚠️ Choose features to begin supervised network weight configurations.")
            else:
                df_model = df_clean[[target_variable] + chosen_features].dropna()
                
                if len(df_model) < 20:
                    st.error("❌ Training dataset sample limits reached. Ingest more comprehensive data arrays to compute models.")
                else:
                    X = df_model[chosen_features]
                    if selected_cat_feats:
                        X = pd.get_dummies(X, columns=selected_cat_feats, drop_first=True)
                    
                    y = df_model[target_variable]
                    
                    if chosen_task == "Regression":
                        try:
                            y = y.astype(float)
                        except ValueError:
                            st.error(f"❌ **Data Type Defect:** Target `{target_variable}` contains values that cannot be cast to floats. Switch to Classification mode above.")
                            st.stop()
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # ROUTE A: CATEGORICAL CLASSIFICATION CAPABILITY
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
                            
                        st.markdown("#### Random Forest Comprehensive Classification Report")
                        st.text(classification_report(y_test, m_rf.predict(X_test)))
                    
                    # ROUTE B: CONTINUOUS COEFFICIENT REGRESSION CAPABILITY
                    else:
                        m_lin = LinearRegression().fit(X_train, y_train)
                        preds = m_lin.predict(X_test)
                        
                        r2 = r2_score(y_test, preds)
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.markdown(f'<div class="metric-card" style="border-left-color: #10b981;"><h5>Adjusted R² Score</h5><h3>{r2:.4f}</h3></div>', unsafe_allow_html=True)
                        with rc2:
                            st.markdown(f'<div class="metric-card" style="border-left-color: #ef4444;"><h5>Root Mean Squared Error (RMSE)</h5><h3>{rmse:.2f}</h3></div>', unsafe_allow_html=True)
