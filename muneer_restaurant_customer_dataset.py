import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(page_title="Restaurant Customer Analytics Matrix", layout="wide", page_icon="🍽️")

st.title("🍽️ Restaurant Customer Insights & Machine Learning Dashboard")
st.markdown("Upload your restaurant operational data to run automated pipelines, clustering, and predictive models.")
st.markdown("---")

# 1. Web Data Loader Layer
uploaded_file = st.sidebar.file_uploader("📂 Upload Restaurant CSV Dataset", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    # Standardize column headers to mitigate casing/whitespace errors
    df.columns = df.columns.str.lower().str.strip()
    
    st.subheader("📋 Raw Data Audit Stream (First 5 Records)")
    st.dataframe(df.head(), use_container_width=True)

    # 2. Data Quality & Profiling
    st.header("⚙️ 1. Data Quality Profile & Outlier Removal")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Dimensions & Types:**")
        buffer = []
        df.info(buf=buffer.write)
        st.text("\n".join(buffer.getvalue().split("\n")[3:-2]))
        
    with col2:
        st.write("**Missing Values Analysis:**")
        st.write(df.isnull().sum())

    # Outlier Detection via Numeric Features Z-Score
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        z_scores = np.abs(zscore(df[numeric_cols]))
        # Keep rows where all numeric z-scores are less than 3
        filtered_entries = (z_scores < 3).all(axis=1)
        df_clean = df[filtered_entries]
        
        st.success(f"✅ **Outlier Filtering Completed:** Rows reduced from {len(df)} to {len(df_clean)} (Removed {len(df) - len(df_clean)} anomalies).")
    else:
        df_clean = df.copy()

    # 3. Exploratory Data Analysis & Dynamic Visuals
    st.header("📊 2. Operational EDA & Trend Analytics")
    
    c_vis1, c_vis2 = st.columns(2)
    
    # Chart 1: Customer Ratings Metric Distribution
    with c_vis1:
        if 'customer_rating' in df_clean.columns:
            st.write("**Distribution of Customer Ratings:**")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            sns.countplot(data=df_clean, x='customer_rating', palette='viridis', ax=ax1)
            ax1.set_title("Customer Rating Count Matrix")
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("💡 Column 'customer_rating' not found for plotting.")

    # Chart 2: Correlation Heatmap
    with c_vis2:
        if len(numeric_cols) > 1:
            st.write("**Numerical Feature Cross-Correlation Map:**")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
            ax2.set_title("Correlation Evaluation Matrix")
            st.pyplot(fig2)
            plt.close(fig2)

    # 4. Unsupervised Machine Learning: Customer Clustering
    st.header("🧬 3. Unsupervised K-Means Behavioral Clustering")
    
    # Dynamically select two numerical columns for clustering mapping
    if len(numeric_cols) >= 2:
        cluster_features = st.multiselect(
            "Select 2 Features for Behavioral Segmentation Mapping:", 
            options=numeric_cols, 
            default=numeric_cols[:2]
        )
        
        if len(cluster_features) == 2:
            X_clust = df_clean[cluster_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clust)
            
            # Setup K-Means Model
            kmeans = KMeans(n_clusters=3, random_state=42)
            df_clean['cluster_id'] = kmeans.fit_predict(X_scaled)
            
            # Scatter Plotting Profiles
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sns.scatterplot(
                data=df_clean, x=cluster_features[0], y=cluster_features[1], 
                hue='cluster_id', palette='Set1', alpha=0.8, ax=ax3
            )
            ax3.set_title(f"Customer Clusters: {cluster_features[0]} vs {cluster_features[1]}")
            st.pyplot(fig3)
            plt.close(fig3)
            
            st.write("**Segmented Demographic Cohort Profiles:**")
            st.dataframe(df_clean.groupby('cluster_id')[cluster_features].mean(), use_container_width=True)

    # 5. Supervised Machine Learning Engines
    st.header("🤖 4. Supervised Predictive Modeling Core")
    
    target_col = st.selectbox("Select Target Variable for Predictive Machine Learning:", options=df_clean.columns)
    
    if target_col:
        # Separate features vs labels
        X_ml = df_clean.drop(columns=[target_col])
        # Drop unique/uninformative objects or identifiers
        X_ml = X_ml.select_dtypes(include=[np.number]) 
        y_ml = df_clean[target_col]
        
        if not X_ml.empty and len(X_ml.columns) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
            
            # CHECKCASE: Target is discrete/categorical/object -> Classification Routing
            if df_clean[target_col].dtype == 'object' or df_clean[target_col].nunique() < 10:
                st.subheader(f"🏷️ Classification Analysis for Target: `{target_col}`")
                
                # Train Models
                log_reg = LogisticRegression(max_iter=1000)
                rf_clf = RandomForestClassifier(random_state=42)
                
                log_reg.fit(X_train, y_train)
                rf_clf.fit(X_train, y_train)
                
                # Predictions
                y_pred_log = log_reg.predict(X_test)
                y_pred_rf = rf_clf.predict(X_test)
                
                # UI Metrics Display
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric(label="Logistic Regression Accuracy", value=f"{accuracy_score(y_test, y_pred_log):.4f}")
                with mc2:
                    st.metric(label="Random Forest Accuracy", value=f"{accuracy_score(y_test, y_pred_rf):.4f}")
                
                st.write("**Random Forest Exhaustive Classification Breakdown:**")
                st.text(classification_report(y_test, y_pred_rf))
                
            # CHECKCASE: Target is continuous/float -> Regression Routing
            else:
                st.subheader(f"📈 Continuous Regression Analysis for Target: `{target_col}`")
                
                lin_reg = LinearRegression()
                lin_reg.fit(X_train, y_train)
                y_pred_lin = lin_reg.predict(X_test)
                
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.metric(label="Model Variance Score (R²)", value=f"{r2_score(y_test, y_pred_lin):.4f}")
                with rc2:
                    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{np.sqrt(mean_squared_error(y_test, y_pred_lin)):.2f}")
                    
        else:
            st.warning("⚠️ Insufficient numerical feature data metrics left to evaluate models after isolation.")

else:
    st.info("👋 Welcome! Please upload a valid restaurant transactional `.csv` file in the sidebar filter panel to initialize data streams.")
