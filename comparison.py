# app.py – Streamlit App for CustomerClusterX

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Title
st.set_page_config(page_title="CustomerClusterX", layout="wide")
st.title("🛍️ CustomerClusterX")
st.markdown("""
Segment mall customers into groups based on their demographics and shopping habits.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload your mall customer CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    features = ['age', 'annual_income_(k$)', 'spending_score_(1-100)', 'gender']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    # Plot
    fig = px.scatter(
        df, x='pca1', y='pca2', color='cluster',
        hover_data=features,
        title="Customer Segments Visualization",
        color_continuous_scale='Turbo'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster summary
    st.subheader("Cluster-wise Summary")
    st.dataframe(df.groupby('cluster')[features].mean().round(2))

    # Regression Comparison
    st.subheader("📈 Regression Model Comparison")
    X_reg = df[['age', 'annual_income_(k$)', 'gender']]
    y_reg = df['spending_score_(1-100)']
    X_scaled_reg = scaler.fit_transform(X_reg)

    # Linear Regression
    linreg = LinearRegression()
    linreg.fit(X_scaled_reg, y_reg)
    y_pred_lin = linreg.predict(X_scaled_reg)
    r2_lin = r2_score(y_reg, y_pred_lin)
    mse_lin = mean_squared_error(y_reg, y_pred_lin)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled_reg, y_reg)
    y_pred_rf = rf.predict(X_scaled_reg)
    r2_rf = r2_score(y_reg, y_pred_rf)
    mse_rf = mean_squared_error(y_reg, y_pred_rf)

    st.markdown(f"**Linear Regression:** R² = {r2_lin:.3f}, MSE = {mse_lin:.2f}")
    st.markdown(f"**Random Forest:** R² = {r2_rf:.3f}, MSE = {mse_rf:.2f}")

else:
    st.info("📌 Please upload the Mall_Customers.csv file to proceed.")
