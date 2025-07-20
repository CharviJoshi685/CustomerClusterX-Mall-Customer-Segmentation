# app.py – Streamlit App for CustomerClusterX

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

else:
    st.info("📌 Please upload the Mall_Customers.csv file to proceed.")
