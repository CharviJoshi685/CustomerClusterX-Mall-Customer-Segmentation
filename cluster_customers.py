# cluster_customers.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# === Step 1: Load Dataset ===
df = pd.read_csv("data/Mall_Customers.csv")

# Rename columns for consistency
df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

# === Step 2: Encode categorical ===
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# === Step 3: Feature Selection ===
X = df[['age', 'annual_income_(k$)', 'spending_score_(1-100)', 'gender']]

# === Step 4: Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: KMeans Clustering ===
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['cluster'] = kmeans_labels

# === Step 6: PCA for 2D Visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# === Step 7: Plot Clusters ===
fig = px.scatter(
    df, x='pca1', y='pca2', color='cluster', 
    hover_data=['age', 'annual_income_(k$)', 'spending_score_(1-100)'],
    title="Customer Segments using KMeans + PCA",
    color_continuous_scale='Viridis'
)
fig.write_html("cluster_plot.html")
print("✅ Clustering complete. Interactive plot saved as 'cluster_plot.html'")
