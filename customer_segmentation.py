import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = 'mall_customers.csv'  # Ensure the dataset is in the same directory
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Head:")
print(df.head())

# Select relevant features for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {cluster}')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Describe customer groups
for cluster in range(5):
    print(f"Cluster {cluster}:")
    print(df[df['Cluster'] == cluster][features].describe())
    print("\n")