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

# ------------------------
# Netflix-like KNN recommendation example
# ------------------------

def cosine_similarity_matrix(matrix):
    # compute pairwise cosine similarity for rows
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    # avoid division by zero
    norm[norm == 0] = 1
    return (matrix @ matrix.T) / (norm @ norm.T)


def recommend_from_user_knn(ratings, user_index, k=2):
    """Simple user-based KNN recommendation with toy ratings."""
    sim = cosine_similarity_matrix(ratings)
    user_sims = sim[user_index]

    # Do not include self
    user_sims[user_index] = -1

    # find k nearest similar users
    neighbors = np.argsort(user_sims)[::-1][:k]
    print(f"Nearest neighbors for user {user_index}: {neighbors}")

    # Predict missing ratings where user has 0
    user_ratings = ratings[user_index]
    unseen = np.where(user_ratings == 0)[0]
    preds = {}
    for item in unseen:
        weighted_sum = 0.0
        sim_sum = 0.0
        for n in neighbors:
            weighted_sum += user_sims[n] * ratings[n, item]
            sim_sum += abs(user_sims[n])
        if sim_sum > 0:
            preds[item] = weighted_sum / sim_sum
        else:
            preds[item] = 0.0

    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds


def netflix_knn_demo():
    print("\n--- Netflix-like KNN recommendation demo ---")
    # toy user-item ratings (rows=users, cols=movies), 0=not rated
    ratings = np.array([
        [5, 4, 0, 1],  # Alice
        [4, 5, 1, 2],  # Bob
        [1, 2, 5, 4],  # Carol
        [2, 1, 4, 5],  # Dave
    ], dtype=float)

    user_index = 0  # Alice
    print("User ratings matrix:")
    print(ratings)
    recommendations = recommend_from_user_knn(ratings, user_index, k=2)
    print(f"Recommendations for Alice (user 0) sorted by predicted score:")
    for item, score in recommendations:
        print(f"  Movie {item+1}: predicted score {score:.2f}")

    print("\nItem-based similarity (cosine) for movies")
    movie_sim = cosine_similarity_matrix(ratings.T)
    print(movie_sim)


if __name__ == '__main__':
    netflix_knn_demo()