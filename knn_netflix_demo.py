import numpy as np


def cosine_similarity_matrix(matrix):
    """Compute row-wise cosine similarity matrix."""
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (matrix @ matrix.T) / (norm @ norm.T)


def recommend_from_user_knn(ratings, user_index, k=2):
    """User-based KNN predictions for unrated items (0=unrated)."""
    sim = cosine_similarity_matrix(ratings)
    user_sim = sim[user_index].copy()
    user_sim[user_index] = -1.0

    neighbors = np.argsort(user_sim)[::-1][:k]
    print(f"Nearest neighbors for user {user_index}: {neighbors}")

    unseen_items = np.where(ratings[user_index] == 0)[0]
    predictions = {}
    for item in unseen_items:
        weighted = 0.0
        weights = 0.0
        for n in neighbors:
            weighted += user_sim[n] * ratings[n, item]
            weights += abs(user_sim[n])
        predictions[item] = weighted / weights if weights > 0 else 0.0

    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)


def main():
    print("Netflix-like KNN demo (user-based collaborative filtering)\n")

    ratings = np.array([
        [5, 4, 0, 1],  # Alice
        [4, 5, 1, 2],  # Bob
        [1, 2, 5, 4],  # Carol
        [2, 1, 4, 5],  # Dave
    ], dtype=float)

    print("Ratings matrix (rows=users, cols=movies):")
    print(ratings)

    user_index = 0
    recs = recommend_from_user_knn(ratings, user_index, k=2)
    print("\nPredicted recommendations for Alice (user 0):")
    for item, score in recs:
        print(f"  Movie {item + 1}: predicted score {score:.2f}")

    print("\nMovie-to-movie cosine similarity:")
    movie_sim = cosine_similarity_matrix(ratings.T)
    print(np.round(movie_sim, 3))

    print("\nInterpretation:")
    print("- KNN finds nearest users by similarity and predicts missing ratings.")
    print("- In real Netflix systems, this is scaled to millions of users/items and often combined with matrix factorization.")


if __name__ == '__main__':
    main()
