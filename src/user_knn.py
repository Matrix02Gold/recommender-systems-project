"""Simple user-based collaborative filtering recommender."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_user_knn(train, movies):
    """Build user similarities from training ratings only."""
    user_ids = np.sort(train["user_id"].unique())
    movie_ids = movies["movie_id"].to_numpy()

    user_index = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    ratings_matrix = csr_matrix(
        (
            train["rating"].to_numpy(dtype=np.float32),
            (
                train["user_id"].map(user_index),
                train["movie_id"].map(movie_index),
            ),
        ),
        shape=(len(user_ids), len(movie_ids)),
    )

    similarities = cosine_similarity(ratings_matrix)
    np.fill_diagonal(similarities, 0)

    # Small fallback score for movies without ratings from neighbours.
    popularity = (
        train.groupby("movie_id").size()
        .reindex(movie_ids, fill_value=0)
        .to_numpy(dtype=float)
    )
    popularity = popularity / (popularity.max() + 1)

    return user_ids, movie_ids, ratings_matrix, similarities, popularity


def recommend_user_knn(user_id, model, k_neighbors=20, top_n=10):
    """Recommend movies using ratings from similar users."""
    user_ids, movie_ids, ratings_matrix, similarities, popularity = model

    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")

    scores = popularity.copy()
    user_matches = np.where(user_ids == user_id)[0]

    if len(user_matches):
        user_index = user_matches[0]
        similarity_row = similarities[user_index]

        # Find the most similar other users.
        neighbors = np.argsort(-similarity_row)[:k_neighbors]
        neighbors = neighbors[similarity_row[neighbors] > 0]

        if len(neighbors):
            weights = similarity_row[neighbors]
            neighbor_ratings = ratings_matrix[neighbors]

            # Similarity-weighted mean of neighbour ratings.
            weighted_sum = neighbor_ratings.T @ weights
            observed = neighbor_ratings.copy()
            observed.data[:] = 1
            total_weight = observed.T @ weights

            np.divide(
                weighted_sum, total_weight,
                out=scores, where=total_weight > 0,
            )

        # Never recommend a movie this user already rated in training.
        scores[ratings_matrix[user_index].indices] = -np.inf

    # Scores descending; break ties by movie ID.
    best = np.lexsort((movie_ids, -scores))[:top_n]
    return pd.DataFrame({"movie_id": movie_ids[best], "score": scores[best]})
