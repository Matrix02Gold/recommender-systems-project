
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_item_similarity(train, method="adjusted_cosine"):
    """Calculate similarities between training movies."""

    user_ids = sorted(train["user_id"].unique())
    movie_ids = sorted(train["movie_id"].unique())

    user_index = {
        user_id: i for i, user_id in enumerate(user_ids)
    }
    movie_index = {
        movie_id: i for i, movie_id in enumerate(movie_ids)
    }

    if method == "cosine":
        values = train["rating"].astype(float)

    elif method == "adjusted_cosine":
        user_means = (
            train.groupby("user_id")["rating"].transform("mean")
        )
        values = train["rating"] - user_means

    else:
        raise ValueError(f"Unknown similarity method: {method}")

    # Create sparse user-item matrix
    user_item = csr_matrix(
        (
            values.to_numpy(),
            (
                train["user_id"].map(user_index).to_numpy(),
                train["movie_id"].map(movie_index).to_numpy(),
            ),
        ),
        shape=(len(user_ids), len(movie_ids)),
    )

    similarities = cosine_similarity(user_item.T)
    np.fill_diagonal(similarities, 0)

    return similarities, movie_ids


def recommend_item_knn(
    user_id,
    train,
    similarity_matrix,
    movie_ids,
    k_neighbors=20,
    top_n=10,
    shrinkage=1.0,
):
    """Recommend unseen movies using item-based kNN."""

    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")

    if shrinkage < 0:
        raise ValueError("shrinkage cannot be negative")

    # Get the user's training ratings
    user_ratings = train.loc[
        train["user_id"] == user_id,
        ["movie_id", "rating"]
    ]

    if user_ratings.empty:
        return pd.DataFrame(columns=["movie_id", "score"])

    # Map movie IDs to similarity matrix indices
    movie_index = {
        movie_id: i for i, movie_id in enumerate(movie_ids)
    }

    rated_indices = (
        user_ratings["movie_id"]
        .map(movie_index)
        .to_numpy(dtype=int)
    )

    ratings = user_ratings["rating"].to_numpy(dtype=float)

    # Get positive similarities for rated movies
    weights = np.maximum(
        similarity_matrix[:, rated_indices], 0.0
    )

    # Select k nearest neighbours
    k = min(k_neighbors, len(rated_indices))

    neighbors = np.argpartition(
        weights, -k, axis=1
    )[:, -k:]

    neighbor_weights = np.take_along_axis(
        weights, neighbors, axis=1
    )

    # Calculate regularized prediction scores
    numerator = (
        neighbor_weights * ratings[neighbors]
    ).sum(axis=1)

    denominator = neighbor_weights.sum(axis=1)

    scores = np.divide(
        numerator + shrinkage * ratings.mean(),
        denominator + shrinkage,
        out=np.full(len(movie_ids), -np.inf),
        where=denominator > 0,
    )

    # Create recommendation list
    recommendations = pd.DataFrame({
        "movie_id": movie_ids,
        "score": scores
    })

    # Remove invalid predictions and already-rated movies
    recommendations = recommendations.loc[
        np.isfinite(recommendations["score"])
        & ~recommendations["movie_id"].isin(
            user_ratings["movie_id"]
        )
    ]

    # Return the highest-scoring unseen movies
    return (
        recommendations.sort_values(
            ["score", "movie_id"],
            ascending=[False, True]
        )
        .head(top_n)
        .copy()
    )
