"""Content-based recommendation using MovieLens genres and release decades.

The user profile follows the mean-centred rating aggregation in Lecture 2.2.
"""

import numpy as np
import pandas as pd


def build_content_model(train, movies, year_weight=0.5):
    """Build normalized movie features and user taste profiles from training data."""
    if year_weight < 0:
        raise ValueError("year_weight cannot be negative")

    # Genre labels are multi-valued, e.g. "Action|Adventure|Sci-Fi".
    genres = movies["genre"].fillna("").str.get_dummies(sep="|")

    # MovieLens stores release years at the end of movie titles.
    years = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)\s*$")[0],
        errors="coerce",
    )
    decades = (years // 10 * 10).fillna(-1).astype(int).astype(str)
    decade_features = pd.get_dummies(decades, prefix="decade")

    features = np.column_stack([
        genres.to_numpy(dtype=float),
        year_weight * decade_features.to_numpy(dtype=float),
    ])

    # Unit-length features enable cosine-style similarity scoring.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.divide(
        features, norms, out=np.zeros_like(features), where=norms > 0
    )

    movie_ids = movies["movie_id"].to_numpy()
    movie_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    profiles = {}
    seen_by_user = {}

    for user_id, history in train.groupby("user_id"):
        indices = history["movie_id"].map(movie_index).to_numpy(dtype=int)
        ratings = history["rating"].to_numpy(dtype=float)
        centered = ratings - ratings.mean()
        profile = centered @ features[indices]

        # With constant ratings, centering gives a zero profile.
        if np.linalg.norm(profile) < 1e-12:
            liked_indices = indices[ratings >= 4]
            if len(liked_indices) == 0:
                liked_indices = indices
            profile = features[liked_indices].mean(axis=0)

        profile_norm = np.linalg.norm(profile)
        if profile_norm > 0:
            profile = profile / profile_norm

        profiles[user_id] = profile
        seen_by_user[user_id] = set(history["movie_id"])

    return movie_ids, features, profiles, seen_by_user


def recommend_content_based(user_id, model, top_n=10):
    """Return the user's top_n unseen movies from the full catalogue."""
    movie_ids, features, profiles, seen_by_user = model
    profile = profiles.get(user_id)

    if profile is None:
        return pd.DataFrame(columns=["movie_id", "score"])

    scores = features @ profile
    recommendations = pd.DataFrame({"movie_id": movie_ids, "score": scores})
    recommendations = recommendations.loc[
        ~recommendations["movie_id"].isin(seen_by_user[user_id])
    ]

    return (
        recommendations.sort_values(["score", "movie_id"], ascending=[False, True])
        .head(top_n)
        .copy()
    )
