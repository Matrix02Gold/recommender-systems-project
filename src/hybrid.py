"""Simple hybrid recommender: combine content and popularity rankings."""

import pandas as pd
from .content_based import recommend_content_based


def recommend_hybrid(user_id, content_model, popular_movies, alpha=0.5, top_n=10):
    """Combine two full-catalogue rankings; alpha weights content (0 to 1)."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    seen = content_model[3].get(user_id, set())
    content = recommend_content_based(
        user_id, content_model, top_n=len(popular_movies)
    )
    if content.empty:
        return pd.DataFrame(columns=["movie_id", "score"])

    popular = popular_movies.loc[
        ~popular_movies["movie_id"].isin(seen)
    ]

    # Higher ranks give higher scores, with equal scale for both models.
    content_scores = pd.Series(
        range(len(content), 0, -1), index=content["movie_id"]
    )
    popular_scores = pd.Series(
        range(len(popular), 0, -1), index=popular["movie_id"]
    )

    scores = alpha * content_scores + (1 - alpha) * popular_scores
    recommendations = pd.DataFrame({
        "movie_id": scores.index,
        "score": scores.to_numpy(),
    })
    return (
        recommendations.sort_values(
            ["score", "movie_id"], ascending=[False, True]
        )
        .head(top_n)
        .copy()
    )
