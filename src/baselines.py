
def most_popular(train, movies, positive_only=False):

    # Decide which ratings contribute to popularity
    if positive_only:
        selected_ratings = train[train["rating"] >= 4]
    else:
        selected_ratings = train

    # Count the number of selected ratings per movie
    popularity = selected_ratings.groupby("movie_id").size()

    # Start with the complete movie catalogue
    ranked_movies = movies[["movie_id"]].copy()

    # Assign each movie its popularity score
    ranked_movies["score"] = (
        ranked_movies["movie_id"]
        .map(popularity)
        .fillna(0)
        .astype(int)
    )

    # Rank by popularity, resolving ties by movie ID
    ranked_movies = ranked_movies.sort_values(
        by=["score", "movie_id"],
        ascending=[False, True]
    )

    return ranked_movies



def recommend_most_popular(user_id, ranked_movies, train, k=10):

    # Find movies the user already rated in training
    seen_movies = set(
        train.loc[
            train["user_id"] == user_id,
            "movie_id"
        ]
    )

    # Exclude previously rated movies
    candidates = ranked_movies[
        ~ranked_movies["movie_id"].isin(seen_movies)
    ]

    # Return the highest-ranked k unseen movies
    recommendations = candidates.head(k).copy()

    return recommendations
