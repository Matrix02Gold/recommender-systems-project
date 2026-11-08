
def temporal_split(ratings):
    """Split ratings chronologically into 80% train, 10% validation, 10% test."""

    # Find the chronological cutoffs
    timestamps = ratings["timestamp"]
    cutoffs = timestamps.quantile([0.8, 0.9])

    train_cutoff = cutoffs.loc[0.8]
    val_cutoff = cutoffs.loc[0.9]

    # Divide interactions by timestamp
    train = ratings.loc[
        timestamps <= train_cutoff
    ].copy()

    validation = ratings.loc[
        (timestamps > train_cutoff) & (timestamps <= val_cutoff)
    ].copy()

    test = ratings.loc[
        timestamps > val_cutoff
    ].copy()

    # Verify the split
    assert len(train) + len(validation) + len(test) == len(ratings)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()

    return train, validation, test


def analyze_cold_start(train, validation, test):
    """Report unseen users, movies, and warm interactions."""

    train_users = set(train["user_id"])
    train_movies = set(train["movie_id"])

    print("\nCold-Start Analysis")
    print("--------------------")

    for name, dataset in [
        ("Validation", validation),
        ("Testing", test)
    ]:
        # Interactions involving unseen users or movies
        cold_users = ~dataset["user_id"].isin(train_users)
        cold_movies = ~dataset["movie_id"].isin(train_movies)

        # Interactions where both user and movie are known
        warm_count = (~(cold_users | cold_movies)).sum()

        print(f"\n{name}:")
        print("Cold-start users:", dataset.loc[cold_users, "user_id"].nunique())
        print("Cold-start movies:", dataset.loc[cold_movies, "movie_id"].nunique())
        print("Interactions with unseen users:", cold_users.sum())
        print("Interactions with unseen movies:", cold_movies.sum())
        print("Warm interactions:", warm_count)
        print(f"Warm interaction percentage: {warm_count / len(dataset):.2%}")
