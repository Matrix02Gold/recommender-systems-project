
import math


def evaluate_recommender(
    train, validation, movies, recommend_fn, k=10, min_history=5
):
    """Evaluate Top-k recommendations on users with enough training history."""

    seen_by_user = (
        train.groupby("user_id")["movie_id"].agg(set).to_dict()
    )
    history_counts = train.groupby("user_id").size()

    relevant_by_user = (
        validation[validation["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    eligible_users = sorted(
        user_id for user_id in relevant_by_user
        if history_counts.get(user_id, 0) >= min_history
    )

    if not eligible_users:
        raise ValueError("No eligible evaluation users.")

    catalogue = set(movies["movie_id"])
    scores = {"precision@10": [], "recall@10": [], "ndcg@10": []}
    all_recommended = set()

    for user_id in eligible_users:
        recommended = recommend_fn(user_id, k)["movie_id"].tolist()
        recommended_set = set(recommended)

        seen = seen_by_user[user_id]
        relevant = relevant_by_user[user_id]

        # Verify the recommendations
        assert len(recommended) == k
        assert len(recommended_set) == k
        assert not (recommended_set & seen), (
            f"Seen movie recommended to user {user_id}"
        )
        assert recommended_set <= catalogue

        # Precision and recall
        hits = len(recommended_set & relevant)

        # NDCG
        dcg = sum(
            1 / math.log2(rank + 2)
            for rank, movie_id in enumerate(recommended)
            if movie_id in relevant
        )

        ideal_hits = min(k, len(relevant))
        idcg = sum(
            1 / math.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        scores["precision@10"].append(hits / k)
        scores["recall@10"].append(hits / len(relevant))
        scores["ndcg@10"].append(dcg / idcg)

        all_recommended.update(recommended_set)

    n_users = len(eligible_users)

    return {
        "users_evaluated": n_users,
        **{
            name: sum(values) / n_users
            for name, values in scores.items()
        },
        "coverage": len(all_recommended) / len(catalogue),
    }
