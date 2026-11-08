import argparse

from src.baselines import most_popular, recommend_most_popular
from src.data_analysis import (
    analyze_dataset,
    analyze_interactions,
    analyze_timestamps,
    plot_interactions,
    plot_rating_patterns,
)
from src.data_split import analyze_cold_start, temporal_split
from src.evaluation import evaluate_recommender
from src.item_knn import build_item_similarity, recommend_item_knn
from src.load_data import load_movielens


def show_results(name, results):
    """Print ranking metrics in a consistent format."""
    print(f"\n{name}")
    print(f"Users: {results['users_evaluated']}")
    for metric in ("precision@10", "recall@10", "ndcg@10", "coverage"):
        print(f"{metric}: {results[metric]:.4f}")


def make_knn_recommender(train, similarities, movie_ids, neighbors, shrinkage):
    """Configure item-kNN for the common evaluation function."""

    def recommend(user_id, top_n):
        return recommend_item_knn(
            user_id=user_id,
            train=train,
            similarity_matrix=similarities,
            movie_ids=movie_ids,
            k_neighbors=neighbors,
            top_n=top_n,
            shrinkage=shrinkage,
        )

    return recommend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    # Load and explore MovieLens 1M
    ratings, movies, users = load_movielens(args.data_dir)
    analyze_dataset(ratings, movies, users)
    analyze_interactions(ratings, movies, users)
    plot_interactions(ratings, movies)
    plot_rating_patterns(ratings, movies)
    analyze_timestamps(ratings)

    # Split chronologically and examine cold-start cases
    train, validation, test = temporal_split(ratings)
    analyze_cold_start(train, validation, test)

    print(
        f"\nSplit sizes: train={len(train)}, "
        f"val={len(validation)}, test={len(test)}"
    )

    # Common evaluation for all models
    def evaluate(recommender, held_out):
        return evaluate_recommender(
            train=train,
            validation=held_out,
            movies=movies,
            recommend_fn=recommender,
            k=10,
            min_history=5,
        )

    # Two popularity baselines
    popularity_models = {
        "Most Popular (All)": most_popular(
            train, movies, positive_only=False
        ),
        "Most Popular (Positive)": most_popular(
            train, movies, positive_only=True
        ),
    }

    def popular_recommender(ranking):
        return lambda user_id, top_n: recommend_most_popular(
            user_id, ranking, train, k=top_n
        )

    print("\nPOPULARITY — VALIDATION")

    for name, ranking in popularity_models.items():
        show_results(
            name,
            evaluate(popular_recommender(ranking), validation)
        )

    # Tune item-based kNN on validation
    matrices = {}
    tuning_results = {}

    print("\nITEM-KNN — VALIDATION TUNING")

    for method in ("cosine", "adjusted_cosine"):
        similarities, movie_ids = build_item_similarity(
            train, method=method
        )
        matrices[method] = (similarities, movie_ids)

        neighbors_to_try = (
            [5, 10, 20, 50, 100]
            if method == "cosine"
            else [5, 10, 20]
        )

        settings = [
            (neighbors, 1.0)
            for neighbors in neighbors_to_try
        ]

        if method == "adjusted_cosine":
            settings += [
                (10, value)
                for value in (0.0, 0.1, 0.5, 2.0, 5.0)
            ]

        for neighbors, shrinkage in settings:
            recommender = make_knn_recommender(
                train, similarities, movie_ids,
                neighbors, shrinkage
            )

            results = evaluate(recommender, validation)
            tuning_results[(method, neighbors, shrinkage)] = results

            show_results(
                f"{method}, k={neighbors}, shrinkage={shrinkage}",
                results
            )

    # Select the best kNN configuration using validation NDCG
    best = max(
        tuning_results,
        key=lambda setting: tuning_results[setting]["ndcg@10"]
    )

    method, neighbors, shrinkage = best
    similarities, movie_ids = matrices[method]

    final_knn = make_knn_recommender(
        train, similarities, movie_ids,
        neighbors, shrinkage
    )

    print(
        f"\nSelected item-kNN: {method}, "
        f"k={neighbors}, shrinkage={shrinkage}"
    )

    # Final test evaluation
    print("\nFINAL TEST RESULTS")

    for name, ranking in popularity_models.items():
        show_results(
            name,
            evaluate(popular_recommender(ranking), test)
        )

    show_results("Item-kNN", evaluate(final_knn, test))


if __name__ == "__main__":
    main()
