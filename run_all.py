import argparse
import numpy as np
from src.baselines import most_popular, recommend_most_popular
from src.data_analysis import (
    analyze_dataset,
    analyze_interactions,
    analyze_timestamps,
    plot_interactions,
    plot_rating_patterns,
)
from src.data_split import analyze_cold_start, temporal_split
from src.evaluation import evaluate_recommender, bootstrap_ci
from src.item_knn import build_item_similarity, recommend_item_knn
from src.load_data import load_movielens
from src.content_based import (
    build_content_model,
    recommend_content_based
)
from src.user_knn import build_user_knn, recommend_user_knn
from src.hybrid import recommend_hybrid



def show_results(name, results):
    print(f"\n{name}")
    print(f"Users: {results['users_evaluated']}")
    for metric in ("precision@10", "recall@10", "ndcg@10", "coverage"):
        print(f"{metric}: {results[metric]:.4f}")


def make_knn_recommender(train, similarities, movie_ids, neighbors, shrinkage):
    def recommend(user_id, top_n):
        return recommend_item_knn(
            user_id=user_id, train=train, similarity_matrix=similarities,
            movie_ids=movie_ids, k_neighbors=neighbors, top_n=top_n,
            shrinkage=shrinkage,
        )
    return recommend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # 1. Load, explore and split the data
    ratings, movies, users = load_movielens(args.data_dir)
    analyze_dataset(ratings, movies, users)
    analyze_interactions(ratings, movies, users)
    plot_interactions(ratings, movies)
    plot_rating_patterns(ratings, movies)
    analyze_timestamps(ratings)

    train, validation, test = temporal_split(ratings)
    analyze_cold_start(train, validation, test)
    print(f"\nSplit sizes: train={len(train)}, val={len(validation)}, test={len(test)}")

    def evaluate(recommender, held_out):
        return evaluate_recommender(
            train=train, validation=held_out, movies=movies,
            recommend_fn=recommender, k=10, min_history=5,
        )

    # 2. Popularity baselines
    popularity_models = {
        "Most Popular (All)": most_popular(train, movies, positive_only=False),
        "Most Popular (Positive)": most_popular(train, movies, positive_only=True),
    }

    def popular_recommender(ranking):
        return lambda user_id, top_n: recommend_most_popular(
            user_id, ranking, train, k=top_n
        )

    print("\nPOPULARITY — VALIDATION")
    for name, ranking in popularity_models.items():
        show_results(name, evaluate(popular_recommender(ranking), validation))

    # 3. Item-kNN tuning (skipped with --quick)
    matrices, tuning_results = {}, {}
    print("\nITEM-KNN — VALIDATION TUNING")

    for method in ([] if args.quick else ("cosine", "adjusted_cosine")):
        similarities, movie_ids = build_item_similarity(train, method=method)
        matrices[method] = (similarities, movie_ids)
        neighbors_to_try = [5, 10, 20, 50, 100] if method == "cosine" else [5, 10, 20]
        settings = [(neighbors, 1.0) for neighbors in neighbors_to_try]
        if method == "adjusted_cosine":
            settings += [(10, value) for value in (0.0, 0.1, 0.5, 2.0, 5.0)]

        for neighbors, shrinkage in settings:
            recommender = make_knn_recommender(
                train, similarities, movie_ids, neighbors, shrinkage
            )
            results = evaluate(recommender, validation)
            tuning_results[(method, neighbors, shrinkage)] = results
            show_results(f"{method}, k={neighbors}, shrinkage={shrinkage}", results)

    if args.quick:
        best = ("adjusted_cosine", 10, 5.0)
        matrices["adjusted_cosine"] = build_item_similarity(
            train, method="adjusted_cosine"
        )
    else:
        best = max(tuning_results, key=lambda setting: tuning_results[setting]["ndcg@10"])

    method, neighbors, shrinkage = best
    similarities, movie_ids = matrices[method]
    final_knn = make_knn_recommender(
        train, similarities, movie_ids, neighbors, shrinkage
    )
    print(f"\nSelected item-kNN: {method}, k={neighbors}, shrinkage={shrinkage}")

    # 4. Part 1 final test results
    print("\nFINAL TEST RESULTS")
    test_results = {}
    for name, ranking in popularity_models.items():
        results = evaluate(popular_recommender(ranking), test)
        test_results[name] = results
        show_results(name, results)

    results = evaluate(final_knn, test)
    test_results["Item-kNN"] = results
    show_results("Item-kNN", results)

    # 5. Content-based tuning
    print("\nCONTENT-BASED — VALIDATION")
    content_results, content_models = {}, {}
    for year_weight in [0.0, 0.5, 1.0]:
        model = build_content_model(train, movies, year_weight=year_weight)

        def content_recommend(user_id, top_n):
            return recommend_content_based(user_id, model, top_n=top_n)

        results = evaluate(content_recommend, validation)
        content_results[year_weight] = results
        content_models[year_weight] = model
        show_results(f"Content-based, year_weight={year_weight}", results)

    best_year_weight = max(
        content_results, key=lambda weight: content_results[weight]["ndcg@10"]
    )
    best_content_model = content_models[best_year_weight]
    print(f"\nSelected content-based year weight: {best_year_weight}")

    # 6. User-kNN tuning
    print("\nUSER-KNN — VALIDATION")
    user_model = build_user_knn(train, movies)
    user_knn_results = {}
    for neighbors in [10, 20, 50]:
        def user_recommend(user_id, top_n):
            return recommend_user_knn(
                user_id, user_model, k_neighbors=neighbors, top_n=top_n
            )

        results = evaluate(user_recommend, validation)
        user_knn_results[neighbors] = results
        show_results(f"User-kNN, k={neighbors}", results)

    best_user_k = max(
        user_knn_results, key=lambda k: user_knn_results[k]["ndcg@10"]
    )
    print(f"\nSelected user-kNN neighbours: {best_user_k}")

    # 7. Hybrid tuning
    print("\nHYBRID — VALIDATION")
    hybrid_results = {}
    for alpha in [0.25, 0.5, 0.75]:
        def hybrid_recommend(user_id, top_n):
            return recommend_hybrid(
                user_id=user_id, content_model=best_content_model,
                popular_movies=popularity_models["Most Popular (Positive)"],
                alpha=alpha, top_n=top_n,
            )

        results = evaluate(hybrid_recommend, validation)
        hybrid_results[alpha] = results
        show_results(f"Hybrid, alpha={alpha}", results)

    best_hybrid_alpha = max(
        hybrid_results, key=lambda a: hybrid_results[a]["ndcg@10"]
    )
    print(f"\nSelected hybrid alpha: {best_hybrid_alpha}")

    # 8. Part 2 final test results
    print("\nPART 2 — FINAL TEST RESULTS")

    def content_final(user_id, top_n):
        return recommend_content_based(user_id, best_content_model, top_n=top_n)

    def user_knn_final(user_id, top_n):
        return recommend_user_knn(
            user_id, user_model, k_neighbors=best_user_k, top_n=top_n
        )

    def hybrid_final(user_id, top_n):
        return recommend_hybrid(
            user_id=user_id, content_model=best_content_model,
            popular_movies=popularity_models["Most Popular (Positive)"],
            alpha=best_hybrid_alpha, top_n=top_n,
        )

    for name, recommender in [
        ("Content-based", content_final),
        ("User-kNN", user_knn_final),
        ("Hybrid", hybrid_final),
    ]:
        results = evaluate(recommender, test)
        test_results[name] = results
        show_results(name, results)

    # 9. Bootstrap confidence intervals and paired comparison
    print("\n95% CONFIDENCE INTERVALS — NDCG@10")
    for name, results in test_results.items():
        low, high = bootstrap_ci(results["per_user_ndcg"])
        print(f"{name}: {results['ndcg@10']:.4f} [{low:.4f}, {high:.4f}]")

    popular = test_results["Most Popular (Positive)"]["per_user_ndcg"]
    item_knn = test_results["Item-kNN"]["per_user_ndcg"]
    differences = {user_id: popular[user_id] - item_knn[user_id] for user_id in popular}
    low, high = bootstrap_ci(differences)
    print("\nPaired NDCG difference: Popular Positive - Item-kNN")
    print(f"Mean difference: {np.mean(list(differences.values())):.4f}")
    print(f"95% CI: [{low:.4f}, {high:.4f}]")

    # 10. Low- versus high-activity users
    print("\nUSER ACTIVITY ANALYSIS")
    user_activity = train.groupby("user_id").size()
    evaluated_users = list(test_results["Item-kNN"]["per_user_ndcg"])
    threshold = user_activity.loc[evaluated_users].median()
    print(f"Activity threshold: {threshold}")

    for name, results in test_results.items():
        low_activity, high_activity = [], []
        for user_id, ndcg in results["per_user_ndcg"].items():
            if user_activity[user_id] <= threshold:
                low_activity.append(ndcg)
            else:
                high_activity.append(ndcg)

        print(f"\n{name}")
        print(f"Low activity ({len(low_activity)} users): {np.mean(low_activity):.4f}")
        print(f"High activity ({len(high_activity)} users): {np.mean(high_activity):.4f}")

    # 11. Early versus late test interactions
    print("\nEARLY VS LATE TEST ANALYSIS")
    cutoff = test["timestamp"].median()
    test_periods = {
        "Early": test[test["timestamp"] <= cutoff],
        "Late": test[test["timestamp"] > cutoff],
    }
    period_models = {
        "Most Popular (Positive)": popular_recommender(
            popularity_models["Most Popular (Positive)"]
        ),
        "Item-kNN": final_knn,
        "Hybrid": hybrid_final,
    }

    for period, held_out in test_periods.items():
        print(f"\n{period} period:")
        for name, recommender in period_models.items():
            results = evaluate(recommender, held_out)
            print(
                f"{name}: Users={results['users_evaluated']}, "
                f"NDCG@10={results['ndcg@10']:.4f}"
            )

    # 12. Sparsity ablation: 100%, 50%, 20% of training interactions
    print("\nSPARSITY ABLATION")
    shuffled = train.sample(frac=1, random_state=42)
    training_sets = {
        fraction: shuffled.iloc[:int(len(train) * fraction)].copy()
        for fraction in [1.0, 0.5, 0.2]
    }
    history_20 = training_sets[0.2].groupby("user_id").size()
    eligible_ids = set(history_20[history_20 >= 5].index)
    common_test = test[test["user_id"].isin(eligible_ids)]

    for fraction, small_train in training_sets.items():
        print(f"\nTraining data: {fraction:.0%}")
        pop_all = most_popular(small_train, movies)
        pop_pos = most_popular(small_train, movies, positive_only=True)
        sim, movie_ids = build_item_similarity(small_train, method="adjusted_cosine")
        content = build_content_model(
            small_train, movies, year_weight=best_year_weight
        )
        user_model_small = build_user_knn(small_train, movies)

        models = {
            "Popular All": lambda u, k: recommend_most_popular(
                u, pop_all, small_train, k=k
            ),
            "Popular Positive": lambda u, k: recommend_most_popular(
                u, pop_pos, small_train, k=k
            ),
            "Item-kNN": lambda u, k: recommend_item_knn(
                u, small_train, sim, movie_ids,
                k_neighbors=10, top_n=k, shrinkage=5.0
            ),
            "Content-based": lambda u, k: recommend_content_based(
                u, content, top_n=k
            ),
            "User-kNN": lambda u, k: recommend_user_knn(
                u, user_model_small, k_neighbors=best_user_k, top_n=k
            ),
            "Hybrid": lambda u, k: recommend_hybrid(
                user_id=u, content_model=content, popular_movies=pop_pos,
                alpha=best_hybrid_alpha, top_n=k
            ),
        }

        scores = {}
        for name, recommender in models.items():
            result = evaluate_recommender(
                train=small_train, validation=common_test, movies=movies,
                recommend_fn=recommender, k=10, min_history=5,
            )
            scores[name] = result["ndcg@10"]
            print(
                f"{name}: NDCG@10={result['ndcg@10']:.4f}, "
                f"Users={result['users_evaluated']}"
            )

        print("Model ranking:")
        for name in sorted(scores, key=scores.get, reverse=True):
            print(f"  {name}: {scores[name]:.4f}")


if __name__ == "__main__":
    main()
