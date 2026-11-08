
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Directory for generated figures
FIGURE_DIR = Path("results/figures")


def save_figure(filename):
    """Save and close the current figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def movie_rating_counts(ratings, movies):
    """Count ratings per movie, including unrated movies."""
    return (
        ratings.groupby("movie_id").size()
        .reindex(movies["movie_id"], fill_value=0)
    )


def analyze_dataset(ratings, movies, users):
    """Print basic dataset statistics and sparsity."""

    n_users = users["user_id"].nunique()
    n_movies = movies["movie_id"].nunique()
    n_ratings = len(ratings)
    n_rated_movies = ratings["movie_id"].nunique()

    density = n_ratings / (n_users * n_movies)

    print("\nDataset Statistics")
    print("--------------------")
    print("Users:", n_users)
    print("Movies:", n_movies)
    print("Rated movies:", n_rated_movies)
    print("Ratings:", n_ratings)
    print(f"Density: {density:.2%}")
    print(f"Sparsity: {1 - density:.2%}")


def analyze_interactions(ratings, movies, users):
    """Summarize user activity and movie popularity."""

    ratings_per_user = ratings.groupby("user_id").size()
    ratings_per_movie = movie_rating_counts(ratings, movies)

    print("\nUser Activity Statistics")
    print("--------------------")
    print(ratings_per_user.describe())

    print("\nMovie Popularity Statistics")
    print("--------------------")
    print(ratings_per_movie.describe())


def plot_interactions(ratings, movies):
    """Plot ratings per user and per movie."""

    ratings_per_user = ratings.groupby("user_id").size()
    ratings_per_movie = movie_rating_counts(ratings, movies)

    # User activity
    plt.figure(figsize=(9, 5))
    plt.hist(ratings_per_user, bins=50, edgecolor="black")
    plt.title("Distribution of Ratings per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")
    save_figure("ratings_per_user.png")

    # Movie popularity
    plt.figure(figsize=(9, 5))
    plt.hist(ratings_per_movie, bins=50, edgecolor="black")
    plt.title("Distribution of Ratings per Movie")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Movies")
    save_figure("ratings_per_movie.png")


def plot_rating_patterns(ratings, movies):
    """Plot rating distribution and movie popularity long tail."""

    # Rating distribution
    rating_counts = ratings["rating"].value_counts().sort_index()

    plt.figure(figsize=(9, 5))
    plt.bar(rating_counts.index, rating_counts.values, edgecolor="black")
    plt.title("Distribution of Rating Values")
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Number of Ratings")
    plt.xticks([1, 2, 3, 4, 5])
    save_figure("rating_distribution.png")

    # Long-tail distribution
    movie_counts = movie_rating_counts(ratings, movies)
    movie_counts = movie_counts.sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(movie_counts) + 1), movie_counts.values)
    plt.title("Movie Popularity: Long-Tail Distribution")
    plt.xlabel("Movie Rank (Most to Least Popular)")
    plt.ylabel("Number of Ratings")
    save_figure("movie_long_tail.png")


def analyze_timestamps(ratings):
    """Summarize rating activity and plot its monthly distribution."""

    rating_dates = pd.to_datetime(
        ratings["timestamp"], unit="s", utc=True
    )

    ratings_per_year = rating_dates.dt.year.value_counts().sort_index()

    print("\nTemporal Statistics")
    print("--------------------")
    print("First rating:", rating_dates.min())
    print("Last rating:", rating_dates.max())
    print("\nRatings per Year:")
    print(ratings_per_year)

    # Monthly rating activity
    monthly_ratings = (
        pd.DataFrame({"date": rating_dates})
        .set_index("date")
        .resample("MS")
        .size()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_ratings.index, monthly_ratings.values)
    plt.title("MovieLens Rating Activity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Ratings per Month")
    save_figure("rating_activity_over_time.png")
