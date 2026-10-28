from pathlib import Path
import pandas as pd

def load_movielens(data_dir):
    data_dir = Path(data_dir)

    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genre"]
    )

    user = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "gender", "age", "occupation", "zip_code"]

    )

    return ratings, movies, user