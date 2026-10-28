
import argparse
from src.load_data import load_movielens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", 
        required=True
    )
    args = parser.parse_args()

    ratings, movies, users = load_movielens(args.data_dir)

    print("MovieLens 1M Dataset")
    print("--------------------")
    print("Users:", users["user_id"].nunique())
    print("Movies:", movies["movie_id"].nunique())
    print("Ratings:", len(ratings))
    print("\nSample ratings:")
    print(ratings.head())
     


if __name__ == "__main__":
    main()
