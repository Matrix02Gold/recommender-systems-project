# Recommender Systems — MovieLens 1M

Individual course project for **Recommender Systems** at Tampere University.

This repository contains the **Part 1** implementation: dataset exploration, chronological splitting, two popularity baselines, item-based k-nearest-neighbour (kNN) collaborative filtering, and Top-10 evaluation. The repository will be extended in Parts 2 and 3.

## Dataset

The project uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) from GroupLens:

- 6,040 users and 3,883 catalogue movies
- 1,000,209 ratings on a 1–5 scale
- 4.26% user–item matrix density (95.74% sparsity)

Download and extract MovieLens 1M so that the project contains:

```text
data/
└── ml-1m/
    ├── ratings.dat
    ├── movies.dat
    └── users.dat
```

The dataset is not committed to the repository.

## Setup

Run these commands from the repository root. On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you already have a working `.venv`, activate it and install the requirements; you do not need to recreate it.

## Run the experiments

```powershell
python run_all.py --data-dir data/ml-1m
```

The script:

1. Loads the MovieLens 1M ratings, movie catalogue and user metadata.
2. Prints dataset statistics and generates five exploratory plots.
3. Creates a global chronological **80% train / 10% validation / 10% test** split and reports cold-start statistics.
4. Evaluates two Most Popular variants (all ratings; ratings of 4 or 5 stars).
5. Tunes item-based kNN on validation using cosine or adjusted-cosine similarity, neighbourhood size and shrinkage.
6. Selects the kNN configuration with the highest validation **NDCG@10**, then evaluates all three models on the test set.

The evaluation considers users with at least **five training ratings** and at least **one relevant held-out rating** (4 or 5 stars). It excludes already-rated training movies and reports **Precision@10, Recall@10, NDCG@10 and catalogue coverage**.

The script prints numerical results to the terminal. It **does not currently write the results template spreadsheet or a results CSV automatically**.

## Project structure

```text
.
├── data/                    # Put ml-1m/ here (not tracked by Git)
├── results/
│   └── figures/             # Generated exploratory figures
├── src/
│   ├── load_data.py         # Load MovieLens data files
│   ├── data_analysis.py     # Dataset statistics and figures
│   ├── data_split.py        # Temporal split and cold-start analysis
│   ├── baselines.py         # Most Popular recommenders
│   ├── item_knn.py          # Similarity and item-kNN recommendations
│   └── evaluation.py        # Ranking evaluation metrics
├── run_all.py               # Run the full Part 1 workflow
├── requirements.txt
└── README.md
```

The five figures generated in `results/figures/` are:

- `ratings_per_user.png`
- `ratings_per_movie.png`
- `rating_distribution.png`
- `movie_long_tail.png`
- `rating_activity_over_time.png`

## Part 1 results

The selected kNN configuration is **adjusted cosine**, **10 neighbours**, and **shrinkage = 5.0**, selected on validation data. Final test results (986 eligible users):

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|
| Most Popular — All ratings | 0.1731 | 0.0482 | 0.1860 | 3.50% |
| Most Popular — Positive ratings | **0.1815** | **0.0533** | **0.1969** | 3.32% |
| Item-based kNN | 0.1657 | 0.0498 | 0.1807 | **18.00%** |

The positive-rating popularity baseline performs best on the ranking accuracy metrics, while item-based kNN achieves higher catalogue coverage.

**Evaluation limitation:** The popularity methods rank the full movie catalogue. The current kNN model scores only movies observed in training that have positive-neighbour evidence. Consequently, the candidate sets are not strictly identical; this needs addressing for a fully comparable full-catalogue evaluation.

## Reproducibility and AI use

The train/validation/test split is based on interaction timestamps; there is no random sampling in the current Part 1 workflow. Direct Python dependencies are version-pinned in `requirements.txt`.

ChatGPT was used for conceptual explanations, debugging support, code cleanup, and drafting documentation. The code and reported results were reviewed and executed as part of this individual project.

## Author

Youssef Zougari-Belkhayat — individual project.
