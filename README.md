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
Use **Python 3.13** (the project was checked with Python 3.13.5). The direct dependencies and their pinned versions are listed in `requirements.txt`.

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

**Quick run** (uses the previously selected item-kNN settings instead of repeating its validation search):

```powershell
python run_all.py --data-dir data/ml-1m --quick
```

Both options still run the Part 2 tuning, final evaluations, confidence intervals, user-group analysis, early/late analysis, and sparsity experiments. **Quick mode skips only the Part 1 item-kNN tuning search**, so it may still take some time.

## Project structure

```text
.
├── data/                     # Put the ml-1m/ dataset here
├── results/
│   └── figures/              # Figures created by the script
├── src/
│   ├── load_data.py          # Load the MovieLens data
│   ├── data_analysis.py      # Data statistics and plots
│   ├── data_split.py         # Chronological split and cold-start analysis
│   ├── baselines.py          # Most Popular models
│   ├── item_knn.py           # Item-based collaborative filtering
│   ├── content_based.py      # Content-based recommendations
│   ├── user_knn.py           # User-based collaborative filtering
│   ├── hybrid.py             # Hybrid recommendations
│   └── evaluation.py         # Evaluation metrics and bootstrap intervals
├── run_all.py                # Run the Part 1 and Part 2 experiments
├── requirements.txt
└── README.md
```

The five figures generated in `results/figures/` are:

- `ratings_per_user.png`
- `ratings_per_movie.png`
- `rating_distribution.png`
- `movie_long_tail.png`
- `rating_activity_over_time.png`

### Part 1

- Explore the MovieLens dataset and examine cold-start cases.
- Make a **global chronological split**: 80% training (800,168 ratings), 10% validation (100,020), and 10% testing (100,021).
- Compare **Most Popular (All)**, **Most Popular (Positive)** (ratings of 4 or 5), and **item-based kNN**.
- Tune item-kNN on validation using cosine or adjusted-cosine similarity, neighbour counts, and shrinkage. The chosen settings are **adjusted cosine, 10 neighbours, shrinkage 5.0**.

### Part 2

Three more recommenders were added:

- **Content-based (self-implemented):** Builds user profiles using movie genres and release decades. The selected decade weight is **0.5** (validation search: 0.0, 0.5, 1.0).
- **User-kNN:** Recommends movies based on ratings from similar users. The selected number of neighbours is **10** (validation search: 10, 20, 50).
- **Hybrid:** Combines the content-based and positive-popularity rankings. The selected content weight is **0.25**, with **0.75** weight on popularity (validation search: 0.25, 0.5, 0.75).

Part 2 also includes:

- **95% bootstrap confidence intervals** for NDCG@10, plus a paired comparison of the positive-popularity baseline against item-kNN.
- Results for **lower-activity and higher-activity users**, based on their number of training ratings.
- A comparison of the **early and late** halves of the test interactions.
- **Sparsity experiments** using 100%, 50%, and 20% of the training interactions, evaluating the same eligible test users across all three settings.

### Evaluation rules

- A relevant movie is one rated **4 or 5 stars** in the held-out data.
- Evaluation includes users with **at least five training ratings** and at least one relevant held-out movie.
- Movies already rated in training are excluded from recommendations.
- Metrics are **Precision@10**, **Recall@10**, **NDCG@10**, and **catalogue coverage**.
- Hyperparameters are selected using **validation data**, then final performance is measured on the **test set**.

## Final test results

These are the Part 1 and Part 2 results for **986 eligible test users**.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|
| Most Popular (All) | 0.1731 | 0.0482 | 0.1860 | 3.50% |
| **Most Popular (Positive)** | **0.1815** | **0.0533** | **0.1969** | 3.32% |
| Item-kNN | 0.1657 | 0.0498 | 0.1807 | 18.00% |
| Content-based | 0.0244 | 0.0077 | 0.0253 | **42.24%** |
| User-kNN | 0.0620 | 0.0164 | 0.0679 | 34.77% |
| Hybrid | 0.1262 | 0.0345 | 0.1386 | 9.45% |

The positive-popularity baseline has the highest ranking accuracy, while content-based filtering covers the largest part of the catalogue. The paired NDCG@10 difference between positive popularity and item-kNN is **0.0162**, with a **95% confidence interval of [0.0053, 0.0275]**.

In the sparsity test, the NDCG@10 results were:

| Model | 100% train | 50% train | 20% train |
|---|---:|---:|---:|
| Most Popular (All) | 0.1821 | 0.1023 | 0.0791 |
| Most Popular (Positive) | 0.1927 | 0.1108 | 0.0878 |
| Item-kNN | 0.1761 | 0.1008 | 0.0323 |
| Content-based | 0.0240 | 0.0216 | 0.0219 |
| User-kNN | 0.0612 | 0.0714 | 0.0648 |
| Hybrid | 0.1336 | 0.0922 | 0.0870 |

These sparsity values use the same **896 eligible test users** at each training-data level, so they are slightly different from the main full-data test results.

## Reproducibility

- The main train/validation/test split is chronological, with no random splitting.
- The **random seed is 42** for training-data subsampling and bootstrap confidence intervals.
- The bootstrap calculation uses **1,000 samples**.
- All paths in the run commands are relative to the project folder.
- `run_all.py` prints its numerical results to the terminal and saves data-exploration figures, but **does not automatically export `results.csv` or fill the course Excel template**. The required results file and completed spreadsheet must be added separately when preparing the submission.

**Current limitation:** The popularity, content-based, user-kNN, and hybrid recommenders can rank the full movie catalogue, but the current **item-kNN** implementation only returns training movies with positive similarity support. Its smaller candidate set means the models are not yet fully comparable under a strict full-catalogue evaluation. Cold-start users are also excluded from the ranking evaluation.

## AI use

ChatGPT was used for conceptual explanations, some code implementation and debugging support, code cleanup, and drafting documentation.

## Author

Youssef Zougari-Belkhayat — individual project.
