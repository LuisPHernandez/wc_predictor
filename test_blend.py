from src.loader import (
    load_pool_data,
    get_wc_teams,
    load_kaggle_data,
)

from src.model import (
    DixonColes,
    outcome_probs_from_matrix,
    blend_matrix_outcomes,
)

# Tuned settings
DECAY_LAMBDA = 0.2
REGULARIZATION = 0.0010

# Load 2022 pool
pool = load_pool_data("data/pool", 2022)
wc_teams = get_wc_teams(pool)

# Same training window used for 2022 backtests
kaggle_df = load_kaggle_data(
    "data/kaggle/results.csv",
    wc_teams,
    "2010-11-20",
    "2022-11-19",
    decay_lambda=DECAY_LAMBDA,
)

print(f"Training matches: {len(kaggle_df)}")

# Fit model
model = DixonColes(
    kaggle_df,
    decay_lambda=DECAY_LAMBDA,
    regularization=REGULARIZATION,
)

model.fit()

# Generate score matrix
matrix, _, _ = model.score_matrix(
    "Argentina",
    "France",
    neutral=True,
)

print("\nModel outcome probabilities:")
print(outcome_probs_from_matrix(matrix))

# Fake bookmaker probabilities for testing
bookmaker_probs = {
    "home": 0.40,
    "draw": 0.30,
    "away": 0.30,
}

# Blend
blended_matrix = blend_matrix_outcomes(
    matrix,
    bookmaker_probs,
    alpha=0.5,
)

print("\nBlended outcome probabilities:")
print(outcome_probs_from_matrix(blended_matrix))