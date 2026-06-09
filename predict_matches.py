import pandas as pd
from pathlib import Path

from src.loader import (
    load_kaggle_data,
)
from src.model import DixonColes
from src.odds_loader import _shin_probs
from src.mappings import code_to_name

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

KAGGLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "kaggle"
    / "results.csv"
)

TEAMS_PATH = (
    PROJECT_ROOT
    / "data"
    / "pool"
    / "2026_teams.csv"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "input_matches.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "predictions.csv"
)

WC_START_DATE = "2026-06-11"

TRAINING_YEARS = 12
DECAY_LAMBDA = 0.20
REGULARIZATION = 0.0010

# ============================================================
# LOAD WC TEAMS
# ============================================================

teams = pd.read_csv(
    TEAMS_PATH,
    header=None,
    names=[
        "code",
        "group",
    ],
)

teams["name"] = (
    teams["code"]
    .apply(code_to_name)
)

wc_teams = (
    teams["name"]
    .tolist()
)

print(
    f"Loaded {len(wc_teams)} World Cup teams"
)

# ============================================================
# TRAINING WINDOW
# ============================================================

end_date = (
    pd.Timestamp(WC_START_DATE)
    - pd.Timedelta(days=1)
)

start_date = (
    end_date
    - pd.DateOffset(years=TRAINING_YEARS)
)

start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")

print(
    f"Training window: "
    f"{start_date} → {end_date}"
)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

kaggle_df = load_kaggle_data(
    KAGGLE_PATH,
    wc_teams,
    start_date,
    end_date,
    DECAY_LAMBDA,
)

print(
    f"Training matches: {len(kaggle_df)}"
)

# ============================================================
# FIT MODEL
# ============================================================

print()
print("=" * 60)
print("FITTING MODEL")
print("=" * 60)

model = DixonColes(
    kaggle_df,
    decay_lambda=DECAY_LAMBDA,
    regularization=REGULARIZATION,
)

model.fit()

# ============================================================
# LOAD INPUT MATCHES
# ============================================================

matches = pd.read_csv(
    INPUT_PATH
)

required_columns = [
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "ou_line",
]

missing = [
    c
    for c in required_columns
    if c not in matches.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# ============================================================
# PREDICT
# ============================================================

results = []

print()
print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

for row in matches.itertuples():

    p_home, p_draw, p_away = _shin_probs(
        row.home_odds,
        row.draw_odds,
        row.away_odds,
    )

    bookmaker_probs = {
        "home": p_home,
        "draw": p_draw,
        "away": p_away,
    }

    pred = model.predict(
        row.home_team,
        row.away_team,
        neutral=True,
        market_total_goals=row.ou_line,
        bookmaker_probs=bookmaker_probs,
    )

    results.append({

        # --------------------------------------
        # Match
        # --------------------------------------

        "home_team":
            row.home_team,

        "away_team":
            row.away_team,

        # --------------------------------------
        # Market Inputs
        # --------------------------------------

        "home_odds":
            row.home_odds,

        "draw_odds":
            row.draw_odds,

        "away_odds":
            row.away_odds,

        "ou_line":
            row.ou_line,

        # --------------------------------------
        # Optimal Prediction
        # --------------------------------------

        "pred_home":
            pred["pred_home"],

        "pred_away":
            pred["pred_away"],

        "prediction":
            pred["prediction"],

        # --------------------------------------
        # Expected Points
        # --------------------------------------

        "expected_pts":
            round(
                pred["expected_pts"],
                4,
            ),

        # --------------------------------------
        # Runner-Up Prediction
        # --------------------------------------

        "second_best_prediction":
            pred["second_prediction"],

        "second_best_expected_pts":
            round(
                pred["second_expected_pts"],
                4,
            ),

        "decision_margin":
            round(
                pred["decision_margin"],
                4,
            ),

        # --------------------------------------
        # Final Lambdas
        # --------------------------------------

        "lambda_home":
            round(
                pred["lambda_home"],
                4,
            ),

        "lambda_away":
            round(
                pred["lambda_away"],
                4,
            ),

        "lambda_total":
            round(
                pred["lambda_home"] + pred["lambda_away"],
                4,
            ),

        # --------------------------------------
        # Outcome Probabilities
        # --------------------------------------

        "home_win":
            round(
                pred["home_win"],
                4,
            ),

        "draw":
            round(
                pred["draw"],
                4,
            ),

        "away_win":
            round(
                pred["away_win"],
                4,
            ),
    })

# ============================================================
# SAVE OUTPUT
# ============================================================

output = pd.DataFrame(results)

output.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("=" * 60)
print("PREDICTIONS")
print("=" * 60)

print(
    output.to_string(
        index=False
    )
)

print()
print(
    f"Saved: {OUTPUT_PATH}"
)