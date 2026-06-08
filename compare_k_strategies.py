# compare_k_strategies.py

import pandas as pd
from pathlib import Path

from src.loader import (
    load_kaggle_data,
    load_pool_data,
    get_wc_teams,
)

from src.model import DixonColes
from src.scoring import points_for_prediction

PROJECT_ROOT = Path(__file__).resolve().parent

KAGGLE_PATH = PROJECT_ROOT / "data" / "kaggle" / "results.csv"
POOL_PATH   = PROJECT_ROOT / "data" / "pool"

TRAINING_YEARS = 12
DECAY_LAMBDA   = 0.20
REGULARIZATION = 0.0010

WC_START_DATES = {
    2002: "2002-05-31",
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

YEARS = [
    2002,
    2006,
    2010,
    2014,
    2018,
    2022,
]


def get_training_window(year):

    end = (
        pd.Timestamp(
            WC_START_DATES[year]
        )
        - pd.Timedelta(days=1)
    )

    start = (
        end
        - pd.DateOffset(
            years=TRAINING_YEARS
        )
    )

    return (
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


rows = []

for year in YEARS:

    print(f"\n{year}")

    pool = load_pool_data(
        POOL_PATH,
        year,
    )

    games = pool["games"]
    scores = pool["scores"]

    wc_teams = get_wc_teams(year)

    start, end = get_training_window(year)

    train_df = load_kaggle_data(
        KAGGLE_PATH,
        wc_teams,
        start,
        end,
        DECAY_LAMBDA,
    )

    # -------------------------
    # Constant k model
    # -------------------------

    model_constant = DixonColes(
        train_df,
        decay_lambda=DECAY_LAMBDA,
        regularization=REGULARIZATION,
        goal_inflation=1.15,
    )

    model_constant.fit()

    # -------------------------
    # Piecewise model
    # -------------------------

    model_piecewise = DixonColes(
        train_df,
        decay_lambda=DECAY_LAMBDA,
        regularization=REGULARIZATION,
        goal_inflation=1.0,
    )

    model_piecewise.fit()

    for game in games.itertuples():

        actual = scores[
            scores["game_id"]
            == game.game_id
        ]

        actual_home = int(
            actual.iloc[0]["score1"]
        )

        actual_away = int(
            actual.iloc[0]["score2"]
        )

        # -------------------------
        # Constant k prediction
        # -------------------------

        pred_const = model_constant.predict(
            game.team1,
            game.team2,
            neutral=True,
        )

        pts_const = points_for_prediction(
            pred_const["pred_home"],
            pred_const["pred_away"],
            actual_home,
            actual_away,
        )

        # -------------------------
        # Piecewise prediction
        # -------------------------

        lh, la = model_piecewise._get_lambda(
            game.team1,
            game.team2,
            True,
        )

        lambda_total = lh + la

        if lambda_total < 3.0:
            model_piecewise.goal_inflation = 1.15
        else:
            model_piecewise.goal_inflation = 0.90

        pred_piece = model_piecewise.predict(
            game.team1,
            game.team2,
            neutral=True,
        )

        pts_piece = points_for_prediction(
            pred_piece["pred_home"],
            pred_piece["pred_away"],
            actual_home,
            actual_away,
        )

        if (
            pred_const["prediction"]
            !=
            pred_piece["prediction"]
        ):

            rows.append({

                "year": year,

                "team1": game.team1,
                "team2": game.team2,

                "lambda_total": lambda_total,

                "prediction_constant":
                    pred_const["prediction"],

                "prediction_piecewise":
                    pred_piece["prediction"],

                "actual":
                    f"{actual_home}-{actual_away}",

                "points_constant":
                    pts_const,

                "points_piecewise":
                    pts_piece,

                "delta":
                    pts_piece
                    - pts_const,
            })

df = pd.DataFrame(rows)

df.to_csv(
    "changed_k_matches.csv",
    index=False,
)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    df["delta"]
    .value_counts()
    .sort_index()
)

print()

print(
    df.groupby("year")["delta"]
    .sum()
)

print()
print(
    "Saved changed_k_matches.csv"
)