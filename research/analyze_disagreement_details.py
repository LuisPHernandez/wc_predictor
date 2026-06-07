import pandas as pd
import numpy as np

from src.loader import (
    load_pool_data,
    get_wc_teams,
    load_kaggle_data,
)

from src.model import (
    DixonColes,
    outcome_probs_from_matrix,
)

from src.odds_loader import load_wc_odds_lookup

from src.scoring import points_for_prediction

YEARS = [2006, 2010, 2014, 2018, 2022]

DECAY_LAMBDA = 0.2
REGULARIZATION = 0.0010

WC_START_DATES = {
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}


def training_window(year):

    end = (
        pd.Timestamp(WC_START_DATES[year])
        - pd.Timedelta(days=1)
    )

    start = end - pd.DateOffset(years=12)

    return (
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


def favorite(prob_dict):
    return max(prob_dict, key=prob_dict.get)


rows = []

for year in YEARS:

    print(f"\n{year}")

    pool = load_pool_data(
        "data/pool",
        year,
    )

    wc_teams = get_wc_teams(pool)

    start_date, end_date = training_window(year)

    kaggle_df = load_kaggle_data(
        "data/kaggle/results.csv",
        wc_teams,
        start_date,
        end_date,
        decay_lambda=DECAY_LAMBDA,
    )

    model = DixonColes(
        kaggle_df,
        decay_lambda=DECAY_LAMBDA,
        regularization=REGULARIZATION,
    )

    model.fit()

    odds_lookup = load_wc_odds_lookup(year)

    scores = pool["scores"]

    actuals = (
        scores
        .set_index("game_id")[["score1", "score2"]]
        .to_dict("index")
    )

    for game in pool["games"].itertuples():

        bookmaker_probs = odds_lookup.get(
            game.game_id
        )

        if bookmaker_probs is None:
            continue

        matrix, _, _ = model.score_matrix(
            game.team1,
            game.team2,
            neutral=True,
        )

        model_probs = outcome_probs_from_matrix(
            matrix
        )

        disagreement = (
            abs(
                model_probs["home"]
                - bookmaker_probs["home"]
            )
            +
            abs(
                model_probs["draw"]
                - bookmaker_probs["draw"]
            )
            +
            abs(
                model_probs["away"]
                - bookmaker_probs["away"]
            )
        ) / 2

        model_pred = model.predict(
            game.team1,
            game.team2,
            neutral=True,
            alpha=1.0,
        )

        blend_pred = model.predict(
            game.team1,
            game.team2,
            neutral=True,
            bookmaker_probs=bookmaker_probs,
            alpha=0.20,
        )

        actual = actuals[
            game.game_id
        ]

        model_pts = points_for_prediction(
            model_pred["pred_home"],
            model_pred["pred_away"],
            actual["score1"],
            actual["score2"],
        )

        blend_pts = points_for_prediction(
            blend_pred["pred_home"],
            blend_pred["pred_away"],
            actual["score1"],
            actual["score2"],
        )

        model_fav = favorite(
            model_probs
        )

        book_fav = favorite(
            bookmaker_probs
        )

        rows.append({

            "year": year,

            "game_id": game.game_id,

            "team1": game.team1,
            "team2": game.team2,

            "disagreement": disagreement,

            "model_home": model_probs["home"],
            "model_draw": model_probs["draw"],
            "model_away": model_probs["away"],

            "book_home": bookmaker_probs["home"],
            "book_draw": bookmaker_probs["draw"],
            "book_away": bookmaker_probs["away"],

            "model_favorite": model_fav,
            "book_favorite": book_fav,

            "favorite_flip":
                model_fav != book_fav,

            "model_prediction":
                model_pred["prediction"],

            "blend_prediction":
                blend_pred["prediction"],

            "prediction_changed":
                model_pred["prediction"]
                !=
                blend_pred["prediction"],

            "actual":
                f"{actual['score1']}-{actual['score2']}",

            "model_points":
                model_pts,

            "blend_points":
                blend_pts,

            "delta":
                blend_pts - model_pts,
        })

df = pd.DataFrame(rows)

df.to_csv(
    "disagreement_analysis.csv",
    index=False,
)

print("\n")
print("=" * 80)
print("TOP 50 DISAGREEMENTS")
print("=" * 80)

print(
    df.sort_values(
        "disagreement",
        ascending=False,
    )
    .head(50)
    [
        [
            "year",
            "team1",
            "team2",
            "disagreement",
            "favorite_flip",
            "model_prediction",
            "blend_prediction",
            "actual",
            "delta",
        ]
    ]
)

print("\n")
print("=" * 80)
print("FAVORITE FLIPS")
print("=" * 80)

flips = df[
    df["favorite_flip"]
]

print(
    f"Favorite flips: "
    f"{len(flips)}"
)

print(
    f"Average delta: "
    f"{flips['delta'].mean():.3f}"
)

print(
    f"Changed prediction rate: "
    f"{flips['prediction_changed'].mean():.3f}"
)

print("\n")
print("=" * 80)
print("NO FAVORITE FLIP")
print("=" * 80)

no_flips = df[
    ~df["favorite_flip"]
]

print(
    f"Average delta: "
    f"{no_flips['delta'].mean():.3f}"
)

print(
    f"Changed prediction rate: "
    f"{no_flips['prediction_changed'].mean():.3f}"
)

print("\n")
print("=" * 80)
print("DISAGREEMENT BUCKETS")
print("=" * 80)

df["bucket"] = pd.cut(
    df["disagreement"],
    bins=[
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        1.00,
    ]
)

print(
    df.groupby("bucket")
      .agg(
          matches=("delta", "count"),
          avg_delta=("delta", "mean"),
          changed_rate=("prediction_changed", "mean"),
      )
)