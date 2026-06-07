import pandas as pd

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
            alpha=0.2,
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

        rows.append({
            "year": year,
            "game_id": game.game_id,
            "team1": game.team1,
            "team2": game.team2,
            "disagreement": disagreement,
            "delta": blend_pts - model_pts,
            "changed": (
                model_pred["prediction"]
                != blend_pred["prediction"]
            ),
        })

df = pd.DataFrame(rows)

print("\n")
print("=" * 70)
print("TOP 25 DISAGREEMENTS")
print("=" * 70)

print(
    df.sort_values(
        "disagreement",
        ascending=False,
    )
    .head(25)
)

print("\n")
print("=" * 70)
print("CHANGED PREDICTIONS")
print("=" * 70)

changed = df[
    df["changed"]
]

print(
    changed[
        [
            "year",
            "team1",
            "team2",
            "disagreement",
            "delta",
        ]
    ]
    .sort_values(
        "disagreement",
        ascending=False,
    )
)

print("\n")
print("=" * 70)
print("AVERAGES")
print("=" * 70)

print(
    "Changed disagreement:",
    changed["disagreement"].mean()
)

print(
    "Unchanged disagreement:",
    df[
        ~df["changed"]
    ]["disagreement"].mean()
)

print(
    "Average delta when changed:",
    changed["delta"].mean()
)