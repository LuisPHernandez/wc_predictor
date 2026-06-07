import pandas as pd
import numpy as np
from pathlib import Path

from src.loader import (
    load_kaggle_base_data,
    build_competition_weights,
    build_confederation_weights,
)

from src.model import (
    DixonColes,
    outcome_probs_from_matrix,
)

from src.odds_loader import (
    _resolve_team,
    _shin_probs,
    _matchup_key
)

from src.scoring import points_for_prediction

PROJECT_ROOT = Path(__file__).resolve().parent

ODDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "odds"
    / "continental_odds.csv"
)

KAGGLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "kaggle"
    / "results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "continental_analysis.csv"
)

# ---------------------------------------------
# Best tuned parameters
# ---------------------------------------------

DECAY_LAMBDA = 0.2
TRAINING_YEARS = 12
REGULARIZATION = 0.0010

CONTINENTAL = 1.0
QUALIFIER = 0.5
REGIONAL = 0.3
FRIENDLY = 0.3

CONMEBOL = 1.0
UEFA = 1.0
CAF = 1.10
CONCACAF = 1.05
AFC = 0.95
OFC = 0.90

ALPHAS = [
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
]


# ---------------------------------------------
# Load odds
# ---------------------------------------------

odds = pd.read_csv(ODDS_PATH)

odds["home_team"] = (
    odds["home_team"]
    .astype(str)
    .str.strip()
    .apply(_resolve_team)
)

odds["away_team"] = (
    odds["away_team"]
    .astype(str)
    .str.strip()
    .apply(_resolve_team)
)

odds["matchup_key"] = odds.apply(
    lambda r: _matchup_key(
        r["date"],
        r["home_team"],
        r["away_team"],
    ),
    axis=1,
)

# bookmaker probabilities

probs = odds.apply(
    lambda r: _shin_probs(
        r["h_odds_avg"],
        r["d_odds_avg"],
        r["a_odds_avg"],
    ),
    axis=1,
)

odds["book_home"] = [x[0] for x in probs]
odds["book_draw"] = [x[1] for x in probs]
odds["book_away"] = [x[2] for x in probs]

# ---------------------------------------------
# Join to Kaggle
# ---------------------------------------------

kaggle = pd.read_csv(KAGGLE_PATH)

kaggle["matchup_key"] = kaggle.apply(
    lambda r: _matchup_key(
        r["date"],
        r["home_team"],
        r["away_team"],
    ),
    axis=1,
)

matches = odds.merge(
    kaggle,
    on="matchup_key",
    how="inner",
)

print(
    "Reversed:",
    (
        (matches["home_team_x"] != matches["home_team_y"])
        |
        (matches["away_team_x"] != matches["away_team_y"])
    ).sum()
)

matches = matches.rename(
    columns={
        "date_x": "date",
        "home_team_x": "home_team",
        "away_team_x": "away_team",
    }
)

print(
    f"Matched "
    f"{len(matches)} "
    f"of {len(odds)} odds rows"
)

# ---------------------------------------------
# Tournament editions
# ---------------------------------------------

editions = (
    matches[
        [
            "tournament",
            "year",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["tournament", "year"]
    )
)

print(
    f"{len(editions)} tournament editions"
)

# ---------------------------------------------
# Weight builders
# ---------------------------------------------

competition_weights = (
    build_competition_weights(
        CONTINENTAL,
        QUALIFIER,
        REGIONAL,
        FRIENDLY,
    )
)

confederation_weights = (
    build_confederation_weights(
        CONMEBOL,
        CAF,
        CONCACAF,
        AFC,
        OFC,
    )
)

rows = []

# ---------------------------------------------
# Fit once per tournament edition
# ---------------------------------------------

for edition in editions.itertuples():

    tournament = edition.tournament
    year = edition.year

    edition_matches = matches[
        (matches["tournament"] == tournament)
        &
        (matches["year"] == year)
    ].copy()

    start_date = (
        pd.to_datetime(
            edition_matches["date"]
        )
        .min()
    )

    training_end = (
        start_date
        - pd.Timedelta(days=1)
    )

    training_start = (
        training_end
        - pd.DateOffset(
            years=TRAINING_YEARS
        )
    )

    print(
        f"\n{tournament} {year}"
    )

    print(
        training_start.date(),
        "->",
        training_end.date(),
    )

    wc_teams = sorted(
        set(
            edition_matches["home_team"]
        )
        |
        set(
            edition_matches["away_team"]
        )
    )

    base_df = load_kaggle_base_data(
        KAGGLE_PATH,
        wc_teams,
        training_start.strftime("%Y-%m-%d"),
        training_end.strftime("%Y-%m-%d"),
        DECAY_LAMBDA,
    )

    base_df["competition_weight"] = (
        base_df["tournament"]
        .map(
            competition_weights
        )
    )

    home_conf = (
        base_df["home_confederation"]
        .map(
            confederation_weights
        )
        .fillna(1.0)
    )

    away_conf = (
        base_df["away_confederation"]
        .map(
            confederation_weights
        )
        .fillna(1.0)
    )

    base_df["confederation_weight"] = (
        np.sqrt(
            home_conf
            * away_conf
        )
    )

    base_df["weight"] = (
        base_df["recency_weight"]
        * base_df["competition_weight"]
        * base_df["confederation_weight"]
    )

    train_df = base_df[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "neutral",
            "weight",
        ]
    ]

    model = DixonColes(
        train_df,
        decay_lambda=DECAY_LAMBDA,
        regularization=REGULARIZATION,
    )

    model.fit()

    # -----------------------------------------
    # Process matches
    # -----------------------------------------

    for match in edition_matches.itertuples():

        matrix, _, _ = model.score_matrix(
            match.home_team,
            match.away_team,
            neutral=True,
        )

        model_probs = (
            outcome_probs_from_matrix(
                matrix
            )
        )

        book_probs = {
            "home": match.book_home,
            "draw": match.book_draw,
            "away": match.book_away,
        }

        tvd = (
            abs(
                model_probs["home"]
                - book_probs["home"]
            )
            +
            abs(
                model_probs["draw"]
                - book_probs["draw"]
            )
            +
            abs(
                model_probs["away"]
                - book_probs["away"]
            )
        ) / 2

        model_favorite = max(
            model_probs,
            key=model_probs.get,
        )

        book_favorite = max(
            book_probs,
            key=book_probs.get,
        )

        favorite_flip = (
            model_favorite
            !=
            book_favorite
        )

        actual_home = int(
            match.home_score
        )

        actual_away = int(
            match.away_score
        )

        row = {
            "tournament": tournament,
            "year": year,

            "date": match.date,

            "home_team": match.home_team,
            "away_team": match.away_team,

            "actual_home": actual_home,
            "actual_away": actual_away,

            "model_home": model_probs["home"],
            "model_draw": model_probs["draw"],
            "model_away": model_probs["away"],

            "book_home": book_probs["home"],
            "book_draw": book_probs["draw"],
            "book_away": book_probs["away"],

            "tvd": tvd,

            "model_favorite": model_favorite,
            "book_favorite": book_favorite,

            "favorite_flip": favorite_flip,
        }

        # -----------------------------
        # Pure model
        # -----------------------------

        model_pred = model.predict(
            match.home_team,
            match.away_team,
            neutral=True,
            alpha=1.0,
        )

        row.update({

            "model_prediction":
                model_pred["prediction"],

            "model_points":
                points_for_prediction(
                    model_pred["pred_home"],
                    model_pred["pred_away"],
                    actual_home,
                    actual_away,
                ),

            "model_decision_margin":
                model_pred[
                    "decision_margin"
                ],
        })

        # -----------------------------
        # Blend alphas
        # -----------------------------

        for alpha in ALPHAS:

            pred = model.predict(
                match.home_team,
                match.away_team,
                neutral=True,
                bookmaker_probs=book_probs,
                alpha=alpha,
            )

            tag = str(alpha).replace(
                ".",
                ""
            )

            row[
                f"blend{tag}_prediction"
            ] = pred["prediction"]

            row[
                f"blend{tag}_points"
            ] = points_for_prediction(
                pred["pred_home"],
                pred["pred_away"],
                actual_home,
                actual_away,
            )

            row[
                f"blend{tag}_decision_margin"
            ] = pred[
                "decision_margin"
            ]

        rows.append(row)
    
    pd.DataFrame(rows).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved checkpoint "
        f"({len(rows)} rows)"
    )

    print(
        f"Completed "
        f"{tournament} {year} "
        f"({len(edition_matches)} matches)"
    )

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    "\nSaved:",
    OUTPUT_PATH
)

print(
    "Rows:",
    len(df)
)