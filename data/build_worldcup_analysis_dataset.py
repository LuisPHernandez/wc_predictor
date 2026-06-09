"""
build_wc_analysis_dataset.py

Builds a flat CSV containing every World Cup match (2006–2022) with:
  - match metadata
  - actual scores
  - model outcome probabilities (model_home / draw / away)
  - bookmaker probabilities (book_home / draw / away)
  - raw odds and Shin stats
  - disagreement signals (TVD, favorite_flip)
  - model lambda estimates
  - pure model best + second-best prediction with expected pts and decision margin
  - blend{tag}_pred_home/away, _prediction, _points, _expected_pts, _decision_margin
    for every alpha in ALPHAS

Re-uses:
  - src/loader.py    → load_kaggle_data(), load_pool_data(), get_wc_teams()
  - src/model.py     → DixonColes, outcome_probs_from_matrix
  - src/odds_loader.py → load_wc_odds_lookup()
  - src/scoring.py   → points_for_prediction()
  - backtest.py      → WC_START_DATES training-window logic (copied here)

Run from project root:
    py -3 build_wc_analysis_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.loader import (
    load_kaggle_data,
    load_pool_data,
    get_wc_teams,
)
from src.model import (
    DixonColes,
    outcome_probs_from_matrix,
)
from src.odds_loader import load_wc_odds_lookup
from src.scoring import points_for_prediction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

KAGGLE_PATH = PROJECT_ROOT / "data" / "kaggle" / "results.csv"
POOL_PATH   = PROJECT_ROOT / "data" / "pool"
OUTPUT_PATH = PROJECT_ROOT / "wc_analysis_rho.csv"

# ---------------------------------------------------------------------------
# Hyperparameters  (current production values from handoff doc)
# ---------------------------------------------------------------------------

DECAY_LAMBDA   = 0.2
TRAINING_YEARS = 12
REGULARIZATION = 0.0010

# Alpha grid — dense enough to avoid future refits
ALPHAS = [round(a, 2) for a in np.arange(0.0, 1.05, 0.05)]

# World Cups to process (must have odds + pool data)
WC_YEARS = [2006, 2010, 2014, 2018, 2022]

# First-game date per WC — used to define the training window cutoff
# (same dict as backtest.py)
WC_START_DATES = {
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_training_window(wc_year):
    """
    Returns (start_date, end_date) strings for load_kaggle_data().
    end_date   = day before the WC's first game  (no data leakage)
    start_date = TRAINING_YEARS years before that
    """
    end   = pd.Timestamp(WC_START_DATES[wc_year]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=TRAINING_YEARS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def alpha_tag(alpha):
    """
    Converts a float alpha to the column-name suffix used throughout.
    0.0 -> '00', 0.05 -> '005', 0.5 -> '05', 1.0 -> '10'
    Matches the convention in build_continental_analysis_dataset.py.
    """
    return str(alpha).replace(".", "")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

all_rows = []

for year in WC_YEARS:
    print(f"\n{'='*60}")
    print(f"Processing {year} World Cup")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # 1. Pool data (games schedule + actual scores)
    # -----------------------------------------------------------------------
    pool      = load_pool_data(POOL_PATH, year)
    wc_teams  = get_wc_teams(pool)
    games     = pool["games"]       # game_id, phase, datetime, team1, team2
    scores    = pool["scores"]      # game_id, score1, score2

    # Build a fast score lookup: game_id -> (score1, score2)
    score_lookup = (
        scores
        .set_index("game_id")[["score1", "score2"]]
        .to_dict("index")
    )

    print(f"  Teams : {len(wc_teams)}")
    print(f"  Games : {len(games)}")

    # -----------------------------------------------------------------------
    # 2. Training data + model fit
    # -----------------------------------------------------------------------
    start_date, end_date = get_training_window(year)
    print(f"  Training window: {start_date} → {end_date}")

    kaggle_df = load_kaggle_data(
        KAGGLE_PATH,
        wc_teams,
        start_date,
        end_date,
        decay_lambda=DECAY_LAMBDA,
    )
    print(f"  Training matches: {len(kaggle_df)}")

    model = DixonColes(
        kaggle_df,
        decay_lambda=DECAY_LAMBDA,
        regularization=REGULARIZATION,
        goal_inflation=1,
    )
    model.fit()

    # -----------------------------------------------------------------------
    # 3. Bookmaker odds
    # -----------------------------------------------------------------------
    odds_lookup = load_wc_odds_lookup(year)   # game_id -> {home, draw, away}
    print(f"  Odds matched: {len(odds_lookup)} / {len(games)}")

    # -----------------------------------------------------------------------
    # 4. Build one row per game
    # -----------------------------------------------------------------------
    for game in games.itertuples():

        game_id = game.game_id
        home    = game.team1
        away    = game.team2
        phase   = game.phase
        date    = game.datetime

        actual = score_lookup.get(game_id)
        if actual is None:
            print(f"  WARNING: no score for game_id={game_id} ({home} vs {away}) — skipping")
            continue

        actual_home = int(actual["score1"])
        actual_away = int(actual["score2"])

        book_probs = odds_lookup.get(game_id)   # may be None if odds missing

        # -------------------------------------------------------------------
        # Model score matrix + outcome probs
        # -------------------------------------------------------------------
        matrix, lh, la = model.score_matrix(home, away, neutral=True)
        model_probs     = outcome_probs_from_matrix(matrix)

        rho = float(
            model.fitted_params[
                2 * model.n_teams + 1
            ]
        )

        # -------------------------------------------------------------------
        # Disagreement signals (only meaningful when we have book odds)
        # -------------------------------------------------------------------
        if book_probs is not None:
            tvd = (
                abs(model_probs["home"] - book_probs["home"])
                + abs(model_probs["draw"] - book_probs["draw"])
                + abs(model_probs["away"] - book_probs["away"])
            ) / 2

            model_fav = max(model_probs, key=model_probs.get)
            book_fav  = max(book_probs,  key=book_probs.get)
            fav_flip  = model_fav != book_fav

            # Signed differences (model − book) — useful for regression later
            diff_home = model_probs["home"] - book_probs["home"]
            diff_draw = model_probs["draw"] - book_probs["draw"]
            diff_away = model_probs["away"] - book_probs["away"]
        else:
            tvd       = np.nan
            model_fav = max(model_probs, key=model_probs.get)
            book_fav  = np.nan
            fav_flip  = np.nan
            diff_home = np.nan
            diff_draw = np.nan
            diff_away = np.nan

        # -------------------------------------------------------------------
        # Base row — metadata + model probs + book probs + signals
        # -------------------------------------------------------------------
        row = {
            # Match metadata
            "year":       year,
            "date":       date,
            "phase":      phase,
            "game_id":    game_id,
            "home_team":  home,
            "away_team":  away,

            # Result
            "actual_home": actual_home,
            "actual_away": actual_away,

            # Model outcome probabilities
            "model_home": round(model_probs["home"], 6),
            "model_draw": round(model_probs["draw"], 6),
            "model_away": round(model_probs["away"], 6),

            # Model expected goals (useful for future lambda-blending research)
            "lambda_home": round(lh, 4),
            "lambda_away": round(la, 4),
            "rho": round(rho, 6),

            # Bookmaker probabilities (Shin-adjusted)
            "book_home": round(book_probs["home"], 6) if book_probs else np.nan,
            "book_draw": round(book_probs["draw"], 6) if book_probs else np.nan,
            "book_away": round(book_probs["away"], 6) if book_probs else np.nan,

            # Disagreement signals
            "tvd":            round(tvd, 6) if book_probs else np.nan,
            "model_favorite": model_fav,
            "book_favorite":  book_fav,
            "favorite_flip":  fav_flip,

            # Signed prob differences (model − book)
            "diff_home": round(diff_home, 6) if book_probs else np.nan,
            "diff_draw": round(diff_draw, 6) if book_probs else np.nan,
            "diff_away": round(diff_away, 6) if book_probs else np.nan,
        }

        # -------------------------------------------------------------------
        # Pure model prediction (alpha=1.0)
        # Stored separately so it's always available even if book odds missing
        # -------------------------------------------------------------------
        pure = model.predict(home, away, neutral=True, alpha=1.0)

        row.update({
            "model_pred_home":    pure["pred_home"],
            "model_pred_away":    pure["pred_away"],
            "model_prediction":   pure["prediction"],
            "model_expected_pts": round(pure["expected_pts"], 6),
            "model_decision_margin": round(pure["decision_margin"], 6),

            # Second-best prediction — key for decision margin research
            "model_second_pred_home":    pure["second_pred_home"],
            "model_second_pred_away":    pure["second_pred_away"],
            "model_second_prediction":   pure["second_prediction"],
            "model_second_expected_pts": round(pure["second_expected_pts"], 6),

            # Points earned by pure model
            "model_points": points_for_prediction(
                pure["pred_home"], pure["pred_away"],
                actual_home, actual_away,
            ),
        })

        # -------------------------------------------------------------------
        # Blend predictions — one block per alpha
        # Only computed when book odds are available; NaN otherwise.
        # -------------------------------------------------------------------
        for alpha in ALPHAS:
            tag = alpha_tag(alpha)

            if book_probs is not None:
                pred = model.predict(
                    home, away,
                    neutral=True,
                    bookmaker_probs=book_probs,
                    alpha=alpha,
                )

                row[f"blend{tag}_pred_home"]      = pred["pred_home"]
                row[f"blend{tag}_pred_away"]      = pred["pred_away"]
                row[f"blend{tag}_prediction"]     = pred["prediction"]
                row[f"blend{tag}_expected_pts"]   = round(pred["expected_pts"], 6)
                row[f"blend{tag}_decision_margin"]= round(pred["decision_margin"], 6)
                row[f"blend{tag}_points"]         = points_for_prediction(
                    pred["pred_home"], pred["pred_away"],
                    actual_home, actual_away,
                )

                # Second-best prediction for blend (useful for margin research)
                row[f"blend{tag}_second_prediction"]   = pred["second_prediction"]
                row[f"blend{tag}_second_pred_home"]    = pred["second_pred_home"]
                row[f"blend{tag}_second_pred_away"]    = pred["second_pred_away"]
                row[f"blend{tag}_second_expected_pts"] = round(pred["second_expected_pts"], 6)

            else:
                # No odds — every alpha collapses to pure model
                row[f"blend{tag}_pred_home"]           = pure["pred_home"]
                row[f"blend{tag}_pred_away"]           = pure["pred_away"]
                row[f"blend{tag}_prediction"]          = pure["prediction"]
                row[f"blend{tag}_expected_pts"]        = round(pure["expected_pts"], 6)
                row[f"blend{tag}_decision_margin"]     = round(pure["decision_margin"], 6)
                row[f"blend{tag}_points"]              = row["model_points"]
                row[f"blend{tag}_second_prediction"]   = pure["second_prediction"]
                row[f"blend{tag}_second_pred_home"]    = pure["second_pred_home"]
                row[f"blend{tag}_second_pred_away"]    = pure["second_pred_away"]
                row[f"blend{tag}_second_expected_pts"] = round(pure["second_expected_pts"], 6)

        all_rows.append(row)

    # Checkpoint after each WC year
    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
    print(f"  Checkpoint saved ({len(all_rows)} rows total)")

# ---------------------------------------------------------------------------
# Final save + summary
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*60}")
print(f"DONE — {len(df)} rows saved to {OUTPUT_PATH}")
print(f"{'='*60}")

# Quick sanity: points by year for pure model and best fixed alphas
summary_cols = (
    ["model_points"]
    + [f"blend{alpha_tag(a)}_points" for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
)
# Only include cols that actually exist (they won't if all odds were missing)
summary_cols = [c for c in summary_cols if c in df.columns]

print("\nPoints by year (pure model + key blends, NaN games excluded):")
print(
    df.groupby("year")[summary_cols]
    .sum()
    .to_string()
)
print(f"\nTotals across all years:")
print(df[summary_cols].sum().to_string())