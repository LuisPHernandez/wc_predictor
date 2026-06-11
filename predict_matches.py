import pandas as pd
import numpy as np
from pathlib import Path

from src.loader import (
    load_kaggle_data,
)
from src.model import DixonColes
from src.odds_loader import _shin_probs
from src.mappings import code_to_name
from implied_xg import implied_expected_goals

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

HIST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "predictions_history.csv"
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
    "match_date",
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "ou_line",
    "over_odds",
    "under_odds",
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
run_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

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

    exact_implied_xg = implied_expected_goals(
        line=float(row.ou_line),
        over_odds=float(row.over_odds),
        under_odds=float(row.under_odds)
    )

    pred = model.predict(
        row.home_team,
        row.away_team,
        neutral=True,
        market_total_goals=exact_implied_xg,
        bookmaker_probs=bookmaker_probs,
    )

    results.append({

        # --------------------------------------
        # History Context Data
        # --------------------------------------

        "match_date":
            row.match_date,

        "prediction_timestamp":
            run_timestamp,

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

        "over_odds":
            row.over_odds,

        "under_odds":
            row.under_odds,

        "calculated_implied_xg":
            round(exact_implied_xg, 4),

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
# SAVE CURRENT SOURCE OF TRUTH (Overwrites predictions.csv)
# ============================================================

output = pd.DataFrame(results)

output.to_csv(
    OUTPUT_PATH,
    index=False,
)

# ============================================================
# SAVE HISTORICAL LOG (Appends ONLY on meaningful changes)
# ============================================================

if HIST_OUTPUT_PATH.exists():
    historical_df = pd.read_csv(HIST_OUTPUT_PATH)
    
    # Isolate the single absolute latest chronological record per fixture
    # Note: match_date, home_team, and away_team become the index levels here
    latest_hist = historical_df.sort_values("prediction_timestamp").groupby(
        ["match_date", "home_team", "away_team"], observed=False
    ).last()
    
    # FIX: Exclude the index keys along with the transient timestamp
    exclude_columns = ["prediction_timestamp", "match_date", "home_team", "away_team"]
    compare_columns = [c for c in output.columns if c not in exclude_columns]
    filtered_new_rows = []
    
    for row in output.to_dict('records'):
        key = (str(row["match_date"]), str(row["home_team"]), str(row["away_team"]))
        
        if key in latest_hist.index:
            hist_row = latest_hist.loc[key]
            has_changed = False
            
            for col in compare_columns:
                v_new = row[col]
                v_old = hist_row[col]
                
                if pd.isna(v_new) and pd.isna(v_old):
                    continue
                if pd.isna(v_new) or pd.isna(v_old):
                    has_changed = True
                    break
                    
                # Handle floating-point tolerances safely
                try:
                    if np.isclose(float(v_new), float(v_old), atol=1e-6):
                        continue
                except (ValueError, TypeError):
                    pass
                    
                if v_new != v_old:
                    has_changed = True
                    break
            
            if has_changed:
                filtered_new_rows.append(row)
        else:
            # Completely fresh match fixture not found in history logs yet
            filtered_new_rows.append(row)
            
    if filtered_new_rows:
        new_history_df = pd.DataFrame(filtered_new_rows)
        combined_history = pd.concat([historical_df, new_history_df], ignore_index=True)
        print(f"--> Detected market line or prediction changes for {len(filtered_new_rows)} fixtures. Updating log.")
    else:
        combined_history = historical_df
        print("--> All match inputs and predictions match their last recorded state. History log untouched.")
else:
    combined_history = output
    print("--> No existing history log found. Creating fresh predictions_history.csv log.")

# Sort chronologically by match groups so snapshots track cleanly over time
combined_history = combined_history.sort_values(
    by=["match_date", "home_team", "away_team", "prediction_timestamp"],
    ascending=[True, True, True, True]
)

combined_history.to_csv(
    HIST_OUTPUT_PATH,
    index=False,
)

# ============================================================
# PRINT OUTPUT SUMMARY
# ============================================================

print()
print("=" * 60)
print("PREDICTIONS (SOURCE OF TRUTH)")
print("=" * 60)

print(
    output.to_string(
        index=False
    )
)

print()
print(f"Saved current source of truth: {OUTPUT_PATH}")
print(f"Updated chronological history log: {HIST_OUTPUT_PATH}")