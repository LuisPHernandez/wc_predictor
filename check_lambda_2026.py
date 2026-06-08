"""
check_lambda_2026.py
────────────────────
Sanity check for k=1.15 before locking it in for predict_2026.py.

Trains the Dixon-Coles model on data up to June 10 2026 and
computes lambda_home + lambda_away for every group stage match.

Compares the average predicted total lambda against the 2022
baseline (2.3440) that k=1.15 was calibrated against.

If 2026 baseline is close to 2.3440  → k=1.15 is correct.
If 2026 baseline has drifted upward  → reduce k accordingly.
See the decision table printed at the end.

Requires:
    data/wc2026_group_stage.csv
    Two columns: home_team, away_team
    72 rows — one per group stage match.
    All team names should match FIFA naming (will be resolved
    via _resolve_team automatically).

    Fill this in from your WorldCup2026.xlsx or any schedule
    page. Home/away designation does not matter since all
    matches are run with neutral=True.

Usage:
    py -3 check_lambda_2026.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from src.loader import (
    load_kaggle_base_data,
    build_competition_weights,
    build_confederation_weights,
)
from src.model import DixonColes
from src.odds_loader import _resolve_team

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
KAGGLE_PATH    = PROJECT_ROOT / "data" / "kaggle" / "results.csv"
MATCHES_PATH   = PROJECT_ROOT / "data" / "wc2026_group_stage.csv"

# ─────────────────────────────────────────────────────────────
# Best tuned parameters
# ─────────────────────────────────────────────────────────────
DECAY_LAMBDA   = 0.2
TRAINING_YEARS = 12
REGULARIZATION = 0.0010

CONTINENTAL = 1.0
QUALIFIER   = 0.5
REGIONAL    = 0.3
FRIENDLY    = 0.3

CONMEBOL = 1.0
UEFA     = 1.0
CAF      = 1.10
CONCACAF = 1.05
AFC      = 0.95
OFC      = 0.90

TOURNAMENT_START = "2026-06-11"

# ─────────────────────────────────────────────────────────────
# Reference values from the k analysis
# ─────────────────────────────────────────────────────────────
LAMBDA_2022_BASELINE  = 2.3440   # avg lambda at k=1.00 in 2022
LAMBDA_RECENT_WC_AVG  = 2.6875   # actual avg goals in 2022 (target)
K_CURRENT             = 1.15     # k we intend to use


# ─────────────────────────────────────────────────────────────
# 1. Load group stage matchups
# ─────────────────────────────────────────────────────────────
if not MATCHES_PATH.exists():
    raise FileNotFoundError(
        f"\nFile not found: {MATCHES_PATH}\n\n"
        "Create a CSV with two columns:\n"
        "    home_team, away_team\n"
        "72 rows — one per group stage match.\n"
        "Fill from your WorldCup2026.xlsx or any schedule page.\n"
        "Home/away designation doesn't matter (all run neutral=True).\n"
    )

matches = pd.read_csv(MATCHES_PATH)
matches["home_team"] = matches["home_team"].astype(str).str.strip().apply(_resolve_team)
matches["away_team"] = matches["away_team"].astype(str).str.strip().apply(_resolve_team)

assert len(matches) == 72, (
    f"Expected 72 group stage matches, got {len(matches)}. "
    "Check your CSV."
)

wc_teams = sorted(
    set(matches["home_team"]) | set(matches["away_team"])
)

print(f"Matches loaded : {len(matches)}")
print(f"Unique teams   : {len(wc_teams)}")

# ─────────────────────────────────────────────────────────────
# 2. Train model on all data up to day before tournament
# ─────────────────────────────────────────────────────────────
print("\nTraining Dixon-Coles model...")

training_end   = pd.Timestamp(TOURNAMENT_START) - pd.Timedelta(days=1)
training_start = training_end - pd.DateOffset(years=TRAINING_YEARS)

print(f"Training window: {training_start.date()} → {training_end.date()}")

competition_weights   = build_competition_weights(CONTINENTAL, QUALIFIER, REGIONAL, FRIENDLY)
confederation_weights = build_confederation_weights(CONMEBOL, CAF, CONCACAF, AFC, OFC)

base_df = load_kaggle_base_data(
    KAGGLE_PATH,
    wc_teams,
    training_start.strftime("%Y-%m-%d"),
    training_end.strftime("%Y-%m-%d"),
    DECAY_LAMBDA,
)

base_df["competition_weight"] = base_df["tournament"].map(competition_weights)

home_conf = base_df["home_confederation"].map(confederation_weights).fillna(1.0)
away_conf = base_df["away_confederation"].map(confederation_weights).fillna(1.0)
base_df["confederation_weight"] = np.sqrt(home_conf * away_conf)

base_df["weight"] = (
    base_df["recency_weight"]
    * base_df["competition_weight"]
    * base_df["confederation_weight"]
)

train_df = base_df[[
    "date", "home_team", "away_team",
    "home_score", "away_score",
    "neutral", "weight",
]]

model = DixonColes(train_df, decay_lambda=DECAY_LAMBDA, regularization=REGULARIZATION)
model.fit()
print("Model fitted.")

# ─────────────────────────────────────────────────────────────
# 3. Compute lambda for every group stage match
# ─────────────────────────────────────────────────────────────
print("\nComputing lambdas for all 72 group stage matches...")

rows = []
missing = []

for _, match in matches.iterrows():
    home = match["home_team"]
    away = match["away_team"]
    try:
        _, lh, la = model.score_matrix(home, away, neutral=True)
        rows.append({
            "home_team"   : home,
            "away_team"   : away,
            "lambda_home" : round(lh, 4),
            "lambda_away" : round(la, 4),
            "lambda_total": round(lh + la, 4),
        })
    except Exception as e:
        missing.append(f"{home} vs {away}: {e}")
        rows.append({
            "home_team"   : home,
            "away_team"   : away,
            "lambda_home" : None,
            "lambda_away" : None,
            "lambda_total": None,
        })

result_df = pd.DataFrame(rows)

if missing:
    print(f"\nWARNING — {len(missing)} matches could not be computed:")
    for m in missing:
        print(f"  {m}")

# ─────────────────────────────────────────────────────────────
# 4. Summary statistics
# ─────────────────────────────────────────────────────────────
valid = result_df.dropna(subset=["lambda_total"])

avg_lambda    = valid["lambda_total"].mean()
median_lambda = valid["lambda_total"].median()
p25           = valid["lambda_total"].quantile(0.25)
p75           = valid["lambda_total"].quantile(0.75)

# Implied k if 2026 model has drifted from 2022 baseline
k_implied = LAMBDA_RECENT_WC_AVG / avg_lambda

print(f"\n{'=' * 60}")
print("LAMBDA SUMMARY — 2026 GROUP STAGE (72 matches)")
print(f"{'=' * 60}")
print(f"\n  Matches computed     : {len(valid)} / 72")
print(f"  Mean   λ total       : {avg_lambda:.4f}")
print(f"  Median λ total       : {median_lambda:.4f}")
print(f"  P25 / P75            : {p25:.4f} / {p75:.4f}")
print(f"\n  2022 baseline λ      : {LAMBDA_2022_BASELINE:.4f}  (what k=1.15 was calibrated against)")
print(f"  Drift vs 2022        : {avg_lambda - LAMBDA_2022_BASELINE:+.4f}")
print(f"\n  Target actual avg    : {LAMBDA_RECENT_WC_AVG:.4f}  (recent WC goal average)")
print(f"  Current k            : {K_CURRENT:.2f}")
print(f"  Implied k from drift : {k_implied:.2f}  (= {LAMBDA_RECENT_WC_AVG:.4f} / {avg_lambda:.4f})")

# ─────────────────────────────────────────────────────────────
# 5. Decision table
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("DECISION TABLE")
print(f"{'=' * 60}")
print(f"""
  Your 2026 avg λ is {avg_lambda:.4f}.

  λ ≈ 2.30–2.39  →  k=1.15 is correct. Model is in the same
                      regime as when k was calibrated. Use k=1.15.

  λ ≈ 2.40–2.49  →  k=1.10 is more appropriate. Model has
                      partially self-corrected via recent training
                      data. Reduce k slightly.

  λ ≈ 2.50–2.59  →  k=1.05. Model has largely adjusted on its
                      own. Only a small inflation correction needed.

  λ ≈ 2.60–2.67  →  k=1.00. Model already predicts at recent WC
                      average. No inflation correction needed.

  λ > 2.67       →  k < 1.00 would be implied. Unusual result —
                      double check training data and parameters
                      before trusting this.

  Implied k for your run: {k_implied:.2f}
""")

# ─────────────────────────────────────────────────────────────
# 6. Print full match table (sorted by lambda_total descending)
# ─────────────────────────────────────────────────────────────
print(f"{'=' * 60}")
print("FULL MATCH TABLE (sorted by total lambda, highest first)")
print(f"{'=' * 60}")
print(f"\n  {'Home':<26}  {'Away':<26}  {'λ home':>7}  {'λ away':>7}  {'λ total':>8}")
print(f"  {'─' * 26}  {'─' * 26}  {'─' * 7}  {'─' * 7}  {'─' * 8}")

for _, row in valid.sort_values("lambda_total", ascending=False).iterrows():
    print(
        f"  {row['home_team']:<26}  {row['away_team']:<26}  "
        f"{row['lambda_home']:>7.4f}  {row['lambda_away']:>7.4f}  "
        f"{row['lambda_total']:>8.4f}"
    )

# ─────────────────────────────────────────────────────────────
# 7. Save
# ─────────────────────────────────────────────────────────────
out_path = PROJECT_ROOT / "data" / "wc2026_lambda_check.csv"
result_df.to_csv(out_path, index=False)
print(f"\nFull results saved: {out_path}")