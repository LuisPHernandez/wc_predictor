"""
team_ranks_2026.py
──────────────────
Compares Dixon-Coles model team strength ranking against
bookmaker outright winner odds for WC 2026.

Prints a side-by-side ranking table and saves it to:
    data/odds/strength_comparison.csv

Usage:
    py -3 rank_teams_2026.py

Requirements:
    data/odds/wc2026_outrights.csv
        Columns: team (string), odds (decimal)
        One row per qualified team. Get odds from any major
        bookmaker (Pinnacle preferred) before the tournament.
        Example:
            team,odds
            Brazil,6.00
            France,7.00
            Argentina,6.50
            ...
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr # pyrefly: ignore [missing-import]

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
OUTRIGHTS_PATH = PROJECT_ROOT / "data" / "odds" / "wc2026_outrights.csv"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "odds" / "strength_comparison.csv"

# ─────────────────────────────────────────────────────────────
# Best tuned parameters — do not change
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

# WC 2026 kicks off June 11, 2026
TOURNAMENT_START = "2026-06-11"

# ─────────────────────────────────────────────────────────────
# Reference team for strength computation.
#
# Every team is scored as: expected goal difference vs this
# opponent on a neutral field.  The reference team just needs
# to be stable, data-rich, and reliably mid-table at WC level.
# Switzerland qualifies for almost every tournament and sits
# comfortably in the middle of WC strength distributions.
#
# Important: this team must appear in the kaggle results.csv
# with enough matches to have well-fitted parameters.
# ─────────────────────────────────────────────────────────────
REFERENCE_TEAM = "Switzerland"

# ─────────────────────────────────────────────────────────────
# 1. Load and normalise outright odds
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — OUTRIGHT ODDS")
print("=" * 65)

if not OUTRIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"\nFile not found: {OUTRIGHTS_PATH}\n\n"
        "Create a CSV with two columns:\n"
        "    team   — FIFA name (will be resolved automatically)\n"
        "    odds   — decimal odds from your preferred bookmaker\n\n"
        "Example:\n"
        "    team,odds\n"
        "    Brazil,6.00\n"
        "    France,7.00\n"
        "    Argentina,6.50\n"
    )

outrights = pd.read_csv(OUTRIGHTS_PATH)
outrights["team"] = (
    outrights["team"]
    .astype(str)
    .str.strip()
    .apply(_resolve_team)
)
outrights["odds"] = outrights["odds"].astype(float)
outrights["raw_prob"] = 1.0 / outrights["odds"]

# Remove overround via normalization.
# The Shin model is designed for 3-outcome markets.
# For a 48-team outright market, simple normalization is correct.
overround = outrights["raw_prob"].sum()
outrights["book_prob"] = outrights["raw_prob"] / overround

print(f"Teams loaded  : {len(outrights)}")
print(f"Overround     : {overround:.4f}  ({(overround - 1) * 100:.1f}% bookmaker margin removed)")

wc_teams = sorted(outrights["team"].tolist())

# ─────────────────────────────────────────────────────────────
# 2. Train Dixon-Coles model on data up to tournament start
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — TRAIN DIXON-COLES MODEL")
print("=" * 65)

training_end   = pd.Timestamp(TOURNAMENT_START) - pd.Timedelta(days=1)
training_start = training_end - pd.DateOffset(years=TRAINING_YEARS)

print(f"Training window : {training_start.date()} → {training_end.date()}")
print(f"Teams           : {len(wc_teams)}")

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

model = DixonColes(
    train_df,
    decay_lambda=DECAY_LAMBDA,
    regularization=REGULARIZATION,
)
model.fit()
print("Model fitted.")

# ─────────────────────────────────────────────────────────────
# 3. Compute model strength for each team
#
# For each team we call score_matrix() twice:
#   (a) team  as "home" vs REFERENCE_TEAM  → lambda_for_h,  lambda_against_h
#   (b) REFERENCE_TEAM as "home" vs team   → lambda_for_a,  lambda_against_a
#
# Both calls use neutral=True, so home advantage is suppressed.
# Averaging both orderings removes any residual asymmetry in the
# Dixon-Coles rho correction for low-scoring games.
#
# Final metrics:
#   lambda_for     = average expected goals scored    vs reference
#   lambda_against = average expected goals conceded  vs reference
#   strength_score = lambda_for - lambda_against
#                    (positive → better than reference,
#                     negative → worse than reference)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — COMPUTE MODEL STRENGTH")
print("=" * 65)
print(f"Reference team : {REFERENCE_TEAM}\n")

if REFERENCE_TEAM not in wc_teams:
    print(f"NOTE: {REFERENCE_TEAM} is not in the WC teams list.")
    print("It will be used as a reference only and excluded from rankings.\n")

strength_records = []
missing_teams    = []

for team in wc_teams:
    if team == REFERENCE_TEAM:
        continue
    try:
        # (a) team at home vs reference — neutral=True suppresses home advantage
        _, lam_for_h, lam_against_h = model.score_matrix(
            team, REFERENCE_TEAM, neutral=True
        )
        # (b) reference at home vs team
        _, lam_against_a, lam_for_a = model.score_matrix(
            REFERENCE_TEAM, team, neutral=True
        )

        lambda_for     = (lam_for_h     + lam_for_a)     / 2
        lambda_against = (lam_against_h + lam_against_a) / 2

        strength_records.append({
            "team"           : team,
            "lambda_for"     : round(lambda_for,     4),
            "lambda_against" : round(lambda_against, 4),
            "strength_score" : round(lambda_for - lambda_against, 4),
        })

    except Exception as e:
        missing_teams.append(team)
        print(f"  WARNING — no model data for {team}: {e}")

if missing_teams:
    print(f"\nExcluded (no model parameters): {missing_teams}")

strength_df = (
    pd.DataFrame(strength_records)
    .sort_values("strength_score", ascending=False)
    .reset_index(drop=True)
)
strength_df["model_rank"] = strength_df.index + 1

# ─────────────────────────────────────────────────────────────
# 4. Merge model ranking with bookmaker ranking
# ─────────────────────────────────────────────────────────────
outrights_ranked = (
    outrights[["team", "book_prob", "odds"]]
    .sort_values("book_prob", ascending=False)
    .reset_index(drop=True)
    .assign(book_rank=lambda df: df.index + 1)
)

comparison = (
    strength_df
    .merge(outrights_ranked, on="team", how="inner")
    .assign(rank_diff=lambda df: df["book_rank"] - df["model_rank"])
    .sort_values("model_rank")
    .reset_index(drop=True)
)

unmatched = set(wc_teams) - set(comparison["team"])
if unmatched:
    print(f"\nWARNING — teams in odds but missing from model: {sorted(unmatched)}")

# ─────────────────────────────────────────────────────────────
# 5. Print ranking table
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MODEL vs BOOKMAKER STRENGTH RANKING")
print("=" * 65)
print(
    f"\n  {'#':>3}  {'Team':<26}  {'Strength':>8}  "
    f"{'λ for':>5}  {'λ vs':>5}  "
    f"{'Book#':>5}  {'BookProb':>8}  {'Odds':>6}  {'Δ':>5}"
)
print("  " + "─" * 78)

for _, row in comparison.iterrows():
    delta     = int(row["rank_diff"])
    delta_str = f"+{delta}" if delta > 0 else str(delta)

    # Flag large disagreements
    if abs(delta) >= 8:
        flag = "  ◄◄◄"
    elif abs(delta) >= 5:
        flag = "  ◄◄"
    else:
        flag = ""

    print(
        f"  {int(row['model_rank']):>3}  "
        f"{row['team']:<26}  "
        f"{row['strength_score']:>8.3f}  "
        f"{row['lambda_for']:>5.3f}  "
        f"{row['lambda_against']:>5.3f}  "
        f"{int(row['book_rank']):>5}  "
        f"{row['book_prob']:>8.4f}  "
        f"{row['odds']:>6.2f}  "
        f"{delta_str:>5}"
        f"{flag}"
    )

# ─────────────────────────────────────────────────────────────
# 6. Disagreement summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("DISAGREEMENT SUMMARY  (|Δ rank| ≥ 5)")
print("=" * 65)

model_overrates  = comparison[comparison["rank_diff"] >= 5].sort_values("rank_diff", ascending=False)
market_overrates = comparison[comparison["rank_diff"] <= -5].sort_values("rank_diff")

print(f"\nModel ranks MUCH HIGHER than market  (model thinks team is underrated by book):")
if model_overrates.empty:
    print("  None")
else:
    for _, r in model_overrates.iterrows():
        print(
            f"  {r['team']:<26}  "
            f"Model #{int(r['model_rank']):<3}  "
            f"Book #{int(r['book_rank']):<3}  "
            f"Δ = {int(r['rank_diff']):+}"
        )

print(f"\nMarket ranks MUCH HIGHER than model  (book thinks team is underrated by model):")
if market_overrates.empty:
    print("  None")
else:
    for _, r in market_overrates.iterrows():
        print(
            f"  {r['team']:<26}  "
            f"Model #{int(r['model_rank']):<3}  "
            f"Book #{int(r['book_rank']):<3}  "
            f"Δ = {int(r['rank_diff']):+}"
        )

# ─────────────────────────────────────────────────────────────
# 7. Spearman rank correlation — key summary statistic
#
# ρ close to 1.0 → model and bookmaker essentially agree on team order
#                 → lambda multiplier unlikely to add meaningful value
# ρ below ~0.90  → structural disagreements exist worth investigating
# ─────────────────────────────────────────────────────────────
rho, pval = spearmanr(comparison["model_rank"], comparison["book_rank"])

print("\n" + "=" * 65)
print("OVERALL AGREEMENT METRIC")
print("=" * 65)
print(f"\n  Spearman ρ = {rho:.4f}   (p = {pval:.4f})")
print()
if rho >= 0.95:
    print("  → Rankings are nearly identical.")
    print("    Model and bookmaker agree strongly on team order.")
    print("    Lambda multiplier is unlikely to add meaningful value.")
    print("    Recommendation: skip the lambda calibration step.")
elif rho >= 0.88:
    print("  → Rankings broadly agree with some notable disagreements.")
    print("    Review the flagged teams above — they are the candidates")
    print("    where a lambda multiplier might add value.")
    print("    Recommendation: prototype the simplified k = (book/model)^0.3 approach")
    print("    on the flagged teams before committing to the full optimisation loop.")
else:
    print("  → Substantial structural disagreements between model and market.")
    print("    The lambda multiplier approach is worth implementing properly.")
    print("    Start with the simplified proxy, validate on 2022, then decide.")

# ─────────────────────────────────────────────────────────────
# 8. Save
# ─────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")