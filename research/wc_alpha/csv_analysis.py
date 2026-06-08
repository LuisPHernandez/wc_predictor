"""
analyse_phase_pre_and_1.py

Pre-Analysis: Data Characterization
Phase 1:      Fixed Alpha Profile

Run from project root:
    py -3 analyse_phase_pre_and_1.py

Paste the full output back for interpretation before moving to Phase 2.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "analysis" / "wc_analysis.csv"

TRAINING_YEARS = [2006, 2010, 2014, 2018]
HOLDOUT_YEAR   = 2022
ALL_YEARS      = TRAINING_YEARS + [HOLDOUT_YEAR]

ALPHAS = [round(a, 2) for a in np.arange(0.0, 1.05, 0.05)]

def alpha_tag(alpha):
    return str(alpha).replace(".", "")

def points_col(alpha):
    return f"blend{alpha_tag(alpha)}_points"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Games per year:\n{df['year'].value_counts().sort_index().to_string()}")

# Convenience: split into training and holdout
train = df[df["year"].isin(TRAINING_YEARS)].copy()
hold  = df[df["year"] == HOLDOUT_YEAR].copy()

# Games with and without odds
has_odds = df["book_home"].notna()
print(f"\nGames with odds : {has_odds.sum()} / {len(df)}")
print(f"No-odds breakdown by year:")
print(df.groupby("year")["book_home"].apply(lambda x: x.isna().sum()).to_string())

SEP = "=" * 65

# ===========================================================================
# PRE-ANALYSIS: Data Characterization
# ===========================================================================

print(f"\n{SEP}")
print("PRE-ANALYSIS: DATA CHARACTERIZATION")
print(SEP)

# ---------------------------------------------------------------------------
# TVD summary by year
# ---------------------------------------------------------------------------

print("\n--- TVD summary statistics by year (games with odds only) ---")
tvd_stats = (
    df[has_odds]
    .groupby("year")["tvd"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
        p95=lambda x: x.quantile(0.95),
    )
    .round(4)
)
print(tvd_stats.to_string())

print("\n--- TVD summary across ALL training years combined ---")
train_odds = train[train["book_home"].notna()]
print(f"  count  : {len(train_odds)}")
print(f"  mean   : {train_odds['tvd'].mean():.4f}")
print(f"  median : {train_odds['tvd'].median():.4f}")
print(f"  p75    : {train_odds['tvd'].quantile(0.75):.4f}")
print(f"  p90    : {train_odds['tvd'].quantile(0.90):.4f}")
print(f"  p95    : {train_odds['tvd'].quantile(0.95):.4f}")

# ---------------------------------------------------------------------------
# Flip rate by year
# ---------------------------------------------------------------------------

print("\n--- Favorite flip rate by year (games with odds only) ---")
flip_stats = (
    df[has_odds]
    .groupby("year")["favorite_flip"]
    .agg(
        games="count",
        flips="sum",
        flip_rate=lambda x: x.mean().round(3),
    )
)
print(flip_stats.to_string())

# ---------------------------------------------------------------------------
# Trigger rates at different thresholds
# ---------------------------------------------------------------------------

print("\n--- Rule trigger rates across TRAINING years (games with odds) ---")
print("How many games each rule would act on (alpha_low branch)\n")

thresholds = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

header = f"{'Threshold':>10} | {'Rule B (TVD)':>14} | {'Rule A (Flip+TVD)':>18}"
print(header)
print("-" * len(header))

for t in thresholds:
    rule_b = (train_odds["tvd"] > t).sum()
    rule_a = (train_odds["favorite_flip"] & (train_odds["tvd"] > t)).sum()
    rule_b_pct = rule_b / len(train_odds) * 100
    rule_a_pct = rule_a / len(train_odds) * 100
    print(
        f"{t:>10.2f} | "
        f"{rule_b:>5} / {len(train_odds)} ({rule_b_pct:4.1f}%) | "
        f"{rule_a:>5} / {len(train_odds)} ({rule_a_pct:4.1f}%)"
    )

# ---------------------------------------------------------------------------
# TVD vs model error correlation (training years)
# ---------------------------------------------------------------------------

print("\n--- TVD vs model prediction error on training years ---")
print("(Does high TVD predict that the model prediction was wrong?)\n")

# model_points are 0,1,2,3 — lower = more wrong
# We'll look at whether high TVD correlates with lower model points
train_odds = train[train["book_home"].notna()].copy()

corr = train_odds[["tvd", "model_points"]].corr().loc["tvd", "model_points"]
print(f"  Pearson correlation (TVD vs model_points): {corr:.4f}")

# Bucket analysis
bins   = [0, 0.08, 0.12, 0.16, 0.20, 1.0]
labels = ["0–0.08", "0.08–0.12", "0.12–0.16", "0.16–0.20", "0.20+"]
train_odds["tvd_bucket"] = pd.cut(train_odds["tvd"], bins=bins, labels=labels)

tvd_model = (
    train_odds
    .groupby("tvd_bucket", observed=True)["model_points"]
    .agg(count="count", mean_pts="mean")
    .round(3)
)
print("\n  Model points by TVD bucket (training years):")
print(tvd_model.to_string())

# ===========================================================================
# PHASE 1: Fixed Alpha Profile
# ===========================================================================

print(f"\n{SEP}")
print("PHASE 1: FIXED ALPHA PROFILE")
print(SEP)

# ---------------------------------------------------------------------------
# Points matrix: year x alpha
# ---------------------------------------------------------------------------

# Build the full matrix
alpha_cols  = [points_col(a) for a in ALPHAS]
points_matrix = (
    df.groupby("year")[alpha_cols]
    .sum()
    .rename(columns={points_col(a): a for a in ALPHAS})
)

# Add training total row
points_matrix.loc["2006-2018"] = points_matrix.loc[
    points_matrix.index.isin(TRAINING_YEARS)
].sum()

print("\n--- Points matrix: year x alpha ---")
print("(rows = years + training total, columns = alpha values)\n")

# Print in readable chunks (alpha 0.0 to 0.50, then 0.55 to 1.0)
alphas_lo = [a for a in ALPHAS if a <= 0.50]
alphas_hi = [a for a in ALPHAS if a  > 0.50]

def print_matrix_chunk(matrix, alphas, label):
    chunk = matrix[[a for a in alphas]]
    chunk.columns = [f"{a:.2f}" for a in chunk.columns]
    print(f"  {label}")
    print(chunk.to_string())
    print()

print_matrix_chunk(points_matrix, alphas_lo, "Alpha 0.00 – 0.50")
print_matrix_chunk(points_matrix, alphas_hi, "Alpha 0.55 – 1.00")

# ---------------------------------------------------------------------------
# Best alpha per year
# ---------------------------------------------------------------------------

print("--- Best alpha per year ---\n")
for idx in points_matrix.index:
    row    = points_matrix.loc[idx]
    best_a = row.idxmax()
    best_p = row.max()
    # also find plateau: alphas within 2 pts of best
    plateau = sorted([a for a in ALPHAS if row[a] >= best_p - 2])
    print(
        f"  {idx}: best alpha = {best_a:.2f}  "
        f"({best_p:.0f} pts)  |  "
        f"plateau (within 2 pts): {[f'{a:.2f}' for a in plateau]}"
    )

# ---------------------------------------------------------------------------
# Training best fixed alpha (the primary baseline for all subsequent phases)
# ---------------------------------------------------------------------------

train_row  = points_matrix.loc["2006-2018"]
best_fixed = float(train_row.idxmax())
best_fixed_pts = int(train_row.max())

print(f"\n{'='*65}")
print(f"TRAINING BEST FIXED ALPHA : {best_fixed:.2f}  ({best_fixed_pts} pts over 2006-2018)")
print(f"This is the primary baseline for all subsequent phases.")
print(f"{'='*65}")

# ---------------------------------------------------------------------------
# Alpha curve shape on combined training
# ---------------------------------------------------------------------------

print("\n--- Alpha curve shape on combined training (2006–2018) ---\n")
print(f"  {'Alpha':>6}  {'Points':>7}  {'vs best':>8}")
print(f"  {'------':>6}  {'-------':>7}  {'--------':>8}")
for a in ALPHAS:
    pts  = int(train_row[a])
    diff = pts - best_fixed_pts
    marker = " <-- best" if a == best_fixed else ""
    print(f"  {a:>6.2f}  {pts:>7}  {diff:>+8}{marker}")

# ---------------------------------------------------------------------------
# Consistency check: rank of best training alpha in each individual year
# ---------------------------------------------------------------------------

print("\n--- Rank of training best fixed alpha in each individual year ---")
print(f"  (training best alpha = {best_fixed:.2f})\n")
for yr in ALL_YEARS:
    row      = points_matrix.loc[yr]
    yr_best  = float(row.idxmax())
    yr_pts   = int(row.max())
    alpha_pts = int(row[best_fixed])
    rank     = int((row > alpha_pts).sum()) + 1
    label    = "HOLDOUT" if yr == HOLDOUT_YEAR else "train"
    print(
        f"  {yr} ({label}): "
        f"best={yr_best:.2f} ({yr_pts} pts) | "
        f"alpha={best_fixed:.2f} scores {alpha_pts} pts | "
        f"rank {rank} of {len(ALPHAS)}"
    )