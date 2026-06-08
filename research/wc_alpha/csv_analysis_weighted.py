"""
retune_alpha_weighted.py

Retunes the optimal fixed alpha and dynamic rule parameters using
recency-weighted evaluation on wc_analysis_k115.csv.

Year weights reflect that recent WCs are more predictive of 2026:
    2006 : 0.50  (different betting market, old goal regime)
    2010 : 0.75  (transitional year)
    2014 : 0.90  (modern goal regime begins)
    2018 : 1.00  (most predictive of 2026 alongside 2022)
    2022 : HOLDOUT — not used in tuning

The weighted objective is:
    sum(year_weight * points_that_year)

This is used for alpha selection only. Holdout (2022) is always
evaluated unweighted at the end.

Run from project root:
    py -3 retune_alpha_weighted.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "analysis" / "wc_analysis_k115.csv"
TRAINING_YEARS = [2006, 2010, 2014, 2018]
HOLDOUT_YEAR   = 2022
ALL_YEARS      = TRAINING_YEARS + [HOLDOUT_YEAR]

ALPHAS = [round(a, 2) for a in np.arange(0.0, 1.05, 0.05)]

# Recency weights — 2022 excluded from tuning (holdout)
YEAR_WEIGHTS = {
    2006: 0.50,
    2010: 0.75,
    2014: 0.90,
    2018: 1.00,
    2022: 1.00,   # used only for holdout evaluation, not tuning
}

def alpha_tag(a):
    return str(a).replace(".", "")

def points_col(a):
    return f"blend{alpha_tag(a)}_points"

SEP = "=" * 65

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
print(f"\nYear weights used for tuning:")
for yr, w in YEAR_WEIGHTS.items():
    label = " (HOLDOUT — not tuned)" if yr == HOLDOUT_YEAR else ""
    print(f"  {yr}: {w:.2f}{label}")

train = df[df["year"].isin(TRAINING_YEARS)].copy()
hold  = df[df["year"] == HOLDOUT_YEAR].copy()

SEP = "=" * 65

# ---------------------------------------------------------------------------
# Helper: weighted points total for a given alpha over training years
# ---------------------------------------------------------------------------

def weighted_points(subset, alpha):
    total = 0.0
    for yr in TRAINING_YEARS:
        yr_pts = subset[subset["year"] == yr][points_col(alpha)].sum()
        total += YEAR_WEIGHTS[yr] * yr_pts
    return total

def unweighted_points(subset, alpha):
    return subset[points_col(alpha)].sum()

# ===========================================================================
# SECTION 1: Unweighted vs weighted alpha profile comparison
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 1: UNWEIGHTED vs WEIGHTED ALPHA PROFILE (training years)")
print(SEP)

print(f"\n  {'Alpha':>6}  {'Unweighted':>12}  {'Weighted':>10}  {'2006':>6}  {'2010':>6}  {'2014':>6}  {'2018':>6}  {'2022':>8}")
print(f"  {'------':>6}  {'----------':>12}  {'--------':>10}  {'----':>6}  {'----':>6}  {'----':>6}  {'----':>6}  {'------':>8}")

for a in ALPHAS:
    unw = unweighted_points(train, a)
    w   = weighted_points(train, a)
    pts_by_year = {
        yr: int(df[df["year"] == yr][points_col(a)].sum())
        for yr in ALL_YEARS
    }
    marker = ""
    print(
        f"  {a:>6.2f}  {unw:>12.1f}  {w:>10.2f}  "
        f"{pts_by_year[2006]:>6}  {pts_by_year[2010]:>6}  "
        f"{pts_by_year[2014]:>6}  {pts_by_year[2018]:>6}  "
        f"{pts_by_year[2022]:>8}"
    )

# ===========================================================================
# SECTION 2: Best alpha under each scheme
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 2: BEST ALPHA COMPARISON")
print(SEP)

unw_scores  = {a: unweighted_points(train, a) for a in ALPHAS}
w_scores    = {a: weighted_points(train, a)    for a in ALPHAS}
hold_scores = {a: int(hold[points_col(a)].sum()) for a in ALPHAS}

best_unw_a   = max(unw_scores,  key=unw_scores.get)
best_w_a     = max(w_scores,    key=w_scores.get)
best_hold_a  = max(hold_scores, key=hold_scores.get)

print(f"\n  Best alpha (unweighted training) : {best_unw_a:.2f}  "
      f"({unw_scores[best_unw_a]:.1f} pts unweighted, "
      f"{hold_scores[best_unw_a]} pts on 2022)")

print(f"  Best alpha (weighted training)   : {best_w_a:.2f}  "
      f"({w_scores[best_w_a]:.2f} weighted pts, "
      f"{hold_scores[best_w_a]} pts on 2022)")

print(f"  Best alpha (2022 holdout only)   : {best_hold_a:.2f}  "
      f"({hold_scores[best_hold_a]} pts on 2022)  [for reference only]")

# ===========================================================================
# SECTION 3: Plateau analysis under weighted objective
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 3: WEIGHTED PLATEAU ANALYSIS")
print(SEP)

best_w_pts = w_scores[best_w_a]
plateau_2  = [a for a in ALPHAS if w_scores[a] >= best_w_pts - 2]
plateau_1  = [a for a in ALPHAS if w_scores[a] >= best_w_pts - 1]

print(f"\n  Best weighted score : {best_w_pts:.2f}  at alpha={best_w_a:.2f}")
print(f"  Plateau (within 1)  : {[f'{a:.2f}' for a in plateau_1]}")
print(f"  Plateau (within 2)  : {[f'{a:.2f}' for a in plateau_2]}")

print(f"\n  Weighted curve (full):")
print(f"  {'Alpha':>6}  {'Weighted pts':>13}  {'vs best':>8}  {'2022 pts':>9}")
print(f"  {'------':>6}  {'------------':>13}  {'-------':>8}  {'--------':>9}")
for a in ALPHAS:
    diff   = w_scores[a] - best_w_pts
    marker = " <-- best" if a == best_w_a else ""
    print(f"  {a:>6.2f}  {w_scores[a]:>13.2f}  {diff:>+8.2f}  {hold_scores[a]:>9}{marker}")

# ===========================================================================
# SECTION 4: Per-year rank of best weighted alpha
# ===========================================================================

print(f"\n{SEP}")
print(f"SECTION 4: PER-YEAR RANK OF BEST WEIGHTED ALPHA ({best_w_a:.2f})")
print(SEP)

print(f"\n  {'Year':>6}  {'Best α':>7}  {'Best pts':>9}  "
      f"{'α={:.2f} pts'.format(best_w_a):>12}  {'Rank':>6}  {'Note'}")
print(f"  {'------':>6}  {'------':>7}  {'--------':>9}  "
      f"{'------------':>12}  {'------':>6}")

for yr in ALL_YEARS:
    yr_df     = df[df["year"] == yr]
    yr_scores = {a: int(yr_df[points_col(a)].sum()) for a in ALPHAS}
    yr_best_a = max(yr_scores, key=yr_scores.get)
    yr_best_p = yr_scores[yr_best_a]
    chosen_p  = yr_scores[best_w_a]
    rank      = int(sum(1 for v in yr_scores.values() if v > chosen_p)) + 1
    note      = " (HOLDOUT)" if yr == HOLDOUT_YEAR else ""
    print(
        f"  {yr:>6}  {yr_best_a:>7.2f}  {yr_best_p:>9}  "
        f"{chosen_p:>12}  {rank:>6}{note}"
    )

# ===========================================================================
# SECTION 5: Sensitivity — does the weighted optimum shift with weight choice?
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 5: SENSITIVITY TO YEAR WEIGHT CHOICE")
print(SEP)

weight_schemes = {
    "Equal weights (1/1/1/1)":         {2006: 1.0, 2010: 1.0, 2014: 1.0, 2018: 1.0},
    "Mild recency (0.7/0.8/0.9/1.0)":  {2006: 0.7, 2010: 0.8, 2014: 0.9, 2018: 1.0},
    "Base recency (0.5/0.75/0.9/1.0)": {2006: 0.5, 2010: 0.75, 2014: 0.9, 2018: 1.0},
    "Aggressive (0.3/0.6/0.9/1.0)":    {2006: 0.3, 2010: 0.6,  2014: 0.9, 2018: 1.0},
    "2014-2018 only (0/0/1/1)":         {2006: 0.0, 2010: 0.0,  2014: 1.0, 2018: 1.0},
}

print(f"\n  {'Scheme':40s}  {'Best α':>7}  {'2022 pts':>9}")
print(f"  {'------':40s}  {'------':>7}  {'--------':>9}")

for label, weights in weight_schemes.items():
    scheme_scores = {}
    for a in ALPHAS:
        total = sum(
            weights[yr] * df[df["year"] == yr][points_col(a)].sum()
            for yr in TRAINING_YEARS
        )
        scheme_scores[a] = total
    best_a = max(scheme_scores, key=scheme_scores.get)
    hold_pts = hold_scores[best_a]
    print(f"  {label:40s}  {best_a:>7.2f}  {hold_pts:>9}")

# ===========================================================================
# SECTION 6: Final recommendation
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 6: FINAL RECOMMENDATION FOR 2026")
print(SEP)

print(f"""
  Model parameters:
    goal_inflation : 1.15
    alpha          : {best_w_a:.2f}  (recency-weighted optimum)

  2022 holdout score at recommended alpha : {hold_scores[best_w_a]} pts
  2022 holdout best possible alpha        : {best_hold_a:.2f} ({hold_scores[best_hold_a]} pts)

  Plateau at recommended alpha (within 1 weighted pt):
    {[f'{a:.2f}' for a in plateau_1]}

  If the plateau is wide (5+ alphas), any value in the range is
  equivalent in expectation — pick the midpoint for robustness.

  If the sensitivity table (Section 5) shows the best alpha is
  stable across all weighting schemes, that alpha is robust and
  should be used for 2026 with confidence.
  If it shifts around, use the midpoint of the range that appears
  most frequently across schemes.
""")