"""
analyse_phase_3.py

Phase 3: Grid Search on 2006-2018

Exhaustive grid search for both rule forms on training years only.
Rule A: if favorite_flip AND tvd > threshold -> alpha_low, else alpha_high
Rule B: if tvd > threshold -> alpha_low, else alpha_high

Run from project root:
    py -3 analyse_phase_3.py

Paste full output back for interpretation before Phase 4.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "wc_analysis.csv"
TRAINING_YEARS = [2006, 2010, 2014, 2018]
ALPHAS         = [round(a, 2) for a in np.arange(0.0, 1.05, 0.05)]

# Baseline from Phase 1
BEST_FIXED_ALPHA = 0.15
BEST_FIXED_PTS   = 249

# Trigger count floor — results below this are flagged as unreliable
TRIGGER_FLOOR = 35

def alpha_tag(a):
    return str(a).replace(".", "")

def points_col(a):
    return f"blend{alpha_tag(a)}_points"

SEP = "=" * 65

# ---------------------------------------------------------------------------
# Load training data with odds
# ---------------------------------------------------------------------------

df    = pd.read_csv(CSV_PATH)
train = df[df["year"].isin(TRAINING_YEARS) & df["book_home"].notna()].copy()

# Pre-compute all points columns as a numpy matrix for speed
# rows = games, cols = alphas (in order)
pts_matrix = train[[points_col(a) for a in ALPHAS]].values   # (246, 21)
tvd        = train["tvd"].values
flip       = train["favorite_flip"].values.astype(bool)

alpha_idx  = {a: i for i, a in enumerate(ALPHAS)}

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------

# Thresholds — we include the full grid but flag below-floor results
THRESHOLDS  = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

# alpha_low: full low range
ALPHAS_LOW  = [round(a, 2) for a in np.arange(0.0, 0.45, 0.05)]   # 0.0 – 0.40

# alpha_high: upper range (should be near training best fixed alpha)
ALPHAS_HIGH = [round(a, 2) for a in np.arange(0.15, 1.05, 0.05)]  # 0.15 – 1.00

print(f"Grid dimensions:")
print(f"  Thresholds : {THRESHOLDS}")
print(f"  Alpha low  : {ALPHAS_LOW}")
print(f"  Alpha high : {ALPHAS_HIGH}")
print(f"  Rule A combinations: {len(THRESHOLDS) * len(ALPHAS_LOW) * len(ALPHAS_HIGH)}")
print(f"  Rule B combinations: {len(THRESHOLDS) * len(ALPHAS_LOW) * len(ALPHAS_HIGH)}")

# ---------------------------------------------------------------------------
# Grid search — vectorised over games, loop over parameter combos
# ---------------------------------------------------------------------------

def run_grid(use_flip):
    """
    Runs the full grid search.
    use_flip=True  -> Rule A (flip AND tvd > threshold)
    use_flip=False -> Rule B (tvd > threshold only)
    Returns a DataFrame of results sorted by total_points descending.
    """
    results = []

    for threshold in THRESHOLDS:
        if use_flip:
            trigger_mask = flip & (tvd > threshold)
        else:
            trigger_mask = tvd > threshold

        trigger_count = trigger_mask.sum()

        for alpha_low in ALPHAS_LOW:
            for alpha_high in ALPHAS_HIGH:

                # Points for each game: triggered -> alpha_low, else -> alpha_high
                pts_low  = pts_matrix[:, alpha_idx[alpha_low]]
                pts_high = pts_matrix[:, alpha_idx[alpha_high]]

                total = (
                    np.where(trigger_mask, pts_low, pts_high).sum()
                )

                results.append({
                    "threshold":     threshold,
                    "alpha_low":     alpha_low,
                    "alpha_high":    alpha_high,
                    "total_points":  int(total),
                    "improvement":   int(total) - BEST_FIXED_PTS,
                    "trigger_count": int(trigger_count),
                    "reliable":      trigger_count >= TRIGGER_FLOOR,
                })

    return (
        pd.DataFrame(results)
        .sort_values("total_points", ascending=False)
        .reset_index(drop=True)
    )

print("\nRunning Rule B grid search...")
results_b = run_grid(use_flip=False)

print("Running Rule A grid search...")
results_a = run_grid(use_flip=True)

# ===========================================================================
# RESULTS — Rule B
# ===========================================================================

print(f"\n{SEP}")
print("RULE B RESULTS (TVD only)")
print(f"Baseline: best fixed alpha = {BEST_FIXED_ALPHA} ({BEST_FIXED_PTS} pts)")
print(SEP)

print("\n--- Top 20 overall ---\n")
print(
    results_b.head(20)[
        ["threshold", "alpha_low", "alpha_high",
         "total_points", "improvement", "trigger_count", "reliable"]
    ].to_string(index=False)
)

print("\n--- Top 15 RELIABLE only (trigger_count >= 35) ---\n")
reliable_b = results_b[results_b["reliable"]].head(15)
if len(reliable_b) == 0:
    print("  No reliable Rule B configurations found.")
else:
    print(reliable_b[
        ["threshold", "alpha_low", "alpha_high",
         "total_points", "improvement", "trigger_count"]
    ].to_string(index=False))

# Clustering analysis
print("\n--- Parameter clustering (top 20 reliable) ---")
top_b = results_b[results_b["reliable"]].head(20)
if len(top_b) > 0:
    print(f"  threshold range : {top_b['threshold'].min():.2f} – {top_b['threshold'].max():.2f}")
    print(f"  alpha_low range : {top_b['alpha_low'].min():.2f} – {top_b['alpha_low'].max():.2f}")
    print(f"  alpha_high range: {top_b['alpha_high'].min():.2f} – {top_b['alpha_high'].max():.2f}")
    print(f"  points range    : {top_b['total_points'].min()} – {top_b['total_points'].max()}")

# ===========================================================================
# RESULTS — Rule A
# ===========================================================================

print(f"\n{SEP}")
print("RULE A RESULTS (Flip + TVD)")
print(f"Baseline: best fixed alpha = {BEST_FIXED_ALPHA} ({BEST_FIXED_PTS} pts)")
print(SEP)

print("\n--- Top 20 overall ---\n")
print(
    results_a.head(20)[
        ["threshold", "alpha_low", "alpha_high",
         "total_points", "improvement", "trigger_count", "reliable"]
    ].to_string(index=False)
)

print("\n--- Top 15 RELIABLE only (trigger_count >= 35) ---\n")
reliable_a = results_a[results_a["reliable"]].head(15)
if len(reliable_a) == 0:
    print("  No reliable Rule A configurations found.")
else:
    print(reliable_a[
        ["threshold", "alpha_low", "alpha_high",
         "total_points", "improvement", "trigger_count"]
    ].to_string(index=False))

# Clustering analysis
print("\n--- Parameter clustering (top 20 reliable) ---")
top_a = results_a[results_a["reliable"]].head(20)
if len(top_a) > 0:
    print(f"  threshold range : {top_a['threshold'].min():.2f} – {top_a['threshold'].max():.2f}")
    print(f"  alpha_low range : {top_a['alpha_low'].min():.2f} – {top_a['alpha_low'].max():.2f}")
    print(f"  alpha_high range: {top_a['alpha_high'].min():.2f} – {top_a['alpha_high'].max():.2f}")
    print(f"  points range    : {top_a['total_points'].min()} – {top_a['total_points'].max()}")

# ===========================================================================
# HEAD-TO-HEAD: best reliable rule from each form
# ===========================================================================

print(f"\n{SEP}")
print("HEAD-TO-HEAD: BEST RELIABLE RULE A vs BEST RELIABLE RULE B")
print(SEP)

best_b = results_b[results_b["reliable"]].iloc[0] if len(results_b[results_b["reliable"]]) > 0 else None
best_a = results_a[results_a["reliable"]].iloc[0] if len(results_a[results_a["reliable"]]) > 0 else None

print(f"\n  Fixed alpha {BEST_FIXED_ALPHA:.2f}         : {BEST_FIXED_PTS} pts  (baseline)")

if best_b is not None:
    print(
        f"  Best Rule B (reliable)  : {int(best_b['total_points'])} pts  "
        f"({int(best_b['improvement']):+d} vs baseline)  "
        f"threshold={best_b['threshold']:.2f}  "
        f"α_low={best_b['alpha_low']:.2f}  "
        f"α_high={best_b['alpha_high']:.2f}  "
        f"triggers={int(best_b['trigger_count'])}"
    )
else:
    print("  Best Rule B (reliable)  : NONE — no reliable configurations")

if best_a is not None:
    print(
        f"  Best Rule A (reliable)  : {int(best_a['total_points'])} pts  "
        f"({int(best_a['improvement']):+d} vs baseline)  "
        f"threshold={best_a['threshold']:.2f}  "
        f"α_low={best_a['alpha_low']:.2f}  "
        f"α_high={best_a['alpha_high']:.2f}  "
        f"triggers={int(best_a['trigger_count'])}"
    )
else:
    print("  Best Rule A (reliable)  : NONE — no reliable configurations")

# ===========================================================================
# alpha_high plausibility check
# ===========================================================================

print(f"\n{SEP}")
print("ALPHA_HIGH PLAUSIBILITY CHECK")
print("(alpha_high should be near the training best fixed alpha = 0.15)")
print(SEP)

for label, top_df in [("Rule B top 15 reliable", reliable_b), ("Rule A top 15 reliable", reliable_a)]:
    if len(top_df) == 0:
        print(f"\n  {label}: no data")
        continue
    far_from_baseline = top_df[abs(top_df["alpha_high"] - BEST_FIXED_ALPHA) > 0.20]
    print(f"\n  {label}:")
    print(f"    alpha_high values: {sorted(top_df['alpha_high'].unique().tolist())}")
    if len(far_from_baseline) > 0:
        print(f"    WARNING: {len(far_from_baseline)} configs have alpha_high far from baseline")
    else:
        print(f"    alpha_high values are plausible (within 0.20 of baseline)")

# ===========================================================================
# Per-year breakdown of best reliable rules
# ===========================================================================

print(f"\n{SEP}")
print("PER-YEAR BREAKDOWN OF BEST RELIABLE RULES")
print(SEP)

for label, best_row, use_flip in [
    ("Rule B", best_b, False),
    ("Rule A", best_a, True),
]:
    if best_row is None:
        print(f"\n  {label}: no reliable rule")
        continue

    threshold  = best_row["threshold"]
    alpha_low  = best_row["alpha_low"]
    alpha_high = best_row["alpha_high"]

    print(f"\n  {label}: threshold={threshold:.2f}  α_low={alpha_low:.2f}  α_high={alpha_high:.2f}")
    print(f"  {'Year':>6}  {'Rule pts':>10}  {'Fixed pts':>10}  {'Diff':>6}  {'Triggers':>9}")
    print(f"  {'----':>6}  {'--------':>10}  {'---------':>10}  {'----':>6}  {'--------':>9}")

    for yr in TRAINING_YEARS:
        yr_df = df[df["year"] == yr & df["book_home"].notna() if False else
                   (df["year"] == yr) & (df["book_home"].notna())]

        tvd_yr  = yr_df["tvd"].values
        flip_yr = yr_df["favorite_flip"].values.astype(bool)
        pts_yr  = yr_df[[points_col(a) for a in ALPHAS]].values

        if use_flip:
            mask = flip_yr & (tvd_yr > threshold)
        else:
            mask = tvd_yr > threshold

        pts_lo = pts_yr[:, alpha_idx[alpha_low]]
        pts_hi = pts_yr[:, alpha_idx[alpha_high]]
        rule_pts  = int(np.where(mask, pts_lo, pts_hi).sum())
        fixed_pts = int(yr_df[points_col(BEST_FIXED_ALPHA)].sum())
        triggers  = int(mask.sum())

        print(
            f"  {yr:>6}  {rule_pts:>10}  {fixed_pts:>10}  "
            f"{rule_pts - fixed_pts:>+6}  {triggers:>9}"
        )