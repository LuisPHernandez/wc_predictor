"""
analyse_phase_2.py

Phase 2: Signal Characterization

Checks whether TVD and favorite_flip are genuine signals in WC training data
before running the grid search. Answers:
  - Does the optimal alpha shift systematically with TVD? (Rule B foundation)
  - Does flip add information on top of TVD? (Rule A foundation)
  - How reliable are the cells given sample sizes?

Run from project root:
    py -3 analyse_phase_2.py

Paste full output back for interpretation before Phase 3.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "wc_analysis.csv"
TRAINING_YEARS = [2006, 2010, 2014, 2018]
HOLDOUT_YEAR   = 2022
ALPHAS         = [round(a, 2) for a in np.arange(0.0, 1.05, 0.05)]

# Baseline established in Phase 1
BEST_FIXED_ALPHA = 0.15
BEST_FIXED_PTS   = 249   # over 2006-2018

def alpha_tag(a):
    return str(a).replace(".", "")

def points_col(a):
    return f"blend{alpha_tag(a)}_points"

SEP  = "=" * 65
SEP2 = "-" * 65

# ---------------------------------------------------------------------------
# Load + filter to training years with odds
# ---------------------------------------------------------------------------

df    = pd.read_csv(CSV_PATH)
train = df[df["year"].isin(TRAINING_YEARS) & df["book_home"].notna()].copy()

print(f"Training games with odds: {len(train)}")
print(f"(Baseline: best fixed alpha = {BEST_FIXED_ALPHA}, {BEST_FIXED_PTS} pts)\n")

# ---------------------------------------------------------------------------
# Helper: best alpha and points for a subset of rows
# ---------------------------------------------------------------------------

def best_alpha_for(subset):
    """Returns (best_alpha, best_pts, pts_dict) for a row subset."""
    if len(subset) == 0:
        return None, None, {}
    pts = {a: subset[points_col(a)].sum() for a in ALPHAS}
    best_a = max(pts, key=pts.get)
    return best_a, pts[best_a], pts

def summarise_cell(subset, label, baseline_pts_in_subset):
    """
    Prints a summary line for a cell.
    baseline_pts_in_subset = points the best fixed alpha (0.15) scores
    on this exact subset — used to compute improvement.
    """
    n = len(subset)
    if n == 0:
        print(f"  {label:35s}: n=0  (no data)")
        return

    best_a, best_p, pts = best_alpha_for(subset)
    base_p = subset[points_col(BEST_FIXED_ALPHA)].sum()
    diff   = best_p - base_p

    # Spread across years
    year_counts = subset["year"].value_counts().sort_index().to_dict()
    year_str    = "  ".join(f"{yr}:{cnt}" for yr, cnt in year_counts.items())

    print(
        f"  {label:35s}: n={n:3d} | "
        f"best α={best_a:.2f} ({best_p:.0f} pts) | "
        f"vs fixed α={BEST_FIXED_ALPHA} → {diff:+.0f} | "
        f"years: {year_str}"
    )

# ===========================================================================
# RULE B SIGNAL: TVD buckets
# ===========================================================================

print(SEP)
print("RULE B SIGNAL: TVD BUCKETS")
print("Does the optimal alpha shift with TVD magnitude?")
print(SEP)

tvd_bins   = [0, 0.08, 0.12, 0.16, 0.20, 1.0]
tvd_labels = ["0–0.08", "0.08–0.12", "0.12–0.16", "0.16–0.20", "0.20+"]
train["tvd_bucket"] = pd.cut(train["tvd"], bins=tvd_bins, labels=tvd_labels)

print("\nBest alpha and points improvement over fixed baseline per TVD bucket:\n")
for bucket in tvd_labels:
    subset = train[train["tvd_bucket"] == bucket]
    summarise_cell(subset, f"TVD {bucket}", None)

# Alpha curve per TVD bucket (condensed — show every 0.10)
print("\n\nAlpha curves per TVD bucket (points at key alphas):\n")
key_alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
header = f"  {'Bucket':15s}" + "".join(f"  {a:.2f}" for a in key_alphas)
print(header)
print("  " + "-" * (len(header) - 2))

for bucket in tvd_labels:
    subset = train[train["tvd_bucket"] == bucket]
    if len(subset) == 0:
        continue
    row = f"  {bucket:15s}"
    for a in key_alphas:
        pts = subset[points_col(a)].sum()
        row += f"  {pts:4.0f}"
    print(row)

# ===========================================================================
# RULE A SIGNAL: 2D breakdown — TVD bucket x flip
# ===========================================================================

print(f"\n{SEP}")
print("RULE A SIGNAL: TVD BUCKET x FAVORITE FLIP")
print("Does flip add information on top of TVD alone?")
print(SEP)

print("\n--- Cell sizes (n games per cell) ---\n")
print(f"  {'TVD bucket':15s}  {'flip=False':>12}  {'flip=True':>12}  {'total':>8}")
print("  " + "-" * 55)
for bucket in tvd_labels:
    sub      = train[train["tvd_bucket"] == bucket]
    n_false  = (sub["favorite_flip"] == False).sum()
    n_true   = (sub["favorite_flip"] == True).sum()
    n_total  = len(sub)
    flag = "  <<< THIN" if min(n_false, n_true) < 10 else ""
    print(f"  {bucket:15s}  {n_false:>12}  {n_true:>12}  {n_total:>8}{flag}")

print("\n--- Best alpha per cell ---\n")
print(f"  {'TVD bucket':15s}  {'Flip':>6}  {'n':>4}  {'best α':>7}  {'best pts':>9}  {'vs fixed α':>11}  years")
print("  " + "-" * 80)

for bucket in tvd_labels:
    for flip_val in [False, True]:
        subset = train[
            (train["tvd_bucket"] == bucket) &
            (train["favorite_flip"] == flip_val)
        ]
        n = len(subset)
        if n == 0:
            continue

        best_a, best_p, _ = best_alpha_for(subset)
        base_p = subset[points_col(BEST_FIXED_ALPHA)].sum()
        diff   = best_p - base_p
        year_counts = subset["year"].value_counts().sort_index().to_dict()
        year_str = "  ".join(f"{yr}:{cnt}" for yr, cnt in year_counts.items())
        thin_flag = "  <<< THIN" if n < 15 else ""

        print(
            f"  {bucket:15s}  {str(flip_val):>6}  {n:>4}  "
            f"{best_a:>7.2f}  {best_p:>9.0f}  {diff:>+11.0f}  "
            f"{year_str}{thin_flag}"
        )

# ===========================================================================
# KEY COMPARISON: flip=True high-TVD vs flip=False high-TVD
# ===========================================================================

print(f"\n{SEP}")
print("KEY QUESTION: Does flip separate the high-TVD bucket?")
print("(flip=True,TVD>0.12 vs flip=False,TVD>0.12)")
print(SEP)

for threshold in [0.10, 0.12, 0.14, 0.16]:
    high_tvd    = train[train["tvd"] > threshold]
    flip_true   = high_tvd[high_tvd["favorite_flip"] == True]
    flip_false  = high_tvd[high_tvd["favorite_flip"] == False]

    best_a_t, best_p_t, _ = best_alpha_for(flip_true)
    best_a_f, best_p_f, _ = best_alpha_for(flip_false)

    base_t = flip_true[points_col(BEST_FIXED_ALPHA)].sum()  if len(flip_true)  else 0
    base_f = flip_false[points_col(BEST_FIXED_ALPHA)].sum() if len(flip_false) else 0

    print(f"\n  TVD > {threshold:.2f}:")
    print(
        f"    flip=True  (n={len(flip_true):3d}): "
        f"best α={best_a_t}  ({best_p_t:.0f} pts)  "
        f"vs fixed → {best_p_t - base_t:+.0f}"
        if len(flip_true) else
        f"    flip=True  (n=0): no data"
    )
    print(
        f"    flip=False (n={len(flip_false):3d}): "
        f"best α={best_a_f}  ({best_p_f:.0f} pts)  "
        f"vs fixed → {best_p_f - base_f:+.0f}"
        if len(flip_false) else
        f"    flip=False (n=0): no data"
    )
    print(
        f"    → Same optimal alpha? {'YES' if best_a_t == best_a_f else 'NO — flip separates them'}"
    )

# ===========================================================================
# ADDITIONAL: flip=True breakdown by year
# ===========================================================================

print(f"\n{SEP}")
print("FLIP=TRUE GAMES: breakdown by year and TVD")
print(SEP)

flips = train[train["favorite_flip"] == True].copy()
print(f"\nTotal flip=True games in training: {len(flips)}")
print("\nFlip games by year and TVD bucket:\n")
print(
    pd.crosstab(flips["year"], flips["tvd_bucket"])
    .to_string()
)

print("\nBest alpha for flip=True games overall:")
summarise_cell(flips, "All flip=True", None)

print("\nBest alpha for flip=True, TVD > 0.10:")
summarise_cell(flips[flips["tvd"] > 0.10], "flip=True, TVD>0.10", None)

print("\nBest alpha for flip=True, TVD > 0.12:")
summarise_cell(flips[flips["tvd"] > 0.12], "flip=True, TVD>0.12", None)

# ===========================================================================
# DECISION PREVIEW: is there a signal worth grid-searching?
# ===========================================================================

print(f"\n{SEP}")
print("SIGNAL SUMMARY — READ BEFORE PROCEEDING TO PHASE 3")
print(SEP)

# Does the alpha curve slope downward as TVD increases?
print("\nOptimal alpha by TVD bucket (Rule B foundation):")
for bucket in tvd_labels:
    subset = train[train["tvd_bucket"] == bucket]
    if len(subset) == 0:
        continue
    best_a, best_p, _ = best_alpha_for(subset)
    print(f"  TVD {bucket:10s} (n={len(subset):3d}): best α = {best_a:.2f}")

print("\nIf optimal alpha decreases as TVD increases → Rule B signal exists.")
print("If optimal alpha is flat across buckets → TVD is not a useful splitter.")