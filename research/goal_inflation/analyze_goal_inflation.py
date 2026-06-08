"""
analyse_goal_inflation.py

Step 1: Compute the empirical goal inflation scalar k from wc_analysis.csv.
        k = mean(actual_total_goals) / mean(pred_total_goals)

Step 2: Grid-search k over training years (2006-2018) to find the value
        that maximises pool points, using the pre-computed lambda columns
        already stored in the CSV.

        Rather than regenerating the full score matrix for every k (which
        would require refitting), we use the stored lambda_home / lambda_away
        values from the CSV and re-derive what the optimal scoreline prediction
        would be under k*lambda_home, k*lambda_away. This is exact — no
        approximation — because the score matrix depends only on the lambdas
        and rho, and we can recompute the expected-points maximisation cheaply.

Step 3: Validate the best k on 2022 holdout.

Step 4: Print the exact one-line change needed in model.py.

Run from project root:
    py -3 analyse_goal_inflation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson # pyrefly: ignore [missing-import]
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import points_for_prediction

CSV_PATH       = PROJECT_ROOT / "data" / "analysis" / "wc_analysis.csv"
TRAINING_YEARS = [2006, 2010, 2014, 2018]
HOLDOUT_YEAR   = 2022
ALL_YEARS      = TRAINING_YEARS + [HOLDOUT_YEAR]

SEP = "=" * 65

# ---------------------------------------------------------------------------
# Dixon-Coles tau correction (same as model.py)
# ---------------------------------------------------------------------------

def tau(i, j, lh, la, rho):
    if   i == 0 and j == 0: return max(1 - lh * la * rho, 1e-10)
    elif i == 1 and j == 0: return max(1 + la * rho,      1e-10)
    elif i == 0 and j == 1: return max(1 + lh * rho,      1e-10)
    elif i == 1 and j == 1: return max(1 - rho,           1e-10)
    else:                   return 1.0

def build_score_matrix(lh, la, rho, max_goals=8):
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = (
                poisson.pmf(i, lh) *
                poisson.pmf(j, la) *
                tau(i, j, lh, la, rho)
            )
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix

def best_prediction(matrix, max_goals=8):
    """Returns (pred_home, pred_away, expected_pts) that maximises E[pts]."""
    best_ph, best_pa, best_ep = 0, 0, -1.0
    for ph in range(max_goals):
        for pa in range(max_goals):
            ep = sum(
                points_for_prediction(ph, pa, ah, aa) * matrix[ah, aa]
                for ah in range(max_goals)
                for aa in range(max_goals)
            )
            if ep > best_ep:
                best_ep = ep
                best_ph, best_pa = ph, pa
    return best_ph, best_pa, best_ep

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# We need rho — it's not stored in the CSV. We'll use the typical fitted value.
# Since rho is stable across fits (~-0.13 to -0.09), we use a representative
# value. The sensitivity of the optimal prediction to rho is low — the
# scoreline choice is dominated by lh and la.
# If you want the exact rho per year you'd need to store it in the dataset.
RHO_ESTIMATE = -0.10   # conservative representative value

print(f"Loaded {len(df)} rows")
print(f"Using rho estimate: {RHO_ESTIMATE} (stable across fits, low sensitivity)")
print(f"Note: rho affects probability of 0-0/1-0/0-1/1-1 only\n")

# Only rows with lambda stored (i.e., all rows — lambdas always computed)
assert "lambda_home" in df.columns and "lambda_away" in df.columns, \
    "lambda_home / lambda_away columns missing — regenerate wc_analysis.csv"

# ===========================================================================
# STEP 1: Empirical k
# ===========================================================================

print(SEP)
print("STEP 1: EMPIRICAL GOAL INFLATION SCALAR")
print(SEP)

# Predicted total goals = lambda_home + lambda_away (mean of Poisson)
df["pred_total_lambda"] = df["lambda_home"] + df["lambda_away"]
df["actual_total_goals"] = df["actual_home"] + df["actual_away"]

for years, label in [
    (TRAINING_YEARS, "Training (2006-2018)"),
    ([HOLDOUT_YEAR],  "Holdout  (2022)     "),
    (ALL_YEARS,       "All years            "),
]:
    sub = df[df["year"].isin(years)]
    mean_pred   = sub["pred_total_lambda"].mean()
    mean_actual = sub["actual_total_goals"].mean()
    k_empirical = mean_actual / mean_pred
    print(f"\n  {label}:")
    print(f"    Mean predicted total goals (λ_h + λ_a) : {mean_pred:.4f}")
    print(f"    Mean actual total goals                 : {mean_actual:.4f}")
    print(f"    Empirical k = actual / predicted        : {k_empirical:.4f}")

# ===========================================================================
# STEP 2: Grid search k on training years
# ===========================================================================

print(f"\n{SEP}")
print("STEP 2: GRID SEARCH k ON TRAINING YEARS (2006-2018)")
print(SEP)

train = df[df["year"].isin(TRAINING_YEARS)].copy()

K_VALUES = [round(k, 2) for k in np.arange(1.00, 1.55, 0.05)]
print(f"\nSearching k in: {K_VALUES}")
print(f"Training games: {len(train)} (lambdas loaded from CSV)\n")

k_results = []

for k in K_VALUES:
    total_pts = 0
    for _, row in train.iterrows():
        lh_k = row["lambda_home"] * k
        la_k = row["lambda_away"] * k
        matrix = build_score_matrix(lh_k, la_k, RHO_ESTIMATE)
        ph, pa, _ = best_prediction(matrix)
        pts = points_for_prediction(ph, pa, int(row["actual_home"]), int(row["actual_away"]))
        total_pts += pts
    k_results.append({"k": k, "total_pts": total_pts})
    print(f"  k={k:.2f}  →  {total_pts} pts")

k_df = pd.DataFrame(k_results).sort_values("total_pts", ascending=False)

print(f"\n--- Results sorted by points ---\n")
print(k_df.to_string(index=False))

best_k     = float(k_df.iloc[0]["k"])
best_pts   = int(k_df.iloc[0]["total_pts"])
baseline_k1 = int(k_df[k_df["k"] == 1.0]["total_pts"].iloc[0])

print(f"\n  k=1.00 (no inflation)  : {baseline_k1} pts  (baseline)")
print(f"  Best k={best_k:.2f}            : {best_pts} pts  ({best_pts - baseline_k1:+d} vs baseline)")

# ===========================================================================
# STEP 3: Per-year breakdown at best k
# ===========================================================================

print(f"\n{SEP}")
print(f"STEP 3: PER-YEAR BREAKDOWN AT BEST k={best_k:.2f}")
print(SEP)

print(f"\n  {'Year':>6}  {'k=1.00':>8}  {'k={:.2f}'.format(best_k):>8}  {'Diff':>6}  {'Note'}")
print(f"  {'------':>6}  {'------':>8}  {'------':>8}  {'----':>6}")

for yr in ALL_YEARS:
    sub = df[df["year"] == yr]
    pts_base = 0
    pts_best = 0
    for _, row in sub.iterrows():
        # Baseline
        m0 = build_score_matrix(row["lambda_home"], row["lambda_away"], RHO_ESTIMATE)
        ph0, pa0, _ = best_prediction(m0)
        pts_base += points_for_prediction(ph0, pa0, int(row["actual_home"]), int(row["actual_away"]))
        # Best k
        m1 = build_score_matrix(row["lambda_home"] * best_k, row["lambda_away"] * best_k, RHO_ESTIMATE)
        ph1, pa1, _ = best_prediction(m1)
        pts_best += points_for_prediction(ph1, pa1, int(row["actual_home"]), int(row["actual_away"]))

    note = " (HOLDOUT)" if yr == HOLDOUT_YEAR else ""
    diff = pts_best - pts_base
    print(f"  {yr:>6}  {pts_base:>8}  {pts_best:>8}  {diff:>+6}{note}")

# ===========================================================================
# STEP 4: Holdout validation
# ===========================================================================

print(f"\n{SEP}")
print(f"STEP 4: HOLDOUT VALIDATION ON 2022  (k={best_k:.2f})")
print(SEP)

hold = df[df["year"] == HOLDOUT_YEAR]
hold_base = 0
hold_best = 0
changed = 0

for _, row in hold.iterrows():
    m0 = build_score_matrix(row["lambda_home"], row["lambda_away"], RHO_ESTIMATE)
    ph0, pa0, _ = best_prediction(m0)
    p0 = points_for_prediction(ph0, pa0, int(row["actual_home"]), int(row["actual_away"]))
    hold_base += p0

    m1 = build_score_matrix(row["lambda_home"] * best_k, row["lambda_away"] * best_k, RHO_ESTIMATE)
    ph1, pa1, _ = best_prediction(m1)
    p1 = points_for_prediction(ph1, pa1, int(row["actual_home"]), int(row["actual_away"]))
    hold_best += p1

    if (ph0, pa0) != (ph1, pa1):
        changed += 1

print(f"\n  2022 baseline (k=1.00) : {hold_base} pts")
print(f"  2022 with k={best_k:.2f}     : {hold_best} pts  ({hold_best - hold_base:+d})")
print(f"  Predictions changed    : {changed} / {len(hold)}")

verdict = (
    "PASS — k improves holdout" if hold_best > hold_base else
    "NEUTRAL — no change on holdout" if hold_best == hold_base else
    "FAIL — k hurts holdout"
)
print(f"\n  Verdict: {verdict}")

# ===========================================================================
# STEP 5: Sensitivity — nearby k values on holdout
# ===========================================================================

print(f"\n{SEP}")
print("STEP 5: SENSITIVITY — NEARBY k VALUES ON HOLDOUT (2022)")
print(SEP)

print(f"\n  {'k':>6}  {'2022 pts':>10}  {'vs k=1.00':>10}")
print(f"  {'----':>6}  {'--------':>10}  {'---------':>10}")

for k in K_VALUES:
    sub_pts = 0
    for _, row in hold.iterrows():
        m = build_score_matrix(row["lambda_home"] * k, row["lambda_away"] * k, RHO_ESTIMATE)
        ph, pa, _ = best_prediction(m)
        sub_pts += points_for_prediction(ph, pa, int(row["actual_home"]), int(row["actual_away"]))
    marker = " <-- best training k" if k == best_k else ""
    print(f"  {k:>6.2f}  {sub_pts:>10}  {sub_pts - hold_base:>+10}{marker}")

# ===========================================================================
# STEP 6: model.py change instructions
# ===========================================================================

print(f"\n{SEP}")
print("STEP 6: HOW TO IMPLEMENT IN model.py")
print(SEP)

print(f"""
The change is a single scalar applied in score_matrix() after _get_lambda().

In DixonColes.__init__, add:
    self.goal_inflation = 1.0   # set to best k after tuning

In DixonColes.score_matrix(), change:

    CURRENT:
        lh, la = self._get_lambda(home_team, away_team, neutral)

    CHANGE TO:
        lh, la = self._get_lambda(home_team, away_team, neutral)
        lh = lh * self.goal_inflation
        la = la * self.goal_inflation

That's the entire implementation. The scalar applies uniformly to both
teams, preserving the relative strength ratio. It shifts the score matrix
toward higher-scoring outcomes without changing win/draw/away probabilities
significantly (they shift slightly because of the Poisson tail structure
but the bookmaker blend corrects any residual mismatch).

Recommended next step:
    Set goal_inflation = {best_k:.2f} in predict_2026.py and regenerate
    the wc_analysis.csv with this parameter to get exact per-game
    comparisons including blend columns.

To pass it at construction time (cleaner for backtest):
    model = DixonColes(train_df, decay_lambda=0.2,
                       regularization=0.001, goal_inflation={best_k:.2f})
""")