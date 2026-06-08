"""
analyse_goal_trends.py

Checks whether World Cup goal totals are trending upward over time,
and whether 2022 is an outlier or part of a structural shift.

Also computes the optimal k per year independently, to see if the
best k is increasing over time — which would justify optimizing for
recent years rather than all years combined.

Run from project root:
    py -3 analyse_goal_trends.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson, linregress # pyrefly: ignore [missing-import]

from src.scoring import points_for_prediction

CSV_PATH  = Path(__file__).resolve().parent / "wc_analysis.csv"
KAGGLE    = Path(__file__).resolve().parent / "data" / "kaggle" / "results.csv"

SEP = "=" * 65

df = pd.read_csv(CSV_PATH)
df["pred_total_lambda"]  = df["lambda_home"] + df["lambda_away"]
df["actual_total_goals"] = df["actual_home"] + df["actual_away"]

# ===========================================================================
# SECTION 1: Historical WC goal averages from Kaggle (all WCs since 1966)
# ===========================================================================

print(SEP)
print("SECTION 1: HISTORICAL WC GOALS PER GAME (Kaggle, all years)")
print(SEP)

kaggle = pd.read_csv(KAGGLE)
kaggle["date"] = pd.to_datetime(kaggle["date"])
wc = kaggle[kaggle["tournament"] == "FIFA World Cup"].copy()
wc["year"] = wc["date"].dt.year
wc["total_goals"] = wc["home_score"] + wc["away_score"]
wc = wc.dropna(subset=["home_score", "away_score"])

wc_summary = (
    wc.groupby("year")
    .agg(
        games=("total_goals", "count"),
        total_goals=("total_goals", "sum"),
        avg_goals=("total_goals", "mean"),
    )
    .reset_index()
)

print(f"\n  {'Year':>6}  {'Games':>6}  {'Total goals':>12}  {'Avg per game':>13}")
print(f"  {'------':>6}  {'-----':>6}  {'------------':>12}  {'-------------':>13}")
for _, row in wc_summary.iterrows():
    print(f"  {int(row['year']):>6}  {int(row['games']):>6}  "
          f"{int(row['total_goals']):>12}  {row['avg_goals']:>13.3f}")

# Trend test on years >= 1986 (modern era, consistent 52-64 game formats)
modern = wc_summary[wc_summary["year"] >= 1986]
slope, intercept, r, p, se = linregress(modern["year"], modern["avg_goals"])
print(f"\n  Linear trend (1986 onwards):")
print(f"    Slope     : {slope:+.4f} goals/year")
print(f"    R²        : {r**2:.4f}")
print(f"    p-value   : {p:.4f}")
print(f"    Interpretation: ", end="")
if p < 0.05:
    direction = "INCREASING" if slope > 0 else "DECREASING"
    print(f"statistically significant {direction} trend")
else:
    print(f"no statistically significant trend (p={p:.3f})")

# ===========================================================================
# SECTION 2: Model predicted vs actual per year (from wc_analysis.csv)
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 2: MODEL PREDICTED vs ACTUAL GOALS PER YEAR")
print(SEP)

print(f"\n  {'Year':>6}  {'Games':>6}  {'Pred λ avg':>11}  {'Actual avg':>11}  {'Error':>7}  {'k=actual/pred':>14}")
print(f"  {'------':>6}  {'------':>6}  {'-----------':>11}  {'-----------':>11}  {'-------':>7}  {'--------------':>14}")

k_by_year = {}
for yr in sorted(df["year"].unique()):
    sub = df[df["year"] == yr]
    pred_avg   = sub["pred_total_lambda"].mean()
    actual_avg = sub["actual_total_goals"].mean()
    error      = actual_avg - pred_avg
    k          = actual_avg / pred_avg
    k_by_year[yr] = k
    note = " (HOLDOUT)" if yr == 2022 else ""
    print(f"  {yr:>6}  {len(sub):>6}  {pred_avg:>11.4f}  {actual_avg:>11.4f}  "
          f"{error:>+7.4f}  {k:>14.4f}{note}")

# Trend in k over time
years_list = sorted(k_by_year.keys())
k_list     = [k_by_year[yr] for yr in years_list]
slope_k, intercept_k, r_k, p_k, _ = linregress(years_list, k_list)
print(f"\n  Trend in empirical k over time:")
print(f"    Slope  : {slope_k:+.5f} per year")
print(f"    R²     : {r_k**2:.4f}")
print(f"    p-value: {p_k:.4f}")

# ===========================================================================
# SECTION 3: Optimal k per year independently
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 3: OPTIMAL k PER YEAR (independently tuned)")
print(SEP)

def tau(i, j, lh, la, rho=-0.10):
    if   i == 0 and j == 0: return max(1 - lh * la * rho, 1e-10)
    elif i == 1 and j == 0: return max(1 + la * rho,      1e-10)
    elif i == 0 and j == 1: return max(1 + lh * rho,      1e-10)
    elif i == 1 and j == 1: return max(1 - rho,           1e-10)
    else:                   return 1.0

def score_k(subset, k, rho=-0.10, max_goals=8):
    total = 0
    for _, row in subset.iterrows():
        lh = row["lambda_home"] * k
        la = row["lambda_away"] * k
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la) * tau(i, j, lh, la, rho)
        matrix /= matrix.sum()
        best_ep, best_ph, best_pa = -1.0, 0, 0
        for ph in range(max_goals):
            for pa in range(max_goals):
                ep = sum(
                    points_for_prediction(ph, pa, ah, aa) * matrix[ah, aa]
                    for ah in range(max_goals)
                    for aa in range(max_goals)
                )
                if ep > best_ep:
                    best_ep, best_ph, best_pa = ep, ph, pa
        total += points_for_prediction(best_ph, best_pa,
                                       int(row["actual_home"]), int(row["actual_away"]))
    return total

K_VALUES = [round(k, 2) for k in np.arange(1.00, 1.55, 0.05)]

print(f"\n  {'Year':>6}  {'Best k':>8}  {'Pts at best k':>14}  {'Pts at k=1.00':>14}  {'Gain':>6}  {'Empirical k':>12}")
print(f"  {'------':>6}  {'------':>8}  {'--------------':>14}  {'--------------':>14}  {'------':>6}  {'------------':>12}")

best_k_by_year = {}
for yr in sorted(df["year"].unique()):
    sub = df[df["year"] == yr]
    baseline = score_k(sub, 1.0)
    best_pts, best_k = baseline, 1.0
    for k in K_VALUES[1:]:   # skip 1.0 already computed
        pts = score_k(sub, k)
        if pts > best_pts:
            best_pts = pts
            best_k   = k
    best_k_by_year[yr] = best_k
    gain = best_pts - baseline
    note = " (HOLDOUT)" if yr == 2022 else ""
    print(f"  {yr:>6}  {best_k:>8.2f}  {best_pts:>14}  {baseline:>14}  "
          f"{gain:>+6}  {k_by_year[yr]:>12.4f}{note}")

# Trend in best k
best_k_list = [best_k_by_year[yr] for yr in years_list]
slope_bk, _, r_bk, p_bk, _ = linregress(years_list, best_k_list)
print(f"\n  Trend in best k over time:")
print(f"    Slope  : {slope_bk:+.5f} per year")
print(f"    R²     : {r_bk**2:.4f}")
print(f"    p-value: {p_bk:.4f}")

# ===========================================================================
# SECTION 4: The key question — optimize for all years or recent years?
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 4: OPTIMIZE FOR ALL YEARS vs RECENT YEARS ONLY")
print(SEP)

# Strategy A: k tuned on all 5 years
# Strategy B: k tuned on 2014+2018 only (most recent pre-holdout)
# Strategy C: k tuned on 2010+2014+2018 only
# Applied to 2022 holdout in each case

strategies = {
    "All training (2006-2018)": [2006, 2010, 2014, 2018],
    "Recent training (2010-2018)": [2010, 2014, 2018],
    "Most recent (2014-2018)": [2014, 2018],
}

hold = df[df["year"] == 2022]
hold_base = score_k(hold, 1.0)

print(f"\n  2022 holdout baseline (k=1.00): {hold_base} pts\n")
print(f"  {'Strategy':35s}  {'Best k':>7}  {'Train pts':>10}  {'2022 pts':>9}  {'2022 gain':>10}")
print(f"  {'--------':35s}  {'------':>7}  {'---------':>10}  {'--------':>9}  {'---------':>10}")

for label, train_years in strategies.items():
    train_sub = df[df["year"].isin(train_years)]
    baseline_train = score_k(train_sub, 1.0)
    best_pts_train, best_k_strat = baseline_train, 1.0
    for k in K_VALUES[1:]:
        pts = score_k(train_sub, k)
        if pts > best_pts_train:
            best_pts_train = pts
            best_k_strat   = k
    hold_pts = score_k(hold, best_k_strat)
    print(f"  {label:35s}  {best_k_strat:>7.2f}  {best_pts_train:>10}  "
          f"{hold_pts:>9}  {hold_pts - hold_base:>+10}")

# Also show the full 2022 k curve for reference
print(f"\n  Full 2022 curve (for reference):")
print(f"  {'k':>6}  {'2022 pts':>10}  {'vs k=1.00':>10}")
for k in K_VALUES:
    pts = score_k(hold, k)
    print(f"  {k:>6.2f}  {pts:>10}  {pts - hold_base:>+10}")

# ===========================================================================
# SECTION 5: Verdict
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 5: VERDICT")
print(SEP)

print(f"""
Key questions to answer before deciding on k for 2026:

1. Is there a statistically significant upward trend in WC goals per game?
   (see Section 1 — if yes, 2022 is not an outlier, it's the new normal)

2. Is the optimal k per year increasing over time?
   (see Section 3 — if best k is 1.0 in 2006 but 1.2 in 2022, recent tuning wins)

3. Does a k tuned on recent years only validate better on 2022?
   (see Section 4 — if recent-tuned k beats all-years k on 2022, use recent years)

Decision framework:
  - If goals are trending up AND best k increases over time:
      Use k tuned on 2014+2018 only. 2026 will likely resemble recent WCs.
  - If goals are flat/noisy AND best k is inconsistent across years:
      k=1.0 is the honest answer. The inflation is noise, not signal.
  - If 2022 is a genuine outlier (goals much higher than trend):
      Use k=1.0 but note that 2026 is unpredictable in this dimension.
""")