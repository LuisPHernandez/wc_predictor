"""
live_monitor_2026.py

Live goal rate monitor for the 2026 World Cup.

Tracks the running empirical k (actual goals / model predicted goals)
as results come in, and recommends whether to adjust goal_inflation
before locking the next set of predictions.

Usage
-----
1. Before the tournament: run once to verify setup and see pre-tournament state.

2. After each matchday: add results to COMPLETED_RESULTS below and re-run.
   The script will update the running k estimate and print recommendations.

3. Before locking predictions for upcoming games: check the recommendation
   section and decide whether to adjust goal_inflation in predict_2026.py.

Data entry format
-----------------
Add each completed match to COMPLETED_RESULTS as a tuple:
    (home_team, away_team, actual_home, actual_away)

Team names must match FIFA names used in the model (same as wc_analysis_k115.csv).
The script will look up lambda_home and lambda_away from the pre-tournament
model predictions stored in wc_analysis_k115.csv (or a 2026 equivalent if
you generate one before the tournament).

Run from project root:
    py -3 live_monitor_2026.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson # pyrefly: ignore [missing-import]

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pre-tournament model predictions CSV
# If you generate a 2026-specific predictions file, point here instead
WC_ANALYSIS_CSV = PROJECT_ROOT / "wc_analysis_k115.csv"

# Current production parameters
CURRENT_K     = 1.15
CURRENT_ALPHA = 0.30

# Historical baseline (from analysis) — modern era 2014-2022
HISTORICAL_K_MODERN = 1.146   # mean empirical k across 2014, 2018, 2022
HISTORICAL_AVG_GOALS = 2.667  # mean actual goals per game 2014-2022

# Minimum games before trusting the running estimate
MIN_GAMES_FOR_SIGNAL  = 20
MIN_GAMES_FOR_ACTION  = 30

# Threshold: if running k deviates from current k by this much, flag it
K_SHIFT_THRESHOLD = 0.07

# ---------------------------------------------------------------------------
# Completed results — fill in as tournament progresses
# Format: (home_team, away_team, actual_home_goals, actual_away_goals)
# ---------------------------------------------------------------------------

COMPLETED_RESULTS = [
    # --- Matchday 1 ---
    # ("United States", "Serbia", 1, 0),   # example — replace with real results
    # ("Mexico", "Uruguay", 0, 1),
    # ("Canada", "Belgium", 2, 2),

    # --- Matchday 2 ---

    # --- Matchday 3 ---

    # --- Round of 32 ---

    # --- Quarter-finals ---

    # --- Semi-finals ---

    # --- Third place / Final ---
]

# ---------------------------------------------------------------------------
# Alternative: load from a simple CSV if you prefer not to edit this file
# Format: home_team, away_team, actual_home, actual_away (no header needed)
# ---------------------------------------------------------------------------

RESULTS_CSV = PROJECT_ROOT / "data" / "live" / "2026_results.csv"

SEP  = "=" * 65
SEP2 = "-" * 45

# ---------------------------------------------------------------------------
# Load pre-tournament lambda predictions
# We use the historical WC analysis CSV as a proxy for lambda values.
# These lambdas come from the Dixon-Coles model fitted before the tournament
# and represent the model's pre-game expected goals for each match.
#
# For 2026: replace this with a freshly generated 2026 predictions CSV
# that stores lambda_home and lambda_away for each scheduled match.
# ---------------------------------------------------------------------------

def load_lambda_lookup(csv_path):
    """
    Loads (home_team, away_team) -> (lambda_home, lambda_away) from CSV.
    Uses the most recent year available as a proxy if 2026 not present.
    """
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found.")
        print("Lambda lookup unavailable — goals-per-game will be used instead.")
        return None

    df = pd.read_csv(csv_path)

    # If 2026 data exists in the file, use it; otherwise use all years
    if "year" in df.columns and 2026 in df["year"].values:
        df = df[df["year"] == 2026]
    elif "year" in df.columns:
        # Use most recent year as structural proxy for lambda scale
        most_recent = df["year"].max()
        df = df[df["year"] == most_recent]
        print(f"Note: Using {most_recent} lambda values as proxy for 2026.")

    lookup = {}
    for _, row in df.iterrows():
        key = (row["home_team"], row["away_team"])
        lookup[key] = (row["lambda_home"], row["lambda_away"])
        # Also store reverse for robustness
        key_rev = (row["away_team"], row["home_team"])
        if key_rev not in lookup:
            lookup[key_rev] = (row["lambda_away"], row["lambda_home"])

    return lookup

# ---------------------------------------------------------------------------
# Load results from CSV if file exists (alternative to hardcoding above)
# ---------------------------------------------------------------------------

def load_results_from_csv(path):
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, header=None,
                         names=["home_team", "away_team", "actual_home", "actual_away"])
        return list(df.itertuples(index=False, name=None))
    except Exception as e:
        print(f"Could not load results CSV: {e}")
        return []

# ---------------------------------------------------------------------------
# Core: compute running empirical k
# ---------------------------------------------------------------------------

def compute_running_k(results, lambda_lookup):
    """
    For each completed result, finds the model's predicted lambdas and
    computes empirical k = actual_total / predicted_total.

    Returns a DataFrame with per-game details and running totals.
    """
    rows = []
    missing_lambda = []

    for result in results:
        home, away, ah, aa = result
        actual_total = ah + aa

        # Look up pre-tournament lambda
        key = (home, away)
        key_rev = (away, home)

        if lambda_lookup and key in lambda_lookup:
            lh, la = lambda_lookup[key]
            lambda_source = "direct"
        elif lambda_lookup and key_rev in lambda_lookup:
            la, lh = lambda_lookup[key_rev]
            lambda_source = "reversed"
        else:
            # Fall back to historical average lambda if not found
            lh, la = HISTORICAL_AVG_GOALS / 2, HISTORICAL_AVG_GOALS / 2
            lambda_source = "fallback"
            missing_lambda.append(f"{home} vs {away}")

        pred_total = lh + la

        rows.append({
            "home_team":     home,
            "away_team":     away,
            "actual_home":   ah,
            "actual_away":   aa,
            "actual_total":  actual_total,
            "lambda_home":   round(lh, 4),
            "lambda_away":   round(la, 4),
            "pred_total":    round(pred_total, 4),
            "game_k":        round(actual_total / pred_total, 4),
            "lambda_source": lambda_source,
        })

    if missing_lambda:
        print(f"\nWARNING: Lambda not found for {len(missing_lambda)} games "
              f"(used fallback):")
        for m in missing_lambda:
            print(f"  {m}")

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# K recommendation logic
# ---------------------------------------------------------------------------

def recommend_k(running_k, n_games, current_k):
    """
    Returns recommended k and rationale based on running estimate.
    """
    if n_games < MIN_GAMES_FOR_SIGNAL:
        return current_k, f"Too few games ({n_games} < {MIN_GAMES_FOR_SIGNAL}) — hold current k"

    deviation = running_k - current_k

    if abs(deviation) < K_SHIFT_THRESHOLD:
        return current_k, f"Running k={running_k:.3f} within tolerance — hold k={current_k}"

    if n_games < MIN_GAMES_FOR_ACTION:
        direction = "below" if deviation < 0 else "above"
        return current_k, (
            f"Signal emerging (running k={running_k:.3f} is {direction} current k={current_k}), "
            f"but only {n_games} games — wait for {MIN_GAMES_FOR_ACTION}+ before acting"
        )

    # Recommend adjustment
    if running_k < current_k - K_SHIFT_THRESHOLD:
        # Tournament running low — step down conservatively
        if running_k < current_k - 0.15:
            rec_k = round(current_k - 0.10, 2)
        else:
            rec_k = round(current_k - 0.05, 2)
        rec_k = max(rec_k, 1.00)
        return rec_k, (
            f"Tournament is LOW-SCORING (running k={running_k:.3f}). "
            f"Recommend reducing to k={rec_k}"
        )
    else:
        # Tournament running high — step up
        if running_k > current_k + 0.15:
            rec_k = round(current_k + 0.10, 2)
        else:
            rec_k = round(current_k + 0.05, 2)
        rec_k = min(rec_k, 1.40)
        return rec_k, (
            f"Tournament is HIGH-SCORING (running k={running_k:.3f}). "
            f"Recommend increasing to k={rec_k}"
        )

# ---------------------------------------------------------------------------
# How many predictions change under a different k?
# ---------------------------------------------------------------------------

def count_prediction_changes(lambda_lookup, remaining_games, old_k, new_k, rho=-0.10):
    """
    Estimates how many upcoming predictions would change if k shifted.
    Uses a simplified best-scoreline check.
    """
    from scipy.stats import poisson # pyrefly: ignore [missing-import]
    from src.scoring import points_for_prediction

    if not remaining_games or lambda_lookup is None:
        return None

    changes = 0
    max_goals = 8

    def tau(i, j, lh, la, rho):
        if   i==0 and j==0: return max(1 - lh*la*rho, 1e-10)
        elif i==1 and j==0: return max(1 + la*rho,    1e-10)
        elif i==0 and j==1: return max(1 + lh*rho,    1e-10)
        elif i==1 and j==1: return max(1 - rho,       1e-10)
        else:                return 1.0

    def best_pred(lh, la):
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i,j] = poisson.pmf(i,lh) * poisson.pmf(j,la) * tau(i,j,lh,la,rho)
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
        return best_ph, best_pa

    for home, away in remaining_games:
        key = (home, away)
        key_rev = (away, home)
        if key in lambda_lookup:
            lh, la = lambda_lookup[key]
        elif key_rev in lambda_lookup:
            la, lh = lambda_lookup[key_rev]
        else:
            continue

        ph_old, pa_old = best_pred(lh * old_k, la * old_k)
        ph_new, pa_new = best_pred(lh * new_k, la * new_k)

        if (ph_old, pa_old) != (ph_new, pa_new):
            changes += 1

    return changes

# ===========================================================================
# MAIN
# ===========================================================================

print(SEP)
print("2026 WORLD CUP — LIVE GOAL RATE MONITOR")
print(SEP)

print(f"\nCurrent production parameters:")
print(f"  goal_inflation (k) : {CURRENT_K}")
print(f"  alpha              : {CURRENT_ALPHA}")
print(f"  Historical baseline: k={HISTORICAL_K_MODERN:.3f} (2014-2022 average)")

# Load lambda lookup
lambda_lookup = load_lambda_lookup(WC_ANALYSIS_CSV)

# Load results — prefer CSV if it exists, else use hardcoded list
csv_results = load_results_from_csv(RESULTS_CSV)
all_results  = csv_results if csv_results else COMPLETED_RESULTS

print(f"\nCompleted results loaded: {len(all_results)}")

if len(all_results) == 0:
    print("\nNo results entered yet.")
    print("Add completed matches to COMPLETED_RESULTS in this script,")
    print(f"or create {RESULTS_CSV} with columns: home_team, away_team, actual_home, actual_away")
    print("\nPre-tournament state:")
    print(f"  Recommended k     : {CURRENT_K} (pre-tournament calibration)")
    print(f"  Recommended alpha : {CURRENT_ALPHA}")
    print(f"\nRe-run after entering results from each matchday.")
    exit(0)

# Compute running stats
games_df = compute_running_k(all_results, lambda_lookup)

n_games         = len(games_df)
total_actual    = games_df["actual_total"].sum()
total_pred      = games_df["pred_total"].sum()
running_k       = total_actual / total_pred
avg_actual      = games_df["actual_total"].mean()
avg_pred        = games_df["pred_total"].mean()

fallback_count  = (games_df["lambda_source"] == "fallback").sum()

# ===========================================================================
# SECTION 1: Running summary
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 1: RUNNING GOAL RATE SUMMARY")
print(SEP)

print(f"\n  Games completed         : {n_games}")
print(f"  Total actual goals      : {int(total_actual)}")
print(f"  Total predicted (λ sum) : {total_pred:.2f}")
print(f"  Avg actual per game     : {avg_actual:.3f}")
print(f"  Avg predicted per game  : {avg_pred:.3f}")
print(f"  Running empirical k     : {running_k:.4f}")
print(f"  Current production k    : {CURRENT_K}")
print(f"  Deviation from current k: {running_k - CURRENT_K:+.4f}")
print(f"  vs historical baseline  : {running_k - HISTORICAL_K_MODERN:+.4f} (baseline={HISTORICAL_K_MODERN})")

if fallback_count > 0:
    print(f"\n  WARNING: {fallback_count} games used fallback lambda "
          f"(team not found in lookup) — k estimate may be slightly off")

# ===========================================================================
# SECTION 2: Per-game breakdown
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 2: PER-GAME BREAKDOWN")
print(SEP)

print(f"\n  {'Home':20s} {'Away':20s} {'Score':>6} {'Pred λ':>8} {'Act':>4} {'Game k':>7}")
print(f"  {'-'*20} {'-'*20} {'------':>6} {'------':>8} {'---':>4} {'------':>7}")

running_actual = 0
running_pred   = 0
for _, row in games_df.iterrows():
    running_actual += row["actual_total"]
    running_pred   += row["pred_total"]
    r_k = running_actual / running_pred
    flag = " ***" if abs(row["game_k"] - CURRENT_K) > 0.40 else ""
    print(
        f"  {row['home_team']:20s} {row['away_team']:20s} "
        f"{int(row['actual_home'])}-{int(row['actual_away']):>1}    "
        f"{row['pred_total']:>8.3f} "
        f"{int(row['actual_total']):>4} "
        f"{row['game_k']:>7.3f}{flag}"
    )

# ===========================================================================
# SECTION 3: Running k over time (simple trend)
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 3: RUNNING k OVER TIME")
print(SEP)

cumulative_actual = games_df["actual_total"].cumsum()
cumulative_pred   = games_df["pred_total"].cumsum()
cumulative_k      = cumulative_actual / cumulative_pred

print(f"\n  {'After game':>11}  {'Running k':>10}  {'Signal strength':>16}")
print(f"  {'-----------':>11}  {'---------':>10}  {'---------------':>16}")

checkpoints = list(range(5, n_games + 1, 5))
if n_games not in checkpoints:
    checkpoints.append(n_games)

for i in checkpoints:
    ck = cumulative_k.iloc[i - 1]
    if i < MIN_GAMES_FOR_SIGNAL:
        strength = "too few games"
    elif i < MIN_GAMES_FOR_ACTION:
        strength = "emerging signal"
    else:
        strength = "actionable"
    print(f"  {i:>11}  {ck:>10.4f}  {strength:>16}")

# ===========================================================================
# SECTION 4: Recommendation
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 4: RECOMMENDATION")
print(SEP)

rec_k, rationale = recommend_k(running_k, n_games, CURRENT_K)

print(f"\n  {rationale}")
print(f"\n  Current k  : {CURRENT_K}")
print(f"  Running k  : {running_k:.4f}")
print(f"  Recommended: {rec_k}")

if rec_k != CURRENT_K:
    print(f"\n  ACTION REQUIRED:")
    print(f"    Change goal_inflation from {CURRENT_K} to {rec_k} in predict_2026.py")
    print(f"    Re-run predictions for all unplayed games before next kickoff")
else:
    print(f"\n  NO ACTION REQUIRED — hold current parameters")

# ===========================================================================
# SECTION 5: Impact of potential k change
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 5: IMPACT OF POTENTIAL k CHANGES")
print("(How sensitive are upcoming predictions to k adjustments?)")
print(SEP)

# Show points impact of different k values on completed games
# (retrospective — how would the completed games have scored?)
print(f"\n  Retrospective: completed games under different k values")
print(f"  (using stored model predictions from wc_analysis_k115.csv)\n")

if lambda_lookup:
    print(f"  {'k value':>8}  {'Avg pred goals':>15}  {'vs current k':>13}")
    print(f"  {'-------':>8}  {'--------------':>15}  {'------------':>13}")

    for test_k in [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
        avg_pred_k = games_df["pred_total"].mean() * (test_k / CURRENT_K)
        diff_str   = f"{avg_pred_k - avg_actual:+.3f} vs actual"
        marker     = " <-- current" if test_k == CURRENT_K else ""
        print(f"  {test_k:>8.2f}  {avg_pred_k:>15.3f}  {diff_str:>13}{marker}")

print(f"\n  Actual average goals per game so far: {avg_actual:.3f}")
print(f"  Historical modern era average        : {HISTORICAL_AVG_GOALS:.3f}")
print(f"  Difference vs historical             : {avg_actual - HISTORICAL_AVG_GOALS:+.3f}")

# ===========================================================================
# SECTION 6: Decision guide
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 6: DECISION GUIDE")
print(SEP)

print(f"""
  Running k interpretation:
    > {CURRENT_K + K_SHIFT_THRESHOLD:.2f}  : Tournament running HOT  → consider increasing k
    {CURRENT_K - K_SHIFT_THRESHOLD:.2f} – {CURRENT_K + K_SHIFT_THRESHOLD:.2f} : Within tolerance      → hold k={CURRENT_K}
    < {CURRENT_K - K_SHIFT_THRESHOLD:.2f}  : Tournament running COLD → consider decreasing k

  Confidence by games played:
    < {MIN_GAMES_FOR_SIGNAL} games : Signal unreliable — do not act
    {MIN_GAMES_FOR_SIGNAL}–{MIN_GAMES_FOR_ACTION} games : Emerging signal — monitor closely
    > {MIN_GAMES_FOR_ACTION} games : Actionable — adjust if threshold breached

  Conservative adjustment steps:
    Running k suggests lower : try k={CURRENT_K - 0.05:.2f} first, then {CURRENT_K - 0.10:.2f} if sustained
    Running k suggests higher: try k={CURRENT_K + 0.05:.2f} first, then {CURRENT_K + 0.10:.2f} if sustained

  Never go below k=1.00 (model already calibrated on modern era data)
  Never go above k=1.30 (outside the range validated by historical analysis)

  Remember: each k adjustment should be applied to ALL remaining unplayed
  games before their kickoff. Check which predictions change and verify
  they make intuitive sense before submitting.
""")