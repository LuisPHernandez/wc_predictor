"""
analyse_winning_margin_bias.py

Checks whether the model systematically underpredicts the winning margin
when it correctly identifies the winner.

If the model predicts the right winner but consistently undershoots the
margin (e.g. predicts 1-0 when the actual is 2-1 or 3-1), that is a
structural lambda calibration bias worth fixing.

If the errors are centered around zero, it is variance and no fix helps
in expectation.

Run from project root:
    py -3 analyse_winning_margin_bias.py

Paste full output back for interpretation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "wc_analysis.csv"
ALL_YEARS      = [2006, 2010, 2014, 2018, 2022]
TRAINING_YEARS = [2006, 2010, 2014, 2018]

SEP  = "=" * 65
SEP2 = "-" * 65

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows\n")

# Use the pure model prediction (alpha=1.0) for this analysis —
# we want to understand the model's own bias, not the blended prediction
df["pred_home"] = df["model_pred_home"]
df["pred_away"] = df["model_pred_away"]

# ---------------------------------------------------------------------------
# Classify each game
# ---------------------------------------------------------------------------

def outcome(h, a):
    if h > a:   return "home"
    elif a > h: return "away"
    else:       return "draw"

df["pred_outcome"]   = df.apply(lambda r: outcome(r["pred_home"],   r["pred_away"]),   axis=1)
df["actual_outcome"] = df.apply(lambda r: outcome(r["actual_home"], r["actual_away"]), axis=1)
df["correct_winner"] = df["pred_outcome"] == df["actual_outcome"]

# Winning margin (goals difference, signed from winner's perspective)
# For a home win: actual margin = actual_home - actual_away
# For an away win: actual margin = actual_away - actual_home
# We want: predicted margin vs actual margin, both from winner's POV

def winner_margin(home, away):
    """Returns goals for winner minus goals for loser. 0 for draws."""
    if home > away:   return home - away
    elif away > home: return away - home
    else:             return 0

def winner_goals(home, away):
    """Returns goals scored by the winning team. 0 for draws."""
    return max(home, away) if home != away else 0

def loser_goals(home, away):
    """Returns goals scored by the losing team. 0 for draws."""
    return min(home, away) if home != away else 0

df["pred_margin"]   = df.apply(lambda r: winner_margin(r["pred_home"],   r["pred_away"]),   axis=1)
df["actual_margin"] = df.apply(lambda r: winner_margin(r["actual_home"], r["actual_away"]), axis=1)
df["margin_error"]  = df["pred_margin"] - df["actual_margin"]   # negative = underpredicted margin

df["pred_winner_goals"]   = df.apply(lambda r: winner_goals(r["pred_home"],   r["pred_away"]),   axis=1)
df["actual_winner_goals"] = df.apply(lambda r: winner_goals(r["actual_home"], r["actual_away"]), axis=1)
df["winner_goals_error"]  = df["pred_winner_goals"] - df["actual_winner_goals"]

df["pred_loser_goals"]    = df.apply(lambda r: loser_goals(r["pred_home"],   r["pred_away"]),    axis=1)
df["actual_loser_goals"]  = df.apply(lambda r: loser_goals(r["actual_home"], r["actual_away"]),  axis=1)
df["loser_goals_error"]   = df["pred_loser_goals"] - df["actual_loser_goals"]

# Total goals
df["pred_total_goals"]   = df["pred_home"]   + df["pred_away"]
df["actual_total_goals"] = df["actual_home"] + df["actual_away"]
df["total_goals_error"]  = df["pred_total_goals"] - df["actual_total_goals"]

# ===========================================================================
# SECTION 1: Overall bias — all games, correct winner only
# ===========================================================================

print(SEP)
print("SECTION 1: WINNING MARGIN BIAS (correct winner predictions only)")
print(SEP)

correct = df[df["correct_winner"] & (df["actual_outcome"] != "draw")].copy()
print(f"\nGames where model predicted correct winner (excl. draws): {len(correct)} / {len(df)}")
print(f"  of which training years (2006-2018): {len(correct[correct['year'].isin(TRAINING_YEARS)])}")
print(f"  of which holdout (2022)            : {len(correct[correct['year'] == 2022])}")

print(f"\n--- Margin error (pred_margin - actual_margin) ---")
print(f"  Negative = model underpredicted margin (scored fewer goals than actual)")
print(f"  Positive = model overpredicted margin\n")

for years, label in [
    (ALL_YEARS,      "All years    "),
    (TRAINING_YEARS, "Training only"),
    ([2022],         "2022 holdout "),
]:
    sub = correct[correct["year"].isin(years)]
    if len(sub) == 0:
        continue
    me = sub["margin_error"]
    print(f"  {label} (n={len(sub):3d}): "
          f"mean={me.mean():+.3f}  "
          f"median={me.median():+.3f}  "
          f"std={me.std():.3f}  "
          f"pct_negative={100*(me < 0).mean():.1f}%  "
          f"pct_zero={100*(me == 0).mean():.1f}%  "
          f"pct_positive={100*(me > 0).mean():.1f}%")

# ===========================================================================
# SECTION 2: Winner goals vs loser goals — where is the bias?
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 2: WHERE IS THE BIAS? Winner goals vs Loser goals")
print(SEP)

print(f"\n--- Winner goals error (pred - actual) ---")
for years, label in [
    (ALL_YEARS,      "All years    "),
    (TRAINING_YEARS, "Training only"),
    ([2022],         "2022 holdout "),
]:
    sub = correct[correct["year"].isin(years)]
    if len(sub) == 0:
        continue
    e = sub["winner_goals_error"]
    print(f"  {label} (n={len(sub):3d}): mean={e.mean():+.3f}  median={e.median():+.3f}  std={e.std():.3f}")

print(f"\n--- Loser goals error (pred - actual) ---")
for years, label in [
    (ALL_YEARS,      "All years    "),
    (TRAINING_YEARS, "Training only"),
    ([2022],         "2022 holdout "),
]:
    sub = correct[correct["year"].isin(years)]
    if len(sub) == 0:
        continue
    e = sub["loser_goals_error"]
    print(f"  {label} (n={len(sub):3d}): mean={e.mean():+.3f}  median={e.median():+.3f}  std={e.std():.3f}")

print(f"\n--- Total goals error (pred - actual, ALL games incl. draws) ---")
for years, label in [
    (ALL_YEARS,      "All years    "),
    (TRAINING_YEARS, "Training only"),
    ([2022],         "2022 holdout "),
]:
    sub = df[df["year"].isin(years)]
    e = sub["total_goals_error"]
    print(f"  {label} (n={len(sub):3d}): mean={e.mean():+.3f}  median={e.median():+.3f}  std={e.std():.3f}")

# ===========================================================================
# SECTION 3: Margin error by year
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 3: MARGIN BIAS BY YEAR")
print(SEP)

print(f"\n  {'Year':>6}  {'n':>4}  {'mean error':>11}  {'median':>8}  {'% under':>8}  {'% exact':>8}  {'% over':>8}")
print(f"  {'------':>6}  {'----':>4}  {'-----------':>11}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}")

for yr in ALL_YEARS:
    sub = correct[correct["year"] == yr]
    if len(sub) == 0:
        continue
    me = sub["margin_error"]
    label = " (HOLDOUT)" if yr == 2022 else ""
    print(
        f"  {yr:>6}  {len(sub):>4}  "
        f"{me.mean():>+11.3f}  "
        f"{me.median():>+8.3f}  "
        f"{100*(me < 0).mean():>7.1f}%  "
        f"{100*(me == 0).mean():>7.1f}%  "
        f"{100*(me > 0).mean():>7.1f}%"
        f"{label}"
    )

# ===========================================================================
# SECTION 4: Predicted margin distribution vs actual margin distribution
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 4: PREDICTED vs ACTUAL MARGIN DISTRIBUTION (correct winners, training)")
print(SEP)

train_correct = correct[correct["year"].isin(TRAINING_YEARS)]

print(f"\n  {'Margin':>8}  {'Pred count':>12}  {'Actual count':>13}  {'Diff':>6}")
print(f"  {'-------':>8}  {'----------':>12}  {'------------':>13}  {'----':>6}")

for margin in range(0, 7):
    pred_count   = (train_correct["pred_margin"]   == margin).sum()
    actual_count = (train_correct["actual_margin"] == margin).sum()
    diff = pred_count - actual_count
    diff_str = f"{diff:+d}" if diff != 0 else "0"
    print(f"  {margin:>8}  {pred_count:>12}  {actual_count:>13}  {diff_str:>6}")

# ===========================================================================
# SECTION 5: Specific scoreline analysis — 1-0 predictions
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 5: WHAT ACTUALLY HAPPENS WHEN MODEL PREDICTS 1-0")
print("(The most common conservative prediction)")
print(SEP)

# Find all games where model predicted 1-0 or 0-1 (1-goal margin wins)
pred_1goal = df[
    ((df["pred_home"] == 1) & (df["pred_away"] == 0)) |
    ((df["pred_home"] == 0) & (df["pred_away"] == 1))
].copy()

pred_1goal_correct = pred_1goal[pred_1goal["correct_winner"]]

print(f"\nGames where model predicted a 1-goal win (1-0 or 0-1): {len(pred_1goal)}")
print(f"  Correct winner: {len(pred_1goal_correct)} ({100*len(pred_1goal_correct)/len(pred_1goal):.1f}%)")

print(f"\nActual margin distribution when model predicted 1-goal win AND was correct:")
margin_dist = pred_1goal_correct["actual_margin"].value_counts().sort_index()
for margin, count in margin_dist.items():
    pct = 100 * count / len(pred_1goal_correct)
    pts_exact = 3 if margin == 1 else (2 if margin == 1 else 1)
    bar = "█" * count
    print(f"  Actual margin {margin}: {count:3d} ({pct:5.1f}%)  {bar}")

print(f"\nPoints breakdown when model predicted 1-goal win AND was correct:")
pts_dist = pred_1goal_correct["model_points"].value_counts().sort_index()
for pts, count in pts_dist.items():
    pct = 100 * count / len(pred_1goal_correct)
    print(f"  {pts} points: {count:3d} ({pct:5.1f}%)")

# ===========================================================================
# SECTION 6: How many extra points would a +1 goal adjustment give?
# ===========================================================================

print(f"\n{SEP}")
print("SECTION 6: SIMULATED +1 GOAL ADJUSTMENT ON WINNER'S GOALS")
print("(What if we bumped winner's predicted goals up by 1?)")
print(SEP)

print(f"\nFor each game where model predicted correct winner and scored < 3 pts,")
print(f"check if adding 1 goal to the winner would have improved the score.\n")

gains = []
for _, row in correct[correct["model_points"] < 3].iterrows():
    # Determine which team is the predicted winner and bump their goals
    if row["pred_home"] > row["pred_away"]:
        new_pred_home = row["pred_home"] + 1
        new_pred_away = row["pred_away"]
    else:
        new_pred_home = row["pred_home"]
        new_pred_away = row["pred_away"] + 1

    # Score the adjusted prediction
    from src.scoring import points_for_prediction
    adj_pts = points_for_prediction(
        int(new_pred_home), int(new_pred_away),
        int(row["actual_home"]), int(row["actual_away"]),
    )
    gain = adj_pts - row["model_points"]
    gains.append({
        "year": row["year"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "pred": row["model_prediction"],
        "actual": f"{int(row['actual_home'])}-{int(row['actual_away'])}",
        "orig_pts": row["model_points"],
        "adj_pts": adj_pts,
        "gain": gain,
    })

gains_df = pd.DataFrame(gains)
total_gain = gains_df["gain"].sum()
positive   = (gains_df["gain"] > 0).sum()
negative   = (gains_df["gain"] < 0).sum()
zero       = (gains_df["gain"] == 0).sum()

print(f"Games analysed (correct winner, < 3 pts): {len(gains_df)}")
print(f"  Would gain points : {positive}")
print(f"  Would lose points : {negative}")
print(f"  No change         : {zero}")
print(f"  Net point change  : {total_gain:+d}")

print(f"\nBy year:")
print(gains_df.groupby("year")["gain"].sum().to_string())

print(f"\nCases where +1 goal adjustment helps (gain > 0):")
print(gains_df[gains_df["gain"] > 0][
    ["year", "home_team", "away_team", "pred", "actual", "orig_pts", "adj_pts", "gain"]
].to_string(index=False))

print(f"\nCases where +1 goal adjustment hurts (gain < 0):")
print(gains_df[gains_df["gain"] < 0][
    ["year", "home_team", "away_team", "pred", "actual", "orig_pts", "adj_pts", "gain"]
].to_string(index=False))

# ===========================================================================
# SECTION 7: Summary verdict
# ===========================================================================

print(f"\n{SEP}")
print("SUMMARY VERDICT")
print(SEP)

mean_margin_error = correct["margin_error"].mean()
pct_under = 100 * (correct["margin_error"] < 0).mean()

print(f"""
Mean margin error (all years, correct winners): {mean_margin_error:+.3f}
% of games where model underpredicted margin  : {pct_under:.1f}%
Net gain from naive +1 winner goals adjustment: {total_gain:+d} pts

If mean margin error is more negative than -0.3 AND % underpredicted > 55%:
    Systematic bias exists. Lambda calibration (scaling up winner's expected
    goals) is worth prototyping. The +1 simulation gives an upper bound on
    the addressable gain.

If mean margin error is between -0.3 and +0.3, or % underpredicted is near 50%:
    No meaningful structural bias. The errors are variance, not bias.
    Lambda calibration will not help in expectation.

If net gain from +1 adjustment is negative:
    Even a perfect upward adjustment hurts more than it helps — the model's
    conservative predictions are already earning points on tight results that
    a higher prediction would miss.
""")