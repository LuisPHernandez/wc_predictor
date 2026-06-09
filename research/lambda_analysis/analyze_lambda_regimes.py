"""
analyze_lambda_regimes.py

Analyzes the relationship between the model's predicted lambda (expected
goals), market over/under lines, and actual goals scored — across
historical World Cups (2014-2022) and the 2026 tournament.

Core question: is the goal-inflation factor k best applied as a flat
scalar, or should it vary with the model's predicted lambda? Evidence
shows the model overshoots goals in high-lambda games and undershoots
in low-lambda games, so a per-bucket k may be more appropriate for 2026.

Inputs
------
wc_analysis_rho.csv                    — historical WC dataset (2014-2022)
data/odds/2014wc_expected_goals.csv    — market O/U lines per match
data/odds/2018wc_expected_goals.csv
data/odds/2022wc_expected_goals.csv
data/odds/2026wc_expected_goals.csv    — market O/U lines for 2026
data/wc2026_lambda_check.csv           — model lambda predictions for 2026

Outputs  (all written to lambda_analysis/)
------------------------------------------
bucket_summary.csv
year_summary.csv
fine_bucket_summary.csv
hist_bucket_counts.csv
bucket_counts.csv
2026_market_vs_model.csv
lambda_distribution.png
market_vs_model_2026.png
fine_bucket_calibration.png

Run from project root:
    py -3 analyze_lambda_regimes.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.mappings import ODDS_NAME_TO_FIFA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

WC_PATH = PROJECT_ROOT / "wc_analysis_rho.csv"

HIST_ODDS_PATHS = {
    2014: PROJECT_ROOT / "data" / "odds" / "2014wc_expected_goals.csv",
    2018: PROJECT_ROOT / "data" / "odds" / "2018wc_expected_goals.csv",
    2022: PROJECT_ROOT / "data" / "odds" / "2022wc_expected_goals.csv",
}

ODDS_2026_PATH   = PROJECT_ROOT / "data" / "odds" / "2026wc_expected_goals.csv"
LAMBDA_2026_PATH = PROJECT_ROOT / "data" / "wc2026_lambda_check.csv"

OUTDIR = PROJECT_ROOT / "lambda_analysis"
OUTDIR.mkdir(exist_ok=True)

SEP = "=" * 70

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def normalize_teams(df):
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype(str).str.strip().replace(ODDS_NAME_TO_FIFA)
    return df


def summarize(sub):
    """Computes lambda, market O/U, actual goal averages and derived k values."""
    avg_model  = sub["lambda_total"].mean()
    avg_market = sub["ou_lines"].mean()
    avg_actual = sub["actual_goals"].mean()
    return pd.Series({
        "matches":          len(sub),
        "avg_model_lambda": round(avg_model,  3),
        "avg_market_ou":    round(avg_market, 3),
        "avg_actual_goals": round(avg_actual, 3),
        "model_k":          round(avg_actual / avg_model,  3),
        "market_k":         round(avg_actual / avg_market, 3),
    })

# ---------------------------------------------------------------------------
# 1. Load historical WC data
# ---------------------------------------------------------------------------

section("1. LOADING HISTORICAL WC DATA")

df = pd.read_csv(WC_PATH)
df["lambda_total"] = df["lambda_home"] + df["lambda_away"]
df["actual_goals"] = df["actual_home"] + df["actual_away"]

# Coarse bucket (used throughout)
df["bucket"] = pd.cut(
    df["lambda_total"],
    bins=[0, 2.5, 3.0, 100],
    labels=["<2.5", "2.5-3.0", ">3.0"],
)

print(f"\nLoaded {len(df)} rows | years: {sorted(df['year'].unique())}")

# ---------------------------------------------------------------------------
# 2. Load and merge historical O/U odds
# ---------------------------------------------------------------------------

section("2. LOADING AND MERGING HISTORICAL O/U ODDS")

odds_frames = []
for year, path in HIST_ODDS_PATHS.items():
    o = pd.read_csv(path)
    o = normalize_teams(o)
    o["year"] = year
    odds_frames.append(o[["year", "home_team", "away_team", "ou_lines"]])

hist_odds = pd.concat(odds_frames, ignore_index=True)

hist = df.merge(hist_odds, on=["year", "home_team", "away_team"], how="inner")
hist["market_gap"]  = hist["lambda_total"] - hist["ou_lines"]
hist["fine_bucket"] = pd.cut(
    hist["lambda_total"],
    bins=[0, 2.0, 2.5, 3.0, 3.5, 100],
    labels=["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", ">3.5"],
)

print(f"\nMatched {len(hist)} of {len(df)} historical rows to O/U odds")

# ---------------------------------------------------------------------------
# 3. Bucket summary
# ---------------------------------------------------------------------------

section("3. BUCKET SUMMARY  (lambda < 2.5 / 2.5–3.0 / > 3.0)")

bucket_summary = hist.groupby("bucket", observed=False).apply(
    summarize, include_groups=False
)
print(f"\n{bucket_summary.to_string()}")
bucket_summary.to_csv(OUTDIR / "bucket_summary.csv")

# ---------------------------------------------------------------------------
# 4. Year summary
# ---------------------------------------------------------------------------

section("4. YEAR SUMMARY")

year_summary = hist.groupby("year", observed=False).apply(
    summarize, include_groups=False
)
print(f"\n{year_summary.to_string()}")
year_summary.to_csv(OUTDIR / "year_summary.csv")

# ---------------------------------------------------------------------------
# 5. Model–market gap by year
# ---------------------------------------------------------------------------

section("5. MODEL − MARKET GAP BY YEAR")

gap_by_year = hist.groupby("year")["market_gap"].mean().round(4)
print(f"\n{gap_by_year.to_string()}")

# ---------------------------------------------------------------------------
# 6. Historical bucket counts (modern era: 2014-2022)
# ---------------------------------------------------------------------------

section("6. HISTORICAL BUCKET COUNTS (2014–2022)")

hist_bucket_counts = (
    hist.groupby(["year", "bucket"], observed=False)
    .size()
    .unstack(fill_value=0)
)
hist_bucket_pct = (
    hist_bucket_counts
    .div(hist_bucket_counts.sum(axis=1), axis=0)
    * 100
)

print(f"\nCounts:\n{hist_bucket_counts.to_string()}")
print(f"\nPercentages:\n{hist_bucket_pct.round(1).to_string()}")
hist_bucket_counts.to_csv(OUTDIR / "hist_bucket_counts.csv")

# All years in df (for broader context)
all_bucket_counts = (
    df.groupby(["year", "bucket"], observed=False)
    .size()
    .unstack(fill_value=0)
)
all_bucket_pct = (
    all_bucket_counts
    .div(all_bucket_counts.sum(axis=1), axis=0)
    * 100
)

print(f"\nAll years counts:\n{all_bucket_counts.to_string()}")
print(f"\nAll years percentages:\n{all_bucket_pct.round(1).to_string()}")
all_bucket_counts.to_csv(OUTDIR / "bucket_counts.csv")

# ---------------------------------------------------------------------------
# 7. Fine bucket calibration
# ---------------------------------------------------------------------------

section("7. FINE BUCKET CALIBRATION")

fine_summary = hist.groupby("fine_bucket", observed=False).apply(
    summarize, include_groups=False
)
print(f"\n{fine_summary.to_string()}")

section("7b. FINE BUCKET K VALUES")
print(f"\n{fine_summary[['matches', 'model_k', 'market_k']].to_string()}")
fine_summary.to_csv(OUTDIR / "fine_bucket_summary.csv")

# ---------------------------------------------------------------------------
# 8. Load 2026 lambdas and O/U odds
# ---------------------------------------------------------------------------

section("8. LOADING 2026 DATA")

wc26 = pd.read_csv(LAMBDA_2026_PATH)
wc26["home_team"] = wc26["home_team"].astype(str).str.strip()
wc26["away_team"] = wc26["away_team"].astype(str).str.strip()
wc26["lambda_total"] = wc26["lambda_home"] + wc26["lambda_away"]

o26 = pd.read_csv(ODDS_2026_PATH)
o26 = normalize_teams(o26)

print(f"\nLambda rows (2026): {len(wc26)}")
print(f"O/U rows    (2026): {len(o26)}")

wc26 = wc26.merge(o26[["home_team", "away_team", "ou_lines"]],
                  on=["home_team", "away_team"], how="inner")
wc26["market_gap"] = wc26["lambda_total"] - wc26["ou_lines"]
wc26["bucket"] = pd.cut(
    wc26["lambda_total"],
    bins=[0, 2.5, 3.0, 100],
    labels=["<2.5", "2.5-3.0", ">3.0"],
)

print(f"Matched     (2026): {len(wc26)}")
print(f"\nFirst 5 rows:")
print(wc26[["home_team", "away_team", "lambda_total", "ou_lines", "market_gap"]]
      .head().to_string(index=False))

# ---------------------------------------------------------------------------
# 9. 2026 market vs model
# ---------------------------------------------------------------------------

section("9. 2026 MARKET VS MODEL BY BUCKET")

bucket26 = (
    wc26.groupby("bucket", observed=False)
    .agg(
        matches=("bucket", "count"),
        avg_model=("lambda_total", "mean"),
        avg_market=("ou_lines", "mean"),
        avg_gap=("market_gap", "mean"),
    )
)
print(f"\n{bucket26.to_string()}")
bucket26.to_csv(OUTDIR / "2026_market_vs_model.csv")

section("9b. 2026 BUCKET SHARE (%)")
bucket_share = (bucket26["matches"] / bucket26["matches"].sum() * 100).round(1)
print(f"\n{bucket_share.to_string()}")

# ---------------------------------------------------------------------------
# 10. Key takeaways
# ---------------------------------------------------------------------------

section("10. KEY TAKEAWAYS")

hist_model_k  = hist["actual_goals"].mean() / hist["lambda_total"].mean()
hist_market_k = hist["actual_goals"].mean() / hist["ou_lines"].mean()

print(f"""
Historical 2014–2022:
  Avg model lambda  : {hist["lambda_total"].mean():.3f}
  Avg market O/U    : {hist["ou_lines"].mean():.3f}
  Avg actual goals  : {hist["actual_goals"].mean():.3f}
  Model k  (overall): {hist_model_k:.3f}
  Market k (overall): {hist_market_k:.3f}

2026 pre-tournament:
  Avg model lambda  : {wc26["lambda_total"].mean():.3f}
  Avg market O/U    : {wc26["ou_lines"].mean():.3f}
  Avg model−market gap: {wc26["market_gap"].mean():.3f}

Interpretation:
  The model predicts {wc26["lambda_total"].mean():.3f} goals/game for 2026 vs
  a market O/U of {wc26["ou_lines"].mean():.3f} — a gap of {wc26["market_gap"].mean():+.3f}.
  Historically, model k drops below 1.0 when lambda > 3.0 (model overshoots).
  A flat k=1.15 would OVERPREDICT goals for high-lambda 2026 games.
  Per-bucket k calibration is recommended.

Bucket share comparison:
  Historical <2.5 share: {hist_bucket_pct["<2.5"].mean():.1f}%  vs  2026: {bucket_share.get("<2.5", 0):.1f}%
  Historical >3.0 share: {hist_bucket_pct[">3.0"].mean():.1f}%  vs  2026: {bucket_share.get(">3.0", 0):.1f}%
""")

# Per-bucket k recommendation
print("Per-bucket k recommendation for 2026 (from historical calibration):")
print(f"\n  {'Bucket':12s}  {'Hist matches':>13}  {'Model k':>8}  {'Market k':>9}  {'Suggested action'}")
print(f"  {'-'*12}  {'-'*13}  {'-'*8}  {'-'*9}  {'-'*25}")
for bucket, row in fine_summary.iterrows():
    if row["matches"] < 5:
        action = "insufficient data"
    elif row["model_k"] >= 1.08:
        action = f"inflate  → use k ≈ {row['model_k']:.2f}"
    elif row["model_k"] <= 0.97:
        action = f"deflate  → use k ≈ {row['model_k']:.2f}"
    else:
        action = "neutral  → use k ≈ 1.00"
    print(f"  {str(bucket):12s}  {int(row['matches']):>13}  "
          f"{row['model_k']:>8.3f}  {row['market_k']:>9.3f}  {action}")

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------

section("11. GENERATING PLOTS")

# Plot 1: Lambda distribution — historical vs 2026
fig, ax = plt.subplots(figsize=(9, 5))
hist["lambda_total"].hist(bins=20, alpha=0.6, ax=ax, label="2014–2022 (historical)")
wc26["lambda_total"].hist(bins=20, alpha=0.6, ax=ax, label="2026 (pre-tournament)")
ax.axvline(hist["lambda_total"].mean(), color="steelblue", linestyle="--", linewidth=1.2,
           label=f"Hist mean ({hist['lambda_total'].mean():.2f})")
ax.axvline(wc26["lambda_total"].mean(), color="darkorange", linestyle="--", linewidth=1.2,
           label=f"2026 mean ({wc26['lambda_total'].mean():.2f})")
ax.set_xlabel("Model lambda (total expected goals)")
ax.set_ylabel("Matches")
ax.set_title("Lambda Distribution: Historical (2014–2022) vs 2026")
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "lambda_distribution.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: lambda_distribution.png")

# Plot 2: 2026 market O/U vs model lambda scatter
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(wc26["lambda_total"], wc26["ou_lines"], alpha=0.7, s=40, color="steelblue")
mx = max(wc26["lambda_total"].max(), wc26["ou_lines"].max()) + 0.3
ax.plot([0, mx], [0, mx], "k--", linewidth=1, label="y = x (perfect agreement)")
ax.set_xlabel("Model lambda (total)")
ax.set_ylabel("Market O/U line")
ax.set_title("2026: Model Lambda vs Market O/U")
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "market_vs_model_2026.png", dpi=150)
plt.close(fig)
print(f"  Saved: market_vs_model_2026.png")

# Plot 3: Fine bucket calibration — model vs market vs actual
valid = fine_summary[fine_summary["matches"] >= 3]
if len(valid) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(valid))
    ax.plot(x, valid["avg_model_lambda"],  marker="o", label="Model lambda")
    ax.plot(x, valid["avg_market_ou"],     marker="s", label="Market O/U")
    ax.plot(x, valid["avg_actual_goals"],  marker="^", label="Actual goals")
    ax.set_xticks(x)
    ax.set_xticklabels(valid.index)
    ax.set_xlabel("Lambda bucket")
    ax.set_ylabel("Goals per game")
    ax.set_title("Fine Bucket Calibration (2014–2022): Model vs Market vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fine_bucket_calibration.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: fine_bucket_calibration.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

section("DONE")
print(f"\nAll outputs written to: {OUTDIR}")
print(f"\nFiles saved:")
for f in sorted(OUTDIR.iterdir()):
    print(f"  {f.name}")