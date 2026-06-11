import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mappings import ODDS_NAME_TO_FIFA

CSV_PATH = PROJECT_ROOT / "data" / "analysis" / "wc_analysis_rho.csv"

BETAS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
]

df = pd.read_csv(CSV_PATH)

df["model_total"] = (
    df["lambda_home"]
    +
    df["lambda_away"]
)

df["actual_goals"] = (
    df["actual_home"]
    +
    df["actual_away"]
)

# ============================================================
# LOAD MARKET O/U
# ============================================================

market_frames = []

for year in [2014, 2018, 2022]:

    odds = pd.read_csv(
        f"data/odds/{year}wc_expected_goals.csv"
    )

    odds["year"] = year

    odds["home_team"] = (
        odds["home_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    odds["away_team"] = (
        odds["away_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    market_frames.append(
        odds[
            [
                "year",
                "home_team",
                "away_team",
                "implied_xg",
            ]
        ]
    )

market = pd.concat(
    market_frames,
    ignore_index=True,
)

merged = df.merge(
    market,
    on=[
        "year",
        "home_team",
        "away_team",
    ],
    how="inner",
)

print()
print(
    f"Matched {len(merged)} of {len(df)} rows"
)

# ============================================================
# BUCKETS
# ============================================================

def bucketize(x):
    if x < 2.0:
        return "<2.0"

    if x < 2.5:
        return "2.0-2.5"

    if x < 3.0:
        return "2.5-3.0"

    if x < 3.5:
        return "3.0-3.5"

    return ">3.5"


# ============================================================
# EVALUATE BETAS
# ============================================================

for beta in BETAS:

    print()
    print("=" * 80)
    print(f"BETA = {beta:.2f}")
    print("=" * 80)

    work = merged.copy()

    work["blend_total"] = (
        beta * work["model_total"]
        +
        (1 - beta) * work["implied_xg"]
    )

    work["bucket"] = (
        work["blend_total"]
        .apply(bucketize)
    )

    summary = (
        work.groupby("bucket")
        .agg(
            matches=(
                "bucket",
                "count",
            ),

            avg_blend=(
                "blend_total",
                "mean",
            ),

            avg_actual=(
                "actual_goals",
                "mean",
            ),
        )
    )

    summary["k"] = (
        summary["avg_actual"]
        /
        summary["avg_blend"]
    )

    print()
    print(
        summary.round(3)
    )

    k_spread = (
        summary["k"].max()
        -
        summary["k"].min()
    )

    print()
    print(
        f"K spread: {k_spread:.3f}"
    )

    print(
        f"Mean k   : {summary['k'].mean():.3f}"
    )