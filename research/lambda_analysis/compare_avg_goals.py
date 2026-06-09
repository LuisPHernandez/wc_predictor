# compare_model_market_actual.py

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mappings import ODDS_NAME_TO_FIFA

WC_PATH = PROJECT_ROOT / "wc_analysis_rho.csv"

ODDS_FILES = {
    2014: PROJECT_ROOT / "data" / "odds" / "2014wc_expected_goals.csv",
    2018: PROJECT_ROOT / "data" / "odds" / "2018wc_expected_goals.csv",
    2022: PROJECT_ROOT / "data" / "odds" / "2022wc_expected_goals.csv",
}

df = pd.read_csv(WC_PATH)

# --------------------------------------------------
# Load O/U lines
# --------------------------------------------------

odds_frames = []

for year, path in ODDS_FILES.items():

    o = pd.read_csv(path)

    o["home_team"] = (
        o["home_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    o["away_team"] = (
        o["away_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    o["year"] = year

    odds_frames.append(
        o[
            [
                "year",
                "home_team",
                "away_team",
                "ou_lines",
            ]
        ]
    )

odds = pd.concat(
    odds_frames,
    ignore_index=True,
)

# --------------------------------------------------
# Merge
# --------------------------------------------------

merged = df.merge(
    odds,
    on=[
        "year",
        "home_team",
        "away_team",
    ],
    how="inner",
)

print(
    f"Matched: "
    f"{len(merged)} / {len(df)}"
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

merged["model_lambda"] = (
    merged["lambda_home"]
    +
    merged["lambda_away"]
)

merged["actual_goals"] = (
    merged["actual_home"]
    +
    merged["actual_away"]
)

merged["bucket"] = pd.cut(
    merged["model_lambda"],
    bins=[
        0,
        2.5,
        3.0,
        100,
    ],
    labels=[
        "<2.5",
        "2.5-3.0",
        ">3.0",
    ],
)

# --------------------------------------------------
# Summary function
# --------------------------------------------------

def summarize(sub):

    avg_model = sub["model_lambda"].mean()

    avg_market = sub["ou_lines"].mean()

    avg_actual = sub["actual_goals"].mean()

    return pd.Series({

        "matches":
            len(sub),

        "avg_model_lambda":
            round(avg_model, 3),

        "avg_market_ou":
            round(avg_market, 3),

        "avg_actual_goals":
            round(avg_actual, 3),

        "model_k":
            round(
                avg_actual
                /
                avg_model,
                3,
            ),

        "market_k":
            round(
                avg_actual
                /
                avg_market,
                3,
            ),
    })

# --------------------------------------------------
# By bucket
# --------------------------------------------------

print()
print("=" * 70)
print("BY BUCKET")
print("=" * 70)

bucket_summary = (
    merged
    .groupby("bucket")
    .apply(summarize)
)

print(
    bucket_summary
    .to_string()
)

# --------------------------------------------------
# Overall
# --------------------------------------------------

print()
print("=" * 70)
print("OVERALL")
print("=" * 70)

overall = summarize(merged)

print(overall)

# --------------------------------------------------
# By year
# --------------------------------------------------

print()
print("=" * 70)
print("BY YEAR")
print("=" * 70)

year_summary = (
    merged
    .groupby("year")
    .apply(summarize)
)

print(
    year_summary
    .to_string()
)