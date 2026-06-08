from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT   = Path(__file__).resolve().parent
CSV_PATH     = PROJECT_ROOT / "data" / "analysis" / "wc_analysis.csv"

df = pd.read_csv(CSV_PATH)

df["lambda_total"] = (
    df["lambda_home"]
    +
    df["lambda_away"]
)

df["actual_goals"] = (
    df["actual_home"]
    +
    df["actual_away"]
)

df["error"] = (
    df["actual_goals"]
    -
    df["lambda_total"]
)

df["bucket"] = pd.cut(
    df["lambda_total"],
    bins=[
        0,
        2.5,
        3.0,
        3.5,
        100,
    ],
    labels=[
        "<2.5",
        "2.5-3.0",
        "3.0-3.5",
        ">3.5",
    ],
)

print()
print("=" * 70)
print("WORLD CUP LAMBDA CALIBRATION")
print("=" * 70)

summary = (
    df.groupby("bucket")
    .agg(
        matches=("bucket", "count"),

        avg_lambda=(
            "lambda_total",
            "mean",
        ),

        avg_actual=(
            "actual_goals",
            "mean",
        ),

        avg_error=(
            "error",
            "mean",
        ),
    )
)

summary["implied_k"] = (
    summary["avg_actual"]
    /
    summary["avg_lambda"]
)

print(summary)

print()
print("=" * 70)
print("BY YEAR")
print("=" * 70)

for year in sorted(df["year"].unique()):

    sub = df[
        df["year"] == year
    ]

    avg_lambda = (
        sub["lambda_total"]
        .mean()
    )

    avg_actual = (
        sub["actual_goals"]
        .mean()
    )

    implied_k = (
        avg_actual
        /
        avg_lambda
    )

    print(
        f"{year}: "
        f"lambda={avg_lambda:.3f} "
        f"actual={avg_actual:.3f} "
        f"k={implied_k:.3f}"
    )

print()
print("=" * 70)
print("EXTREME MISMATCHES")
print("=" * 70)

print(
    df[
        df["lambda_total"] > 3.5
    ][
        [
            "year",
            "home_team",
            "away_team",
            "lambda_total",
            "actual_goals",
            "error",
        ]
    ]
    .sort_values(
        "lambda_total",
        ascending=False,
    )
    .to_string(index=False)
)