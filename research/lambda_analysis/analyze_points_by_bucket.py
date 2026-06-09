import pandas as pd
import numpy as np

CSV_PATH = "k_function_results.csv"

df = pd.read_csv(CSV_PATH)

# ============================================================
# BUCKETS
# ============================================================

df["bucket"] = pd.cut(
    df["lambda_total"],
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

# ============================================================
# OVERALL
# ============================================================

print()
print("=" * 70)
print("POINTS BY LAMBDA BUCKET")
print("=" * 70)

summary = (
    df.groupby("bucket")
    .agg(
        matches=(
            "bucket",
            "count",
        ),

        total_points=(
            "points",
            "sum",
        ),

        avg_points=(
            "points",
            "mean",
        ),
    )
)

summary["share_of_matches"] = (
    summary["matches"]
    /
    summary["matches"].sum()
)

summary["share_of_points"] = (
    summary["total_points"]
    /
    summary["total_points"].sum()
)

print(
    summary.round(3)
)

# ============================================================
# BY YEAR
# ============================================================

print()
print("=" * 70)
print("POINTS BY BUCKET AND YEAR")
print("=" * 70)

year_summary = (
    df.groupby(
        [
            "year",
            "bucket",
        ]
    )
    .agg(
        matches=(
            "bucket",
            "count",
        ),

        total_points=(
            "points",
            "sum",
        ),

        avg_points=(
            "points",
            "mean",
        ),
    )
)

print(
    year_summary.round(3)
)

# ============================================================
# SIMPLE TABLE
# ============================================================

print()
print("=" * 70)
print("POINTS PER MATCH")
print("=" * 70)

ppm = (
    df.groupby("bucket")["points"]
    .mean()
    .sort_index()
)

for bucket, value in ppm.items():

    print(
        f"{bucket:>7} : "
        f"{value:.3f}"
    )