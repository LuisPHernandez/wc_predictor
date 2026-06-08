import pandas as pd

df = pd.read_csv(
    "continental_analysis.csv"
)

# --------------------------------------------------
# Point columns
# --------------------------------------------------

point_cols = sorted(
    [
        c
        for c in df.columns
        if c.endswith("_points")
    ]
)

# --------------------------------------------------
# Winning alpha per match
# --------------------------------------------------

def winning_alpha(row):

    best_col = row[point_cols].idxmax()

    if best_col == "model_points":
        return "model"

    return (
        best_col
        .replace("blend", "")
        .replace("_points", "")
    )

df["winner"] = df.apply(
    winning_alpha,
    axis=1,
)

# --------------------------------------------------
# Oracle matches only
# --------------------------------------------------

oracle = df[
    df[point_cols].max(axis=1)
    >
    df["blend10_points"]
].copy()

print()
print("=" * 70)
print("ORACLE MATCHES")
print("=" * 70)

print(
    "Count:",
    len(oracle)
)

print()
print(
    oracle["winner"]
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# Split by favorite flip
# --------------------------------------------------

print()
print("=" * 70)
print("ORACLE MATCHES - FAVORITE FLIP")
print("=" * 70)

flip = oracle[
    oracle["favorite_flip"]
]

print(
    flip["winner"]
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# Split by TVD bucket
# --------------------------------------------------

oracle["bucket"] = pd.cut(
    oracle["tvd"],
    bins=[
        0,
        0.05,
        0.10,
        0.15,
        0.20,
        1.0,
    ]
)

print()
print("=" * 70)
print("WINNING ALPHA BY TVD BUCKET")
print("=" * 70)

for bucket, subset in oracle.groupby(
    "bucket"
):

    print()
    print(bucket)

    print(
        subset["winner"]
        .value_counts()
        .sort_index()
    )

# --------------------------------------------------
# Average TVD by winner
# --------------------------------------------------

print()
print("=" * 70)
print("AVERAGE TVD BY WINNING ALPHA")
print("=" * 70)

print(
    oracle.groupby("winner")["tvd"]
    .agg(
        [
            "count",
            "mean",
            "median",
        ]
    )
    .sort_values(
        "mean"
    )
)