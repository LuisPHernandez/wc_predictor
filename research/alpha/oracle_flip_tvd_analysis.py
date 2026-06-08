import pandas as pd

CSV_PATH = "continental_analysis.csv"

THRESHOLDS = [
    0.05,
    0.10,
    0.12,
    0.15,
    0.17,
    0.20,
]

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

point_cols = [
    c
    for c in df.columns
    if c.endswith("_points")
]

# --------------------------------------------------
# Oracle calculations
# --------------------------------------------------

df["oracle_points"] = (
    df[point_cols]
    .max(axis=1)
)

df["oracle_gain"] = (
    df["oracle_points"]
    - df["blend10_points"]
)

# --------------------------------------------------
# Overall summary
# --------------------------------------------------

print()
print("=" * 70)
print("FAVORITE FLIP ORACLE ANALYSIS")
print("=" * 70)

flip_df = df[
    df["favorite_flip"] == True
].copy()

print(
    f"Favorite flip matches: {len(flip_df)}"
)

print(
    f"Total oracle gain in flips: "
    f"{flip_df['oracle_gain'].sum()}"
)

# --------------------------------------------------
# TVD threshold analysis
# --------------------------------------------------

print()
print("=" * 70)
print("FAVORITE FLIP × TVD THRESHOLDS")
print("=" * 70)

for threshold in THRESHOLDS:

    subset = flip_df[
        flip_df["tvd"] > threshold
    ]

    gain_matches = subset[
        subset["oracle_gain"] > 0
    ]

    total_gain = int(
        subset["oracle_gain"].sum()
    )

    gain_rate = (
        100 * len(gain_matches) / len(subset)
        if len(subset) > 0
        else 0.0
    )

    print()
    print(
        f"Threshold = {threshold:.2f}"
    )

    print(
        f"Matches: {len(subset)}"
    )

    print(
        f"Oracle gain matches: "
        f"{len(gain_matches)}"
    )

    print(
        f"Gain rate: "
        f"{gain_rate:.2f}%"
    )

    print(
        f"Total oracle gain: "
        f"{total_gain}"
    )

# --------------------------------------------------
# Gain concentration
# --------------------------------------------------

print()
print("=" * 70)
print("GAIN CONCENTRATION")
print("=" * 70)

total_oracle_gain = int(
    df["oracle_gain"].sum()
)

for threshold in THRESHOLDS:

    subset = flip_df[
        flip_df["tvd"] > threshold
    ]

    gain = int(
        subset["oracle_gain"].sum()
    )

    pct = (
        100 * gain / total_oracle_gain
        if total_oracle_gain > 0
        else 0
    )

    print(
        f"TVD > {threshold:.2f}: "
        f"gain={gain:3d} "
        f"({pct:.1f}% of all oracle gain)"
    )