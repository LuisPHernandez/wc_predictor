import pandas as pd

CSV_PATH = "continental_analysis.csv"

df = pd.read_csv(CSV_PATH)

# --------------------------------------------------
# Alpha point columns
# --------------------------------------------------

point_cols = sorted(
    [
        c
        for c in df.columns
        if c.endswith("_points")
    ]
)

# --------------------------------------------------
# Oracle
# --------------------------------------------------

df["oracle_points"] = (
    df[point_cols]
    .max(axis=1)
)

df["oracle_gain"] = (
    df["oracle_points"]
    - df["blend10_points"]
)

interesting = df[
    df["oracle_gain"] > 0
].copy()

# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total matches: {len(df)}")
print(f"Oracle gain matches: {len(interesting)}")
print(
    f"Oracle gain rate: "
    f"{100 * len(interesting) / len(df):.2f}%"
)

# --------------------------------------------------
# Oracle matches
# --------------------------------------------------

print()
print("=" * 60)
print("ORACLE MATCHES")
print("=" * 60)

print(
    interesting[
        [
            "tvd",
            "model_decision_margin",
            "oracle_gain",
        ]
    ]
    .describe()
)

print()
print(
    "Favorite flip rate "
    f"(oracle matches): "
    f"{interesting['favorite_flip'].mean():.3f}"
)

print(
    "Average TVD "
    f"(oracle matches): "
    f"{interesting['tvd'].mean():.3f}"
)

print(
    "Average decision margin "
    f"(oracle matches): "
    f"{interesting['model_decision_margin'].mean():.4f}"
)

# --------------------------------------------------
# Full dataset comparison
# --------------------------------------------------

print()
print("=" * 60)
print("FULL DATASET")
print("=" * 60)

print(
    "Favorite flip rate "
    f"(all matches): "
    f"{df['favorite_flip'].mean():.3f}"
)

print(
    "Average TVD "
    f"(all matches): "
    f"{df['tvd'].mean():.3f}"
)

print(
    "Average decision margin "
    f"(all matches): "
    f"{df['model_decision_margin'].mean():.4f}"
)

# --------------------------------------------------
# Gain breakdown
# --------------------------------------------------

print()
print("=" * 60)
print("ORACLE GAIN DISTRIBUTION")
print("=" * 60)

print(
    interesting["oracle_gain"]
    .value_counts()
    .sort_index()
)