import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent / "../../data/analysis/continental_analysis.csv"

df = pd.read_csv(CSV_PATH)

# --------------------------------------------------
# All point columns
# --------------------------------------------------

point_cols = sorted(
    [
        c
        for c in df.columns
        if c.endswith("_points")
    ]
)

# --------------------------------------------------
# Oracle metrics
# --------------------------------------------------

best_points = df[point_cols].max(axis=1)

df["oracle_points"] = best_points

df["oracle_gain"] = (
    best_points
    -
    df["blend10_points"]
)

# --------------------------------------------------
# Count how many alphas tie for best
# --------------------------------------------------

def count_best_alphas(row):

    best = row["oracle_points"]

    return sum(
        row[c] == best
        for c in point_cols
    )

df["n_best_alphas"] = df.apply(
    count_best_alphas,
    axis=1,
)

# --------------------------------------------------
# Oracle matches only
# --------------------------------------------------

oracle = df[
    df["oracle_gain"] > 0
].copy()

print()
print("=" * 80)
print("ORACLE SUMMARY")
print("=" * 80)

print(
    f"Oracle matches: "
    f"{len(oracle)}"
)

print(
    f"Total oracle gain: "
    f"{oracle['oracle_gain'].sum()}"
)

print(
    f"Average oracle gain: "
    f"{oracle['oracle_gain'].mean():.3f}"
)

print()

print(
    oracle[
        [
            "oracle_gain",
            "tvd",
            "model_decision_margin",
            "n_best_alphas",
        ]
    ]
    .describe()
)

# --------------------------------------------------
# Tie analysis
# --------------------------------------------------

print()
print("=" * 80)
print("NUMBER OF BEST ALPHAS")
print("=" * 80)

print(
    oracle["n_best_alphas"]
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# Build alpha list
# --------------------------------------------------

def best_alpha_list(row):

    best = row["oracle_points"]

    winners = []

    for col in point_cols:

        if row[col] == best:

            winners.append(
                col
                .replace(
                    "_points",
                    ""
                )
            )

    return ",".join(winners)

oracle["best_alphas"] = oracle.apply(
    best_alpha_list,
    axis=1,
)

# --------------------------------------------------
# Top oracle matches
# --------------------------------------------------

print()
print("=" * 80)
print("TOP ORACLE MATCHES")
print("=" * 80)

cols = [
    "tournament",
    "year",

    "home_team",
    "away_team",

    "favorite_flip",
    "tvd",

    "model_decision_margin",

    "oracle_gain",

    "n_best_alphas",
    "best_alphas",
]

print(
    oracle
    .sort_values(
        [
            "oracle_gain",
            "tvd",
        ],
        ascending=False,
    )[cols]
    .head(30)
    .to_string(index=False)
)

# --------------------------------------------------
# Save full oracle dataset
# --------------------------------------------------

oracle.to_csv(
    "oracle_matches.csv",
    index=False,
)

print()
print(
    "Saved oracle_matches.csv"
)