import pandas as pd

df = pd.read_csv(
    "research/alpha/oracle_matches.csv"
)

# ------------------------------------------
# Truly alpha-sensitive matches
# ------------------------------------------

sensitive = df[
    df["n_best_alphas"] <= 3
].copy()

print()
print("=" * 70)
print("TRULY ALPHA-SENSITIVE MATCHES")
print("=" * 70)

print(
    "Matches:",
    len(sensitive)
)

print(
    "Total oracle gain:",
    sensitive["oracle_gain"].sum()
)

print(
    "Average oracle gain:",
    sensitive["oracle_gain"].mean()
)

print()

print(
    "Favorite flip rate:",
    sensitive["favorite_flip"].mean()
)

print(
    "Average TVD:",
    sensitive["tvd"].mean()
)

print(
    "Average decision margin:",
    sensitive["model_decision_margin"].mean()
)

print()

print("=" * 70)
print("BEST ALPHA COUNTS")
print("=" * 70)

print(
    sensitive["best_alphas"]
    .value_counts()
)

print()

print("=" * 70)
print("MATCHES")
print("=" * 70)

print(
    sensitive[
        [
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
    ]
    .sort_values(
        ["oracle_gain", "tvd"],
        ascending=False,
    )
    .to_string(index=False)
)