import pandas as pd

df = pd.read_csv("continental_analysis.csv")

point_cols = [
    c
    for c in df.columns
    if c.endswith("_points")
]

df["oracle_points"] = (
    df[point_cols]
    .max(axis=1)
)

df["oracle_gain"] = (
    df["oracle_points"]
    - df["blend10_points"]
)

for flip in [False, True]:

    subset = df[
        df["favorite_flip"] == flip
    ]

    gain_matches = subset[
        subset["oracle_gain"] > 0
    ]

    print()
    print("=" * 60)
    print(f"favorite_flip={flip}")
    print("=" * 60)

    print(
        f"matches: {len(subset)}"
    )

    print(
        f"oracle gain matches: "
        f"{len(gain_matches)}"
    )

    print(
        f"gain rate: "
        f"{100 * len(gain_matches) / len(subset):.2f}%"
    )

    print(
        f"total oracle gain: "
        f"{gain_matches['oracle_gain'].sum()}"
    )

    print(
        f"avg tvd: "
        f"{subset['tvd'].mean():.3f}"
    )