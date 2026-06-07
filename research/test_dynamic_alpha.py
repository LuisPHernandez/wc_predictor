import pandas as pd

df = pd.read_csv(
    "continental_analysis.csv"
)

THRESHOLD = 0.15

# --------------------------------------------------
# Dynamic rule
# --------------------------------------------------

use_alpha02 = (
    (df["tvd"] > THRESHOLD)
)

use_alpha06 = ~use_alpha02

# --------------------------------------------------
# Basic counts
# --------------------------------------------------

print()
print("=" * 70)
print("RULE BREAKDOWN")
print("=" * 70)

print(
    "Alpha 0.2 matches:",
    use_alpha02.sum()
)

print(
    "Alpha 0.6 matches:",
    use_alpha06.sum()
)

print(
    "Total matches:",
    len(df)
)

# --------------------------------------------------
# Dynamic score
# --------------------------------------------------

dynamic_points = (
    df.loc[
        use_alpha02,
        "blend02_points"
    ].sum()
    +
    df.loc[
        use_alpha06,
        "blend06_points"
    ].sum()
)

model_points = (
    df["model_points"]
    .sum()
)

blend06_points = (
    df["blend06_points"]
    .sum()
)

print()
print("=" * 70)
print("TOTALS")
print("=" * 70)

print(
    f"Model       : {model_points}"
)

print(
    f"Alpha 0.6   : {blend06_points}"
)

print(
    f"Dynamic     : {dynamic_points}"
)

print(
    f"Dynamic gain over model: "
    f"{dynamic_points - model_points:+}"
)

print(
    f"Dynamic gain over alpha 0.6: "
    f"{dynamic_points - blend06_points:+}"
)

# --------------------------------------------------
# Where did the gain come from?
# --------------------------------------------------

print()
print("=" * 70)
print("SWITCHED MATCHES ONLY")
print("=" * 70)

switched = df[
    use_alpha02
].copy()

print(
    "Matches switched:",
    len(switched)
)

# Dynamic uses 0.2 here
dynamic_switched = (
    switched["blend02_points"]
    .sum()
)

# Fixed 0.6 would use 0.6
fixed_switched = (
    switched["blend06_points"]
    .sum()
)

print(
    "0.2 points:",
    dynamic_switched
)

print(
    "0.6 points:",
    fixed_switched
)

print(
    "Gain from switched matches:",
    dynamic_switched
    -
    fixed_switched
)

# --------------------------------------------------
# Top positive switched matches
# --------------------------------------------------

switched["switch_delta"] = (
    switched["blend02_points"]
    -
    switched["blend06_points"]
)

print()
print("=" * 70)
print("TOP POSITIVE SWITCHES")
print("=" * 70)

print(
    switched
    .sort_values(
        "switch_delta",
        ascending=False,
    )
    [
        [
            "tournament",
            "year",
            "home_team",
            "away_team",
            "tvd",
            "model_prediction",
            "blend02_prediction",
            "blend06_prediction",
            "actual_home",
            "actual_away",
            "switch_delta",
        ]
    ]
    .head(25)
    .to_string(index=False)
)

# --------------------------------------------------
# Top negative switched matches
# --------------------------------------------------

print()
print("=" * 70)
print("TOP NEGATIVE SWITCHES")
print("=" * 70)

print(
    switched
    .sort_values(
        "switch_delta",
        ascending=True,
    )
    [
        [
            "tournament",
            "year",
            "home_team",
            "away_team",
            "tvd",
            "model_prediction",
            "blend02_prediction",
            "blend06_prediction",
            "actual_home",
            "actual_away",
            "switch_delta",
        ]
    ]
    .head(25)
    .to_string(index=False)
)