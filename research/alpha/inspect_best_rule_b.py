import pandas as pd

CSV_PATH = "continental_analysis.csv"

# --------------------------------------------------
# Winning Rule B
# --------------------------------------------------

THRESHOLD = 0.075
ALPHA_LOW = 0.70
ALPHA_HIGH = 0.50

# --------------------------------------------------
# Column mapper
# --------------------------------------------------

def alpha_to_col(alpha):

    mapping = {
        0.00: "blend00_points",
        0.05: "blend005_points",
        0.10: "blend01_points",
        0.15: "blend015_points",
        0.20: "blend02_points",
        0.25: "blend025_points",
        0.30: "blend03_points",
        0.35: "blend035_points",
        0.40: "blend04_points",
        0.45: "blend045_points",
        0.50: "blend05_points",
        0.55: "blend055_points",
        0.60: "blend06_points",
        0.65: "blend065_points",
        0.70: "blend07_points",
        0.75: "blend075_points",
        0.80: "blend08_points",
        0.85: "blend085_points",
        0.90: "blend09_points",
        0.95: "blend095_points",
        1.00: "blend10_points",
    }

    return mapping[round(alpha, 2)]

# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

low_col = alpha_to_col(ALPHA_LOW)
high_col = alpha_to_col(ALPHA_HIGH)

# --------------------------------------------------
# Matches affected by the rule
# --------------------------------------------------

changed = df[
    df["tvd"] > THRESHOLD
].copy()

print()
print("=" * 80)
print("RULE B INSPECTION")
print("=" * 80)

print(
    f"Threshold: {THRESHOLD}"
)

print(
    f"Alpha low : {ALPHA_LOW}"
)

print(
    f"Alpha high: {ALPHA_HIGH}"
)

print(
    f"Changed matches: {len(changed)}"
)

# --------------------------------------------------
# Point difference
# --------------------------------------------------

changed["point_diff"] = (
    changed[low_col]
    - changed[high_col]
)

print()
print("=" * 80)
print("POINT DIFFERENCE SUMMARY")
print("=" * 80)

print(
    changed["point_diff"]
    .describe()
)

# --------------------------------------------------
# Distribution
# --------------------------------------------------

print()
print("=" * 80)
print("POINT DIFFERENCE DISTRIBUTION")
print("=" * 80)

print(
    changed["point_diff"]
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# Most important matches
# --------------------------------------------------

changed = changed.sort_values(
    "point_diff",
    ascending=False,
)

print()
print("=" * 80)
print("TOP POSITIVE MATCHES")
print("=" * 80)

cols = [
    "tournament",
    "year",
    "home_team",
    "away_team",
    "actual_home",
    "actual_away",
    "tvd",
    "favorite_flip",
    "model_prediction",
    "blend05_prediction",
    "blend07_prediction",
    "blend05_points",
    "blend07_points",
    "point_diff",
]

print(
    changed[
        cols
    ]
    .head(25)
    .to_string(index=False)
)

# --------------------------------------------------
# Most negative matches
# --------------------------------------------------

print()
print("=" * 80)
print("TOP NEGATIVE MATCHES")
print("=" * 80)

print(
    changed[
        cols
    ]
    .tail(25)
    .to_string(index=False)
)

# --------------------------------------------------
# Aggregate
# --------------------------------------------------

print()
print("=" * 80)
print("AGGREGATE")
print("=" * 80)

gain = int(
    changed["point_diff"].sum()
)

print(
    f"Net gain from switching "
    f"0.50 -> 0.70 on these matches: {gain:+d}"
)

print(
    f"Positive matches: "
    f"{(changed['point_diff'] > 0).sum()}"
)

print(
    f"Negative matches: "
    f"{(changed['point_diff'] < 0).sum()}"
)

print(
    f"Neutral matches: "
    f"{(changed['point_diff'] == 0).sum()}"
)