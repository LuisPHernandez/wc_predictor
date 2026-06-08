import pandas as pd

CSV_PATH = "continental_analysis.csv"

THRESHOLDS = [
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
]

ALPHAS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

# --------------------------------------------------
# Helpers
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
# Load data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

POINT_COLS = {
    c
    for c in df.columns
    if c.endswith("_points")
}

results = []

# --------------------------------------------------
# Grid search
# --------------------------------------------------

for threshold in THRESHOLDS:

    for alpha_low in ALPHAS:

        for alpha_high in ALPHAS:

            low_col = alpha_to_col(alpha_low)
            high_col = alpha_to_col(alpha_high)

            if low_col not in POINT_COLS:
                raise ValueError(low_col)

            if high_col not in POINT_COLS:
                raise ValueError(high_col)

            use_low = (
                (df["favorite_flip"])
                &
                (df["tvd"] > threshold)
            )

            total_points = (
                df.loc[use_low, low_col].sum()
                +
                df.loc[~use_low, high_col].sum()
            )

            results.append({
                "threshold": threshold,
                "alpha_low": alpha_low,
                "alpha_high": alpha_high,
                "total_points": int(total_points),
            })

# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "total_points",
    ascending=False,
)

print()
print("=" * 70)
print("TOP 50 RULE A RESULTS")
print("=" * 70)

print(
    results_df.head(50).to_string(index=False)
)

print()
print("=" * 70)
print("BEST RULE A")
print("=" * 70)

best = results_df.iloc[0]

print(
    f"threshold   = {best['threshold']}"
)

print(
    f"alpha_low   = {best['alpha_low']}"
)

print(
    f"alpha_high  = {best['alpha_high']}"
)

print(
    f"total_pts   = {best['total_points']}"
)

# --------------------------------------------------
# Compare against best fixed alpha
# --------------------------------------------------

fixed_cols = [
    c
    for c in df.columns
    if c.endswith("_points")
]

fixed_totals = {
    col: int(df[col].sum())
    for col in fixed_cols
}

best_fixed_col = max(
    fixed_totals,
    key=fixed_totals.get
)

best_fixed_total = fixed_totals[
    best_fixed_col
]

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)

print(
    f"Best fixed alpha : "
    f"{best_fixed_col}"
)

print(
    f"Best fixed total : "
    f"{best_fixed_total}"
)

print(
    f"Best Rule A      : "
    f"{best['total_points']}"
)

print(
    f"Improvement      : "
    f"{best['total_points'] - best_fixed_total:+.0f}"
)

use_low = (
    (df["favorite_flip"])
    &
    (df["tvd"] > 0.175)
)

print("\nmatches changed: ", use_low.sum())