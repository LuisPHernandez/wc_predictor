import pandas as pd

CSV_PATH = "continental_analysis.csv"

df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------------------
# Find all alpha point columns
# ------------------------------------------------------------------

point_cols = sorted(
    [
        c
        for c in df.columns
        if c.endswith("_points")
    ]
)

# ------------------------------------------------------------------
# Total score for each alpha
# ------------------------------------------------------------------

totals = {}

for col in point_cols:
    totals[col] = int(df[col].sum())

# ------------------------------------------------------------------
# Model total
# ------------------------------------------------------------------

model_total = totals["model_points"]

# ------------------------------------------------------------------
# Best fixed alpha
# ------------------------------------------------------------------

best_fixed_col = max(
    totals,
    key=totals.get
)

best_fixed_total = totals[best_fixed_col]

# ------------------------------------------------------------------
# Oracle
# ------------------------------------------------------------------

df["oracle_points"] = df[point_cols].max(axis=1)

oracle_total = int(df["oracle_points"].sum())

# --------------------------------------------------
# Oracle gain over pure model
# --------------------------------------------------

df["oracle_gain"] = (
    df["oracle_points"]
    - df["blend10_points"]
)

print()
print("=" * 60)
print("ORACLE GAIN DISTRIBUTION")
print("=" * 60)

print(
    df["oracle_gain"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------------
# Improvement calculations
# ------------------------------------------------------------------

print()
print("=" * 60)
print("FIXED ALPHA TOTALS")
print("=" * 60)

for col, total in sorted(
    totals.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{col:30s} {total}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Model total      : {model_total}")
print(f"Best fixed total : {best_fixed_total}")
print(f"Best fixed alpha : {best_fixed_col}")
print(f"Oracle total     : {oracle_total}")

print()
print("Gain over model:")
print(f"Best fixed : {best_fixed_total - model_total:+d}")
print(f"Oracle     : {oracle_total - model_total:+d}")

print()
print("Remaining room above best fixed:")
print(f"{oracle_total - best_fixed_total:+d}")

# ------------------------------------------------------------------
# Best alpha per match distribution
# ------------------------------------------------------------------

df["best_alpha_col"] = (
    df[point_cols]
    .idxmax(axis=1)
)

print()
print("=" * 60)
print("BEST ALPHA DISTRIBUTION")
print("=" * 60)

dist = (
    df["best_alpha_col"]
    .value_counts()
    .sort_index()
)

for alpha, count in dist.items():
    print(f"{alpha:30s} {count}")