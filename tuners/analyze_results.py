import pandas as pd
import sys

if len(sys.argv) != 2:
    raise ValueError(
        "Usage: python analyze_results.py results.csv"
    )

df = pd.read_csv(sys.argv[1])

margin_cols = [
    c for c in df.columns
    if c.startswith("margin_")
]

df["min_margin"] = df[margin_cols].min(axis=1)
df["max_margin"] = df[margin_cols].max(axis=1)

df["margin_std"] = df[margin_cols].std(axis=1)

df["positive_years"] = (
    df[margin_cols] > 0
).sum(axis=1)

df["robustness_score"] = (
    df["avg_margin"] /
    df["margin_std"].replace(0, 1e-9)
)

df["adjusted_score"] = (
    df["avg_margin"] -
    df["margin_std"]
)

print("\n" + "="*70)
print("BEST AVG MARGIN")
print("="*70)

print(
    df.sort_values(
        "avg_margin",
        ascending=False
    ).head(10)
)

print("\n" + "="*70)
print("BEST WORST-CASE PERFORMANCE")
print("="*70)

print(
    df.sort_values(
        "min_margin",
        ascending=False
    ).head(10)
)

print("\n" + "="*70)
print("LOWEST VARIABILITY")
print("="*70)

print(
    df.sort_values(
        "margin_std",
        ascending=True
    ).head(10)
)

print("\n" + "="*70)
print("BEST ROBUSTNESS SCORE")
print("="*70)

print(
    df.sort_values(
        "robustness_score",
        ascending=False
    ).head(10)
)

print("\n" + "="*70)
print("BEST ADJUSTED SCORE")
print("="*70)

print(
    df.sort_values(
        "adjusted_score",
        ascending=False
    ).head(10)
)