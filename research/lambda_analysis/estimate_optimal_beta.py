import pandas as pd
import numpy as np

from src.mappings import ODDS_NAME_TO_FIFA

CSV_PATH = "wc_analysis_rho.csv"

# ============================================================
# LOAD MODEL DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

df["model_total"] = (
    df["lambda_home"]
    +
    df["lambda_away"]
)

df["actual_goals"] = (
    df["actual_home"]
    +
    df["actual_away"]
)

# ============================================================
# LOAD MARKET DATA
# ============================================================

market_frames = []

for year in [2014, 2018, 2022]:

    odds = pd.read_csv(
        f"data/odds/{year}wc_expected_goals.csv"
    )

    odds["year"] = year

    odds["home_team"] = (
        odds["home_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    odds["away_team"] = (
        odds["away_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    market_frames.append(
        odds[
            [
                "year",
                "home_team",
                "away_team",
                "ou_lines",
            ]
        ]
    )

market = pd.concat(
    market_frames,
    ignore_index=True,
)

df = df.merge(
    market,
    on=[
        "year",
        "home_team",
        "away_team",
    ],
    how="inner",
)

print(
    f"Matched {len(df)} rows"
)

# ============================================================
# SEARCH BETAS
# ============================================================

best_beta = None
best_rmse = 999999

print()
print("=" * 60)
print("BETA SEARCH")
print("=" * 60)

for beta in np.arange(
    0.00,
    1.01,
    0.01,
):

    pred = (
        beta * df["model_total"]
        +
        (1 - beta) * df["ou_lines"]
    )

    rmse = np.sqrt(
        np.mean(
            (
                pred
                -
                df["actual_goals"]
            ) ** 2
        )
    )

    if rmse < best_rmse:
        best_rmse = rmse
        best_beta = beta

print()
print(
    f"Best beta : {best_beta:.2f}"
)

print(
    f"Best RMSE : {best_rmse:.4f}"
)

print()

for beta in [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]:

    pred = (
        beta * df["model_total"]
        +
        (1 - beta) * df["ou_lines"]
    )

    rmse = np.sqrt(
        np.mean(
            (
                pred
                -
                df["actual_goals"]
            ) ** 2
        )
    )

    print(
        f"beta={beta:.2f}  rmse={rmse:.4f}"
    )