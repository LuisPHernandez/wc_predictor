import pandas as pd

from src.loader import load_pool_data
from src.scoring import points_for_prediction

POOL_PATH = "data/pool"
MODEL_RESULTS = "k_function_results.csv"

# ============================================================
# LOAD MODEL RESULTS
# ============================================================

model = pd.read_csv(MODEL_RESULTS)

model["bucket"] = pd.cut(
    model["lambda_total"],
    bins=[
        0,
        2.5,
        3.0,
        100,
    ],
    labels=[
        "<2.5",
        "2.5-3.0",
        ">3.0",
    ],
)

# ============================================================
# BUILD TOP-5 USER RESULTS
# ============================================================

all_rows = []

for year in [2018, 2022]:

    print()
    print("=" * 60)
    print(year)
    print("=" * 60)

    pool = load_pool_data(
        POOL_PATH,
        year,
    )

    preds = pool["predictions"]
    scores = pool["scores"]
    games = pool["games"]

    if preds is None:
        continue

    # --------------------------------------------------------
    # Score every prediction
    # --------------------------------------------------------

    actual_lookup = (
        scores
        .set_index("game_id")
        [["score1", "score2"]]
        .to_dict("index")
    )

    scored_rows = []

    for row in preds.itertuples():

        actual = actual_lookup.get(
            row.game_id
        )

        if actual is None:
            continue

        pts = points_for_prediction(
            row.score1,
            row.score2,
            actual["score1"],
            actual["score2"],
        )

        scored_rows.append({
            "user_id": row.user_id,
            "game_id": row.game_id,
            "points": pts,
        })

    scored = pd.DataFrame(
        scored_rows
    )

    # --------------------------------------------------------
    # Find top 5 users
    # --------------------------------------------------------

    ranking = (
        scored
        .groupby("user_id")["points"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "points": "total_points"
            }
        )
        .sort_values(
            "total_points",
            ascending=False,
        )
    )

    top5 = (
        ranking
        .head(5)["user_id"]
        .tolist()
    )

    print(
        "Top 5 users:",
        top5
    )

    # --------------------------------------------------------
    # Keep only top-5 predictions
    # --------------------------------------------------------

    scored = scored[
        scored["user_id"].isin(top5)
    ].copy()

    # --------------------------------------------------------
    # Average points per game across top 5
    # --------------------------------------------------------

    top5_game_points = (
        scored
        .groupby("game_id")["points"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "points": "top5_avg_points"
            }
        )
    )

    # --------------------------------------------------------
    # Attach team names to game ids
    # --------------------------------------------------------

    top5_game_points = top5_game_points.merge(
        games[
            [
                "game_id",
                "team1",
                "team2",
            ]
        ],
        on="game_id",
        how="left",
    )

    top5_game_points["year"] = year

    # --------------------------------------------------------
    # Merge with model results
    # --------------------------------------------------------

    year_model = (
        model[
            model["year"] == year
        ]
        .copy()
    )

    merged = top5_game_points.merge(
        year_model,
        left_on=[
            "year",
            "team1",
            "team2",
        ],
        right_on=[
            "year",
            "home_team",
            "away_team",
        ],
        how="inner",
    )

    print(
        f"{year}: matched "
        f"{len(merged)} of "
        f"{len(year_model)} games"
    )

    all_rows.append(
        merged[
            [
                "year",
                "bucket",
                "points",
                "top5_avg_points",
            ]
        ]
    )

    merged["top5_avg_points"] = (
        top5_game_points["top5_avg_points"]
    )

    print(year_model.columns.tolist())

    all_rows.append(
        merged[
            [
                "year",
                "bucket",
                "points",
                "top5_avg_points",
            ]
        ]
    )

    if len(merged) != len(year_model):

        missing = (
            year_model.merge(
                merged[
                    [
                        "home_team",
                        "away_team",
                    ]
                ],
                on=[
                    "home_team",
                    "away_team",
                ],
                how="left",
                indicator=True,
            )
        )

        print(
            missing[
                missing["_merge"] == "left_only"
            ][
                [
                    "home_team",
                    "away_team",
                ]
            ]
        )

# ============================================================
# COMBINE
# ============================================================

df = pd.concat(
    all_rows,
    ignore_index=True,
)

# ============================================================
# MODEL BY BUCKET
# ============================================================

model_summary = (
    df.groupby("bucket")
    .agg(
        matches=(
            "bucket",
            "count",
        ),

        model_ppm=(
            "points",
            "mean",
        ),

        top5_ppm=(
            "top5_avg_points",
            "mean",
        ),
    )
)

model_summary["gap"] = (
    model_summary["top5_ppm"]
    -
    model_summary["model_ppm"]
)

model_summary["model_total"] = (
    model_summary["model_ppm"]
    *
    model_summary["matches"]
)

model_summary["top5_total"] = (
    model_summary["top5_ppm"]
    *
    model_summary["matches"]
)

print()
print("=" * 70)
print("MODEL VS TOP-5 USERS")
print("=" * 70)

print(
    model_summary.round(3)
)

# ============================================================
# SIMPLE VIEW
# ============================================================

print()
print("=" * 70)
print("POINTS PER MATCH")
print("=" * 70)

for bucket, row in model_summary.iterrows():

    print(
        f"{bucket:>7} | "
        f"Model={row['model_ppm']:.3f} | "
        f"Top5={row['top5_ppm']:.3f} | "
        f"Gap={row['gap']:.3f}"
    )