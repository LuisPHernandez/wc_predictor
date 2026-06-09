import pandas as pd

from src.scoring import points_for_prediction

POOL_PATH = "data/pool"

YEARS = [2002, 2006, 2010, 2018, 2022]

def score_predictions(predictions_df, scores_df):
    actuals = (
        scores_df
        .set_index("game_id")[["score1", "score2"]]
        .to_dict("index")
    )

    results = []

    for row in predictions_df.itertuples():

        actual = actuals.get(row.game_id)

        if actual is None:
            continue

        pts = points_for_prediction(
            row.score1,
            row.score2,
            actual["score1"],
            actual["score2"],
        )

        results.append(
            {
                "user_id": row.user_id,
                "points": pts,
            }
        )

    return (
        pd.DataFrame(results)
        .groupby("user_id")["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "total_points"})
        .sort_values("total_points", ascending=False)
        .reset_index(drop=True)
    )


first_places = []
second_places = []
third_places = []
top5_cutoffs = []

for year in YEARS:

    preds = pd.read_csv(
        f"{POOL_PATH}/{year}_predictions.csv",
        header=None,
        names=[
            "game_id",
            "user_id",
            "score1",
            "score2",
        ],
    )

    scores_raw = pd.read_csv(
        f"{POOL_PATH}/{year}_scores.csv",
        header=None,
    )

    if scores_raw.shape[1] == 4:
        scores_raw.columns = [
            "team1_code",
            "team2_code",
            "score1",
            "score2",
        ]
    else:
        scores_raw.columns = [
            "phase",
            "team1_code",
            "team2_code",
            "score1",
            "score2",
        ]

    scores_raw["game_id"] = range(
        1,
        len(scores_raw) + 1,
    )

    ranking = score_predictions(
        preds,
        scores_raw,
    )

    print()
    print("=" * 70)
    print(year)
    print("=" * 70)

    print("\nTop 5 Users")
    print(
        ranking.head(5)
        .to_string(index=False)
    )

    first = ranking.iloc[0]["total_points"]
    second = ranking.iloc[1]["total_points"]
    third = ranking.iloc[2]["total_points"]
    top5 = ranking.iloc[4]["total_points"]

    first_places.append(first)
    second_places.append(second)
    third_places.append(third)
    top5_cutoffs.append(top5)

    print()
    print(f"1st place: {first}")
    print(f"2nd place: {second}")
    print(f"3rd place: {third}")

print()
print("=" * 70)
print("AVERAGES")
print("=" * 70)

print(
    f"Average 1st place: {sum(first_places)/len(first_places):.2f}"
)

print(
    f"Average 2nd place: {sum(second_places)/len(second_places):.2f}"
)

print(
    f"Average 3rd place: {sum(third_places)/len(third_places):.2f}"
)

top5_cutoff_avg = (
    sum(top5_cutoffs)
    / len(top5_cutoffs)
)

print(
    f"Average Top 5 cutoff: "
    f"{top5_cutoff_avg:.2f}"
)