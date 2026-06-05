from backtest import run_backtest

print("=" * 70)
print("RUNNING PURE MODEL")
print("=" * 70)

model_result = run_backtest(
    2010,
    alpha=1.0,
)

print("\n" + "=" * 70)
print("RUNNING BLEND")
print("=" * 70)

odds_result = run_backtest(
    2010,
    alpha=0.6,
)

# --------------------------------------------------
# Compare predictions
# --------------------------------------------------

model_preds = model_result["model_preds"][
    [
        "game_id",
        "team1",
        "team2",
        "prediction",
        "actual",
        "points_earned",
    ]
].copy()

odds_preds = odds_result["model_preds"][
    [
        "game_id",
        "prediction",
        "points_earned",
    ]
].copy()

comparison = model_preds.merge(
    odds_preds,
    on="game_id",
    suffixes=("_model", "_odds"),
)

changed = comparison[
    comparison["prediction_model"]
    != comparison["prediction_odds"]
].copy()

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Model points : {model_result['model_points']}")
print(f"Blend points  : {odds_result['model_points']}")

print()
print(
    f"Predictions changed: "
    f"{len(changed)} / {len(comparison)}"
)

# --------------------------------------------------
# Detailed changes
# --------------------------------------------------

print("\n" + "=" * 70)
print("MATCHES THAT CHANGED")
print("=" * 70)

if len(changed) == 0:
    print("No predictions changed.")
else:
    print(
        changed[
            [
                "game_id",
                "team1",
                "team2",
                "prediction_model",
                "prediction_odds",
                "actual",
                "points_earned_model",
                "points_earned_odds",
            ]
        ].to_string(index=False)
    )

# --------------------------------------------------
# Net gain/loss from changed matches
# --------------------------------------------------

if len(changed) > 0:

    changed["point_delta"] = (
        changed["points_earned_odds"]
        - changed["points_earned_model"]
    )

    print("\n" + "=" * 70)
    print("POINT DELTAS")
    print("=" * 70)

    print(
        changed[
            [
                "team1",
                "team2",
                "prediction_model",
                "prediction_odds",
                "actual",
                "points_earned_model",
                "points_earned_odds",
                "point_delta",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Total delta from changed matches:",
        changed["point_delta"].sum()
    )