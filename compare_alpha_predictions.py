from backtest import run_backtest

print("=" * 70)
print("RUNNING PURE MODEL")
print("=" * 70)

model_result = run_backtest(
    2006,
    alpha=1.0,
)

print("\n" + "=" * 70)
print("RUNNING BLEND")
print("=" * 70)

blend_result = run_backtest(
    2006,
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

blend_preds = blend_result["model_preds"][
    [
        "game_id",
        "prediction",
        "points_earned",
    ]
].copy()

comparison = model_preds.merge(
    blend_preds,
    on="game_id",
    suffixes=("_model", "_blend"),
)

changed = comparison[
    comparison["prediction_model"]
    != comparison["prediction_blend"]
].copy()

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Model points : {model_result['model_points']}")
print(f"Blend points  : {blend_result['model_points']}")

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
                "prediction_blend",
                "actual",
                "points_earned_model",
                "points_earned_blend",
            ]
        ].to_string(index=False)
    )

# --------------------------------------------------
# Net gain/loss from changed matches
# --------------------------------------------------

if len(changed) > 0:

    changed["point_delta"] = (
        changed["points_earned_blend"]
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
                "prediction_blend",
                "actual",
                "points_earned_model",
                "points_earned_blend",
                "point_delta",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Total delta from changed matches:",
        changed["point_delta"].sum()
    )