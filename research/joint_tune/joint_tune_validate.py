"""
joint_tune_validate.py

Merges results from both PCs, ranks all combinations, and validates
the top N on the 2022 holdout year.

Run after both PCs have finished:
    py -3 joint_tune_validate.py

Reads:
    research/joint_tune/results_part1.csv
    research/joint_tune/results_part2.csv
    (or results_full.csv if run on a single machine)

Writes:
    research/joint_tune/validation_results.csv
    research/joint_tune/final_recommendation.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "research" / "joint_tune"

# Current production parameters (baseline to beat)
CURRENT_DECAY         = 0.20
CURRENT_REG           = 0.0010
CURRENT_ELITE         = 1.00
CURRENT_CAF           = 1.10
CURRENT_CONCACAF      = 1.05
CURRENT_AFC           = 0.95
CURRENT_OFC           = 0.90
CURRENT_ALPHA         = 0.30
CURRENT_K             = 1.15
CURRENT_WEIGHTED_SCORE = None   # filled in from results

# How many top combinations to validate on holdout
TOP_N_TO_VALIDATE = 25

HOLDOUTS = [
    2006,
    2010,
    2014,
    2018,
    2022,
]

YEAR_WEIGHTS = {
    2006: 0.50,
    2010: 0.75,
    2014: 0.90,
    2018: 1.00,
}

WC_START_DATES = {
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

SEP = "=" * 70

# ---------------------------------------------------------------------------
# Load and merge results
# ---------------------------------------------------------------------------

def load_results():
    paths = [
        RESULTS_DIR / "results_part1.csv",
        RESULTS_DIR / "results_part2.csv",
        RESULTS_DIR / "results_full.csv",
    ]
    dfs = []
    for p in paths:
        if p.exists():
            dfs.append(pd.read_csv(p))
            print(f"Loaded {len(dfs[-1])} rows from {p.name}")

    if not dfs:
        raise FileNotFoundError(
            f"No results files found in {RESULTS_DIR}. "
            "Run joint_tune_coordinator.py first."
        )

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["best_weighted_score"] > -999]   # drop failed fits
    df = df.drop_duplicates(subset=["combo_id"])
    print(f"Total valid combinations: {len(df)}")
    return df

# ---------------------------------------------------------------------------
# Validate one combination on 2022
# ---------------------------------------------------------------------------

def validate_on_2022(combo_row, holdout_year):
    """
    Fits the model on 2006-2018 with the given parameters and
    evaluates on 2022. Returns points scored on 2022.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from research.joint_tune.joint_tune_worker import (
        _fit_dixon_coles, _get_lambda, _score_matrix,
        _blend_matrix, _best_pred_pts,
        COMPETITION_WEIGHTS, RHO_ESTIMATE,
        K_VALUES, ALPHAS,
    )
    from src.loader      import load_kaggle_data, load_pool_data, get_wc_teams
    from src.odds_loader import load_wc_odds_lookup

    KAGGLE_PATH = PROJECT_ROOT / "data" / "kaggle" / "results.csv"
    POOL_PATH   = PROJECT_ROOT / "data" / "pool"
    HOLDOUT = holdout_year

    decay_lambda   = combo_row["decay_lambda"]
    regularization = combo_row["regularization"]
    best_alpha     = combo_row["best_alpha"]
    best_k         = combo_row["best_k"]

    confederation_weights = {
        "CONMEBOL": combo_row["elite"],
        "UEFA":     combo_row["elite"],
        "CAF":      combo_row["caf"],
        "CONCACAF": combo_row["concacaf"],
        "AFC":      combo_row["afc"],
        "OFC":      combo_row["ofc"],
    }

    # Training window for 2022 holdout fit
    end   = pd.Timestamp(WC_START_DATES[HOLDOUT]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=12)

    pool     = load_pool_data(POOL_PATH, HOLDOUT)
    wc_teams = get_wc_teams(pool)

    kaggle_df = load_kaggle_data(
        KAGGLE_PATH,
        wc_teams,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        decay_lambda=decay_lambda,
        competition_weights=COMPETITION_WEIGHTS,
        confederation_weights=confederation_weights,
    )

    try:
        params, team_idx, n_teams, rho = _fit_dixon_coles(
            kaggle_df, decay_lambda, regularization
        )
    except Exception as e:
        return None, str(e)

    try:
        odds_lookup = load_wc_odds_lookup(HOLDOUT)
    except Exception:
        odds_lookup = {}

    scores_df    = pool["scores"]
    score_lookup = (
        scores_df.set_index("game_id")[["score1", "score2"]].to_dict("index")
    )

    total_pts = 0
    games     = pool["games"]

    for row in games.itertuples():
        actual = score_lookup.get(row.game_id)
        if actual is None:
            continue
        home = row.team1
        away = row.team2
        if home not in team_idx or away not in team_idx:
            continue

        lh, la = _get_lambda(params, team_idx, n_teams, home, away, neutral=True)
        book    = odds_lookup.get(row.game_id)

        matrix  = _score_matrix(lh, la, RHO_ESTIMATE, best_k)
        matrix  = _blend_matrix(matrix, book, best_alpha)
        total_pts += _best_pred_pts(
            matrix, int(actual["score1"]), int(actual["score2"])
        )

    return total_pts, None

def run_candidate_cv(candidate):

    print()
    print(SEP)
    print("LEAVE-ONE-WC-OUT")
    print(SEP)

    rows = []

    for holdout in HOLDOUTS:

        pts, err = validate_on_2022(
            candidate,
            holdout,
        )

        rows.append({
            "holdout": holdout,
            "points": pts,
        })

    return pd.DataFrame(rows)

CURRENT_CANDIDATE = {
    "decay_lambda": 0.20,
    "regularization": 0.001,
    "elite": 1.00,
    "caf": 1.10,
    "concacaf": 1.05,
    "afc": 0.95,
    "ofc": 0.90,
    "best_alpha": 0.30,
    "best_k": 1.15,
}

NEW_CANDIDATE = {
    "decay_lambda": 0.10,
    "regularization": 0.001,
    "elite": 1.00,
    "caf": 1.10,
    "concacaf": 1.05,
    "afc": 0.95,
    "ofc": 0.90,
    "best_alpha": 0.75,
    "best_k": 1.20,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("JOINT TUNE VALIDATION")
    print(SEP)

    df = load_results()
    df = df.sort_values("best_weighted_score", ascending=False).reset_index(drop=True)
    print()
    print(SEP)
    print("GRID SIZE")
    print(SEP)

    print(
        f"Total parameter combinations: "
        f"{len(df)}"
    )

    # ---------------------------------------------------------------------------
    # Find current production score in results for baseline comparison
    # ---------------------------------------------------------------------------

    current_mask = (
        (df["decay_lambda"]   == CURRENT_DECAY) &
        (df["regularization"] == CURRENT_REG) &
        (df["elite"]          == CURRENT_ELITE) &
        (df["caf"]            == CURRENT_CAF) &
        (df["concacaf"]       == CURRENT_CONCACAF) &
        (df["afc"]            == CURRENT_AFC) &
        (df["ofc"]            == CURRENT_OFC)
    )

    if current_mask.any():
        current_row = df[current_mask].iloc[0]
        baseline_weighted = current_row["best_weighted_score"]
        baseline_rank     = df[current_mask].index[0] + 1
        print(f"\nCurrent production params found at rank {baseline_rank}")
        print(f"  Weighted score: {baseline_weighted:.4f}")
        print(f"  Best alpha    : {current_row['best_alpha']}")
        print(f"  Best k        : {current_row['best_k']}")
    else:
        baseline_weighted = None
        print("\nCurrent production params not found in results "
              "(may not have been in this PC's half — check other CSV)")

    # ---------------------------------------------------------------------------
    # Top combinations overview
    # ---------------------------------------------------------------------------

    print(f"\n{SEP}")
    print(f"TOP {min(30, len(df))} TRAINING COMBINATIONS (by weighted score)")
    print(SEP)

    display_cols = [
        "combo_id", "decay_lambda", "regularization",
        "elite", "caf", "concacaf", "afc", "ofc",
        "best_alpha", "best_k", "best_weighted_score",
        "score_2006", "score_2010", "score_2014", "score_2018",
    ]
    print(df.head(30)[display_cols].to_string(index=False))

    # ---------------------------------------------------------------------------
    # Parameter distribution in top 50
    # ---------------------------------------------------------------------------

    print(f"\n{SEP}")
    print(
        f"PARAMETER DISTRIBUTION IN TOP {len(df)} COMBINATIONS"
    )
    print(SEP)

    top50 = df.copy()
    for col in ["decay_lambda", "regularization", "elite", "caf",
                "concacaf", "afc", "ofc", "best_alpha", "best_k"]:
        vc = top50[col].value_counts().sort_index()
        print(f"\n  {col}:")
        for val, cnt in vc.items():
            bar = "█" * cnt
            print(f"    {val:>8}: {cnt:3d}  {bar}")

    # ---------------------------------------------------------------------------
    # Validate top N on 2022
    # ---------------------------------------------------------------------------

    print(f"\n{SEP}")
    print(f"VALIDATING TOP {TOP_N_TO_VALIDATE} COMBINATIONS ON 2022 HOLDOUT")
    print(SEP)

    top_combos    = df.head(TOP_N_TO_VALIDATE)
    val_results   = []

    for i, (_, row) in enumerate(top_combos.iterrows()):
        print(f"  Validating {i+1}/{TOP_N_TO_VALIDATE}: combo_id={int(row['combo_id'])}  "
              f"(train weighted={row['best_weighted_score']:.2f}) ...", end=" ", flush=True)
        pts_2022, err = validate_on_2022(
            row,
            2022,
        )
        if err:
            print(f"ERROR: {err}")
        else:
            print(f"{pts_2022} pts on 2022")
        val_results.append({**row.to_dict(), "pts_2022": pts_2022, "val_error": err})

    val_df = pd.DataFrame(val_results)
    val_df = val_df[val_df["pts_2022"].notna()].sort_values(
        "pts_2022", ascending=False
    ).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # Validation summary
    # ---------------------------------------------------------------------------

    print(f"\n{SEP}")
    print("VALIDATION RESULTS — RANKED BY 2022 HOLDOUT SCORE")
    print(SEP)

    val_display = [
        "combo_id", "decay_lambda", "regularization",
        "elite", "caf", "concacaf", "afc", "ofc",
        "best_alpha", "best_k",
        "best_weighted_score", "pts_2022",
        "score_2014", "score_2018",
    ]
    print(val_df[val_display].to_string(index=False))

    # ---------------------------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------------------------

    print(f"\n{SEP}")
    print("FINAL RECOMMENDATION")
    print(SEP)

    # Primary: best on 2022 holdout
    best_val = val_df.iloc[0]

    # Also show best balanced (average rank on training + holdout)
    val_df["combined"] = (
        val_df["best_weighted_score"] / val_df["best_weighted_score"].max() +
        val_df["pts_2022"] / val_df["pts_2022"].max()
    )
    best_balanced = val_df.sort_values("combined", ascending=False).iloc[0]

    print(f"\n  CURRENT PRODUCTION PARAMS:")
    print(f"    decay_lambda={CURRENT_DECAY}  reg={CURRENT_REG}")
    print(f"    elite={CURRENT_ELITE}  caf={CURRENT_CAF}  "
          f"concacaf={CURRENT_CONCACAF}  afc={CURRENT_AFC}  ofc={CURRENT_OFC}")
    print(f"    alpha={CURRENT_ALPHA}  k={CURRENT_K}")
    if baseline_weighted:
        print(f"    Weighted training score: {baseline_weighted:.4f}")

    print(f"\n  BEST ON 2022 HOLDOUT:")
    for col in ["decay_lambda", "regularization", "elite", "caf",
                "concacaf", "afc", "ofc", "best_alpha", "best_k",
                "best_weighted_score", "pts_2022"]:
        print(f"    {col}: {best_val[col]}")

    print(f"\n  BEST BALANCED (training + holdout):")
    for col in ["decay_lambda", "regularization", "elite", "caf",
                "concacaf", "afc", "ofc", "best_alpha", "best_k",
                "best_weighted_score", "pts_2022"]:
        print(f"    {col}: {best_balanced[col]}")

    # Save
    val_df.to_csv(RESULTS_DIR / "validation_results.csv", index=False)

    rec_lines = [
        "JOINT TUNE FINAL RECOMMENDATION",
        "=" * 50,
        "",
        "BEST ON 2022 HOLDOUT:",
        f"  decay_lambda  = {best_val['decay_lambda']}",
        f"  regularization= {best_val['regularization']}",
        f"  elite         = {best_val['elite']}",
        f"  caf           = {best_val['caf']}",
        f"  concacaf      = {best_val['concacaf']}",
        f"  afc           = {best_val['afc']}",
        f"  ofc           = {best_val['ofc']}",
        f"  alpha         = {best_val['best_alpha']}",
        f"  goal_inflation= {best_val['best_k']}",
        f"  2022 pts      = {best_val['pts_2022']}",
        f"  weighted train= {best_val['best_weighted_score']:.4f}",
        "",
        "BEST BALANCED:",
        f"  decay_lambda  = {best_balanced['decay_lambda']}",
        f"  regularization= {best_balanced['regularization']}",
        f"  elite         = {best_balanced['elite']}",
        f"  caf           = {best_balanced['caf']}",
        f"  concacaf      = {best_balanced['concacaf']}",
        f"  afc           = {best_balanced['afc']}",
        f"  ofc           = {best_balanced['ofc']}",
        f"  alpha         = {best_balanced['best_alpha']}",
        f"  goal_inflation= {best_balanced['best_k']}",
        f"  2022 pts      = {best_balanced['pts_2022']}",
        f"  weighted train= {best_balanced['best_weighted_score']:.4f}",
    ]

    rec_path = RESULTS_DIR / "final_recommendation.txt"
    rec_path.write_text("\n".join(rec_lines))
    print(f"\nRecommendation saved to {rec_path}")
    print(f"Full validation results saved to {RESULTS_DIR / 'validation_results.csv'}")

    print()
    print(SEP)
    print("CURRENT VS NEW")
    print(SEP)

    current_cv = run_candidate_cv(
        CURRENT_CANDIDATE
    )

    new_cv = run_candidate_cv(
        NEW_CANDIDATE
    )

    comparison = (
        current_cv
        .merge(
            new_cv,
            on="holdout",
            suffixes=(
                "_current",
                "_new",
            )
        )
    )

    comparison["delta"] = (
        comparison["points_new"]
        -
        comparison["points_current"]
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Current total:",
        comparison[
            "points_current"
        ].sum()
    )

    print(
        "New total:",
        comparison[
            "points_new"
        ].sum()
    )

    print(
        "Delta:",
        comparison["delta"].sum()
    )


if __name__ == "__main__":
    main()