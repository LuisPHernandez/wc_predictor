import pickle
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.odds_loader import _shin_probs
from implied_xg import implied_expected_goals

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CACHE_PATH = PROJECT_ROOT / "data" / "model" / "dixon_coles_model_2026.pkl"
HIST_OUTPUT_PATH = PROJECT_ROOT / "predictions_history.csv"

def predict_single_match(
    home_team,
    away_team,
    match_date,
    home_odds,
    draw_odds,
    away_odds,
    ou_line,
    over_odds,
    under_odds,
    alpha=0.40,
    home_ou_line=None,
    home_over_odds=None,
    home_under_odds=None, 
    away_ou_line=None,
    away_over_odds=None,
    away_under_odds=None
):
    if not MODEL_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Fitted model state file not found at {MODEL_CACHE_PATH.name}.\n"
            f"Please run predict_matches.py once to export the serialized pickle state first."
        )

    # 1. Load frozen model parameters instantly (No fitting overhead)
    with open(MODEL_CACHE_PATH, "rb") as f:
        model = pickle.load(f)

    # 2. De-bias outcome odds using Shin's iteration method
    p_home, p_draw, p_away = _shin_probs(home_odds, draw_odds, away_odds)
    bookmaker_probs = {"home": p_home, "draw": p_draw, "away": p_away}

    # 3. Calculate exact un-juiced total goal volume via Brent's method
    exact_implied_xg = implied_expected_goals(
        line=ou_line,
        over_odds=over_odds,
        under_odds=under_odds
    )

    # 3b. Calculate explicitly requested Team Totals (for Override Engine)
    market_home_lambda = None
    market_away_lambda = None
    
    if all(v is not None for v in [home_ou_line, home_over_odds, home_under_odds]):
        market_home_lambda = implied_expected_goals(home_ou_line, home_over_odds, home_under_odds)
        
    if all(v is not None for v in [away_ou_line, away_over_odds, away_under_odds]):
        market_away_lambda = implied_expected_goals(away_ou_line, away_over_odds, away_under_odds)

    # 4. Generate predictions with optimized calibration parameters
    pred = model.predict(
        home_team,
        away_team,
        neutral=True,
        market_total_goals=exact_implied_xg,
        market_home_lambda=market_home_lambda,
        market_away_lambda=market_away_lambda,
        bookmaker_probs=bookmaker_probs,
        alpha=alpha
    )

    # Standardize runtime execution timestamp
    run_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # 5. Build output dictionary structure exactly matching history columns
    match_result = {
        "match_date": match_date,
        "prediction_timestamp": run_timestamp,
        "alpha_used": alpha,
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "ou_line": ou_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "calculated_implied_xg": round(exact_implied_xg, 4),
        "pred_home": pred["pred_home"],
        "pred_away": pred["pred_away"],
        "prediction": pred["prediction"],
        "expected_pts": round(pred["expected_pts"], 4),
        "second_best_prediction": pred["second_prediction"],
        "second_best_expected_pts": round(pred["second_expected_pts"], 4),
        "decision_margin": round(pred["decision_margin"], 4),
        "lambda_home": round(pred["lambda_home"], 4),
        "lambda_away": round(pred["lambda_away"], 4),
        "lambda_total": round(pred["lambda_home"] + pred["lambda_away"], 4),
        "home_win": round(pred["home_win"], 4),
        "draw": round(pred["draw"], 4),
        "away_win": round(pred["away_win"], 4),
        "raw_model_home": round(pred["raw_model_home"], 4),
        "raw_model_draw": round(pred["raw_model_draw"], 4),
        "raw_model_away": round(pred["raw_model_away"], 4),
        "raw_market_home": round(p_home, 4),
        "raw_market_draw": round(p_draw, 4),
        "raw_market_away": round(p_away, 4),
    }

    return match_result

def append_to_history_if_changed(new_record):
    """
    Appends the prediction to predictions_history.csv only if the market 
    parameters or model output changed from the last recorded entry.
    """
    new_df = pd.DataFrame([new_record])

    if HIST_OUTPUT_PATH.exists():
        historical_df = pd.read_csv(HIST_OUTPUT_PATH)
        
        # Isolate the absolute latest record for this specific matchup
        latest_hist = historical_df.sort_values("prediction_timestamp").groupby(
            ["match_date", "home_team", "away_team"], observed=False
        ).last()
        
        key = (str(new_record["match_date"]), str(new_record["home_team"]), str(new_record["away_team"]))
        
        if key in latest_hist.index:
            hist_row = latest_hist.loc[key]
            has_changed = False
            
            # Exclude contextual columns from identity matching
            exclude_columns = ["prediction_timestamp", "match_date", "home_team", "away_team"]
            compare_columns = [c for c in new_df.columns if c not in exclude_columns]
            
            for col in compare_columns:
                v_new = new_record[col]
                v_old = hist_row[col]
                
                if pd.isna(v_new) and pd.isna(v_old):
                    continue
                if pd.isna(v_new) or pd.isna(v_old):
                    has_changed = True
                    break
                    
                try:
                    if np.isclose(float(v_new), float(v_old), atol=1e-6):
                        continue
                except (ValueError, TypeError):
                    pass
                    
                if v_new != v_old:
                    has_changed = True
                    break
            
            if has_changed:
                combined_history = pd.concat([historical_df, new_df], ignore_index=True)
                print(f"--> Market shift or prediction update detected. Appended entry to history log.")
            else:
                combined_history = historical_df
                print("--> Match variables are identical to the last logged snapshot. History file left un-appended.")
        else:
            # Completely new fixture line entry
            combined_history = pd.concat([historical_df, new_df], ignore_index=True)
            print("--> Fresh fixture instance noted. Appended entry to history log.")
    else:
        combined_history = new_df
        print("--> No prior file found. Initializing predictions_history.csv.")

    # Re-sort to maintain clean timeline sequencing blocks inside the sheet
    combined_history = combined_history.sort_values(
        by=["match_date", "home_team", "away_team", "prediction_timestamp"],
        ascending=[True, True, True, True]
    )
    combined_history.to_csv(HIST_OUTPUT_PATH, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone command-line interface for individual fixture evaluations.")
    
    parser.add_argument("--home_team", required=True, type=str)
    parser.add_argument("--away_team", required=True, type=str)
    parser.add_argument("--match_date", required=True, type=str)
    parser.add_argument("--home_odds", required=True, type=float)
    parser.add_argument("--draw_odds", required=True, type=float)
    parser.add_argument("--away_odds", required=True, type=float)
    parser.add_argument("--ou_line", required=True, type=float)
    parser.add_argument("--over_odds", required=True, type=float)
    parser.add_argument("--under_odds", required=True, type=float)

    args = parser.parse_args()

    try:
        record = predict_single_match(
            home_team=args.home_team, away_team=args.away_team, match_date=args.match_date,
            home_odds=args.home_odds, draw_odds=args.draw_odds, away_odds=args.away_odds,
            ou_line=args.ou_line, over_odds=args.over_odds, under_odds=args.under_odds
        )
        
        print("\n" + "=" * 60)
        print(f"EVALUATION: {record['home_team']} vs {record['away_team']} ({record['match_date']})")
        print("=" * 60)
        print(f"  -> RECOMMENDED SCORELINE : {record['prediction']} ({record['expected_pts']} EV Pts)")
        print(f"  -> RUNNER-UP OPTIONS     : {record['second_best_prediction']} ({record['second_best_expected_pts']} EV Pts)")
        print(f"  -> PRECISE IMPLIED XG    : {record['calculated_implied_xg']}")
        print(f"  -> DE-BIASED WIN RATIO   : H: {record['home_win']*100:.1f}% | D: {record['draw']*100:.1f}% | A: {record['away_win']*100:.1f}%\n")
        print("\n" + "=" * 60)
        print(f"EVALUATION: {record['home_team']} vs {record['away_team']} ({record['match_date']})")
        print("=" * 60)
        print(f"  -> RECOMMENDED SCORELINE : {record['prediction']} ({record['expected_pts']} EV Pts)")
        print(f"  -> PRECISE IMPLIED XG    : {record['calculated_implied_xg']}")
        print(f"  -> LAMBDAS (O/U Blended) : H: {record['lambda_home']} | A: {record['lambda_away']}")
        
        print("\n--- PROBABILITY TUG-OF-WAR ---")
        print(f"  RAW MODEL (12yr Hist) : H: {record['raw_model_home']*100:.1f}% | D: {record['raw_model_draw']*100:.1f}% | A: {record['raw_model_away']*100:.1f}%")
        print(f"  RAW MARKET (Vegas)    : H: {record['raw_market_home']*100:.1f}% | D: {record['raw_market_draw']*100:.1f}% | A: {record['raw_market_away']*100:.1f}%")
        print(f"  FINAL BLEND (a=0.40)  : H: {record['home_win']*100:.1f}% | D: {record['draw']*100:.1f}% | A: {record['away_win']*100:.1f}%\n")
        
        # Safe history flushing step
        append_to_history_if_changed(record)

    except Exception as e:
        print(f"Runtime Execution Error: {e}")