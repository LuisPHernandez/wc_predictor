"""
analyse_knockout_refit.py

2D Grid Search: Tests the hypothesis of refitting the Dixon-Coles model 
exactly once (after the Group Stage) using a combination of Recency Weight 
and strict L2 Regularization to capture "Live Tournament Form" safely.

Run from project root:
    py -3 analyse_knockout_refit.py
"""

import os
# Suppress numpy/scipy multithreading within workers to prevent CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from multiprocessing import Pool as ProcessPool, freeze_support

PROJECT_ROOT = Path(__file__).resolve().parent
KAGGLE_PATH = PROJECT_ROOT / 'data' / 'kaggle' / 'results.csv'
POOL_PATH   = PROJECT_ROOT / 'data' / 'pool'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_kaggle_data, load_pool_data, get_wc_teams
from src.model import DixonColes, PRODUCTION_ALPHA
from src.scoring import points_for_prediction
from src.odds_loader import load_wc_odds_lookup, load_wc_expected_goals_lookup
from backtest import get_training_window, DECAY_LAMBDA, REGULARIZATION, WC_START_DATES

# Years to backtest the knockout strategy
KNOCKOUT_YEARS = [2014, 2018, 2022]

# --- 2D GRID SEARCH PARAMETERS ---
# How heavily should the 48 Group Stage matches be weighted relative to history?
WEIGHT_SEARCH = [2.0, 3.0, 5.0, 8.0, 10.0]

# How strictly should we punish the solver for changing the pre-tournament baseline?
# (Baseline is 0.0010. We need higher friction here to prevent overfitting to 3 games)
REG_SEARCH = [0.0050, 0.0100, 0.0250, 0.0500, 0.1000]

SEP = "=" * 65

def evaluate_refit_combo(params):
    """
    Worker function to evaluate a single (weight, regularization) combination 
    across all specified World Cups.
    """
    live_weight, live_reg = params
    
    # Mute standard print statements to keep console clean during multiprocessing
    import sys as _sys
    class NullWriter:
        def write(self, text): pass
        def flush(self): pass
    old_stdout = _sys.stdout
    _sys.stdout = NullWriter()
    
    total_static_pts = 0
    total_live_pts = 0
    year_breakdown = {}

    for year in KNOCKOUT_YEARS:
        try:
            pool = load_pool_data(POOL_PATH, year)
            wc_teams = get_wc_teams(pool)
            games = pool['games']
            scores = pool['scores']
            actuals = scores.set_index('game_id')[['score1', 'score2']].to_dict('index')

            # Load Market Data
            odds_lookup = None
            xg_lookup = None
            try: odds_lookup = load_wc_odds_lookup(year)
            except: pass
            try: xg_lookup = load_wc_expected_goals_lookup(year)
            except: pass

            # 1. Train the Pre-Tournament Baseline Model
            start_date, end_date = get_training_window(year)
            kaggle_df = load_kaggle_data(KAGGLE_PATH, wc_teams, start_date, end_date, DECAY_LAMBDA)
            
            static_model = DixonColes(kaggle_df, decay_lambda=DECAY_LAMBDA, regularization=REGULARIZATION)
            static_model.fit()

            # Separate Group Stage and Knockout Stage matches
            group_games = games[games['phase'] == 'G']
            knockout_games = games[games['phase'] != 'G']

            year_static_pts = 0
            year_live_pts = 0
            
            group_stage_rows = []

            # 2. Process Group Stage (Both pipelines use Static Model here)
            for row in group_games.itertuples():
                actual = actuals.get(row.game_id)
                if not actual: continue
                
                ah, aa = int(actual['score1']), int(actual['score2'])
                bk_probs = odds_lookup.get(row.game_id) if odds_lookup else None
                m_total = xg_lookup.get(row.game_id) if xg_lookup else None

                pred = static_model.predict(row.team1, row.team2, neutral=True, 
                                            market_total_goals=m_total, bookmaker_probs=bk_probs, 
                                            alpha=PRODUCTION_ALPHA)
                
                pts = points_for_prediction(pred['pred_home'], pred['pred_away'], ah, aa)
                year_static_pts += pts
                year_live_pts += pts # Live pipeline scores exactly the same in Group Stage

                # Collect the completed match for the upcoming refit
                group_stage_rows.append({
                    'date': pd.Timestamp(WC_START_DATES[year]), # Treat all group games as happening at tournament start
                    'home_team': row.team1,
                    'away_team': row.team2,
                    'home_score': ah,
                    'away_score': aa,
                    'neutral': True,
                    'weight': live_weight,
                })

            # 3. The Halftime Refit
            wc_df = pd.DataFrame(group_stage_rows)
            live_df = pd.concat([kaggle_df, wc_df], ignore_index=True)
            
            live_model = DixonColes(live_df, decay_lambda=DECAY_LAMBDA, regularization=live_reg)
            live_model.fit()

            # 4. Process Knockout Stage (Pipelines diverge)
            for row in knockout_games.itertuples():
                actual = actuals.get(row.game_id)
                if not actual: continue
                
                ah, aa = int(actual['score1']), int(actual['score2'])
                bk_probs = odds_lookup.get(row.game_id) if odds_lookup else None
                m_total = xg_lookup.get(row.game_id) if xg_lookup else None

                # Static Prediction
                pred_static = static_model.predict(row.team1, row.team2, neutral=True, 
                                                   market_total_goals=m_total, bookmaker_probs=bk_probs, 
                                                   alpha=PRODUCTION_ALPHA)
                
                # Live Refit Prediction
                pred_live = live_model.predict(row.team1, row.team2, neutral=True, 
                                               market_total_goals=m_total, bookmaker_probs=bk_probs, 
                                               alpha=PRODUCTION_ALPHA)

                year_static_pts += points_for_prediction(pred_static['pred_home'], pred_static['pred_away'], ah, aa)
                year_live_pts += points_for_prediction(pred_live['pred_home'], pred_live['pred_away'], ah, aa)

            year_breakdown[year] = {
                'static': year_static_pts,
                'live': year_live_pts,
                'gain': year_live_pts - year_static_pts
            }
            total_static_pts += year_static_pts
            total_live_pts += year_live_pts

        except Exception as e:
            _sys.stdout = old_stdout
            print(f"\n[CRASH on {year}] w={live_weight}, r={live_reg}: {e}")
            _sys.stdout = NullWriter()

    _sys.stdout = old_stdout

    return {
        "weight": live_weight,
        "reg": live_reg,
        "total_static": total_static_pts,
        "total_live": total_live_pts,
        "total_gain": total_live_pts - total_static_pts,
        "breakdown": year_breakdown
    }

def main():
    combos = [(w, r) for w in WEIGHT_SEARCH for r in REG_SEARCH]

    print(f"\n{SEP}")
    print("POST-GROUP STAGE REFIT: 2D GRID SEARCH")
    print(f"Target Years: {KNOCKOUT_YEARS}")
    print(f"Total Combos: {len(combos)} iterations")
    print(f"Cores Used  : {max(1, os.cpu_count() - 2)}")
    print(f"{SEP}\n")

    start_time = time.time()
    results = []

    print("Spinning up worker pool... (This will take a few minutes)\n")

    n_workers = max(1, os.cpu_count() - 2)
    with ProcessPool(processes=n_workers) as pool:
        for count, res in enumerate(pool.imap_unordered(evaluate_refit_combo, combos), 1):
            if count % 5 == 0 or count == len(combos):
                best_gain = max([r['total_gain'] for r in results] + [res['total_gain']])
                print(f"[{count}/{len(combos)}] Processed... Best Gain so far: {best_gain:+} pts")
            results.append(res)

    elapsed = time.time() - start_time
    print(f"\nGrid Search finished in {elapsed:.1f} seconds.")

    # Leaderboard Processing
    df = pd.DataFrame(results)
    df = df.sort_values(by="total_gain", ascending=False).reset_index(drop=True)

    print(f"\n{SEP}")
    print("TOP 15 KNOCKOUT REFIT CONFIGURATIONS")
    print(f"{SEP}")
    print(df[['weight', 'reg', 'total_static', 'total_live', 'total_gain']].head(15).to_string(index=False))

    best = df.iloc[0]
    print(f"\n{SEP}")
    print("WINNING CONFIGURATION ANALYSIS")
    print(f"{SEP}")
    
    if best['total_gain'] <= 0:
        print("Verdict: The post-group stage refit FAILED to beat the static baseline.")
        print("Recommendation: Do not change parameters mid-tournament.")
    else:
        print(f"Live Weight Multiplier : {best['weight']}")
        print(f"Live Regularization    : {best['reg']} (Baseline was 0.001)")
        print(f"Net Gain in Knockouts  : +{best['total_gain']} points")
        print("\nYearly Breakdown of the Winning Rule:")
        
        bd = best['breakdown']
        for y in KNOCKOUT_YEARS:
            if y in bd:
                print(f"  {y}: Static {bd[y]['static']} | Live {bd[y]['live']} | Gain {bd[y]['gain']:+}")

if __name__ == "__main__":
    freeze_support()
    main()