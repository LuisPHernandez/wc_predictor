"""
analyse_market_blend.py

2D Grid Search: Optimizes the global hyperparameters alpha (1X2 market trust) 
and beta (Total Goals market trust) across the modern World Cup era.
Includes the partially completed 2026 tournament to adapt to current market conditions.

Run from project root:
    py -3 analyse_market_blend.py
"""

import os
# Suppress numpy/scipy multithreading to prevent CPU thrashing
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

# Import your loaders and standard Dixon-Coles model
from src.loader import load_kaggle_data, load_pool_data, get_wc_teams
from src.model import DixonColes, PRODUCTION_ALPHA, PRODUCTION_GOAL_BLEND_BETA
from src.scoring import points_for_prediction
from src.odds_loader import load_wc_odds_lookup, load_wc_expected_goals_lookup
from backtest import get_training_window, DECAY_LAMBDA, REGULARIZATION

# Years to include in the master tune
TARGET_YEARS = [2014, 2018, 2022, 2026]

# --- 2D GRID SEARCH SPACE ---
# Alpha: 1.0 = Pure Model, 0.0 = Pure Market 1X2
ALPHAS = [round(a, 2) for a in np.arange(0.20, 0.65, 0.05)]

# Beta: 1.0 = Pure Model, 0.0 = Pure Market Total Goals
BETAS = [round(b, 2) for b in np.arange(0.00, 0.35, 0.05)]

SEP = "=" * 65

def evaluate_blend_combo(params):
    """
    Worker function: Evaluates a single (alpha, beta) combination across all target years.
    Returns the total points yielded.
    """
    test_alpha, test_beta = params
    
    total_pts = 0
    year_pts = {}

    for year in TARGET_YEARS:
        try:
            pool = load_pool_data(POOL_PATH, year)
            wc_teams = get_wc_teams(pool)
            games = pool['games']
            scores = pool['scores']
            actuals = scores.set_index('game_id')[['score1', 'score2']].to_dict('index')

            # Load Odds and xG
            odds_lookup = None
            xg_lookup = None
            try: odds_lookup = load_wc_odds_lookup(year)
            except: pass
            try: xg_lookup = load_wc_expected_goals_lookup(year)
            except: pass

            # Train the Model
            start_date, end_date = get_training_window(year)
            kaggle_df = load_kaggle_data(KAGGLE_PATH, wc_teams, start_date, end_date, DECAY_LAMBDA)
            
            # 🚀 INJECT THE TEST BETA INTO THE MODEL
            model = DixonColes(kaggle_df, decay_lambda=DECAY_LAMBDA, regularization=REGULARIZATION, goal_blend_beta=test_beta)
            model.fit()

            # Predict and Score
            y_pts = 0
            for row in games.itertuples():
                actual = actuals.get(row.game_id)
                if not actual: continue # This skips the unplayed 2026 matches automatically
                
                ah, aa = int(actual['score1']), int(actual['score2'])
                bk_probs = odds_lookup.get(row.game_id) if odds_lookup else None
                m_total = xg_lookup.get(row.game_id) if xg_lookup else None

                # 🚀 INJECT THE TEST ALPHA INTO PREDICT
                pred = model.predict(row.team1, row.team2, neutral=True, 
                                     market_total_goals=m_total, bookmaker_probs=bk_probs, 
                                     alpha=test_alpha)
                
                y_pts += points_for_prediction(pred['pred_home'], pred['pred_away'], ah, aa)

            year_pts[year] = y_pts
            total_pts += y_pts

        except Exception as e:
            print(f"\n[CRASH on {year}] alpha={test_alpha}, beta={test_beta}: {e}")

    return {
        "alpha": test_alpha,
        "beta": test_beta,
        "total_points": total_pts,
        "2014_pts": year_pts.get(2014, 0),
        "2018_pts": year_pts.get(2018, 0),
        "2022_pts": year_pts.get(2022, 0),
        "2026_pts": year_pts.get(2026, 0)
    }

def main():
    combos = [(a, b) for a in ALPHAS for b in BETAS]

    print(f"\n{SEP}")
    print("MARKET BLEND 2D GRID SEARCH (INCLUDING 2026 LIVE DATA)")
    print(f"Target Years : {TARGET_YEARS}")
    print(f"Total Combos : {len(combos)} iterations")
    print(f"Cores Used   : {max(1, os.cpu_count() - 2)}")
    print(f"Base Prod    : Alpha = {PRODUCTION_ALPHA:.2f}, Beta = {PRODUCTION_GOAL_BLEND_BETA:.2f}")
    print(f"{SEP}\n")

    start_time = time.time()
    results = []

    print("Spinning up worker pool... (This will take a few minutes)\n")

    n_workers = max(1, os.cpu_count() - 2)
    with ProcessPool(processes=n_workers) as pool:
        for count, res in enumerate(pool.imap_unordered(evaluate_blend_combo, combos), 1):
            best_pts = max([r['total_points'] for r in results] + [res['total_points']])
            print(f"[{count}/{len(combos)}] Finished α={res['alpha']:.2f}, β={res['beta']:.2f} | Max Pts: {best_pts}")
            results.append(res)

    elapsed = time.time() - start_time
    print(f"\nGrid Search finished in {elapsed:.1f} seconds.")

    # Leaderboard Processing
    df = pd.DataFrame(results)
    df = df.sort_values(by="total_points", ascending=False).reset_index(drop=True)

    print(f"\n{SEP}")
    print("TOP 15 MARKET BLEND CONFIGURATIONS")
    print(f"{SEP}")
    print(df[['alpha', 'beta', 'total_points', '2014_pts', '2018_pts', '2022_pts', '2026_pts']].head(15).to_string(index=False))

    best = df.iloc[0]
    
    # Calculate what the current production parameters scored
    prod_row = df[(df['alpha'] == PRODUCTION_ALPHA) & (df['beta'] == PRODUCTION_GOAL_BLEND_BETA)]
    
    print(f"\n{SEP}")
    print("CONCLUSION")
    print(f"{SEP}")
    
    if not prod_row.empty:
        prod_pts = prod_row.iloc[0]['total_points']
        print(f"Current Baseline (a={PRODUCTION_ALPHA:.2f}, b={PRODUCTION_GOAL_BLEND_BETA:.2f}) Yielded : {prod_pts} pts")
    else:
        print("Current Baseline was not included in the grid search sweep.")
        prod_pts = 0

    print(f"Optimal 2026 Era (a={best['alpha']:.2f}, b={best['beta']:.2f}) Yielded : {best['total_points']} pts")
    
    diff = best['total_points'] - prod_pts
    if diff > 0:
        print(f"\nVERDICT: The 2026 market has shifted. Updating parameters provides a +{diff} point edge.")
    else:
        print(f"\nVERDICT: The market has not shifted fundamentally. Your current production parameters are mathematically optimal.")

if __name__ == "__main__":
    freeze_support()
    main()