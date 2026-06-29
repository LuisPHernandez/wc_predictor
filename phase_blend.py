"""
analyse_phase_blend.py

2D Grid Search Phase Split: Optimizes alpha and beta separately for 
Group Stage ('G') and Knockout Stage matches across 2014-2026.

Run from project root:
    py -3 analyse_phase_blend.py
"""

import os
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
from src.model import DixonColes, PRODUCTION_ALPHA, PRODUCTION_GOAL_BLEND_BETA
from src.scoring import points_for_prediction
from src.odds_loader import load_wc_odds_lookup, load_wc_expected_goals_lookup
from backtest import get_training_window, DECAY_LAMBDA, REGULARIZATION

TARGET_YEARS = [2014, 2018, 2022, 2026]

# --- 2D GRID SEARCH SPACE ---
ALPHAS = [round(a, 2) for a in np.arange(0.20, 0.65, 0.05)]
BETAS = [round(b, 2) for b in np.arange(0.00, 0.35, 0.05)]

SEP = "=" * 65

def evaluate_phase_combo(params):
    test_alpha, test_beta = params
    
    group_pts = 0
    knockout_pts = 0

    for year in TARGET_YEARS:
        try:
            pool = load_pool_data(POOL_PATH, year)
            wc_teams = get_wc_teams(pool)
            games = pool['games']
            scores = pool['scores']
            actuals = scores.set_index('game_id')[['score1', 'score2']].to_dict('index')

            odds_lookup = None
            xg_lookup = None
            try: odds_lookup = load_wc_odds_lookup(year)
            except: pass
            try: xg_lookup = load_wc_expected_goals_lookup(year)
            except: pass

            start_date, end_date = get_training_window(year)
            kaggle_df = load_kaggle_data(KAGGLE_PATH, wc_teams, start_date, end_date, DECAY_LAMBDA)
            
            model = DixonColes(kaggle_df, decay_lambda=DECAY_LAMBDA, regularization=REGULARIZATION, goal_blend_beta=test_beta)
            model.fit()

            for row in games.itertuples():
                actual = actuals.get(row.game_id)
                if not actual: continue 
                
                ah, aa = int(actual['score1']), int(actual['score2'])
                bk_probs = odds_lookup.get(row.game_id) if odds_lookup else None
                m_total = xg_lookup.get(row.game_id) if xg_lookup else None

                pred = model.predict(row.team1, row.team2, neutral=True, 
                                     market_total_goals=m_total, bookmaker_probs=bk_probs, 
                                     alpha=test_alpha)
                
                pts = points_for_prediction(pred['pred_home'], pred['pred_away'], ah, aa)
                
                # Split points based on tournament phase
                if row.phase == 'G':
                    group_pts += pts
                else:
                    knockout_pts += pts

        except Exception as e:
            print(f"\n[CRASH on {year}] alpha={test_alpha}, beta={test_beta}: {e}")

    return {
        "alpha": test_alpha,
        "beta": test_beta,
        "group_points": group_pts,
        "knockout_points": knockout_pts,
        "total_points": group_pts + knockout_pts
    }

def main():
    combos = [(a, b) for a in ALPHAS for b in BETAS]

    print(f"\n{SEP}")
    print("PHASE-SPLIT MARKET BLEND GRID SEARCH (2014 - 2026)")
    print(f"Target Years : {TARGET_YEARS}")
    print(f"Total Combos : {len(combos)} iterations")
    print(f"{SEP}\n")

    start_time = time.time()
    results = []

    n_workers = max(1, os.cpu_count() - 2)
    with ProcessPool(processes=n_workers) as pool:
        for count, res in enumerate(pool.imap_unordered(evaluate_phase_combo, combos), 1):
            if count % 10 == 0 or count == len(combos):
                print(f"[{count}/{len(combos)}] Finished α={res['alpha']:.2f}, β={res['beta']:.2f}")
            results.append(res)

    elapsed = time.time() - start_time
    print(f"\nGrid Search finished in {elapsed:.1f} seconds.")

    df = pd.DataFrame(results)

    # --- GROUP STAGE LEADERBOARD ---
    df_group = df.sort_values(by="group_points", ascending=False).reset_index(drop=True)
    print(f"\n{SEP}")
    print("TOP 10: GROUP STAGE PARAMETERS")
    print(f"{SEP}")
    print(df_group[['alpha', 'beta', 'group_points']].head(10).to_string(index=False))

    # --- KNOCKOUT STAGE LEADERBOARD ---
    df_ko = df.sort_values(by="knockout_points", ascending=False).reset_index(drop=True)
    print(f"\n{SEP}")
    print("TOP 10: KNOCKOUT STAGE PARAMETERS")
    print(f"{SEP}")
    print(df_ko[['alpha', 'beta', 'knockout_points']].head(10).to_string(index=False))

    # --- CONCLUSION ---
    print(f"\n{SEP}")
    print("PHASE DISCREPANCY ANALYSIS")
    print(f"{SEP}")
    best_group = df_group.iloc[0]
    best_ko = df_ko.iloc[0]
    
    print(f"Optimal Group Stage    : Alpha = {best_group['alpha']:.2f}, Beta = {best_group['beta']:.2f} ({best_group['group_points']} pts)")
    print(f"Optimal Knockout Stage : Alpha = {best_ko['alpha']:.2f}, Beta = {best_ko['beta']:.2f} ({best_ko['knockout_points']} pts)")
    
    if best_group['alpha'] != best_ko['alpha'] or best_group['beta'] != best_ko['beta']:
        print("\nVERDICT: The mathematical profile of the market shifts in the playoffs.")
        print("Recommendation: Implement dynamic parameters in model.py based on match phase.")
    else:
        print("\nVERDICT: A unified blend is mathematically optimal across all phases.")

if __name__ == "__main__":
    freeze_support()
    main()