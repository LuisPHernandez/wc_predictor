import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from scipy.stats import poisson # pyrefly: ignore [missing-import]
import os
import sys
from multiprocessing import Pool, freeze_support

from src.model import blend_matrix_outcomes, DixonColes
from src.scoring import points_for_prediction
from src.loader import load_kaggle_data, load_pool_data, get_wc_teams
from src.odds_loader import load_wc_odds_lookup, load_wc_expected_goals_lookup

PROJECT_ROOT = Path(__file__).resolve().parent
KAGGLE_PATH  = PROJECT_ROOT / 'data' / 'kaggle' / 'results.csv'
POOL_PATH    = PROJECT_ROOT / 'data' / 'pool'
CACHE_PATH   = PROJECT_ROOT / 'data' / 'analysis' / 'dc_baselines_cache.csv'

TRAINING_YEARS = 12
REGULARIZATION = 0.0010
WC_START_DATES = {2014: '2014-06-12', 2018: '2018-06-14', 2022: '2022-11-20'}

GLOBAL_RECORDS = None

def get_training_window(wc_year):
    end = pd.Timestamp(WC_START_DATES[wc_year]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=TRAINING_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def get_or_create_persistent_cache(years=[2014, 2018, 2022]):
    if CACHE_PATH.exists():
        print(f"--> Found persistent cache at {CACHE_PATH.name}. Loading baselines instantly...")
        return pd.read_csv(CACHE_PATH)
        
    print("--> No persistent cache found. Initiating one-time Dixon-Coles model fitting phase...")
    flat_rows = []
    
    for year in years:
        pool = load_pool_data(POOL_PATH, year)
        wc_teams = get_wc_teams(pool)
        start_date, end_date = get_training_window(year)
        
        kaggle_df = load_kaggle_data(KAGGLE_PATH, wc_teams, start_date, end_date, decay_lambda=0.2)
        model = DixonColes(kaggle_df, decay_lambda=0.2, regularization=REGULARIZATION)
        model.fit()
        
        odds_lookup = load_wc_odds_lookup(year)
        expected_goals_lookup = load_wc_expected_goals_lookup(year)
        scores = pool['scores'].set_index('game_id')[['score1', 'score2']].to_dict('index')
        
        for row in pool['games'].itertuples():
            actual = scores.get(row.game_id)
            if actual is None:
                continue
                
            lh_raw, la_raw = model._get_lambda(row.team1, row.team2, neutral=True)
            probs = odds_lookup.get(row.game_id, {'home': np.nan, 'draw': np.nan, 'away': np.nan})
            
            flat_rows.append({
                'year': year,
                'game_id': row.game_id,
                'lh_raw': lh_raw,
                'la_raw': la_raw,
                'market_total': expected_goals_lookup.get(row.game_id, lh_raw + la_raw),
                'p_home': probs.get('home', np.nan),
                'p_draw': probs.get('draw', np.nan),
                'p_away': probs.get('away', np.nan),
                'actual_home': int(actual['score1']),
                'actual_away': int(actual['score2']),
                'rho': model.fitted_params[2 * model.n_teams + 1]
            })
            
    df_cache = pd.DataFrame(flat_rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_cache.to_csv(CACHE_PATH, index=False)
    print(f"--> Generated and saved persistent cache to {CACHE_PATH}.")
    return df_cache

def evaluate_all_layers(row, alpha, beta, k_low, k_high, lambda_low, lambda_high, max_goals=8):
    model_total = row['lh_raw'] + row['la_raw']
    
    blend_total = beta * model_total + (1 - beta) * row['market_total']
    lh = blend_total * (row['lh_raw'] / model_total)
    la = blend_total * (row['la_raw'] / model_total)
    
    total = lh + la
    if total <= lambda_low:
        k = k_low
    elif total >= lambda_high:
        k = k_high
    else:
        frac = (total - lambda_low) / (lambda_high - lambda_low)
        k = k_low + frac * (k_high - k_low)
        
    lh *= k
    la *= k
    
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = (
                poisson.pmf(i, lh) * poisson.pmf(j, la) * DixonColes._tau(i, j, lh, la, row['rho'])
            )
            
    if not np.isnan(row['p_home']):
        bookmaker_probs = {'home': row['p_home'], 'draw': row['p_draw'], 'away': row['p_away']}
        matrix = blend_matrix_outcomes(matrix, bookmaker_probs, alpha)
        
    best_pred = (0, 0)
    best_ep = -1.0
    for ph in range(max_goals):
        for pa in range(max_goals):
            ep = sum(
                matrix[ah, aa] * points_for_prediction(ph, pa, ah, aa)
                for ah in range(max_goals)
                for aa in range(max_goals)
            )
            if ep > best_ep:
                best_ep = ep
                best_pred = (ph, pa)
                
    return points_for_prediction(best_pred[0], best_pred[1], int(row['actual_home']), int(row['actual_away']))

def init_worker(records):
    global GLOBAL_RECORDS
    GLOBAL_RECORDS = records

def worker_sweep(combo):
    alpha, beta, l_low, l_high, k_low, k_high = combo
    yearly_scores = {2014: 0, 2018: 0, 2022: 0}
    
    for match in GLOBAL_RECORDS:
        pts = evaluate_all_layers(match, alpha, beta, k_low, k_high, l_low, l_high)
        yearly_scores[match['year']] += pts
        
    target_score = yearly_scores[2018] + yearly_scores[2022]
    return target_score, combo, yearly_scores

def run_global_tuning():
    df_matches = get_or_create_persistent_cache()
    records = df_matches.to_dict('records')
    
    alpha_space       = [0.40]
    beta_space        = [0.10]
    lambda_low_space  = [2.0]       
    lambda_high_space = [2.5]
    k_low_space       = [0.93]
    k_high_space      = [1.00, 1.01]
    
    raw_combinations = list(product(alpha_space, beta_space, lambda_low_space, lambda_high_space, k_low_space, k_high_space))
    combinations = [c for c in raw_combinations if c[2] < c[3]]
    
    total_combos = len(combinations)
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Deploying parallel grid search across {num_workers} CPU cores...")
    print(f"Total valid structural variations to evaluate: {total_combos}\n")
    
    best_score = -1
    best_params = {}
    completed = 0
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(records,)) as pool:
        for total_score, combo, yearly_scores in pool.imap_unordered(worker_sweep, combinations, chunksize=4):
            completed += 1
            
            if total_score >= best_score:
                best_score = total_score
                alpha, beta, l_low, l_high, k_low, k_high = combo
                best_params = {
                    'alpha': alpha, 'beta': beta,
                    'l_low': l_low, 'l_high': l_high,
                    'k_low': k_low, 'k_high': k_high,
                    'breakdown': yearly_scores
                }
                # Clear line and print peak results on a new, permanent line
                sys.stdout.write('\r' + ' ' * 90 + '\r')
                print(f"--> [CORE PEAK] {total_score} pts | α={alpha}, β={beta}, Limits=({l_low}, {l_high}), Scalars=({k_low}, {k_high}) | {yearly_scores}")
            
            # Continuous dynamic inline progress reporting
            pct_complete = (completed / total_combos) * 100
            sys.stdout.write(f"\rProgress: {completed}/{total_combos} evaluated ({pct_complete:.1f}%) | Current Best: {best_score if best_score != -1 else 'N/A'} pts")
            sys.stdout.flush()

    print("\n\n" + "="*60)
    print("GLOBAL MULTI-CORE MAXIMIZATION COMPLETE")
    print("="*60)
    print(f"Max Global Pool Points: {best_score} (Old Benchmark: 189)")
    print(f"Optimized Configuration Matrix:")
    print(f"  PRODUCTION_ALPHA           = {best_params['alpha']:.2f}")
    print(f"  PRODUCTION_GOAL_BLEND_BETA = {best_params['beta']:.2f}")
    print(f"  PRODUCTION_K_LOW_LAMBDA    = {best_params['l_low']:.1f}")
    print(f"  PRODUCTION_K_HIGH_LAMBDA   = {best_params['l_high']:.1f}")
    print(f"  PRODUCTION_K_LOW           = {best_params['k_low']:.2f}")
    print(f"  PRODUCTION_K_HIGH          = {best_params['k_high']:.2f}")
    print(f"Yearly Breakdown: {best_params['breakdown']}")

if __name__ == '__main__':
    freeze_support()
    run_global_tuning()