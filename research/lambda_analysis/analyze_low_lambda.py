import numpy as np
import pandas as pd
import sys
from scipy.stats import poisson # pyrefly: ignore [missing-import]
from pathlib import Path

from src.model import blend_matrix_outcomes, DixonColes
from src.scoring import points_for_prediction

CACHE_PATH = Path("data/analysis/dc_baselines_cache.csv")

if not CACHE_PATH.exists():
    print("Please run find_global_max.py first to generate the persistent cache file.")
    sys.exit()

def evaluate_low_lambda_pool(k_test, max_goals=8):
    # Fixed production constants from your optimal 2018+2022 run
    ALPHA = 0.42
    BETA = 0.14
    THRESHOLD = 2.2
    
    df = pd.read_csv(CACHE_PATH)
    records = df.to_dict('records')
    
    low_lambda_matches = 0
    total_points = 0
    scoreline_counts = {}
    prediction_distribution = {}
    
    for row in records:
        model_total = row['lh_raw'] + row['la_raw']
        blend_total = BETA * model_total + (1 - BETA) * row['market_total']
        
        # Only analyze the low-scoring regime
        if blend_total > THRESHOLD:
            continue
            
        low_lambda_matches += 1
        
        # Track actual scorelines in this bucket
        actual_str = f"{int(row['actual_home'])}-{int(row['actual_away'])}"
        scoreline_counts[actual_str] = scoreline_counts.get(actual_str, 0) + 1
        
        # Apply the test k multiplier directly to the blended baseline
        lh = blend_total * (row['lh_raw'] / model_total) * k_test
        la = blend_total * (row['la_raw'] / model_total) * k_test
        
        # Build score probability matrix
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = (
                    poisson.pmf(i, lh) * poisson.pmf(j, la) * DixonColes._tau(i, j, lh, la, row['rho'])
                )
                
        if not np.isnan(row['p_home']):
            bookmaker_probs = {'home': row['p_home'], 'draw': row['p_draw'], 'away': row['p_away']}
            matrix = blend_matrix_outcomes(matrix, bookmaker_probs, ALPHA)
            
        # Argmax selection logic
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
                    
        pred_str = f"{best_pred[0]}-{best_pred[1]}"
        prediction_distribution[pred_str] = prediction_distribution.get(pred_str, 0) + 1
        
        total_points += points_for_prediction(best_pred[0], best_pred[1], int(row['actual_home']), int(row['actual_away']))
        
    return total_points, low_lambda_matches, scoreline_counts, prediction_distribution

if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING TARGETED LOW-LAMBDA PARADOX ANALYSIS")
    print("=" * 70)
    
    # 1. Inspect the points curve across various k values
    k_values = [0.85, 0.90, 0.93, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
    results = []
    
    sample_counts = 0
    actual_scores = None
    
    for k in k_values:
        pts, count, actual_scores, pred_dist = evaluate_low_lambda_pool(k)
        sample_counts = count
        results.append({'k_multiplier': k, 'pool_points': pts, 'top_predictions': sorted(pred_dist.items(), key=lambda x: x[1], reverse=True)[:3]})
        
    df_results = pd.DataFrame(results)
    print(f"\nAnalyzed {sample_counts} matches where Blended Lambda <= 2.2")
    print("\nPOINTS EARNED BY SCALAR STRATEGY:")
    print(df_results.to_string(index=False))
    
    # 2. Print out the true underlying distribution of reality
    print("\n" + "-"*50)
    print("TOP 10 MOST FREQUENT ACTUAL SCORE LINES IN THIS BUCKET:")
    print("-"*50)
    sorted_actuals = sorted(actual_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for score, freq in sorted_actuals:
        percentage = (freq / sample_counts) * 100
        print(f"  Scoreline {score:5s} : {freq:2d} occurrences ({percentage:.1f}%)")