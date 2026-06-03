import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import itertools
import pandas as pd
import numpy as np
from multiprocessing import Pool as ProcessPool
from src.loader import load_pool_data, get_wc_teams, build_competition_weights, load_kaggle_base_data
from src.model import DixonColes
from src.scoring import points_for_prediction

KAGGLE_PATH = 'data/kaggle/results.csv'
POOL_PATH   = 'data/pool'
POOL_CACHE = None
MEAN_USER_CACHE = None
BASE_KAGGLE_CACHE = None

WC_START_DATES = {
    2002: '2002-05-31',
    2006: '2006-06-09',
    2010: '2010-06-11',
    2014: '2014-06-12',
    2018: '2018-06-14',
    2022: '2022-11-20',
}

# Years with pool predictions to score against
TUNING_YEARS = [2002, 2006, 2010, 2018, 2022]

# Best hyperparameters found with hyperparameter_tune.py
DECAY_LAMBDA   = 0.2
TRAINING_YEARS = 12
REGULARIZATION = 0.0001

# Competition weights' search space
SEARCH_SPACE = {
    "continental": [0.8, 0.9, 1.0],
    "qualifier":   [0.5, 0.7, 1.0],
    "regional":    [0.3, 0.5, 1.0],
    "friendly":    [0.0, 0.1, 0.3, 1.0],
}

def get_training_window(wc_year, training_years):
    end = pd.Timestamp(WC_START_DATES[wc_year]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=training_years)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def compute_mean_user_points(year):
    """
    Computes the average total points scored by pool users in a given year.
    This is fixed, doesn't depend on any hyperparameters, so we compute
    it once upfront and reuse it across all tuning runs.
    """
    preds = pd.read_csv(f'{POOL_PATH}/{year}_predictions.csv', header=None,
                        names=['game_id', 'user_id', 'score1', 'score2'])

    scores_raw = pd.read_csv(f'{POOL_PATH}/{year}_scores.csv', header=None)
    if scores_raw.shape[1] == 4:
        scores_raw.columns = ['team1_code', 'team2_code', 'score1', 'score2']
    else:
        scores_raw.columns = ['phase', 'team1_code', 'team2_code', 'score1', 'score2']
    scores_raw['game_id'] = range(1, len(scores_raw) + 1)

    actuals = scores_raw.set_index('game_id')[['score1', 'score2']].to_dict('index')
    n_users = preds['user_id'].nunique()

    total_pts = sum(
        points_for_prediction(
            r.score1, r.score2,
            actuals[r.game_id]['score1'],
            actuals[r.game_id]['score2']
        )
        for r in preds.itertuples()
        if r.game_id in actuals
    )

    return total_pts / n_users

def run_single_year(year, continental, qualifier, regional, friendly, pool_cache, mean_user_cache):
    """
    Fits the model and scores it for a single WC year.
    Returns the model's margin over the mean user.

    pool_cache      : preloaded pool data per year (avoids re-reading CSVs)
    mean_user_cache : precomputed mean user points per year
    """
    t0 = time.time()
    
    pool = pool_cache[year]

    competition_weights = build_competition_weights(
        continental,
        qualifier,
        regional,
        friendly
    )

    kaggle_df = BASE_KAGGLE_CACHE[year].copy()

    kaggle_df['competition_weight'] = (
        kaggle_df['tournament']
        .map(competition_weights)
    )

    kaggle_df['weight'] = (
        kaggle_df['competition_weight']
        * kaggle_df['recency_weight']
    )

    kaggle_df = kaggle_df[
        [
            'date',
            'home_team',
            'away_team',
            'home_score',
            'away_score',
            'neutral',
            'weight'
        ]
    ]

    model = DixonColes(kaggle_df, decay_lambda=DECAY_LAMBDA,
                       regularization=REGULARIZATION)
    model.fit()
    print(f"Model fitted for {year}")

    # Score model predictions against actual results
    games  = pool['games']
    scores = pool['scores']
    actuals = scores.set_index('game_id')[['score1', 'score2']].to_dict('index')

    model_points = 0
    for row in games.itertuples():
        pred   = model.predict(row.team1, row.team2, neutral=True)
        actual = actuals.get(row.game_id)
        if actual is None:
            continue
        model_points += points_for_prediction(
            pred['pred_home'], pred['pred_away'],
            int(actual['score1']), int(actual['score2'])
        )

    margin = model_points - mean_user_cache[year]
    elapsed  = time.time() - t0
    return model_points, margin, elapsed

def init_worker():
    """
    Runs once when each worker process starts.
    Preloads data that never changes between combinations.
    """
    global POOL_CACHE
    global MEAN_USER_CACHE
    global BASE_KAGGLE_CACHE

    POOL_CACHE = {
        year: load_pool_data(POOL_PATH, year)
        for year in TUNING_YEARS
    }

    MEAN_USER_CACHE = {
        year: compute_mean_user_points(year)
        for year in TUNING_YEARS
    }

    BASE_KAGGLE_CACHE = {}

    for year in TUNING_YEARS:
        pool = POOL_CACHE[year]
        wc_teams = get_wc_teams(pool)

        start_date, end_date = get_training_window(
            year,
            TRAINING_YEARS
        )

        BASE_KAGGLE_CACHE[year] = load_kaggle_base_data(
            KAGGLE_PATH,
            wc_teams,
            start_date,
            end_date,
            DECAY_LAMBDA
        )

    print(
        f"Worker initialized "
        f"(pid={os.getpid()})"
    )

def run_combo(args):
    continental, qualifier, regional, friendly = args

    year_margins = {}
    year_points  = {}
    year_times   = {}

    for year in TUNING_YEARS:
        pts, margin, elapsed = run_single_year(
            year,
            continental, 
            qualifier, 
            regional, 
            friendly,
            POOL_CACHE, 
            MEAN_USER_CACHE
        )
        year_margins[year] = margin
        year_points[year]  = pts
        year_times[year]   = elapsed

    avg_margin = np.mean(list(year_margins.values()))
    combo_time = sum(year_times.values())

    row = {
        "continental": continental,
        "qualifier": qualifier,
        "regional": regional,
        "friendly": friendly,
        "avg_margin": avg_margin,
        'combo_time_s':   round(combo_time, 1),
    }
    for year in TUNING_YEARS:
        row[f'pts_{year}']    = year_points[year]
        row[f'margin_{year}'] = year_margins[year]
        row[f'time_{year}']   = round(year_times[year], 1)

    return row

def run_tuning():
    """
    Grid search over all competition weight combinations.
    For each combination, fits the model on all tuning years and
    computes the average margin over mean user.
    Saves results to competition_tune_results.csv as it goes.
    """
    mean_user_cache  = {year: compute_mean_user_points(year) for year in TUNING_YEARS}

    print("\nMean user points per year:")
    for year, pts in mean_user_cache.items():
        print(f"  {year}: {pts:.1f}")

    # Build all combinations
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    # Load existing results if resuming
    try:
        existing = pd.read_csv('competition_tune_results_v2.csv')
        completed = set(
            zip(existing['continental'],
                existing['qualifier'],
                existing['regional'],
                existing['friendly'])
        )
        results = existing.to_dict('records')
        print(f"Resuming... {len(completed)} combinations already done")
    except FileNotFoundError:
        completed = set()
        results   = []

    # Filter to only remaining combos
    remaining = [
        (dict(zip(keys, combo))['continental'],
         dict(zip(keys, combo))['qualifier'],
         dict(zip(keys, combo))['regional'],
         dict(zip(keys, combo))['friendly'])
        for combo in combos
        if (dict(zip(keys, combo))['continental'],
            dict(zip(keys, combo))['qualifier'],
            dict(zip(keys, combo))['regional'],
            dict(zip(keys, combo))['friendly']) not in completed
    ]

    print(f"\n{total - len(remaining)} done, {len(remaining)} remaining")
    print(f"Running with {os.cpu_count()} logical cores")
    print("=" * 70)

    # Number of parallel workers: leave 2 cores free for the OS
    n_workers = max(1, os.cpu_count() - 2)

    # Build args list, pool_cache and mean_user_cache passed to each worker
    args_list = [
        (continental, qualifier, regional, friendly)
        for continental, qualifier, regional, friendly in remaining
    ]

    with ProcessPool(processes=n_workers, initializer=init_worker) as p:
        for i, row in enumerate(p.imap_unordered(run_combo, args_list)):
            results.append(row)
            print(f"[{len(results)}/{total}] "
                  f"continental weight={row['continental']}, "
                  f"qualifier weight={row['qualifier']}, "
                  f"regional weight={row['regional']}, "
                  f"friendly weight={row['friendly']} → "
                  f"avg_margin={row['avg_margin']:+.2f} "
                  f"({row['combo_time_s']:.0f}s)")

            # Save after every completed combo
            pd.DataFrame(results).sort_values(
                'avg_margin', ascending=False
            ).to_csv('competition_tune_results_v2.csv', index=False)

    results_df = pd.DataFrame(results).sort_values('avg_margin', ascending=False)

    print(f"\n{'='*70}")
    print("TOP 10 COMBINATIONS")
    print(f"{'='*70}")
    print(results_df.head(10)[[
        'continental', 'qualifier', 'regional', 'friendly', 'avg_margin'
    ]].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest combination:")
    print(f"  continental weight:   {best['continental']}")
    print(f"  qualifier weight: {best['qualifier']}")
    print(f"  regional weight: {best['regional']}")
    print(f"  friendly weight: {best['friendly']}")
    print(f"  avg_margin:     {best['avg_margin']:+.2f}")

    return results_df

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # needed on Windows
    results = run_tuning()