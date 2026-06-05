import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool as ProcessPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_pool_data, get_wc_teams, build_competition_weights, build_confederation_weights, load_kaggle_base_data
from src.model import DixonColes
from src.scoring import points_for_prediction
from src.odds_loader import load_wc_odds_lookup

KAGGLE_PATH = PROJECT_ROOT / 'data' / 'kaggle' / 'results.csv'
POOL_PATH   = PROJECT_ROOT / 'data' / 'pool'
RESULTS_PATH = PROJECT_ROOT / 'tuners' / 'results' / 'odds_blend_tune_results.csv'
POOL_CACHE = None
MEAN_USER_CACHE = None
BASE_KAGGLE_CACHE = None
ODDS_CACHE = None

WC_START_DATES = {
    2002: '2002-05-31',
    2006: '2006-06-09',
    2010: '2010-06-11',
    2014: '2014-06-12',
    2018: '2018-06-14',
    2022: '2022-11-20',
}

# Years with pool predictions to score against
TUNING_YEARS = [2006, 2010, 2018, 2022]

# Best hyperparameters found with hyperparameter_tune.py and weights found with competition_tune.py and confederation_tune.py
DECAY_LAMBDA   = 0.2
TRAINING_YEARS = 12
REGULARIZATION = 0.0010
CONTINENTAL = 1.0
QUALIFIER   = 0.5
REGIONAL    = 0.3
FRIENDLY    = 0.3
CONMEBOL = 1.0
UEFA     = 1.0
CAF      = 1.10
CONCACAF = 1.05
AFC      = 0.95
OFC      = 0.90

# Alpha search space
SEARCH_SPACE = {
    "alpha": [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]
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

def run_single_year(year, alpha, pool_cache, mean_user_cache):
    """
    Fits the model and scores it for a single WC year.
    Returns the model's margin over the mean user.

    pool_cache      : preloaded pool data per year (avoids re-reading CSVs)
    mean_user_cache : precomputed mean user points per year
    """
    t0 = time.time()
    
    pool = pool_cache[year]

    kaggle_df = BASE_KAGGLE_CACHE[year].copy()

    model = DixonColes(kaggle_df, decay_lambda=DECAY_LAMBDA,
                       regularization=REGULARIZATION)
    model.fit()
    print(f"Model fitted for {year}")

    # Score model predictions against actual results
    games  = pool['games']
    scores = pool['scores']
    actuals = scores.set_index('game_id')[['score1', 'score2']].to_dict('index')

    odds_lookup = ODDS_CACHE[year]

    model_points = 0
    for row in games.itertuples():
        bookmaker_probs = odds_lookup.get(
            row.game_id
        )

        pred = model.predict(
            row.team1,
            row.team2,
            neutral=True,
            bookmaker_probs=bookmaker_probs,
            alpha=alpha,
        )

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
    global ODDS_CACHE

    POOL_CACHE = {
        year: load_pool_data(POOL_PATH, year)
        for year in TUNING_YEARS
    }

    MEAN_USER_CACHE = {
        year: compute_mean_user_points(year)
        for year in TUNING_YEARS
    }

    BASE_KAGGLE_CACHE = {}

    ODDS_CACHE = {
        year: load_wc_odds_lookup(year)
        for year in TUNING_YEARS
    }

    competition_weights = build_competition_weights(
        CONTINENTAL,
        QUALIFIER,
        REGIONAL,
        FRIENDLY,
    )

    confederation_weights = build_confederation_weights(
        CONMEBOL,
        CAF,
        CONCACAF,
        AFC,
        OFC,
    )

    for year in TUNING_YEARS:
        pool = POOL_CACHE[year]
        wc_teams = get_wc_teams(pool)

        start_date, end_date = get_training_window(
            year,
            TRAINING_YEARS
        )

        base_df = load_kaggle_base_data(
            KAGGLE_PATH,
            wc_teams,
            start_date,
            end_date,
            DECAY_LAMBDA
        )

        base_df['competition_weight'] = (
            base_df['tournament']
            .map(competition_weights)
        )

        home_conf_weight = (
            base_df['home_confederation']
            .map(confederation_weights)
            .fillna(1.0)
        )

        away_conf_weight = (
            base_df['away_confederation']
            .map(confederation_weights)
            .fillna(1.0)
        )

        base_df['confederation_weight'] = np.sqrt(
            home_conf_weight * away_conf_weight
        )

        base_df['weight'] = (
            base_df['competition_weight']
            * base_df['recency_weight']
            * base_df['confederation_weight']
        )

        BASE_KAGGLE_CACHE[year] = base_df[
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

    print(
        f"Worker initialized "
        f"(pid={os.getpid()})"
    )

def run_combo(args):
    alpha = args

    year_margins = {}
    year_points  = {}
    year_times   = {}

    for year in TUNING_YEARS:
        pts, margin, elapsed = run_single_year(
            year,
            alpha,
            POOL_CACHE, 
            MEAN_USER_CACHE
        )
        year_margins[year] = margin
        year_points[year]  = pts
        year_times[year]   = elapsed

    avg_margin = np.mean(list(year_margins.values()))
    combo_time = sum(year_times.values())

    row = {
        "alpha": alpha,
        "avg_margin": avg_margin,
        "combo_time_s": round(combo_time, 1),
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
    Saves results to confederation_tune_results.csv as it goes.
    """
    mean_user_cache  = {year: compute_mean_user_points(year) for year in TUNING_YEARS}

    print("\nMean user points per year:")
    for year, pts in mean_user_cache.items():
        print(f"  {year}: {pts:.1f}")

    # Build all combinations
    total = len(
        SEARCH_SPACE["alpha"]
    )

    # Load existing results if resuming
    try:
        existing = pd.read_csv(RESULTS_PATH)
        completed = set(
            existing['alpha']
        )
        results = existing.to_dict('records')
        print(f"Resuming... {len(completed)} combinations already done")
    except FileNotFoundError:
        completed = set()
        results   = []

    # Filter to only remaining combos
    remaining = [
        alpha
        for alpha in SEARCH_SPACE["alpha"]
        if alpha not in completed
    ]

    print(f"\n{total - len(remaining)} done, {len(remaining)} remaining")
    print(f"Running with {os.cpu_count()} logical cores")
    print("=" * 70)

    # Number of parallel workers: leave 2 cores free for the OS
    n_workers = max(1, os.cpu_count() - 2)

    # Build args list, pool_cache and mean_user_cache passed to each worker
    args_list = remaining

    with ProcessPool(processes=n_workers, initializer=init_worker) as p:
        for i, row in enumerate(p.imap_unordered(run_combo, args_list)):
            results.append(row)
            print(
                f"[{len(results)}/{total}] "
                f"alpha={row['alpha']:.2f} → "
                f"avg_margin={row['avg_margin']:+.2f} "
                f"({row['combo_time_s']:.0f}s)"
            )

            # Save after every completed combo
            pd.DataFrame(results).sort_values(
                'avg_margin', ascending=False
            ).to_csv(RESULTS_PATH, index=False)

    results_df = pd.DataFrame(results).sort_values('avg_margin', ascending=False)

    print(f"\n{'='*70}")
    print("TOP 10 COMBINATIONS")
    print(f"{'='*70}")
    print(results_df.head(10)[[
        'alpha',
        'avg_margin'
    ]].to_string(index=False))

    best = results_df.iloc[0]
    print("\nBest combination:")
    print(f"  alpha:      {best['alpha']:.2f}")
    print(f"  avg_margin: {best['avg_margin']:+.2f}")

    return results_df

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # needed on Windows
    results = run_tuning()
