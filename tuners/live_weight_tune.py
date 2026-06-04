import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool as ProcessPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_kaggle_data, load_pool_data, get_wc_teams
from src.model import DixonColes
from src.scoring import points_for_prediction

KAGGLE_PATH = PROJECT_ROOT / 'data' / 'kaggle' / 'results.csv'
POOL_PATH   = PROJECT_ROOT / 'data' / 'pool'

WC_START_DATES = {
    2002: '2002-05-31',
    2006: '2006-06-09',
    2010: '2010-06-11',
    2014: '2014-06-12',
    2018: '2018-06-14',
    2022: '2022-11-20',
}

TRAINING_YEARS = 12
DECAY_LAMBDA   = 0.2
REGULARIZATION = 0.0001

WEIGHT_SEARCH = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
TUNING_YEARS  = [2002, 2006, 2010, 2014, 2018, 2022]

def get_training_window(wc_year):
    end   = pd.Timestamp(WC_START_DATES[wc_year]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=TRAINING_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def build_matchday_groups(games_df):
    group_games    = games_df[games_df['phase'] == 'G'].copy()
    knockout_games = games_df[games_df['phase'] != 'G'].copy()

    matchdays = []

    n = len(group_games)
    round_size = n // 3
    for i in range(3):
        start = i * round_size
        end   = start + round_size if i < 2 else n
        matchdays.append(group_games.iloc[start:end])

    for phase in ['8', '4', '2', '1', '0']:
        round_games = knockout_games[knockout_games['phase'] == phase]
        if len(round_games) > 0:
            matchdays.append(round_games)

    return matchdays

def run_year_with_weight(wc_year, wc_match_weight, kaggle_df, pool):
    """
    Runs live retraining simulation for one year with a given weight.
    Returns (static_points, live_points).
    """
    games  = pool['games']
    scores = pool['scores']

    # Static model — trained once
    static_model = DixonColes(
        kaggle_df,
        decay_lambda=DECAY_LAMBDA,
        regularization=REGULARIZATION
    )
    static_model.fit()

    matchdays      = build_matchday_groups(games)
    completed_rows = []
    actuals        = scores.set_index('game_id')[['score1', 'score2']].to_dict('index')

    static_total = 0
    live_total   = 0

    for matchday in matchdays:
        # Build live training data
        if completed_rows:
            wc_df   = pd.DataFrame(completed_rows)
            live_df = pd.concat([kaggle_df, wc_df], ignore_index=True)
        else:
            live_df = kaggle_df

        live_model = DixonColes(
            live_df,
            decay_lambda=DECAY_LAMBDA,
            regularization=REGULARIZATION
        )
        live_model.fit()

        for game in matchday.itertuples():
            actual = actuals.get(game.game_id)
            if actual is None:
                continue

            ah, aa = int(actual['score1']), int(actual['score2'])

            static_pred = static_model.predict(game.team1, game.team2, neutral=True)
            live_pred   = live_model.predict(game.team1, game.team2, neutral=True)

            static_total += points_for_prediction(
                static_pred['pred_home'], static_pred['pred_away'], ah, aa)
            live_total += points_for_prediction(
                live_pred['pred_home'], live_pred['pred_away'], ah, aa)

            completed_rows.append({
                'date':       pd.Timestamp(WC_START_DATES[wc_year]),
                'home_team':  game.team1,
                'away_team':  game.team2,
                'home_score': ah,
                'away_score': aa,
                'neutral':    True,
                'weight':     wc_match_weight,
            })

    return static_total, live_total

def run_weight_combo(args):
    """
    Worker function — runs all years for a single weight value.
    Loads data independently (required for multiprocessing).
    """
    wc_match_weight = args

    year_static = {}
    year_live   = {}

    for year in TUNING_YEARS:
        pool     = load_pool_data(POOL_PATH, year)
        wc_teams = get_wc_teams(pool)

        start_date, end_date = get_training_window(year)
        kaggle_df = load_kaggle_data(
            KAGGLE_PATH, wc_teams, start_date, end_date, DECAY_LAMBDA
        )

        static_pts, live_pts = run_year_with_weight(
            year, wc_match_weight, kaggle_df, pool
        )

        year_static[year] = static_pts
        year_live[year]   = live_pts

    avg_static = np.mean(list(year_static.values()))
    avg_live   = np.mean(list(year_live.values()))
    avg_gain   = avg_live - avg_static

    row = {
        'wc_match_weight': wc_match_weight,
        'avg_static':      round(avg_static, 2),
        'avg_live':        round(avg_live, 2),
        'avg_gain':        round(avg_gain, 2),
    }
    for year in TUNING_YEARS:
        row[f'static_{year}'] = year_static[year]
        row[f'live_{year}']   = year_live[year]
        row[f'gain_{year}']   = year_live[year] - year_static[year]

    print(f"weight={wc_match_weight:<6} → "
          f"avg_static={avg_static:.1f}, "
          f"avg_live={avg_live:.1f}, "
          f"avg_gain={avg_gain:+.2f}")

    return row

def run_tuning():
    print(f"Searching {len(WEIGHT_SEARCH)} weight values across {len(TUNING_YEARS)} years")
    print(f"Running with {os.cpu_count()} logical cores")
    print("=" * 60)

    # Load existing results if resuming
    try:
        existing = pd.read_csv('live_weight_tune_results.csv')
        completed = set(existing['wc_match_weight'].tolist())
        results = existing.to_dict('records')
        print(f"Resuming... {len(completed)} weights already done")
    except FileNotFoundError:
        completed = set()
        results   = []

    remaining = [w for w in WEIGHT_SEARCH if w not in completed]
    print(f"{len(completed)} done, {len(remaining)} remaining")
    print("=" * 60)

    n_workers = max(1, os.cpu_count() - 2)

    with ProcessPool(processes=n_workers) as p:
        for row in p.imap_unordered(run_weight_combo, remaining):
            results.append(row)
            pd.DataFrame(results).sort_values(
                'avg_gain', ascending=False
            ).to_csv('live_weight_tune_results.csv', index=False)
            print(f"Saved {len(results)}/{len(WEIGHT_SEARCH)} completed")

    results_df = pd.DataFrame(results).sort_values('avg_gain', ascending=False)

    print(f"\n{'='*60}")
    print("RESULTS SORTED BY AVG GAIN")
    print(f"{'='*60}")
    gain_cols = ['wc_match_weight', 'avg_static', 'avg_live', 'avg_gain'] + \
                [f'gain_{y}' for y in TUNING_YEARS]
    print(results_df[gain_cols].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest weight: {best['wc_match_weight']} → avg_gain={best['avg_gain']:+.2f}")

    return results_df

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    results = run_tuning()