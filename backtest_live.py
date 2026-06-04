import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool as ProcessPool
from src.loader import load_kaggle_data, load_pool_data, get_wc_teams
from src.model import DixonColes
from src.scoring import points_for_prediction

PROJECT_ROOT = Path(__file__).resolve().parent
KAGGLE_PATH = PROJECT_ROOT / 'data' / 'kaggle' / 'results.csv'
POOL_PATH   = PROJECT_ROOT / 'data' / 'pool'

# First game date per WC
WC_START_DATES = {
    2002: '2002-05-31',
    2006: '2006-06-09',
    2010: '2010-06-11',
    2014: '2014-06-12',
    2018: '2018-06-14',
    2022: '2022-11-20',
}

# Tuned hyperparameters
TRAINING_YEARS = 12
DECAY_LAMBDA   = 0.2
REGULARIZATION = 0.0001

# Weight applied to completed WC matches when retraining
WC_MATCH_WEIGHT = 5.0

def get_training_window(wc_year):
    end   = pd.Timestamp(WC_START_DATES[wc_year]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=TRAINING_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def build_matchday_groups(games_df):
    """
    Splits games into logical matchdays:
      - Group stage: 3 rounds of 16 games each (by order in CSV)
      - Knockout: one round per phase code (8, 4, 2, 1, 0)

    Returns a list of DataFrames, each representing one matchday.
    """
    group_games    = games_df[games_df['phase'] == 'G'].copy()
    knockout_games = games_df[games_df['phase'] != 'G'].copy()

    matchdays = []

    # Group stage — split by position into 3 rounds of 16
    n = len(group_games)
    round_size = n // 3
    for i in range(3):
        start = i * round_size
        end   = start + round_size if i < 2 else n
        matchdays.append(group_games.iloc[start:end])

    # Knockout — one round per phase, in order
    for phase in ['8', '4', '2', '1', '0']:
        round_games = knockout_games[knockout_games['phase'] == phase]
        if len(round_games) > 0:
            matchdays.append(round_games)

    return matchdays

def run_live_backtest(wc_year):
    """
    Simulates live retraining for a single WC year.
    Retrains before each logical matchday using all completed WC games.

    Returns a dict with:
      year              — WC year
      static_points     — points from static pre-tournament model
      live_points       — points from live retraining model
      gain              — live_points - static_points
      game_details      — per-game DataFrame comparing static vs live
    """
    print(f"\n{'='*50}")
    print(f"Live backtest {wc_year}")
    print(f"{'='*50}")

    pool     = load_pool_data(POOL_PATH, wc_year)
    wc_teams = get_wc_teams(pool)
    games    = pool['games']
    scores   = pool['scores']

    start_date, end_date = get_training_window(wc_year)

    # Pre-tournament training data, shared base for all retrains
    kaggle_df = load_kaggle_data(
        KAGGLE_PATH, wc_teams, start_date, end_date, DECAY_LAMBDA
    )

    # Static model, trained once on pre-tournament data only
    static_model = DixonColes(kaggle_df, decay_lambda=DECAY_LAMBDA,
                              regularization=REGULARIZATION)
    static_model.fit()

    matchdays = build_matchday_groups(games)

    completed_game_ids = []   # game_ids seen so far
    completed_rows     = []   # iterable rows for wc training data

    all_game_details = []
    static_points_total = 0
    live_points_total   = 0

    actuals = scores.set_index('game_id')[['score1', 'score2']].to_dict('index')

    for md_idx, matchday in enumerate(matchdays):
        phase_name = _phase_name(matchday.iloc[0]['phase'], md_idx)
        print(f"\n  Matchday {md_idx+1} ({phase_name, wc_year}): {len(matchday)} games")

        # Build live training data: kaggle + completed WC matches
        if completed_rows:
            wc_rows  = pd.DataFrame(completed_rows)
            live_df  = pd.concat([kaggle_df, wc_rows], ignore_index=True)
        else:
            live_df = kaggle_df

        live_model = DixonColes(live_df, decay_lambda=DECAY_LAMBDA,
                                regularization=REGULARIZATION)
        live_model.fit()

        # Predict and score every game in this matchday
        for game in matchday.itertuples():
            actual = actuals.get(game.game_id)
            if actual is None:
                continue

            ah, aa = int(actual['score1']), int(actual['score2'])

            static_pred = static_model.predict(game.team1, game.team2, neutral=True)
            live_pred   = live_model.predict(game.team1, game.team2, neutral=True)

            static_pts = points_for_prediction(
                static_pred['pred_home'], static_pred['pred_away'], ah, aa)
            live_pts = points_for_prediction(
                live_pred['pred_home'], live_pred['pred_away'], ah, aa)

            static_points_total += static_pts
            live_points_total   += live_pts

            all_game_details.append({
                'matchday':      md_idx + 1,
                'phase':         phase_name,
                'team1':         game.team1,
                'team2':         game.team2,
                'static_pred':   static_pred['prediction'],
                'live_pred':     live_pred['prediction'],
                'actual':        f"{ah}-{aa}",
                'static_pts':    static_pts,
                'live_pts':      live_pts,
                'gain':          live_pts - static_pts,
            })

            # Mark as completed for next matchday's retraining
            completed_rows.append({
                'date':       pd.Timestamp(WC_START_DATES[wc_year]),
                'home_team':  game.team1,
                'away_team':  game.team2,
                'home_score': ah,
                'away_score': aa,
                'neutral':    True,
                'weight':     WC_MATCH_WEIGHT,
            })

        print(f"    Static: {static_points_total} pts | "
              f"Live: {live_points_total} pts | "
              f"Gain so far: {live_points_total - static_points_total:+d}")

    gain = live_points_total - static_points_total
    print(f"\n  FINAL — Static: {static_points_total} | "
          f"Live: {live_points_total} | Gain: {gain:+d}")

    return {
        'year':          wc_year,
        'static_points': static_points_total,
        'live_points':   live_points_total,
        'gain':          gain,
        'game_details':  pd.DataFrame(all_game_details),
    }

def _phase_name(phase_code, md_idx):
    if phase_code == 'G':
        return f"Group Round {md_idx + 1}"
    return {
        '8': 'Round of 16',
        '4': 'Quarter-finals',
        '2': 'Semi-finals',
        '1': '3rd Place',
        '0': 'Final',
    }.get(phase_code, phase_code)

def run_all_live_backtests(years=None):
    if years is None:
        years = [2002, 2006, 2010, 2014, 2018, 2022]

    n_workers = min(len(years), max(1, os.cpu_count() - 2))
    print(f"Running {len(years)} live backtests across {n_workers} workers...")

    with ProcessPool(processes=n_workers) as p:
        results = p.map(run_live_backtest, years)

    results.sort(key=lambda r: r['year'])

    print(f"\n{'='*60}")
    print("LIVE RETRAINING BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Year':<6} {'Static':>8} {'Live':>8} {'Gain':>8}")
    print("-" * 34)
    for r in results:
        print(f"{r['year']:<6} {r['static_points']:>8} "
              f"{r['live_points']:>8} {r['gain']:>+8}")

    total_gain = sum(r['gain'] for r in results)
    print("-" * 34)
    print(f"{'Total':<6} {'':>8} {'':>8} {total_gain:>+8}")
    print(f"{'Avg':<6} {'':>8} {'':>8} {total_gain/len(results):>+8.1f}")

    return results

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    results = run_all_live_backtests()