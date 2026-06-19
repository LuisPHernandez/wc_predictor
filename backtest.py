import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool as ProcessPool
from src.loader import load_kaggle_data, load_pool_data, get_wc_teams
from src.model import DixonColes, PRODUCTION_ALPHA, PRODUCTION_GOAL_BLEND_BETA
from src.model import DixonColes
from src.scoring import points_for_prediction
from src.odds_loader import (
    load_wc_odds_lookup,
    load_wc_expected_goals_lookup,
)

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
DECAY_LAMBDA = 0.2
REGULARIZATION = 0.0010

def get_training_window(wc_year):
    """
    Returns (start_date, end_date) for the Kaggle training data.
    end_date   = day before the WC's first game
    start_date = start of the year 8 years before end_date
    """
    end   = pd.Timestamp(WC_START_DATES[wc_year]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=TRAINING_YEARS)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def score_predictions(predictions_df, scores_df):
    """
    Scores all predictions in predictions_df against actual results.

    Parameters
    ----------
    predictions_df : pd.DataFrame with columns game_id, user_id, score1, score2
    scores_df      : pd.DataFrame with columns game_id, score1, score2

    Returns
    -------
    pd.DataFrame with columns user_id, total_points
    sorted descending by total_points
    """
    # Build a quick lookup dict: game_id -> (actual_score1, actual_score2)
    actuals = scores_df.set_index('game_id')[['score1', 'score2']].to_dict('index')

    results = []
    for row in predictions_df.itertuples():
        actual = actuals.get(row.game_id)
        if actual is None:
            continue
        pts = points_for_prediction(
            row.score1, row.score2,
            actual['score1'], actual['score2']
        )
        results.append({'user_id': row.user_id, 'points': pts})

    df = pd.DataFrame(results)
    return (
        df.groupby('user_id')['points']
        .sum()
        .reset_index()
        .rename(columns={'points': 'total_points'})
        .sort_values('total_points', ascending=False)
        .reset_index(drop=True)
    )

def score_model(model_preds_df, scores_df):
    """
    Scores the model's predictions against actual results.

    Parameters
    ----------
    model_preds_df : pd.DataFrame with columns game_id, pred_home, pred_away
    scores_df      : pd.DataFrame with columns game_id, score1, score2

    Returns
    -------
    int — total points scored by the model
    """
    actuals = scores_df.set_index('game_id')[['score1', 'score2']].to_dict('index')

    total = 0
    for row in model_preds_df.itertuples():
        actual = actuals.get(row.game_id)
        if actual is None:
            continue
        total += points_for_prediction(
            row.pred_home, row.pred_away,
            actual['score1'], actual['score2']
        )
    return total

def run_backtest(
    wc_year,
    decay_lambda=DECAY_LAMBDA,
    training_years=TRAINING_YEARS,
    alpha=PRODUCTION_ALPHA
):
    """
    Runs the full backtest for a single WC year.

    Returns a dict with:
      year          — the WC year
      model_points  — total points the model would have scored
      n_games       — number of games scored
      user_ranking  — DataFrame of user scores (None if no predictions available)
      model_rank    — where the model would have ranked (None if no predictions)
      n_users       — number of users in the pool (0 if no predictions)
      model_preds   — DataFrame of model's predictions with actual scores
    """
    print(f"\n{'='*50}")
    print(f"Backtesting {wc_year} World Cup")
    print(f"Alpha: {alpha:.2f}")
    print(f"{'='*50}")

    # --- Load pool data ---
    pool = load_pool_data(POOL_PATH, wc_year)
    wc_teams = get_wc_teams(pool)
    print(f"WC teams: {len(wc_teams)}")

    # --- Training window ---
    start_date, end_date = get_training_window(wc_year)
    print(f"Training window: {start_date} → {end_date}")

    # --- Load and filter Kaggle data ---
    kaggle_df = load_kaggle_data(
        KAGGLE_PATH, wc_teams, start_date, end_date, decay_lambda
    )
    print(f"Training matches: {len(kaggle_df)}")

    # --- Fit model ---
    model = DixonColes(kaggle_df, decay_lambda=decay_lambda, regularization=REGULARIZATION, goal_blend_beta=PRODUCTION_GOAL_BLEND_BETA,)
    print(model.goal_blend_beta)
    model.fit()

    # --- Generate model predictions for every game ---
    games  = pool['games']
    scores = pool['scores']

    odds_lookup = None
    expected_goals_lookup = None

    if alpha < 1.0:
        try:
            odds_lookup = load_wc_odds_lookup(wc_year)
        except (KeyError, FileNotFoundError):
            print(f"  No 1X2 odds available for {wc_year}")
            
        try:
            expected_goals_lookup = load_wc_expected_goals_lookup(wc_year)
        except (KeyError, FileNotFoundError):
            print(f"  No market xG available for {wc_year} — skipping beta blend")

    model_preds = []
    for row in games.itertuples():
        bookmaker_probs = None
        market_total_goals = None

        if odds_lookup is not None:
            bookmaker_probs = odds_lookup.get(row.game_id)

        if expected_goals_lookup is not None:
            market_total_goals = (
                expected_goals_lookup.get(
                    row.game_id
                )
            )
            
        pred = model.predict(
            row.team1,
            row.team2,
            neutral=True,
            market_total_goals=market_total_goals,
            bookmaker_probs=bookmaker_probs,
            alpha=alpha,
        )
        model_preds.append({
            'game_id':   row.game_id,
            'team1':     row.team1,
            'team2':     row.team2,
            'pred_home': pred['pred_home'],
            'pred_away': pred['pred_away'],
            'prediction': pred['prediction'],
            'expected_pts': pred['expected_pts'],
        })

    model_preds_df = pd.DataFrame(model_preds)

    # Attach actual scores for reference
    model_preds_df = model_preds_df.merge(
        scores[['game_id', 'score1', 'score2']],
        on='game_id', how='left'
    )
    model_preds_df['actual'] = (
        model_preds_df['score1'].astype(int).astype(str) + '-' +
        model_preds_df['score2'].astype(int).astype(str)
    )
    model_preds_df['points_earned'] = model_preds_df.apply(
        lambda r: points_for_prediction(
            r['pred_home'], r['pred_away'],
            int(r['score1']), int(r['score2'])
        ), axis=1
    )

    # --- Score the model ---
    model_points = int(model_preds_df['points_earned'].sum())
    n_games      = len(model_preds_df)
    print(f"Model total points: {model_points} / {n_games} games")

    # --- Score pool users (if predictions available) ---
    user_ranking = None
    model_rank   = None
    n_users      = 0

    if pool['predictions'] is not None and len(pool['predictions']) > 0:
        user_ranking = score_predictions(pool['predictions'], scores)
        n_users      = len(user_ranking)

        # Find where model would rank
        # Count users who scored equal or more than the model
        model_rank = int((user_ranking['total_points'] >= model_points).sum()) + 1

        print(f"Model rank: {model_rank} out of {n_users} users "
              f"(top {100*model_rank/n_users:.0f}%)")
    else:
        print("No pool predictions available for this year — model score only")

    if user_ranking is not None and pool['predictions'] is not None:
        top_user_id = user_ranking.iloc[0]['user_id']
        top_user_preds = (
            pool['predictions'][pool['predictions']['user_id'] == top_user_id]
            [['game_id', 'score1', 'score2']]
            .rename(columns={'score1': 'top_pred_home', 'score2': 'top_pred_away'})
        )
        model_preds_df = model_preds_df.merge(top_user_preds, on='game_id', how='left')

        has_pred = model_preds_df['top_pred_home'].notna()

        model_preds_df.loc[has_pred, 'top_prediction'] = (
            model_preds_df.loc[has_pred, 'top_pred_home'].astype(int).astype(str) + '-' +
            model_preds_df.loc[has_pred, 'top_pred_away'].astype(int).astype(str)
        )
        model_preds_df['top_points_earned'] = model_preds_df.apply(
            lambda r: points_for_prediction(
                int(r['top_pred_home']), int(r['top_pred_away']),
                int(r['score1']),        int(r['score2']),
            ) if pd.notna(r['top_pred_home']) else None,
            axis=1
        )
        model_preds_df['point_diff'] = (
            model_preds_df['points_earned'] - model_preds_df['top_points_earned']
        )
        print(f"\nTop user: {top_user_id} ({int(user_ranking.iloc[0]['total_points'])} pts)")
        print(f"Model vs top user per game:")
        print(
            model_preds_df[[
                'team1', 'team2', 'prediction', 'top_prediction',
                'actual', 'points_earned', 'top_points_earned', 'point_diff'
            ]].to_string(index=False)
        )
        model_preds_df['top_user_id'] = top_user_id

    return {
        'year':         wc_year,
        'model_points': model_points,
        'n_games':      n_games,
        'user_ranking': user_ranking,
        'model_rank':   model_rank,
        'n_users':      n_users,
        'model_preds':  model_preds_df,
    }

def run_backtest_worker(args):
    year, decay_lambda, training_years = args
    return run_backtest(year, decay_lambda=decay_lambda, training_years=training_years)

def run_all_backtests(years=None, decay_lambda=DECAY_LAMBDA):
    """
    Runs backtests for all available years and prints a summary table.

    Parameters
    ----------
    years        : list of ints, e.g. [2018, 2022]. Defaults to all available.
    decay_lambda : float — passed to every backtest run
    """
    if years is None:
        years = [2002, 2006, 2010, 2014, 2018, 2022]

    args = [(year, decay_lambda, TRAINING_YEARS) for year in years]

    n_workers = min(len(years), max(1, os.cpu_count() - 2))
    print(f"Running {len(years)} backtests across {n_workers} workers...")

    with ProcessPool(processes=n_workers) as p:
        results = p.map(run_backtest_worker, args)

    results.sort(key=lambda r: r['year'])

    # Summary table
    print(f"\n{'='*50}")
    print("BACKTEST SUMMARY")
    print(f"{'='*50}")
    print(f"{'Year':<6} {'Points':<8} {'Games':<7} {'Rank':<6} {'Users':<7} {'Top %':<8}")
    print("-" * 44)
    for r in results:
        rank_str = str(r['model_rank']) if r['model_rank'] else 'N/A'
        pct_str  = (f"{100*r['model_rank']/r['n_users']:.0f}%"
                    if r['model_rank'] else 'N/A')
        print(f"{r['year']:<6} {r['model_points']:<8} {r['n_games']:<7} "
              f"{rank_str:<6} {r['n_users']:<7} {pct_str:<8}")

    return results


if __name__ == "__main__":
    import sys
    from multiprocessing import freeze_support

    freeze_support()

    # No arguments -> run all backtests
    if len(sys.argv) == 1:
        run_all_backtests()

    # One argument -> year
    elif len(sys.argv) == 2:

        year = int(sys.argv[1])

        result = run_backtest(
            year,
        )

        print(
            f"\nFinal score: "
            f"{result['model_points']} points"
        )

    # Two arguments -> year + alpha
    elif len(sys.argv) == 3:

        year = int(sys.argv[1])
        alpha = float(sys.argv[2])

        result = run_backtest(
            year,
            alpha=alpha
        )

        print(
            f"\nFinal score: "
            f"{result['model_points']} points"
        )

    else:
        raise ValueError(
            "Usage:\n"
            "  py -3 backtest.py\n"
            "  py -3 backtest.py <year>\n"
            "  py -3 backtest.py <year> <alpha>"
        )