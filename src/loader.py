import pandas as pd
import numpy as np # pyrefly: ignore [missing-import] 
from src.mappings import code_to_name, FIFA_NAME_TO_CONFEDERATION

# Competition weights
DEFAULT_COMPETITION_WEIGHTS = {
    'FIFA World Cup':                       1.0,
    'UEFA Euro':                            1.0,
    'Copa América':                         1.0,
    'AFC Asian Cup':                        1.0,
    'Gold Cup':                             1.0,
    'African Cup of Nations':               1.0,
    'Confederations Cup':                   1.0,
    'CONCACAF Championship':                1.0,
    'FIFA World Cup qualification':         0.5,
    'UEFA Euro qualification':              0.5,
    'CONCACAF Nations League':              0.5,
    'UEFA Nations League':                  0.5,
    'African Cup of Nations qualification': 0.5,
    'AFC Asian Cup qualification':          0.5,
    'Gulf Cup':                             0.3,
    'Arab Cup':                             0.3,
    'AFF Championship':                     0.3,
    'CFU Caribbean Cup':                    0.3,
    'Friendly':                             0.3,
}

# Confederation weights
DEFAULT_CONFEDERATION_WEIGHTS = {
    "CONMEBOL": 1.20,
    "UEFA":     1.15,
    "CAF":      1.05,
    "CONCACAF": 1.05,
    "AFC":      1.00,
    "OFC":      0.95,
}

def build_competition_weights(
    continental=0.9,
    qualifier=0.7,
    regional=0.5,
    friendly=0.3
):
    """
    Creates a full tournament->weight mapping from a small
    set of bucket weights.
    """

    return {
        'FIFA World Cup':                       1.0,

        'UEFA Euro':                            continental,
        'Copa América':                         continental,
        'AFC Asian Cup':                        continental,
        'Gold Cup':                             continental,
        'African Cup of Nations':               continental,
        'Confederations Cup':                   continental,
        'CONCACAF Championship':                continental,

        'FIFA World Cup qualification':         qualifier,
        'UEFA Euro qualification':              qualifier,
        'CONCACAF Nations League':              qualifier,
        'UEFA Nations League':                  qualifier,
        'African Cup of Nations qualification': qualifier,
        'AFC Asian Cup qualification':          qualifier,

        'Gulf Cup':                             regional,
        'Arab Cup':                             regional,
        'AFF Championship':                     regional,
        'CFU Caribbean Cup':                    regional,

        'Friendly':                             friendly,
    }

def build_confederation_weights(
    elite=1.15,
    caf=1.05,
    concacaf=1.05,
    afc=1.00,
    ofc=0.95,
):
    """
    Creates a full confederation->weight mapping from a small
    set of bucket weights.
    """
    return {
        "CONMEBOL": elite,
        "UEFA": elite,
        "CAF": caf,
        "CONCACAF": concacaf,
        "AFC": afc,
        "OFC": ofc,
    }

def load_kaggle_data(path, wc_teams, start_date, end_date, decay_lambda=0.2, competition_weights=None, confederation_weights=None):
    """
    Loads the Kaggle results.csv and returns a filtered, weighted DataFrame
    ready for Dixon-Coles fitting.

    Parameters
    ----------
    path         : str  — path to results.csv
    wc_teams     : list — full country names for the WC being backtested
                          (used to filter matches to relevant teams only)
    start_date   : str  — earliest date to include, e.g. '2014-01-01'
    end_date     : str  — latest date to include, e.g. '2022-11-19'
                          (should be the day before the WC's first game)
    decay_lambda : float — controls how fast older matches lose weight.
                           Higher = older matches matter less. Default 0.2.

    Returns
    -------
    pd.DataFrame with columns:
        date, home_team, away_team, home_score, away_score,
        neutral, weight
    """
    if competition_weights is None:
        competition_weights = DEFAULT_COMPETITION_WEIGHTS

    if confederation_weights is None:
        confederation_weights = DEFAULT_CONFEDERATION_WEIGHTS

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    # Filter to known competitions only
    df = df[df['tournament'].isin(competition_weights)].copy()

    # Filter to date window
    df = df[
        (df['date'] >= pd.Timestamp(start_date)) &
        (df['date'] <= pd.Timestamp(end_date))
    ].copy()

    # Get WC teams
    wc_team_set = set(wc_teams)

    # Check for missing confederation mapping
    missing_wc_teams = sorted(wc_team_set - set(FIFA_NAME_TO_CONFEDERATION))
    if missing_wc_teams:
        raise KeyError(
            f"Missing confederation mapping for WC teams: {missing_wc_teams}"
        )

    # Only keep matches where at least one team is in the WC
    df = df[
        df['home_team'].isin(wc_team_set) |
        df['away_team'].isin(wc_team_set)
    ].copy()

    # Drop rows with missing scores
    df = df[df['home_score'].notna() & df['away_score'].notna()].copy()

    # Competition weight
    df['competition_weight'] = df['tournament'].map(competition_weights)

    # Recency weight calculated with exponential decay relative to end_date
    reference_date = pd.Timestamp(end_date)
    df['years_ago'] = (reference_date - df['date']).dt.days / 365.25
    df['recency_weight'] = np.exp(
        -decay_lambda * df['years_ago']
    )

    # Confederation weight
    df['home_confederation'] = df['home_team'].map(FIFA_NAME_TO_CONFEDERATION)
    df['away_confederation'] = df['away_team'].map(FIFA_NAME_TO_CONFEDERATION)

    home_weight = df['home_confederation'].map(confederation_weights).fillna(1.0)
    away_weight = df['away_confederation'].map(confederation_weights).fillna(1.0)

    df['confederation_weight'] = np.sqrt(home_weight * away_weight)

    # Final weight
    df['weight'] = (
        df['competition_weight'] *
        df['recency_weight'] *
        df['confederation_weight']
    )

    return df[['date', 'home_team', 'away_team',
               'home_score', 'away_score', 'neutral', 'weight']].reset_index(drop=True)

def load_kaggle_base_data(
    path,
    wc_teams,
    start_date,
    end_date,
    decay_lambda=0.2,
):
    """
    Loads the Kaggle results.csv and returns a base filtered, weighted DataFrame
    ready for Dixon-Coles fitting.
    """
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    # Filter to known competitions only
    df = df[df['tournament'].isin(DEFAULT_COMPETITION_WEIGHTS)].copy()

    # Filter to date window
    df = df[
        (df['date'] >= pd.Timestamp(start_date)) &
        (df['date'] <= pd.Timestamp(end_date))
    ].copy()

    # Only keep matches where at least one team is in the WC
    wc_team_set = set(wc_teams)
    df = df[
        df['home_team'].isin(wc_team_set) |
        df['away_team'].isin(wc_team_set)
    ].copy()

    # Drop rows with missing scores
    df = df[df['home_score'].notna() & df['away_score'].notna()].copy()

    reference_date = pd.Timestamp(end_date)

    df['years_ago'] = (
        (reference_date - df['date']).dt.days / 365.25
    )

    df['recency_weight'] = np.exp(
        -decay_lambda * df['years_ago']
    )

    missing_wc_teams = sorted(wc_team_set - set(FIFA_NAME_TO_CONFEDERATION))
    if missing_wc_teams:
        raise KeyError(
            f"Missing confederation mapping for WC teams: {missing_wc_teams}"
        )

    df["home_confederation"] = df["home_team"].map(FIFA_NAME_TO_CONFEDERATION)
    df["away_confederation"] = df["away_team"].map(FIFA_NAME_TO_CONFEDERATION)

    return df.reset_index(drop=True)

def load_pool_data(base_path, year):
    """
    Loads all pool CSVs for a given World Cup year and returns them
    as a single dict of DataFrames with consistent column names
    and full country names instead of FIFA codes.

    Parameters
    ----------
    base_path : str — path to the folder containing the pool CSVs
    year      : int — e.g. 2022

    Returns
    -------
    dict with keys: 'games', 'teams', 'scores', 'predictions'
    'predictions' will be None if the file doesn't exist (e.g. 2014)
    """
    # --- Games ---
    games = pd.read_csv(
        f'{base_path}/{year}_games.csv',
        header=None,
        names=['phase', 'datetime', 'team1_code', 'team2_code']
    )
    games['game_id'] = range(1, len(games) + 1)
    games['team1'] = games['team1_code'].apply(code_to_name)
    games['team2'] = games['team2_code'].apply(code_to_name)

    # --- Teams ---
    teams = pd.read_csv(
        f'{base_path}/{year}_teams.csv',
        header=None,
        names=['code', 'group']
    )
    teams['name'] = teams['code'].apply(code_to_name)

    # --- Scores ---
    scores_raw = pd.read_csv(f'{base_path}/{year}_scores.csv', header=None)
    if scores_raw.shape[1] == 4:
        # 2014 format: no phase column
        scores_raw.columns = ['team1_code', 'team2_code', 'score1', 'score2']
        scores_raw.insert(0, 'phase', 'G')  # placeholder, won't be used
    else:
        scores_raw.columns = ['phase', 'team1_code', 'team2_code', 'score1', 'score2']
    scores = scores_raw
    scores['game_id'] = range(1, len(scores) + 1)
    scores['team1'] = scores['team1_code'].apply(code_to_name)
    scores['team2'] = scores['team2_code'].apply(code_to_name)

    # --- Predictions (only if exists) ---
    pred_path = f'{base_path}/{year}_predictions.csv'
    try:
        predictions = pd.read_csv(
            pred_path,
            header=None,
            names=['game_id', 'user_id', 'score1', 'score2']
        )
    except FileNotFoundError:
        predictions = None

    return {
        'games':       games,
        'teams':       teams,
        'scores':      scores,
        'predictions': predictions,
    }


def get_wc_teams(pool_data):
    """
    Extracts the list of full country names for a WC from pool data.
    Used to filter the Kaggle training data to relevant teams.
    """
    return pool_data['teams']['name'].tolist()


def is_knockout(phase):
    """
    Returns True if a game is a knockout phase match.
    Phase codes: G=group, 8=R16, 4=QF, 2=SF, 1=3rd place, 0=Final
    Per the rules, knockout scores are already stored as 120-min results
    in the pool CSVs, so no extra processing needed, this function
    exists purely for documentation and filtering if needed later.
    """
    return phase != 'G'