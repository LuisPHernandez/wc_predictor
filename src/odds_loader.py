import sys
from pathlib import Path
from src.loader import load_pool_data
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mappings import code_to_name, ODDS_NAME_TO_FIFA

# ============================================================
# PRODUCTION ODDS SOURCES
#
# 1X2 odds:
#   load_wc_odds_lookup()
#
# Expected goals:
#   load_wc_expected_goals_lookup()
#
# Production model:
#   alpha uses 1X2 probabilities
#   beta uses expected goals
# ============================================================
ODDS_PATH  = PROJECT_ROOT / "data" / "odds" / 'WorldCup2026.xlsx'
CSV_ODDS_YEARS = {
    2006: PROJECT_ROOT / "data" / "odds" / "2006_odds.csv",
    2010: PROJECT_ROOT / "data" / "odds" / "2010_odds.csv",
}
POOL_PATH  = PROJECT_ROOT / 'data' / 'pool'

WC_SHEETS = {
    2014: 'WorldCup2014',
    2018: 'WorldCup2018',
    2022: 'WorldCup2022',
}

WC_EXPECTED_GOALS_CSV = {
    2014: PROJECT_ROOT / "data" / "odds" / "2014wc_expected_goals.csv",
    2018: PROJECT_ROOT / "data" / "odds" / "2018wc_expected_goals.csv",
    2022: PROJECT_ROOT / "data" / "odds" / "2022wc_expected_goals.csv",
}

def load_wc_odds_lookup_csv(year):
    """
    Loads bookmaker odds from a CSV and returns:

    {
        game_id: {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        }
    }
    """

    path = CSV_ODDS_YEARS[year]

    odds = pd.read_csv(path)

    odds["home_team"] = odds["home_team"].apply(
        _resolve_team
    )

    odds["away_team"] = odds["away_team"].apply(
        _resolve_team
    )

    odds["match_date"] = pd.to_datetime(
        odds["date"]
    ).dt.date

    pool = load_pool_data(
        POOL_PATH,
        year
    )

    games = pool["games"].copy()

    games["match_date"] = pd.to_datetime(
        games["datetime"],
        utc=True
    ).dt.date

    game_lookup = {}

    for row in games.itertuples():
        game_lookup[
            _matchup_key(
                row.match_date,
                row.team1,
                row.team2,
            )
        ] = row.game_id

    odds_lookup = {}

    for _, row in odds.iterrows():
        if (
            pd.isna(row["h_odds_avg"])
            or pd.isna(row["d_odds_avg"])
            or pd.isna(row["a_odds_avg"])
        ):
            continue

        key = _matchup_key(
            row["match_date"],
            row["home_team"],
            row["away_team"],
        )

        game_id = game_lookup.get(key)

        if game_id is None:
            continue

        if (
            row["h_odds_avg"] <= 1.0
            or row["d_odds_avg"] <= 1.0
            or row["a_odds_avg"] <= 1.0
        ):
            continue

        p_home, p_draw, p_away = _shin_probs(
            row["h_odds_avg"],
            row["d_odds_avg"],
            row["a_odds_avg"],
        )

        odds_lookup[game_id] = {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        }

    print(
        f"{year}: matched "
        f"{len(odds_lookup)} "
        f"of {len(games)} matches"
    )

    return odds_lookup

def load_wc_odds_lookup(year, path=ODDS_PATH):
    """
    Returns bookmaker probabilities keyed by pool game_id.

    {
        game_id: {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        }
    }
    """
    if year in CSV_ODDS_YEARS:
        return load_wc_odds_lookup_csv(year)

    odds = pd.read_excel(
        path,
        sheet_name=WC_SHEETS[year]
    )

    odds["match_date"] = pd.to_datetime(
        odds["Date"]
    ).dt.date

    odds["home_team"] = odds["Home"].apply(
        _resolve_team
    )

    odds["away_team"] = odds["Away"].apply(
        _resolve_team
    )

    pool = load_pool_data(
        POOL_PATH,
        year
    )

    games = pool["games"].copy()

    games["match_date"] = pd.to_datetime(
        games["datetime"],
        utc=True
    ).dt.normalize().dt.date

    game_lookup = {}

    for row in games.itertuples():
        game_lookup[
            (
                row.match_date,
                row.team1,
                row.team2,
            )
        ] = row.game_id

    odds_lookup = {}

    missing_matches = []

    for _, row in odds.iterrows():

        h_odds = row["H-Avg"]
        d_odds = row["D-Avg"]
        a_odds = row["A-Avg"]

        if (
            pd.isna(h_odds)
            or pd.isna(d_odds)
            or pd.isna(a_odds)
        ):
            continue

        if (
            h_odds <= 1.0
            or d_odds <= 1.0
            or a_odds <= 1.0
        ):
            continue

        key = (
            row["match_date"],
            row["home_team"],
            row["away_team"],
        )

        game_id = game_lookup.get(key)

        if game_id is None:
            missing_matches.append(key)
            continue

        p_home, p_draw, p_away = _shin_probs(
            h_odds,
            d_odds,
            a_odds,
        )

        odds_lookup[game_id] = {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        }

    print(
        f"{year}: matched "
        f"{len(odds_lookup)} odds rows"
    )

    if missing_matches:
        print(
            f"{year}: WARNING "
            f"{len(missing_matches)} unmatched games"
        )

        for m in missing_matches[:10]:
            print("   ", m)

    return odds_lookup

def load_wc_expected_goals_lookup(year):
    """
    Returns:

    {
        game_id: expected_goals
    }

    Expected goals are derived from bookmaker O/U lines
    and are used by the production beta blend.

    Production:
        model_total =
            beta * model_total
            +
            (1 - beta) * market_total
    """
    path = WC_EXPECTED_GOALS_CSV[year]

    ou = pd.read_csv(path)

    ou["home_team"] = (
        ou["home_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    ou["away_team"] = (
        ou["away_team"]
        .astype(str)
        .str.strip()
        .replace(ODDS_NAME_TO_FIFA)
    )

    pool = load_pool_data(
        POOL_PATH,
        year,
    )

    games = pool["games"]

    game_lookup = {}

    for row in games.itertuples():

        game_lookup[
            (
                row.team1,
                row.team2,
            )
        ] = row.game_id

    expected_goals_lookup = {}

    missing = []

    for row in ou.itertuples():

        key = (
            row.home_team,
            row.away_team,
        )

        game_id = game_lookup.get(key)

        if game_id is None:

            missing.append(key)
            continue

        expected_goals_lookup[
            game_id
        ] = float(row.ou_lines)

    print(
        f"{year}: matched "
        f"{len(expected_goals_lookup)} expected-goals rows"
    )

    if missing:

        print(
            f"{year}: WARNING "
            f"{len(missing)} unmatched expected-goals rows"
        )

        for m in missing[:10]:
            print("   ", m)

    return expected_goals_lookup

def _resolve_team(name):
    return ODDS_NAME_TO_FIFA.get(name, name)

def _matchup_key(match_date, team1, team2):
    return (
        match_date,
        *sorted([team1, team2])
    )

def _shin_probs(h_odds, d_odds, a_odds):
    """
    Converts decimal odds to implied probabilities using Shin's model.
    More accurate than basic normalization for fixed-odds bookmakers
    because it accounts for the presence of insider traders.

    Reference: Strumbelj (2013), Shin (1993)

    Parameters
    ----------
    h_odds, d_odds, a_odds : float — decimal odds for home/draw/away

    Returns
    -------
    (p_home, p_draw, p_away) : floats summing to 1.0
    """
    n = 3  # always 3 outcomes: home win, draw, away win
    pi = np.array([1/h_odds, 1/d_odds, 1/a_odds])
    booksum = pi.sum()

    # Fixed-point iteration to find z (proportion of insider traders)
    # Starts at z=0 and iterates until convergence (usually < 50 iterations)
    z = 0.0
    for _ in range(1000):
        z_new = (
            np.sum(np.sqrt(z**2 + 4*(1-z) * pi**2 / booksum)) - 2
        ) / (n - 2)
        if abs(z_new - z) < 1e-12:
            break
        z = z_new

    # Back-calculate true probabilities using Shin's formula
    probs = (
        np.sqrt(z**2 + 4*(1-z) * pi**2 / booksum) - z
    ) / (2*(1-z))

    return float(probs[0]), float(probs[1]), float(probs[2])


def _load_pool_scores(year):
    """
    Loads the pool scores CSV for a given year and returns a dict mapping
    (team1_full_name, team2_full_name) -> (score1, score2).
    Both orderings are stored so lookup works regardless of home/away assignment.
    """
    scores_raw = pd.read_csv(f'{POOL_PATH}/{year}_scores.csv', header=None)
    if scores_raw.shape[1] == 4:
        scores_raw.columns = ['team1_code', 'team2_code', 'score1', 'score2']
    else:
        scores_raw.columns = ['phase', 'team1_code', 'team2_code', 'score1', 'score2']

    lookup = {}
    for row in scores_raw.itertuples():
        t1 = code_to_name(row.team1_code)
        t2 = code_to_name(row.team2_code)
        lookup[(t1, t2)] = (int(row.score1), int(row.score2))
        # store reverse too — odds file may list teams in different order
        lookup[(t2, t1)] = (int(row.score2), int(row.score1))

    return lookup


def load_wc_odds(path=ODDS_PATH, years=None):
    """
    Loads historical WC odds and joins to pool scores for reliable actuals.

    For each match in the odds Excel file, looks up the actual result from
    the pool scores CSV (which we trust) instead of the Excel result columns.

    Parameters
    ----------
    path  : str  — path to the Excel file
    years : list — e.g. [2018, 2022]. Defaults to all available (2014, 2018, 2022).

    Returns
    -------
    pd.DataFrame with columns:
        year, home_team, away_team,
        p_home, p_draw, p_away,          <- Shin probabilities (vig removed)
        actual_home, actual_away,         <- score from pool CSV (reliable)
        z,                               <- Shin insider proportion (sanity check)
        overround,                        <- bookmaker margin before removal
        h_odds_avg, d_odds_avg, a_odds_avg  <- raw odds for reference
    """
    if years is None:
        years = list(WC_SHEETS.keys())

    all_rows = []

    for year in years:
        sheet     = WC_SHEETS[year]
        df        = pd.read_excel(path, sheet_name=sheet)
        pool_scores = _load_pool_scores(year)

        skipped_odds    = 0
        skipped_scores  = 0

        for _, row in df.iterrows():
            h_odds = row['H-Avg']
            d_odds = row['D-Avg']
            a_odds = row['A-Avg']

            # Skip if odds are missing or invalid
            if pd.isna(h_odds) or pd.isna(d_odds) or pd.isna(a_odds):
                skipped_odds += 1
                continue
            if h_odds <= 1.0 or d_odds <= 1.0 or a_odds <= 1.0:
                skipped_odds += 1
                continue

            home_team = _resolve_team(row['Home'])
            away_team = _resolve_team(row['Away'])

            # Look up actual score from pool CSV
            actual = pool_scores.get((home_team, away_team))
            if actual is None:
                print(f"  WARNING: no pool score found for "
                      f"{home_team} vs {away_team} ({year}) — skipping")
                skipped_scores += 1
                continue

            actual_home, actual_away = actual

            p_home, p_draw, p_away = _shin_probs(h_odds, d_odds, a_odds)
            overround = 1/h_odds + 1/d_odds + 1/a_odds

            # Compute z for reference (re-run one iteration of shin to extract it)
            pi = np.array([1/h_odds, 1/d_odds, 1/a_odds])
            booksum = pi.sum()
            z = 0.0
            for _ in range(1000):
                z_new = (np.sum(np.sqrt(z**2 + 4*(1-z)*pi**2/booksum)) - 2) / 1
                if abs(z_new - z) < 1e-12:
                    break
                z = z_new

            all_rows.append({
                'year':        year,
                'home_team':   home_team,
                'away_team':   away_team,
                'p_home':      round(p_home, 6),
                'p_draw':      round(p_draw, 6),
                'p_away':      round(p_away, 6),
                'actual_home': actual_home,
                'actual_away': actual_away,
                'z':           round(z, 6),
                'overround':   round(overround, 4),
                'h_odds_avg':  h_odds,
                'd_odds_avg':  d_odds,
                'a_odds_avg':  a_odds,
            })

        print(f"{year}: {len(df)} matches in Excel | "
              f"skipped {skipped_odds} bad odds | "
              f"skipped {skipped_scores} missing scores | "
              f"loaded {len(all_rows)} total so far")

    df_out = pd.DataFrame(all_rows)

    print(f"\nTotal loaded: {len(df_out)} matches "
          f"across years {sorted(df_out['year'].unique())}")
    print(f"Avg overround: {df_out['overround'].mean():.4f} "
          f"(vig ~{(df_out['overround'].mean()-1)*100:.1f}%)")
    print(f"Avg z (insider proportion): {df_out['z'].mean():.4f}")

    return df_out


if __name__ == '__main__':
    df = load_wc_odds()
    print(df[['year', 'home_team', 'away_team',
              'p_home', 'p_draw', 'p_away',
              'actual_home', 'actual_away',
              'overround', 'z']].to_string())
