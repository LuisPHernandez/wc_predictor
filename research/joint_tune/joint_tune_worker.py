"""
joint_tune_worker.py

Worker module for the joint parameter grid search.
Each call fits one Dixon-Coles model and exhaustively scores
all (alpha, k) combinations using the recency-weighted objective.

Not run directly — imported by joint_tune_coordinator.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson # pyrefly: ignore [missing-import]
from src.model import DixonColes

PROJECT_ROOT = Path(__file__).resolve().parent

KAGGLE_PATH = PROJECT_ROOT / "data" / "kaggle" / "results.csv"
POOL_PATH   = PROJECT_ROOT / "data" / "pool"

TRAINING_YEARS = [2006, 2010, 2014, 2018]

YEAR_WEIGHTS = {
    2006: 0.50,
    2010: 0.75,
    2014: 0.90,
    2018: 1.00,
}

WC_START_DATES = {
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

# Fixed competition weights — not searched
COMPETITION_WEIGHTS = {
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

ALPHAS        = [round(a, 2) for a in np.arange(0.0, 1.05, 0.05)]
K_VALUES      = [round(k, 2) for k in np.arange(1.00, 1.30, 0.05)]
MAX_GOALS     = 8
RHO_ESTIMATE  = -0.10   # not searched — stable across fits


# ---------------------------------------------------------------------------
# Minimal Dixon-Coles implementation (no external model.py dependency)
# so the worker is fully self-contained and safe for multiprocessing.
# ---------------------------------------------------------------------------

from scipy.optimize import minimize # pyrefly: ignore [missing-import]
from scipy.special  import gammaln # pyrefly: ignore [missing-import]

def _fit_dixon_coles(df, decay_lambda, regularization):
    """
    Fits Dixon-Coles on df. Returns (fitted_params, team_index, n_teams, rho).
    """
    all_teams  = sorted(set(df["home_team"]) | set(df["away_team"]))
    n_teams    = len(all_teams)
    team_index = {t: i for i, t in enumerate(all_teams)}

    h_idx    = np.array([team_index[t] for t in df["home_team"]])
    a_idx    = np.array([team_index[t] for t in df["away_team"]])
    h_goals  = df["home_score"].astype(int).values
    a_goals  = df["away_score"].astype(int).values
    weights  = df["weight"].values
    neutral  = df["neutral"].values.astype(bool)

    def nll(params):
        attack   = params[:n_teams]
        defense  = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho      = params[2*n_teams + 1]

        adv = np.where(neutral, 0.0, home_adv)
        lh  = np.exp(attack[h_idx] + defense[a_idx] + adv)
        la  = np.exp(attack[a_idx] + defense[h_idx])

        log_ph = h_goals * np.log(lh) - lh - gammaln(h_goals + 1)
        log_pa = a_goals * np.log(la) - la - gammaln(a_goals + 1)

        log_tau = np.zeros(len(h_goals))
        is_00   = (h_goals == 0) & (a_goals == 0)
        is_10   = (h_goals == 1) & (a_goals == 0)
        is_01   = (h_goals == 0) & (a_goals == 1)
        is_11   = (h_goals == 1) & (a_goals == 1)

        log_tau[is_00] = np.log(np.maximum(1 - lh[is_00]*la[is_00]*rho, 1e-10))
        log_tau[is_10] = np.log(np.maximum(1 + la[is_10]*rho,           1e-10))
        log_tau[is_01] = np.log(np.maximum(1 + lh[is_01]*rho,           1e-10))
        log_tau[is_11] = np.log(np.maximum(1 - rho,                     1e-10))

        ll  = weights * (log_ph + log_pa + log_tau)
        reg = regularization * (np.sum(attack**2) + np.sum(defense**2))
        return -ll.sum() + reg

    params0 = np.concatenate([
        np.zeros(n_teams),
        np.zeros(n_teams),
        [0.25],
        [-0.10],
    ])

    result = minimize(nll, params0, method="L-BFGS-B",
                      options={"maxiter": 3500, "maxfun": 300000,
                               "ftol": 1e-9, "gtol": 1e-6})

    return result.x, team_index, n_teams, result.x[2*n_teams + 1]


def _get_lambda(params, team_index, n_teams, home, away, neutral=True):
    attack   = params[:n_teams]
    defense  = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams] if not neutral else 0.0
    h_idx    = team_index[home]
    a_idx    = team_index[away]
    lh = np.exp(attack[h_idx] + defense[a_idx] + home_adv)
    la = np.exp(attack[a_idx] + defense[h_idx])
    return float(lh), float(la)


def _tau(i, j, lh, la, rho):
    if   i == 0 and j == 0: return max(1 - lh*la*rho, 1e-10)
    elif i == 1 and j == 0: return max(1 + la*rho,    1e-10)
    elif i == 0 and j == 1: return max(1 + lh*rho,    1e-10)
    elif i == 1 and j == 1: return max(1 - rho,       1e-10)
    else:                   return 1.0


def _score_matrix(lh, la, rho, k, max_goals=MAX_GOALS):
    lhk = lh * k
    lak = la * k
    m   = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            m[i, j] = (poisson.pmf(i, lhk) *
                       poisson.pmf(j, lak) *
                       _tau(i, j, lhk, lak, rho))
    total = m.sum()
    return m / total if total > 0 else m


def _blend_matrix(matrix, book_probs, alpha):
    """Outcome-level blend (same logic as model.py blend_matrix_outcomes)."""
    if book_probs is None or alpha >= 1.0:
        return matrix

    n = matrix.shape[0]
    home_m = float(np.sum(np.tril(matrix, -1)))
    draw_m = float(np.sum(np.diag(matrix)))
    away_m = float(np.sum(np.triu(matrix, 1)))
    total  = home_m + draw_m + away_m

    if total <= 0:
        return matrix

    t_home = alpha * (home_m/total) + (1-alpha) * book_probs["home"]
    t_draw = alpha * (draw_m/total) + (1-alpha) * book_probs["draw"]
    t_away = alpha * (away_m/total) + (1-alpha) * book_probs["away"]

    hf = t_home / max(home_m/total, 1e-12)
    df = t_draw / max(draw_m/total, 1e-12)
    af = t_away / max(away_m/total, 1e-12)

    blended = matrix.copy()
    for i in range(n):
        for j in range(n):
            if   i > j: blended[i, j] *= hf
            elif i == j: blended[i, j] *= df
            else:        blended[i, j] *= af

    s = blended.sum()
    return blended / s if s > 0 else blended


def _best_pred_pts(matrix, actual_home, actual_away, max_goals=MAX_GOALS):
    """Returns points earned by the expected-pts-maximising prediction."""
    from src.scoring import points_for_prediction

    best_ep, best_ph, best_pa = -1.0, 0, 0
    for ph in range(max_goals):
        for pa in range(max_goals):
            ep = sum(
                points_for_prediction(ph, pa, ah, aa) * matrix[ah, aa]
                for ah in range(max_goals)
                for aa in range(max_goals)
            )
            if ep > best_ep:
                best_ep, best_ph, best_pa = ep, ph, pa

    return points_for_prediction(best_ph, best_pa, actual_home, actual_away)


# ---------------------------------------------------------------------------
# Main worker function
# ---------------------------------------------------------------------------

def run_combination(combo):
    """
    Fits the model for one parameter combination and returns the best
    (alpha, k, weighted_score) found across the inner grid.

    Parameters
    ----------
    combo : dict with keys:
        decay_lambda, regularization,
        elite, caf, concacaf, afc, ofc
        (plus combo_id for tracking)

    Returns
    -------
    dict — the combo dict extended with best_alpha, best_k,
           best_weighted_score, best_unweighted_score,
           scores_by_year (dict), converged
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.loader       import load_kaggle_data, load_pool_data, get_wc_teams
    from src.odds_loader  import load_wc_odds_lookup

    decay_lambda   = combo["decay_lambda"]
    regularization = combo["regularization"]

    confederation_weights = {
        "CONMEBOL": combo["elite"],
        "UEFA":     combo["elite"],
        "CAF":      combo["caf"],
        "CONCACAF": combo["concacaf"],
        "AFC":      combo["afc"],
        "OFC":      combo["ofc"],
    }

    # ------------------------------------------------------------------
    # For each training year: fit model, store per-game lambdas + book odds
    # ------------------------------------------------------------------

    year_game_data = {}   # year -> list of game dicts

    for year in TRAINING_YEARS:
        end   = pd.Timestamp(WC_START_DATES[year]) - pd.Timedelta(days=1)
        start = end - pd.DateOffset(years=12)

        pool     = load_pool_data(POOL_PATH, year)
        wc_teams = get_wc_teams(pool)

        kaggle_df = load_kaggle_data(
            KAGGLE_PATH,
            wc_teams,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            decay_lambda=decay_lambda,
            competition_weights=COMPETITION_WEIGHTS,
            confederation_weights=confederation_weights,
        )

        try:
            params, team_idx, n_teams, rho = _fit_dixon_coles(
                kaggle_df, decay_lambda, regularization
            )
            converged = True
        except Exception:
            return {**combo, "error": "fit_failed", "best_weighted_score": -999}

        # Load odds for this year
        try:
            odds_lookup = load_wc_odds_lookup(year)
        except Exception:
            odds_lookup = {}

        scores_df = pool["scores"]
        score_lookup = (
            scores_df.set_index("game_id")[["score1", "score2"]].to_dict("index")
        )

        games = pool["games"]
        game_data = []

        for row in games.itertuples():
            actual = score_lookup.get(row.game_id)
            if actual is None:
                continue

            home = row.team1
            away = row.team2

            if home not in team_idx or away not in team_idx:
                continue

            lh, la = _get_lambda(params, team_idx, n_teams, home, away, neutral=True)
            book    = odds_lookup.get(row.game_id)

            game_data.append({
                "home":        home,
                "away":        away,
                "actual_home": int(actual["score1"]),
                "actual_away": int(actual["score2"]),
                "lh":          lh,
                "la":          la,
                "book":        book,
            })

        year_game_data[year] = game_data

    # ------------------------------------------------------------------
    # Inner grid: score all (alpha, k) combinations
    # ------------------------------------------------------------------

    best_weighted   = -999.0
    best_alpha      = 1.0
    best_k          = 1.0
    best_scores_yr  = {}
    best_unweighted = 0

    for k in K_VALUES:
        for alpha in ALPHAS:
            weighted_total   = 0.0
            unweighted_total = 0
            scores_yr        = {}

            for year in TRAINING_YEARS:
                yr_pts = 0
                for g in year_game_data[year]:
                    matrix = _score_matrix(g["lh"], g["la"], RHO_ESTIMATE, k)
                    matrix = _blend_matrix(matrix, g["book"], alpha)
                    yr_pts += _best_pred_pts(
                        matrix, g["actual_home"], g["actual_away"]
                    )
                scores_yr[year]   = yr_pts
                weighted_total   += YEAR_WEIGHTS[year] * yr_pts
                unweighted_total += yr_pts

            if weighted_total > best_weighted:
                best_weighted   = weighted_total
                best_alpha      = alpha
                best_k          = k
                best_scores_yr  = scores_yr.copy()
                best_unweighted = unweighted_total

    return {
        **combo,
        "best_alpha":           best_alpha,
        "best_k":               best_k,
        "best_weighted_score":  round(best_weighted, 4),
        "best_unweighted_score": best_unweighted,
        "score_2006":           best_scores_yr.get(2006, 0),
        "score_2010":           best_scores_yr.get(2010, 0),
        "score_2014":           best_scores_yr.get(2014, 0),
        "score_2018":           best_scores_yr.get(2018, 0),
        "converged":            converged,
        "error":                None,
    }