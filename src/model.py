import numpy as np # pyrefly: ignore [missing-import]
from scipy.optimize import minimize # pyrefly: ignore [missing-import]
from scipy.stats import poisson # pyrefly: ignore [missing-import]

from src.scoring import points_for_prediction

# ============================================================
# PRODUCTION PARAMETERS
# ============================================================

PRODUCTION_ALPHA = 0.40
PRODUCTION_GOAL_BLEND_BETA = 0.05

PRODUCTION_K_LOW = 0.93
PRODUCTION_K_HIGH = 1.01

PRODUCTION_K_LOW_LAMBDA = 2.0
PRODUCTION_K_HIGH_LAMBDA = 2.5

DECAY_LAMBDA = 0.2
REGULARIZATION = 0.0010

def outcome_probs_from_matrix(matrix):
    """
    Computes home/draw/away probabilities from a score matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Score probability matrix where [i,j] is P(home=i, away=j)

    Returns
    -------
    dict
        {
            'home': float,
            'draw': float,
            'away': float,
        }
    """
    home = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    away = float(np.sum(np.triu(matrix, 1)))

    total = home + draw + away

    return {
        'home': home / total,
        'draw': draw / total,
        'away': away / total,
    }

def normalize_matrix(matrix):
    """
    Renormalizes a score matrix so probabilities sum to 1.
    """
    total = matrix.sum()

    if total <= 0:
        raise ValueError("Matrix probability mass is zero")

    return matrix / total

def blend_matrix_outcomes(matrix, bookmaker_probs, alpha=PRODUCTION_ALPHA):
    """
    Blends Dixon-Coles outcome probabilities with bookmaker probabilities
    while preserving the model's scoreline structure.

    Parameters
    ----------
    matrix : np.ndarray
        Dixon-Coles score probability matrix

    bookmaker_probs : dict
        {
            'home': float,
            'draw': float,
            'away': float,
        }

    alpha : float
        1.0 = pure model
        0.0 = pure bookmaker

    Returns
    -------
    np.ndarray
        Reweighted probability matrix
    """
    model_probs = outcome_probs_from_matrix(matrix)

    target_home = (
        alpha * model_probs['home']
        + (1 - alpha) * bookmaker_probs['home']
    )

    target_draw = (
        alpha * model_probs['draw']
        + (1 - alpha) * bookmaker_probs['draw']
    )

    target_away = (
        alpha * model_probs['away']
        + (1 - alpha) * bookmaker_probs['away']
    )

    home_factor = target_home / max(model_probs['home'], 1e-12)
    draw_factor = target_draw / max(model_probs['draw'], 1e-12)
    away_factor = target_away / max(model_probs['away'], 1e-12)

    blended = matrix.copy()

    n_rows, n_cols = blended.shape

    for i in range(n_rows):
        for j in range(n_cols):

            if i > j:
                blended[i, j] *= home_factor

            elif i == j:
                blended[i, j] *= draw_factor

            else:
                blended[i, j] *= away_factor

    return normalize_matrix(blended)

class DixonColes:
    """
    Weighted Dixon-Coles Poisson model for international football.

    Fits attack and defense parameters for each team, plus a home
    advantage term and a rho correlation correction for low scores.
    Once fitted, predicts optimal scorelines by maximizing expected
    points under the pool's scoring rules.
    """

    def __init__(self, df, decay_lambda=DECAY_LAMBDA, regularization=REGULARIZATION, goal_blend_beta=PRODUCTION_GOAL_BLEND_BETA):
        """
        Parameters
        ----------
        df             : pd.DataFrame — output of loader.load_kaggle_data()
                         must have columns: home_team, away_team,
                         home_score, away_score, neutral, weight
        decay_lambda   : float — recency decay rate (same value used in loader)
        regularization : float — penalizes extreme attack/defense values
                         to avoid overfitting on teams with few matches
        """
        self.df             = df
        self.decay_lambda   = decay_lambda
        self.regularization = regularization

        # Built during fit()
        self.all_teams    = None
        self.team_index   = None
        self.n_teams      = None
        self.fitted_params = None
        self.goal_blend_beta = goal_blend_beta

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self):
        """
        Fits the model to the training data.
        Populates self.all_teams, self.fitted_params, self.team_index, self.n_teams.
        """
        # Build team universe from training data
        self.all_teams  = sorted(
            set(self.df['home_team'].unique()) |
            set(self.df['away_team'].unique())
        )
        self.n_teams    = len(self.all_teams)
        self.team_index = {team: i for i, team in enumerate(self.all_teams)}

        print(f"Fitting Dixon-Coles on {len(self.df)} matches, "
              f"{self.n_teams} teams...")

        # Precompute static arrays once
        self._h_idx    = np.array([self.team_index[t] for t in self.df['home_team']])
        self._a_idx    = np.array([self.team_index[t] for t in self.df['away_team']])
        self._h_goals  = self.df['home_score'].astype(int).values
        self._a_goals  = self.df['away_score'].astype(int).values
        self._weights  = self.df['weight'].values
        self._neutral  = self.df['neutral'].values.astype(bool)

        params0 = self._initialize_params()

        result = minimize(
            self._neg_log_likelihood,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 3500, 'maxfun': 300000,
                     'ftol': 1e-9, 'gtol': 1e-6},
            callback=lambda x: None 
        )
        print(f"Iterations used: {result.nit}")
        print(f"Function evals used: {result.nfev}")
        print(f"Converged: {result.success}")
        print(f"Final NLL: {result.fun:.4f}")

        self.fitted_params = result.x

        return self

    def _initialize_params(self):
        """
        Parameter vector layout:
          [attack_0 ... attack_n,       <- n_teams values
           defense_0 ... defense_n,     <- n_teams values
           home_advantage,              <- 1 value
           rho]                         <- 1 value
        Total length: 2 * n_teams + 2
        """
        return np.concatenate([
            np.zeros(self.n_teams),   # attack params
            np.zeros(self.n_teams),   # defense params
            np.array([0.25]),         # home advantage
            np.array([-0.1]),         # rho
        ])

    def _get_lambda(self, home_team, away_team, neutral):
        """
        Computes expected goals (lambda) for both teams.
        Uses log-linear model: exp(attack + opponent_defense + home_adv)
        Home advantage is zeroed out for neutral venue matches.
        """
        params = self.fitted_params
        h_idx  = self.team_index[home_team]
        a_idx  = self.team_index[away_team]

        attack  = params[:self.n_teams]
        defense = params[self.n_teams:2 * self.n_teams]
        home_adv = params[2 * self.n_teams] if not neutral else 0.0

        lambda_home = np.exp(attack[h_idx] + defense[a_idx] + home_adv)
        lambda_away = np.exp(attack[a_idx] + defense[h_idx])

        return lambda_home, lambda_away

    @staticmethod
    def _tau(home_goals, away_goals, lh, la, rho):
        """
        Dixon-Coles correction for the joint dependency of low scores.
        Adjusts the probability of (0,0), (1,0), (0,1), (1,1).
        All other scorelines are unaffected (returns 1.0).
        """
        if   home_goals == 0 and away_goals == 0: return 1 - lh * la * rho
        elif home_goals == 1 and away_goals == 0: return 1 + la * rho
        elif home_goals == 0 and away_goals == 1: return 1 + lh * rho
        elif home_goals == 1 and away_goals == 1: return 1 - rho
        else:                                     return 1.0

    def _neg_log_likelihood(self, params):
        """
        Vectorized negative weighted log-likelihood of all matches given params.
        We minimize this (minimizing negative = maximizing likelihood).
        Includes L2 regularization on attack/defense to avoid overfitting.
        All match rows are processed as NumPy arrays simultaneously.
        """
        attack   = params[:self.n_teams]
        defense  = params[self.n_teams:2 * self.n_teams]
        home_adv = params[2 * self.n_teams]
        rho      = params[2 * self.n_teams + 1]

        # Expected goals for every match simultaneously neutral matches get zero home advantage
        adv = np.where(self._neutral, 0.0, home_adv)
        lh = np.exp(attack[self._h_idx] + defense[self._a_idx] + adv)
        la = np.exp(attack[self._a_idx] + defense[self._h_idx])

        # Poisson log-probabilities
        from scipy.special import gammaln # pyrefly: ignore [missing-import]
        log_p_home = self._h_goals * np.log(lh) - lh - gammaln(self._h_goals + 1)
        log_p_away = self._a_goals * np.log(la) - la - gammaln(self._a_goals + 1)

        # Dixon-Coles tau correction — only affects 4 scorelines
        # compute for all rows, then overwrite the 4 special cases
        log_tau = np.zeros(len(self._h_goals))

        is_00 = (self._h_goals == 0) & (self._a_goals == 0)
        is_10 = (self._h_goals == 1) & (self._a_goals == 0)
        is_01 = (self._h_goals == 0) & (self._a_goals == 1)
        is_11 = (self._h_goals == 1) & (self._a_goals == 1)

        log_tau[is_00] = np.log(np.maximum(1 - lh[is_00] * la[is_00] * rho, 1e-10))
        log_tau[is_10] = np.log(np.maximum(1 + la[is_10] * rho,             1e-10))
        log_tau[is_01] = np.log(np.maximum(1 + lh[is_01] * rho,             1e-10))
        log_tau[is_11] = np.log(np.maximum(1 - rho,                         1e-10))

        # Weighted log-likelihood
        log_likelihood = self._weights * (log_p_home + log_p_away + log_tau)
        total = np.sum(log_likelihood)

        # L2 regularization
        reg = self.regularization * (np.sum(attack ** 2) + np.sum(defense ** 2))

        return -total + reg

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def score_matrix(self, home_team, away_team, neutral=True, max_goals=8, market_total_goals=None):
        """
        Returns a (max_goals x max_goals) matrix where entry [i,j]
        is the probability of home_team scoring i, away_team scoring j.

        Also returns the expected goals (lh, la) for reference.
        """
        self._check_fitted()

        # Handle teams not seen during training
        home_team = self._resolve_team(home_team)
        away_team = self._resolve_team(away_team)

        lh, la = self._get_lambda(
            home_team,
            away_team,
            neutral,
        )

        model_total = lh + la

        if market_total_goals is not None:

            blend_total = (
                self.goal_blend_beta * model_total
                +
                (1 - self.goal_blend_beta)
                * market_total_goals
            )

            home_ratio = lh / model_total
            away_ratio = la / model_total

            lh = blend_total * home_ratio
            la = blend_total * away_ratio

        total = lh + la

        LOW_K = PRODUCTION_K_LOW
        HIGH_K = PRODUCTION_K_HIGH

        if total <= PRODUCTION_K_LOW_LAMBDA:
            k = LOW_K

        elif total >= PRODUCTION_K_HIGH_LAMBDA:
            k = HIGH_K

        else:

            frac = (
                (total - PRODUCTION_K_LOW_LAMBDA)
                /
                (PRODUCTION_K_HIGH_LAMBDA - PRODUCTION_K_LOW_LAMBDA)
            )

            k = (
                LOW_K
                +
                frac * (
                    HIGH_K
                    -
                    LOW_K
                )
            )

        lh *= k
        la *= k
        
        rho    = self.fitted_params[2 * self.n_teams + 1]

        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = (
                    poisson.pmf(i, lh) *
                    poisson.pmf(j, la) *
                    self._tau(i, j, lh, la, rho)
                )

        return matrix, lh, la

    def predict(
        self,
        home_team,
        away_team,
        neutral=True,
        max_goals=8,
        market_total_goals=None,
        bookmaker_probs=None,
        alpha=PRODUCTION_ALPHA
    ):
        """
        Finds the scoreline prediction that maximizes expected points
        under the pool's scoring rules.
        """
        matrix, lh, la = self.score_matrix(
            home_team,
            away_team,
            neutral,
            max_goals,
            market_total_goals=market_total_goals
        )

        # 1. Capture RAW Model Probabilities (Before Vegas Tilt)
        raw_home_win = np.sum(np.tril(matrix, -1))
        raw_draw     = np.sum(np.diag(matrix))
        raw_away_win = np.sum(np.triu(matrix, 1))

        # 2. Apply Matrix Tilt (If market odds exist)
        if bookmaker_probs is not None:
            final_matrix = blend_matrix_outcomes(matrix, bookmaker_probs, alpha)
        else:
            final_matrix = matrix

        best_pred = (0, 0)
        best_ep   = -1.0

        second_pred = None
        second_ep = -1.0
        
        # 3. Calculate Expected Points using the FINAL TILTED matrix
        for ph in range(max_goals):
            for pa in range(max_goals):
                ep = sum(
                    points_for_prediction(ph, pa, ah, aa) * final_matrix[ah, aa]
                    for ah in range(max_goals)
                    for aa in range(max_goals)
                )
                if ep > best_ep:
                    second_ep = best_ep
                    second_pred = best_pred

                    best_ep = ep
                    best_pred = (ph, pa)

                elif ep > second_ep:
                    second_ep = ep
                    second_pred = (ph, pa)

        # Match outcome probabilities (Using the FINAL TILTED matrix)
        home_win = np.sum(np.tril(final_matrix, -1))
        draw     = np.sum(np.diag(final_matrix))
        away_win = np.sum(np.triu(final_matrix, 1))

        return {
            'home_team': home_team,
            'away_team': away_team,

            'prediction': f"{best_pred[0]}-{best_pred[1]}",
            'pred_home': best_pred[0],
            'pred_away': best_pred[1],
            'second_prediction': (None if second_pred is None else f"{second_pred[0]}-{second_pred[1]}"),
            'expected_pts': best_ep,
            'second_expected_pts': second_ep,
            'decision_margin': best_ep - second_ep,

            # Final Blended Output
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            
            # Diagnostic Raw Data
            'raw_model_home': raw_home_win,
            'raw_model_draw': raw_draw,
            'raw_model_away': raw_away_win,
            'lambda_home': lh,
            'lambda_away': la,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def _resolve_team(self, team):
        """
        If a team wasn't in the training data, we can't look up its index.
        This shouldn't happen in a correctly set up backtest (since we train
        on the WC teams), but if it does, raise a clear error.
        """
        if team not in self.team_index:
            raise KeyError(
                f"'{team}' was not in the training data. "
                f"Check your wc_teams list and mappings."
            )
        return team

    def strength_table(self):
        """
        Returns a DataFrame ranking all teams by overall strength
        (attack - defense). Useful for sanity checking after fitting.
        """
        self._check_fitted()
        import pandas as pd
        attack  = self.fitted_params[:self.n_teams]
        defense = self.fitted_params[self.n_teams:2 * self.n_teams]
        df = pd.DataFrame({
            'team':     self.all_teams,
            'attack':   attack,
            'defense':  defense,
            'strength': attack - defense,
        }).set_index('team').sort_values('strength', ascending=False)
        return df.round(4)