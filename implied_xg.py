from scipy.stats import poisson # pyrefly: ignore [missing-import]
from scipy.optimize import brentq # pyrefly: ignore [missing-import]
import numpy as np


def no_vig_probs(over_odds, under_odds):

    p_over = 1.0 / over_odds
    p_under = 1.0 / under_odds

    total = p_over + p_under

    return (
        p_over / total,
        p_under / total,
    )


def over_probability(mu, line):
    """
    Probability of winning an OVER bet under a Poisson total-goals model,
    normalized for Expected Value (stake splits and pushes).
    """
    frac = line - int(line)
    n = int(np.floor(line))

    # ----------------------------
    # Whole line (e.g., 2.0)
    # Win: X >= n+1. Push: X = n.
    # ----------------------------
    if frac == 0.0:
        p_win = 1.0 - poisson.cdf(n, mu)
        p_exact = poisson.pmf(n, mu)
        return p_win / (1.0 - p_exact)

    # ----------------------------
    # Half line (e.g., 2.5)
    # Win: X >= n+1. No push.
    # ----------------------------
    if frac == 0.5:
        return 1.0 - poisson.cdf(n, mu)

    # ----------------------------
    # Quarter line (e.g., 2.25)
    # Half stake on n (2.0), Half stake on n+0.5 (2.5)
    # Win: X >= n+1. Half-loss/Half-push: X = n.
    # ----------------------------
    if frac == 0.25:
        p_win = 1.0 - poisson.cdf(n, mu)
        p_exact = poisson.pmf(n, mu)
        return p_win / (1.0 - 0.5 * p_exact)

    # ----------------------------
    # Three-quarter line (e.g., 2.75)
    # Half stake on n+0.5 (2.5), Half stake on n+1.0 (3.0)
    # Win: X >= n+2. Half-win/Half-push: X = n+1.
    # ----------------------------
    if frac == 0.75:
        p_win_full = 1.0 - poisson.cdf(n + 1, mu)
        p_exact = poisson.pmf(n + 1, mu)
        return (p_win_full + 0.5 * p_exact) / (1.0 - 0.5 * p_exact)

    raise ValueError(f"Unsupported line: {line}")


def implied_expected_goals(
    line,
    over_odds,
    under_odds,
):

    p_over, _ = no_vig_probs(
        over_odds,
        under_odds,
    )

    def objective(mu):

        return (
            over_probability(mu, line)
            -
            p_over
        )

    return brentq(
        objective,
        0.2,
        8.0,
    )