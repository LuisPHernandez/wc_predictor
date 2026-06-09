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
    Probability of winning an OVER bet under a Poisson total-goals model.
    Supports:
        2.0
        2.25
        2.5
        2.75
        3.0
        etc.
    """

    frac = line - int(line)

    # ----------------------------
    # Whole line
    # ----------------------------

    if frac == 0.0:

        n = int(line)

        p_push = poisson.pmf(n, mu)

        p_under = poisson.cdf(n - 1, mu)

        return 1.0 - p_under - p_push

    # ----------------------------
    # Half line
    # ----------------------------

    if frac == 0.5:

        n = int(np.floor(line))

        return 1.0 - poisson.cdf(n, mu)

    # ----------------------------
    # Quarter line
    # ----------------------------

    if frac == 0.25:

        return (
            over_probability(mu, line - 0.25)
            +
            over_probability(mu, line + 0.25)
        ) / 2

    if frac == 0.75:

        return (
            over_probability(mu, line - 0.25)
            +
            over_probability(mu, line + 0.25)
        ) / 2

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