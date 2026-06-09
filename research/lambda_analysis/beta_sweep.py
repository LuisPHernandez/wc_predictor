from multiprocessing import Pool
from functools import partial
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backtest
from src.model import DixonColes


BETAS = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
]

YEARS = [
    2014,
    2018,
    2022,
]


def run_beta(beta):

    print()
    print("=" * 80)
    print(f"BETA = {beta:.2f}")
    print("=" * 80)

    original_init = DixonColes.__init__

    def patched_init(
        self,
        df,
        decay_lambda=0.2,
        regularization=0.0010,
        goal_inflation=1.15,
        goal_blend_beta=1.0,
    ):
        original_init(
            self,
            df,
            decay_lambda=decay_lambda,
            regularization=regularization,
            goal_inflation=goal_inflation,
            goal_blend_beta=beta,
        )

    DixonColes.__init__ = patched_init

    results = []

    try:

        for year in YEARS:

            r = backtest.run_backtest(
                year,
                alpha=0.30,
            )

            results.append(r)

    finally:

        DixonColes.__init__ = original_init

    total_points = sum(
        r["model_points"]
        for r in results
    )

    return {
        "beta": beta,
        "total": total_points,
        "2014": results[0]["model_points"],
        "2018": results[1]["model_points"],
        "2022": results[2]["model_points"],
    }


if __name__ == "__main__":

    with Pool(
        processes=min(
            len(BETAS),
            5,
        )
    ) as pool:

        results = pool.map(
            run_beta,
            BETAS,
        )

    results = sorted(
        results,
        key=lambda x: x["total"],
        reverse=True,
    )

    print()
    print("=" * 80)
    print("BETA SWEEP RESULTS")
    print("=" * 80)

    print(
        f"{'Beta':<8}"
        f"{'2014':<8}"
        f"{'2018':<8}"
        f"{'2022':<8}"
        f"{'Total':<8}"
    )

    print("-" * 40)

    for r in results:

        print(
            f"{r['beta']:<8.2f}"
            f"{r['2014']:<8}"
            f"{r['2018']:<8}"
            f"{r['2022']:<8}"
            f"{r['total']:<8}"
        )