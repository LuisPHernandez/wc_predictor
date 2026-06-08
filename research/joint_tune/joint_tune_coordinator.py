"""
joint_tune_coordinator.py

Coordinator for the joint parameter grid search.

Usage
-----
Run the full grid on one PC:
    py -3 joint_tune_coordinator.py

Run the first half (PC 1):
    py -3 joint_tune_coordinator.py 1

Run the second half (PC 2):
    py -3 joint_tune_coordinator.py 2

Results are saved to:
    research/joint_tune/results_part1.csv   (PC 1)
    research/joint_tune/results_part2.csv   (PC 2)

After both PCs finish, run joint_tune_validate.py to merge,
rank, and validate the top combinations on 2022.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool as ProcessPool, freeze_support

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "research" / "joint_tune"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Full parameter grid
# ---------------------------------------------------------------------------

DECAY_LAMBDAS   = [0.10, 0.15, 0.20, 0.25, 0.30]
REGULARIZATIONS = [0.0005, 0.001, 0.002, 0.005, 0.010]

# Fixed production confederation weights
ELITES          = [1.00]
CAFS            = [1.10]
CONCACAFS       = [1.05]
AFCS            = [0.95]
OFCS            = [0.90]

def build_full_grid():
    combos = []
    combo_id = 0
    for dl, reg, elite, caf, concacaf, afc, ofc in itertools.product(
        DECAY_LAMBDAS, REGULARIZATIONS,
        ELITES, CAFS, CONCACAFS, AFCS, OFCS
    ):
        combos.append({
            "combo_id":      combo_id,
            "decay_lambda":  dl,
            "regularization": reg,
            "elite":         elite,
            "caf":           caf,
            "concacaf":      concacaf,
            "afc":           afc,
            "ofc":           ofc,
        })
        combo_id += 1
    return combos

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    """Returns set of already-completed combo_ids."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["combo_id"].tolist())
    except Exception:
        return set()

def save_results(results, path):
    df = pd.DataFrame(results)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    freeze_support()

    # Determine which half to run
    if len(sys.argv) > 1:
        part = int(sys.argv[1])
        assert part in (1, 2), "Argument must be 1 or 2"
    else:
        part = 0   # run full grid

    all_combos = build_full_grid()
    total      = len(all_combos)

    if part == 1:
        combos      = all_combos[:total // 2]
        output_path = RESULTS_DIR / "results_part1.csv"
        print(f"PC 1: running combinations 0 – {total//2 - 1}  ({len(combos)} total)")
    elif part == 2:
        combos      = all_combos[total // 2:]
        output_path = RESULTS_DIR / "results_part2.csv"
        print(f"PC 2: running combinations {total//2} – {total - 1}  ({len(combos)} total)")
    else:
        combos      = all_combos
        output_path = RESULTS_DIR / "results_full.csv"
        print(f"Full grid: {len(combos)} combinations")

    print(f"Total grid size: {total} combinations")
    print(
        f"Expected model fits: "
        f"{total * 4}"
    )
    print(f"Output: {output_path}")

    # Resume from checkpoint
    done_ids = load_checkpoint(output_path)
    pending  = [c for c in combos if c["combo_id"] not in done_ids]

    print(f"Already completed: {len(done_ids)}")
    print(f"Remaining        : {len(pending)}")

    if not pending:
        print("All combinations already completed.")
        return

    # Worker pool
    n_workers = max(1, os.cpu_count() - 1)
    print(f"Workers: {n_workers}\n")

    # Import here so multiprocessing doesn't import at module level
    from research.joint_tune.joint_tune_worker import run_combination

    CHECKPOINT_EVERY = 20   # save every N completed combos
    batch_results    = []
    completed        = len(done_ids)

    with ProcessPool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_combination, pending):
            batch_results.append(result)
            completed += 1

            # Progress
            pct = 100 * completed / len(combos)
            best_so_far = max(
                (r["best_weighted_score"] for r in batch_results
                 if r.get("best_weighted_score", -999) > -999),
                default=0,
            )
            print(
                f"  [{completed}/{len(combos)}  {pct:.1f}%]  "
                f"combo_id={result['combo_id']}  "
                f"weighted={result.get('best_weighted_score', 'ERR'):.2f}  "
                f"best_so_far={best_so_far:.2f}  "
                f"α={result.get('best_alpha', '?')}  "
                f"k={result.get('best_k', '?')}"
            )

            # Checkpoint
            if len(batch_results) >= CHECKPOINT_EVERY:
                save_results(batch_results, output_path)
                batch_results = []
                print(f"  >>> Checkpoint saved to {output_path}")

    # Save remaining
    if batch_results:
        save_results(batch_results, output_path)

    print(f"\nDone. Results saved to {output_path}")

    # Quick top-10 preview
    df = pd.read_csv(output_path)
    df = df[df["best_weighted_score"] > -999].sort_values(
        "best_weighted_score", ascending=False
    )
    print(f"\nTop 10 so far:")
    print(
        df.head(10)[[
            "combo_id", "decay_lambda", "regularization",
            "elite", "caf", "concacaf", "afc", "ofc",
            "best_alpha", "best_k", "best_weighted_score",
            "score_2006", "score_2010", "score_2014", "score_2018",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()