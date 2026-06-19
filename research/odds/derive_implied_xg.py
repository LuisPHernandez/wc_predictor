import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implied_xg import implied_expected_goals

FILES = [
   PROJECT_ROOT / "data" / "odds" / "2010wc_expected_goals.csv",
]

for file in FILES:
    df = pd.read_csv(file)

    df["implied_xg"] = df.apply(
        lambda row: implied_expected_goals(
            line=row["ou_lines"],
            over_odds=row["avg_over"],
            under_odds=row["avg_under"],
        ),
        axis=1,
    )

    df.to_csv(file, index=False)

    print(f"Updated {file}")