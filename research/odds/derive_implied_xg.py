import pandas as pd

from research.odds.implied_xg import implied_expected_goals
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FILES = [
   PROJECT_ROOT / "data" / "odds" / "2014wc_expected_goals.csv",
   PROJECT_ROOT / "data" / "odds" / "2018wc_expected_goals.csv",
   PROJECT_ROOT / "data" / "odds" / "2022wc_expected_goals.csv",
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