import pandas as pd

from src.loader import (
    load_kaggle_base_data,
    load_pool_data,
    get_wc_teams,
)

KAGGLE_PATH = "data/kaggle/results.csv"
POOL_PATH   = "data/pool"

WC_START_DATES = {
    2002: "2002-05-31",
    2006: "2006-06-09",
    2010: "2010-06-11",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

TRAINING_YEARS = 12

def get_training_window(wc_year):
    end = pd.Timestamp(WC_START_DATES[wc_year]) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=TRAINING_YEARS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

for year in [2002, 2006, 2010, 2018, 2022]:

    pool = load_pool_data(POOL_PATH, year)
    wc_teams = get_wc_teams(pool)

    start_date, end_date = get_training_window(year)

    df = load_kaggle_base_data(
        KAGGLE_PATH,
        wc_teams,
        start_date,
        end_date,
    )

    print("\n")
    print("=" * 80)
    print(year)
    print("=" * 80)

    print(
        df["tournament"]
        .value_counts()
        .head(30)
    )