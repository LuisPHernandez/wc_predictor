"""
prepare_2026_data.py
Extracts closing lines and perfectly aligns dates to the official 2026 pool.
Run from project root:
    py -3 prepare_2026_data.py
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mappings import code_to_name

# 👈 Update this to your exact raw tracker filename
RAW_LOG_PATH = PROJECT_ROOT / 'merged_history.csv' 
GAMES_PATH   = PROJECT_ROOT / 'data' / 'pool' / '2026_games.csv'

def main():
    if not RAW_LOG_PATH.exists():
        print(f"❌ File not found at: {RAW_LOG_PATH}")
        return

    # 1. Load official pool games to harvest the exact UTC datetime strings
    print("Loading official 2026 pool dates...")
    games = pd.read_csv(GAMES_PATH, header=None, names=['phase', 'datetime', 't1_code', 't2_code'])
    games['team1'] = games['t1_code'].apply(code_to_name)
    games['team2'] = games['t2_code'].apply(code_to_name)

    official_dates = {}
    for row in games.itertuples():
        # Create an alphabetical tuple key of the two teams playing
        key = tuple(sorted([row.team1, row.team2]))
        official_dates[key] = row.datetime 

    # 2. Load and process your tracker data
    print("Processing tracker logs...")
    df = pd.read_csv(RAW_LOG_PATH)
    
    # Sort chronologically to get closing odds at the bottom
    df = df.sort_values('prediction_timestamp')
    
    # Create the exact same alphabetical key to map against the pool
    df['match_key'] = df.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)
    
    # Drop duplicates safely ignoring any home/away inversion
    closing_df = df.drop_duplicates(subset=['match_key'], keep='last').copy()

    # 3. Inject the official pool date string directly into the odds data
    closing_df['official_date'] = closing_df['match_key'].map(official_dates)

    # Filter out any catastrophic mismatches (e.g., severe typos in the tracker)
    valid_df = closing_df.dropna(subset=['official_date']).copy()
    dropped = len(closing_df) - len(valid_df)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} matches that couldn't be mapped to 2026_games.csv")

    # 4. Export the perfectly synced 1X2 CSV
    odds_df = pd.DataFrame({
        'date': valid_df['official_date'],
        'home_team': valid_df['home_team'],
        'away_team': valid_df['away_team'],
        'h_odds_avg': valid_df['home_odds'],
        'd_odds_avg': valid_df['draw_odds'],
        'a_odds_avg': valid_df['away_odds']
    })
    odds_out = PROJECT_ROOT / 'data' / 'odds' / '2026_odds.csv'
    odds_df.to_csv(odds_out, index=False)

    # 5. Export the perfectly synced xG CSV
    xg_df = pd.DataFrame({
        'date': valid_df['official_date'],
        'home_team': valid_df['home_team'],
        'away_team': valid_df['away_team'],
        'implied_xg': valid_df['calculated_implied_xg']
    })
    xg_out = PROJECT_ROOT / 'data' / 'odds' / '2026wc_expected_goals.csv'
    xg_df.to_csv(xg_out, index=False)

    print(f"✅ Successfully exported {len(valid_df)} matches perfectly synced to the pool dates.")

if __name__ == '__main__':
    main()