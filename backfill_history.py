import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from src.odds_loader import _shin_probs

PROJECT_ROOT = Path(__file__).resolve().parent
HIST_PATH = PROJECT_ROOT / "merged_history.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "model" / "dixon_coles_model_2026.pkl"

def main():
    if not HIST_PATH.exists():
        print("❌ No predictions_history.csv found.")
        return
        
    print("Loading historical data and model...")
    df = pd.read_csv(HIST_PATH)
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # Check if the new columns exist; if not, create them with NaNs
    new_cols = [
        "raw_model_home", "raw_model_draw", "raw_model_away",
        "raw_market_home", "raw_market_draw", "raw_market_away"
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Find rows missing the new diagnostic data
    missing_mask = df["raw_model_home"].isna()
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print("✅ History file is already fully populated. No backfill needed.")
        return
        
    print(f"Found {missing_count} legacy rows. Reconstructing raw probabilities...")
    
    for idx, row in df[missing_mask].iterrows():
        # 1. Reconstruct Raw Market Probs via Shin
        p_home, p_draw, p_away = _shin_probs(
            row["home_odds"], row["draw_odds"], row["away_odds"]
        )
        
        # 2. Reconstruct Raw Model Probs via the frozen Dixon-Coles Matrix
        # We use the exact implied XG stored in the history row to recreate the exact matrix state
        matrix, _, _ = model.score_matrix(
            row["home_team"], 
            row["away_team"], 
            neutral=True, 
            market_total_goals=row["calculated_implied_xg"]
        )
        
        raw_model_h = np.sum(np.tril(matrix, -1))
        raw_model_d = np.sum(np.diag(matrix))
        raw_model_a = np.sum(np.triu(matrix, 1))
        
        # Inject the reconstructed data directly into the dataframe
        df.at[idx, "raw_market_home"] = round(p_home, 4)
        df.at[idx, "raw_market_draw"] = round(p_draw, 4)
        df.at[idx, "raw_market_away"] = round(p_away, 4)
        
        df.at[idx, "raw_model_home"] = round(raw_model_h, 4)
        df.at[idx, "raw_model_draw"] = round(raw_model_d, 4)
        df.at[idx, "raw_model_away"] = round(raw_model_a, 4)
        
    # Save the seamlessly patched history file
    df.to_csv(HIST_PATH, index=False)
    print(f"✅ Successfully backfilled {missing_count} rows. merged_history.csv is now fully updated.")

if __name__ == "__main__":
    main()