import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Pull the core architecture and paths straight from your files
from backtest import run_backtest, POOL_PATH
from src.loader import load_pool_data
from src.scoring import points_for_prediction

def build_true_position_charts():
    target_years = [2006, 2010, 2014, 2018, 2022] 
    
    points_trajectories = {}
    position_trajectories = {}
    colors = {2006: '#7f7f7f', 2010: '#ff7f0e', 2014: '#2ca02c', 2018: '#9467bd', 2022: '#d62728'}
    
    # List to aggregate all granular round-by-round csv logs
    all_density_records = []
    
    print("🔄 Pulling data models and calculating leaderboard histories...")
    
    for year in target_years:
        try:
            # 1. Gather the model's game-by-game performance
            res = run_backtest(year)
            df_model = res['model_preds'].sort_values('game_id').reset_index(drop=True)
            
            df_model['match_idx'] = np.arange(1, len(df_model) + 1)
            df_model['cum_points'] = df_model['points_earned'].cumsum()
            points_trajectories[year] = df_model[['match_idx', 'cum_points']].copy()
            
            # 2. Directly load the raw pool data to get round-by-round user picks
            if year == 2014:
                print("ℹ️ Skipping position/density tracking for 2014 (No user logs available).")
                continue
                
            pool = load_pool_data(POOL_PATH, year)
            user_preds = pool.get('predictions')
            
            if user_preds is None or len(user_preds) == 0:
                print(f"⚠️ No user prediction records found in file for {year}.")
                continue
                
            # Build lookups for the loop
            ordered_games = df_model['game_id'].tolist()
            actual_scores = df_model.set_index('game_id')[['score1', 'score2']].to_dict('index')
            
            unique_users = user_preds['user_id'].unique().tolist()
            user_running_totals = {u_id: 0 for u_id in unique_users}
            model_running_total = 0
            
            round_positions = []
            
            # 3. Step chronologically through every single match played
            for idx, game_id in enumerate(ordered_games, start=1):
                actual = actual_scores.get(game_id)
                if actual is None:
                    continue
                
                # Add points to the model
                model_row = df_model[df_model['game_id'] == game_id].iloc[0]
                model_running_total += model_row['points_earned']
                
                # Add points to every single user who predicted this match
                game_user_rows = user_preds[user_preds['game_id'] == game_id]
                for u_row in game_user_rows.itertuples():
                    u_pts = points_for_prediction(
                        u_row.score1, u_row.score2,
                        int(actual['score1']), int(actual['score2'])
                    )
                    user_running_totals[u_row.user_id] += u_pts
                
                # Calculate real-time position (how many users have more points than the model + 1)
                current_scores = list(user_running_totals.values())
                rank = sum(1 for score in current_scores if score > model_running_total) + 1
                
                # Calculate explicit requested density metric: count competitors >= model points
                higher_or_equal_count = sum(1 for score in current_scores if score >= model_running_total)
                
                round_positions.append({'match_idx': idx, 'position': rank})
                
                # Append granular entry to file exporter list
                all_density_records.append({
                    'world_cup_year': year,
                    'match_index': idx,
                    'model_cumulative_points': model_running_total,
                    'actual_leaderboard_rank': rank,
                    'competitors_higher_or_equal_count': higher_or_equal_count,
                    'total_pool_participants': len(current_scores)
                })
                
            position_trajectories[year] = pd.DataFrame(round_positions)
            print(f"✅ Position and density metrics generated for World Cup {year}")
            
        except Exception as e:
            print(f"❌ Failed processing for year {year}: {e}")
            continue

    # ============================================================
    # SAVE EXPORT LAYER: CSV Table Compilation
    # ============================================================
    if all_density_records:
        df_density = pd.DataFrame(all_density_records)
        df_density.to_csv("pool_density_metrics.csv", index=False)
        print("\n💾 Saved data table: pool_density_metrics.csv")
        
        # Display a quick verification sample of the density dataframe in console
        print("\n🔍 Snapshot preview of generated metrics table (Last 5 rows of 2022):")
        print(df_density[df_density['world_cup_year'] == 2022].tail(5).to_string(index=False))
    print("-" * 65)

    # ============================================================
    # CHART 1: Cumulative Points Velocity Curve
    # ============================================================
    plt.figure(figsize=(11, 6))
    for year, df in points_trajectories.items():
        plt.plot(df['match_idx'], df['cum_points'], label=f"World Cup {year}", 
                 color=colors.get(year, '#1f77b4'), linewidth=2.5)
    plt.title("Tournament Points Velocity: Climb Rate Comparison", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Matches Played (#)", fontsize=11)
    plt.ylabel("Total Points Accumulated (#)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("trajectory_points.png", dpi=300)
    plt.close()
    print("💾 Saved Chart 1: trajectory_points.png")

    # ============================================================
    # CHART 2: Actual Standings Position Progression
    # ============================================================
    plt.figure(figsize=(11, 6))
    for year, df in position_trajectories.items():
        if year in position_trajectories and len(position_trajectories[year]) > 0:
            plt.plot(df['match_idx'], df['position'], label=f"World Cup {year}", 
                     color=colors.get(year, '#1f77b4'), linewidth=2.5)
    plt.title("Pool Standings Progression (Top of Graph is 1st Place)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Matches Played (#)", fontsize=11)
    plt.ylabel("Actual Leaderboard Position", fontsize=11)
    plt.gca().invert_yaxis()  # Forces 1st place to sit at the absolute top of the graph
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("trajectory_positions.png", dpi=300)
    plt.close()
    print("💾 Saved Chart 2: trajectory_positions.png")

if __name__ == "__main__":
    build_true_position_charts()