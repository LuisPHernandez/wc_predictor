import pandas as pd
import numpy as np
from pathlib import Path

# Pull your core scoring architecture and mapping layers directly
from src.mappings import code_to_name
from src.scoring import points_for_prediction

def calculate_shin_market_probs(h_odds, d_odds, a_odds):
    """Clean extraction of your project's Shin calculation model."""
    pi = np.array([1.0 / h_odds, 1.0 / d_odds, 1.0 / a_odds])
    booksum = pi.sum()
    z = 0.0
    for _ in range(1000):
        z_new = (np.sum(np.sqrt(z**2 + 4 * (1 - z) * pi**2 / booksum)) - 2) / 1
        if abs(z_new - z) < 1e-12:
            break
        z = z_new
    probs = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / booksum) - z) / (2 * (1 - z))
    return float(probs[0]), float(probs[1]), float(probs[2])

def run_diagnostic():
    # Resolve file paths flexibly
    pred_path = Path("merged_history.csv")
    scores_path = Path("2026_scores.csv") if Path("2026_scores.csv").exists() else Path("data/pool/2026_scores.csv")
    
    if not pred_path.exists() or not scores_path.exists():
        print("❌ Error: Ensure merged_history.csv and 2026_scores.csv exist in your directory paths.")
        return

    # 1. Load and parse raw datasets
    df_pred = pd.read_csv(pred_path)
    
    # Handle optional header variations in your custom scores structure
    df_scores = pd.read_csv(scores_path)
    if 'team1_code' not in df_scores.columns:
        df_scores = pd.read_csv(scores_path, header=None, names=['team1_code', 'team2_code', 'score1', 'score2'])

    # Map team codes to match your ledger's text names
    df_scores['team1_name'] = df_scores['team1_code'].apply(code_to_name)
    df_scores['team2_name'] = df_scores['team2_code'].apply(code_to_name)

    # 2. Establish order-independent lookup keys to prevent home/away flipping issues
    df_scores['lookup_key'] = df_scores.apply(lambda r: tuple(sorted([str(r['team1_name']).strip(), str(r['team2_name']).strip()])), axis=1)
    df_pred['lookup_key'] = df_pred.apply(lambda r: tuple(sorted([str(r['home_team']).strip(), str(r['away_team']).strip()])), axis=1)

    # Deduplicate history ledger: extract only the absolute latest prediction profile per fixture
    if 'prediction_timestamp' in df_pred.columns:
        df_pred = df_pred.sort_values('prediction_timestamp').drop_duplicates('lookup_key', keep='last')

    # Merge profiles together
    df_analysis = pd.merge(df_pred, df_scores, on='lookup_key', how='inner')
    
    if len(df_analysis) == 0:
        print("⚠️ Match mapping failed. Verify that code_to_name translations perfectly align with your ledger values.")
        return

    audit_records = []
    
    # 3. Dynamic mathematical row evaluation loop
    for row in df_analysis.itertuples():
        # Match score alignments to the perspective of the prediction row
        if str(row.home_team).strip() == str(row.team1_name).strip():
            act_home, act_away = int(row.score1), int(row.score2)
        else:
            act_home, act_away = int(row.score2), int(row.score1)

        # Establish actual binary vectors
        if act_home > act_away:
            o_h, o_d, o_a = 1, 0, 0
            actual_outcome = "home"
        elif act_home == act_away:
            o_h, o_d, o_a = 0, 1, 0
            actual_outcome = "draw"
        else:
            o_h, o_d, o_a = 0, 0, 1
            actual_outcome = "away"

        # Evaluate model outcome selection
        pred_h, pred_a = int(row.pred_home), int(row.pred_away)
        model_outcome = "home" if pred_h > pred_a else ("away" if pred_h < pred_a else "draw")
        model_hit = (model_outcome == actual_outcome)

        # Parse market outcome selection via Shin conversion values
        m_p_h, m_p_d, m_p_a = calculate_shin_market_probs(row.home_odds, row.draw_odds, row.away_odds)
        market_idx = np.argmax([m_p_h, m_p_d, m_p_a])
        market_outcome = ["home", "draw", "away"][market_idx]
        market_hit = (market_outcome == actual_outcome)

        # Compute multi-category Brier scores (Lower = Better, 0.0 is flawless prediction)
        model_brier = (row.home_win - o_h)**2 + (row.draw - o_d)**2 + (row.away_win - o_a)**2
        market_brier = (m_p_h - o_h)**2 + (m_p_d - o_d)**2 + (m_p_a - o_a)**2
        
        # Calculate real earned pool points
        pts = points_for_prediction(pred_h, pred_a, act_home, act_away)

        audit_records.append({
            'fixture': f"{row.home_team} vs {row.away_team}",
            'actual': f"{act_home}-{act_away}",
            'pred': row.prediction,
            'pts': pts,
            'm_hit': market_hit,
            'mod_hit': model_hit,
            'mod_brier': model_brier,
            'm_brier': market_brier,
            'implied_xg': row.calculated_implied_xg,
            'real_goals': act_home + act_away
        })

    df_report = pd.DataFrame(audit_records)
    
    # 4. Generate the Summary Report Consolidation
    print("\n" + "="*65)
    print("📋 TOURNAMENT COLD-START AUDIT REPORT")
    print("="*65)
    print(f"Model Outcome Tendency Accuracy : {df_report['mod_hit'].mean() * 100:.1f}%")
    print(f"Vegas/Bookmaker Market Accuracy : {df_report['m_hit'].mean() * 100:.1f}%")
    print(f"Model Mean Brier Penalty Score  : {df_report['mod_brier'].mean():.4f}")
    print(f"Market Mean Brier Penalty Score  : {df_report['m_brier'].mean():.4f}")
    print(f"Average Market Implied Total xG : {df_report['implied_xg'].mean():.2f} goals")
    print(f"Average Real Goals Scored       : {df_report['real_goals'].mean():.2f} goals")
    print("-"*65)
    
    print("\n🔍 FIXTURE BREAKDOWN MATRIX:")
    print(df_report[['fixture', 'actual', 'pred', 'pts', 'mod_hit', 'm_hit']].to_string(index=False))
    print("="*65)
    
    # 5. Diagnostic Verdict Logic
    mod_b = df_report['mod_brier'].mean()
    mar_b = df_report['m_brier'].mean()
    
    if mar_b > 0.24:
        print("💡 CRITICAL DIAGNOSIS: TOURNAMENT-WIDE CHAOS (VARIANCE)")
        print("The betting market's Brier score is deeply penalized. Las Vegas and bookmakers")
        print("are failing to read these opening matches alongside you. The tournament is highly")
        print("volatile right now (heavy underdogs winning, unexpected low-scoring grinds).")
    elif mod_b > mar_b + 0.04:
        print("🚨 CRITICAL DIAGNOSIS: MODEL DEVIATION DETECTED")
        print("The market is predicting outcomes significantly more cleanly than your script.")
        print("Your Dixon-Coles historical parameters are experiencing bias. Consider shifting")
        print("your alpha lower to rely more on real-time market lines.")
    else:
        print("✅ CRITICAL DIAGNOSIS: STANDARD EXACT-SCORE DRIFT")
        print("Your tendency metrics look completely acceptable, but the exact-score sequence")
        print("distribution is landing on low-probability margins. Hold the line.")

if __name__ == "__main__":
    run_diagnostic()