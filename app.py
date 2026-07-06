import streamlit as st # pyrefly: ignore [missing-import]
from pathlib import Path
import pandas as pd

# 1. Place this at the absolute top of the file before ANY other st code
st.set_page_config(page_title="WC 2026 Predictor", page_icon="🔮", layout="centered")

# ============================================================
# SIMPLE PASSWORD PROTECTION LAYER
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def check_password():
    """Returns True if the user is authenticated, otherwise renders login screen."""
    if st.session_state["authenticated"]:
        return True

    st.title("🔒 Restricted Access")
    
    # Password entry mask
    entered_password = st.text_input("Password", type="password", placeholder="Enter admin password...")
    
    if st.button("Sign In", use_container_width=True):
        ADMIN_PASSWORD = "6739431" 
        
        if entered_password == ADMIN_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Invalid password. Access denied.")
            
    return False

# If the password check returns False, stop script execution immediately
if not check_password():
    st.stop()

# ============================================================
# REST OF THE APP (only runs when authenticated)
# ============================================================

# Import your standalone single-prediction pipeline
from predict_match import predict_single_match, append_to_history_if_changed

st.title("2026 World Cup Predictor")
st.write("Evaluate live market odds adjustments instantly.")

# ============================================================
# FORM INPUTS (Optimized for mobile scrolling layout)
# ============================================================
with st.form("prediction_form"):
    st.subheader("Match Identity")
    match_date = st.date_input("Match Date", value=pd.Timestamp("2026-06-11"))
    home_team = st.text_input("Home Team", value=None)
    away_team = st.text_input("Away Team", value=None)
    
    st.markdown("---")
    st.subheader("1X2 Outcome Odds")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_odds = st.number_input("Home Odds", min_value=1.01, value=None, format="%.4f")
    with col2:
        draw_odds = st.number_input("Draw Odds", min_value=1.01, value=None, format="%.4f")
    with col3:
        away_odds = st.number_input("Away Odds", min_value=1.01, value=None, format="%.4f")
        
    st.markdown("---")
    st.subheader("Over / Under Goals Market")
    ou_line = st.selectbox("O/U Goal Line", options=[1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4, 4.25], index=2)
    
    col_o, col_u = st.columns(2)
    with col_o:
        over_odds = st.number_input("Over Odds", min_value=1.01, value=None, format="%.4f")
    with col_u:
        under_odds = st.number_input("Under Odds", min_value=1.01, value=None, format="%.4f")

    st.markdown("---")
    # Interactive option to suppress historical logging for test queries
    log_to_history = st.checkbox("Log to predictions_history.csv if changed", value=True)

    st.markdown("---")
    st.subheader("Advanced Calibration")

    override_alpha = st.checkbox("⚠️ Enable Market Override (Model Blind Spot)", value=False)
    alpha_val = 0.10 if override_alpha else 0.40

    st.caption(f"Current Alpha in use: **{alpha_val}**")
    
    submit_button = st.form_submit_button(label="Compute Optimal Scoreline", use_container_width=True)

# ============================================================
# EVALUATION & UI RENDERING
# ============================================================
if submit_button:
    if None in [home_odds, draw_odds, away_odds, over_odds, under_odds] or home_team == "" or away_team == "":
        st.error("⚠️ All team names and market odds fields must be filled out before computing!")
    else:
        try:
            # Run calculation using pre-compiled model state
            record = predict_single_match(
                home_team=home_team.strip(),
                away_team=away_team.strip(),
                match_date=str(match_date),
                home_odds=float(home_odds),
                draw_odds=float(draw_odds),
                away_odds=float(away_odds),
                ou_line=float(ou_line),
                over_odds=float(over_odds),
                under_odds=float(under_odds),
                alpha=alpha_val
            )
            
            # Display Core Strategy Cards
            st.success(f"### Optimal Prediction ({home_team.strip()} vs {away_team.strip()}): {record['prediction']}")
            
            c_ev1, c_ev2, c_margin = st.columns(3)
            c_ev1.metric("Optimal EV Pts", f"{record['expected_pts']:.4f}")
            c_ev2.metric("Backup Strategy", record['second_best_prediction'])
            c_margin.metric("Decision Margin", f"{record['decision_margin']:.4f}")
            
            # --- DIAGNOSTICS UI ---
            st.markdown("---")
            st.subheader("Probability Tug-of-War")
            
            # Create a clean visual comparison table
            diag_df = pd.DataFrame({
                "Source": ["Raw Model (12yr History)", "Raw Market (Vegas Odds)", f"Final Alpha Blend ({int(alpha_val * 100)}%)"],
                "Home Win": [f"{record['raw_model_home']*100:.1f}%", f"{record['raw_market_home']*100:.1f}%", f"{record['home_win']*100:.1f}%"],
                "Draw": [f"{record['raw_model_draw']*100:.1f}%", f"{record['raw_market_draw']*100:.1f}%", f"{record['draw']*100:.1f}%"],
                "Away Win": [f"{record['raw_model_away']*100:.1f}%", f"{record['raw_market_away']*100:.1f}%", f"{record['away_win']*100:.1f}%"]
            })
            st.dataframe(diag_df, hide_index=True, use_container_width=True)

            with st.expander("🔍 Advanced Engine Diagnostics"):
                st.write("**Goal Distribution (Lambdas)**")
                st.info(f"The Over/Under market expects **{record['calculated_implied_xg']:.3f} total goals**.")
                
                col_l1, col_l2 = st.columns(2)
                col_l1.metric(f"Home λ ({home_team.strip()})", f"{record['lambda_home']:.3f}")
                col_l2.metric(f"Away λ ({away_team.strip()})", f"{record['lambda_away']:.3f}")
                
                st.caption(
                    "Note: In the Matrix Tilt architecture, Lambdas dictate the *shape* of the scoreline based on the O/U market, "
                    "while the 1X2 market tilts the *final probabilities* above."
                )
            
            # Checkpoint History Logging Layer
            if log_to_history:
                append_to_history_if_changed(record)
                st.caption("Checked and synchronized ledger status inside predictions_history.csv.")

        except Exception as e:
            st.error(f"Prediction Failure: {e}")