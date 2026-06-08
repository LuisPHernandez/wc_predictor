"""
backtest_host_advantage.py
──────────────────────────
Tests whether applying real home advantage (neutral=False) for
WC host team matches improves pool point predictions vs treating
all WC matches as neutral (the current default).

For each WC year, runs two prediction passes:
    Baseline  — all matches neutral=True  (current production)
    Adjusted  — host team home matches neutral=False

Both passes use alpha=0.25 (settled production blend).

Covers: 2006, 2010, 2014, 2018, 2022
2002 excluded: no bookmaker odds available for alpha blending.

Note on home/away designation:
    WC matches in the kaggle dataset list the host team as
    home_team for their group stage games. The adjustment is
    only applied when the host appears as home_team — if the
    host is listed as away_team in a match (rare in WC data),
    neutral=True is preserved. Matches where this occurs are
    flagged in the output.

Usage:
    py -3 backtest_host_advantage.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from src.loader import (
    load_kaggle_base_data,
    build_competition_weights,
    build_confederation_weights,
)
from src.model import (
    DixonColes,
    outcome_probs_from_matrix,
)
from src.odds_loader import (
    _resolve_team,
    _shin_probs,
    _matchup_key,
)
from src.scoring import points_for_prediction

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
KAGGLE_PATH  = PROJECT_ROOT / "data" / "kaggle" / "results.csv"

# ─────────────────────────────────────────────────────────────
# Best tuned parameters — do not change
# ─────────────────────────────────────────────────────────────
DECAY_LAMBDA   = 0.2
TRAINING_YEARS = 12
REGULARIZATION = 0.0010

CONTINENTAL = 1.0
QUALIFIER   = 0.5
REGIONAL    = 0.3
FRIENDLY    = 0.3

CONMEBOL = 1.0
UEFA     = 1.0
CAF      = 1.10
CONCACAF = 1.05
AFC      = 0.95
OFC      = 0.90

ALPHA = 0.30   # settled production blend

# ─────────────────────────────────────────────────────────────
# WC host nations
# ─────────────────────────────────────────────────────────────
WC_HOSTS = {
    2002: {"Japan", "South Korea"},
    2006: {"Germany"},
    2010: {"South Africa"},
    2014: {"Brazil"},
    2018: {"Russia"},
    2022: {"Qatar"},
}

WC_YEARS = sorted(WC_HOSTS.keys())   # [2006, 2010, 2014, 2018, 2022]


# ─────────────────────────────────────────────────────────────
# Load and prepare odds (same logic as build_continental script)
# ─────────────────────────────────────────────────────────────
def load_wc_odds(year: int) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        matchup_key, book_home, book_draw, book_away
    using the same odds infrastructure as the rest of the project.
    Raises FileNotFoundError if odds for this year are unavailable.
    """
    from src.odds_loader import load_wc_odds as _load
    try:
        df = _load(year)
        return df
    except Exception as e:
        raise RuntimeError(f"Could not load odds for {year}: {e}")


# ─────────────────────────────────────────────────────────────
# Weight builders (computed once, reused across years)
# ─────────────────────────────────────────────────────────────
competition_weights   = build_competition_weights(CONTINENTAL, QUALIFIER, REGIONAL, FRIENDLY)
confederation_weights = build_confederation_weights(CONMEBOL, CAF, CONCACAF, AFC, OFC)


# ─────────────────────────────────────────────────────────────
# Helper: train model for a given WC year
# Identical training logic to production backtest.
# ─────────────────────────────────────────────────────────────
def train_model_for_year(year: int, wc_teams: list) -> DixonColes:
    # Infer tournament start from earliest match date in kaggle
    kaggle = pd.read_csv(KAGGLE_PATH)
    kaggle["date"] = pd.to_datetime(kaggle["date"])

    wc_matches = kaggle[
        (kaggle["date"].dt.year == year)
        & (kaggle["tournament"] == "FIFA World Cup")
    ]
    start_date = wc_matches["date"].min()

    training_end   = start_date - pd.Timedelta(days=1)
    training_start = training_end - pd.DateOffset(years=TRAINING_YEARS)

    base_df = load_kaggle_base_data(
        KAGGLE_PATH,
        wc_teams,
        training_start.strftime("%Y-%m-%d"),
        training_end.strftime("%Y-%m-%d"),
        DECAY_LAMBDA,
    )

    base_df["competition_weight"] = base_df["tournament"].map(competition_weights)

    home_conf = base_df["home_confederation"].map(confederation_weights).fillna(1.0)
    away_conf = base_df["away_confederation"].map(confederation_weights).fillna(1.0)
    base_df["confederation_weight"] = np.sqrt(home_conf * away_conf)

    base_df["weight"] = (
        base_df["recency_weight"]
        * base_df["competition_weight"]
        * base_df["confederation_weight"]
    )

    train_df = base_df[[
        "date", "home_team", "away_team",
        "home_score", "away_score",
        "neutral", "weight",
    ]]

    model = DixonColes(
        train_df,
        decay_lambda=DECAY_LAMBDA,
        regularization=REGULARIZATION,
    )
    model.fit()
    return model, start_date


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
year_summaries = []
all_match_rows = []

kaggle_full = pd.read_csv(KAGGLE_PATH)
kaggle_full["date"] = pd.to_datetime(kaggle_full["date"])

for year in WC_YEARS:
    hosts = WC_HOSTS[year]

    print(f"\n{'=' * 60}")
    print(f"  {year}  |  Host(s): {', '.join(sorted(hosts))}")
    print(f"{'=' * 60}")

    # ── Identify WC matches for this year ────────────────────
    wc_df = kaggle_full[
        (kaggle_full["date"].dt.year == year)
        & (kaggle_full["tournament"] == "FIFA World Cup")
    ].copy()

    wc_df["home_team"] = wc_df["home_team"].astype(str).str.strip().apply(_resolve_team)
    wc_df["away_team"] = wc_df["away_team"].astype(str).str.strip().apply(_resolve_team)

    wc_teams = sorted(set(wc_df["home_team"]) | set(wc_df["away_team"]))
    print(f"  Matches : {len(wc_df)}")
    print(f"  Teams   : {len(wc_teams)}")

    # ── Load odds ─────────────────────────────────────────────
    try:
        odds_df = load_wc_odds(year)
        odds_df["home_team"] = odds_df["home_team"].astype(str).str.strip().apply(_resolve_team)
        odds_df["away_team"] = odds_df["away_team"].astype(str).str.strip().apply(_resolve_team)
        odds_df["matchup_key"] = odds_df.apply(
            lambda r: _matchup_key(r["date"], r["home_team"], r["away_team"]), axis=1
        )
        probs = odds_df.apply(
            lambda r: _shin_probs(r["h_odds_avg"], r["d_odds_avg"], r["a_odds_avg"]), axis=1
        )
        odds_df["book_home"] = [x[0] for x in probs]
        odds_df["book_draw"] = [x[1] for x in probs]
        odds_df["book_away"] = [x[2] for x in probs]
        odds_lookup = odds_df.set_index("matchup_key")[["book_home", "book_draw", "book_away"]].to_dict("index")
        print(f"  Odds    : {len(odds_df)} matches loaded")
    except Exception as e:
        print(f"  WARNING: could not load odds — {e}")
        odds_lookup = {}

    # ── Train model ───────────────────────────────────────────
    print(f"  Training model...", end=" ", flush=True)
    model, tournament_start = train_model_for_year(year, wc_teams)
    print("done.")

    # ── Host match counts ─────────────────────────────────────
    host_as_home  = wc_df[wc_df["home_team"].isin(hosts)]
    host_as_away  = wc_df[wc_df["away_team"].isin(hosts) & ~wc_df["home_team"].isin(hosts)]

    print(f"\n  Host as home_team : {len(host_as_home)} matches  → will use neutral=False")
    print(f"  Host as away_team : {len(host_as_away)} matches  → neutral=True preserved (flagged)")

    if not host_as_away.empty:
        print(f"  Flagged away matches:")
        for _, r in host_as_away.iterrows():
            print(f"    {r['date'].date()}  {r['home_team']} vs {r['away_team']}")

    # ── Predict with both neutral settings ───────────────────
    print(f"\n  {'Match':<40}  {'Base pts':>8}  {'Adj pts':>8}  {'Δ':>4}  Note")
    print(f"  {'─' * 40}  {'─' * 8}  {'─' * 8}  {'─' * 4}")

    year_base_total = 0
    year_adj_total  = 0
    host_base_total = 0
    host_adj_total  = 0
    n_odds          = 0

    for _, match in wc_df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        actual_home = int(match["home_score"])
        actual_away = int(match["away_score"])
        key = _matchup_key(str(match["date"].date()), home, away)

        # Book probs (None if no odds for this match)
        book_probs = odds_lookup.get(key)

        # ── Baseline: neutral=True always ────────────────────
        base_pred = model.predict(
            home, away,
            neutral=True,
            bookmaker_probs=book_probs,
            alpha=ALPHA if book_probs else 1.0,
        )
        base_pts = points_for_prediction(
            base_pred["pred_home"], base_pred["pred_away"],
            actual_home, actual_away,
        )

        # ── Adjusted: neutral=False for host home matches ────
        is_host_home = home in hosts
        adj_neutral  = not is_host_home

        adj_pred = model.predict(
            home, away,
            neutral=adj_neutral,
            bookmaker_probs=book_probs,
            alpha=ALPHA if book_probs else 1.0,
        )
        adj_pts = points_for_prediction(
            adj_pred["pred_home"], adj_pred["pred_away"],
            actual_home, actual_away,
        )

        year_base_total += base_pts
        year_adj_total  += adj_pts
        if book_probs:
            n_odds += 1

        note = ""
        if is_host_home:
            host_base_total += base_pts
            host_adj_total  += adj_pts
            changed = base_pred["prediction"] != adj_pred["prediction"]
            note = "HOST ◄" + (" PRED CHANGED" if changed else "")

        delta = adj_pts - base_pts
        delta_str = f"+{delta}" if delta > 0 else str(delta)

        # Only print host matches and any match where prediction changed
        if is_host_home or (base_pred["prediction"] != adj_pred["prediction"]):
            match_label = f"{home} vs {away}"[:38]
            print(
                f"  {match_label:<40}  "
                f"{base_pts:>8}  "
                f"{adj_pts:>8}  "
                f"{delta_str:>4}  "
                f"{note}"
            )

        all_match_rows.append({
            "year"          : year,
            "date"          : str(match["date"].date()),
            "home_team"     : home,
            "away_team"     : away,
            "actual_home"   : actual_home,
            "actual_away"   : actual_away,
            "is_host_home"  : is_host_home,
            "has_odds"      : book_probs is not None,
            "base_prediction"  : base_pred["prediction"],
            "adj_prediction"   : adj_pred["prediction"],
            "pred_changed"  : base_pred["prediction"] != adj_pred["prediction"],
            "base_pts"      : base_pts,
            "adj_pts"       : adj_pts,
            "delta_pts"     : adj_pts - base_pts,
        })

    year_summaries.append({
        "year"            : year,
        "hosts"           : ", ".join(sorted(hosts)),
        "host_matches"    : len(host_as_home),
        "total_base"      : year_base_total,
        "total_adj"       : year_adj_total,
        "total_delta"     : year_adj_total - year_base_total,
        "host_base"       : host_base_total,
        "host_adj"        : host_adj_total,
        "host_delta"      : host_adj_total - host_base_total,
    })

    print(f"\n  Year total  |  Baseline: {year_base_total}  →  Adjusted: {year_adj_total}  "
          f"({'+'  if year_adj_total >= year_base_total else ''}{year_adj_total - year_base_total})")
    print(f"  Host only   |  Baseline: {host_base_total}  →  Adjusted: {host_adj_total}  "
          f"({'+'  if host_adj_total >= host_base_total else ''}{host_adj_total - host_base_total})")


# ─────────────────────────────────────────────────────────────
# Overall summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("OVERALL SUMMARY")
print(f"{'=' * 65}")
print(f"\n  {'Year':<6}  {'Host(s)':<20}  {'Host#':>5}  {'Base':>5}  {'Adj':>5}  {'Δ Total':>7}  {'Δ Host':>6}")
print(f"  {'─' * 6}  {'─' * 20}  {'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 7}  {'─' * 6}")

total_base = 0
total_adj  = 0

for s in year_summaries:
    delta_total = s["total_delta"]
    delta_host  = s["host_delta"]
    print(
        f"  {s['year']:<6}  "
        f"{s['hosts']:<20}  "
        f"{s['host_matches']:>5}  "
        f"{s['total_base']:>5}  "
        f"{s['total_adj']:>5}  "
        f"{delta_total:>+7}  "
        f"{delta_host:>+6}"
    )
    total_base += s["total_base"]
    total_adj  += s["total_adj"]

print(f"  {'─' * 6}  {'─' * 20}  {'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 7}  {'─' * 6}")
grand_delta = total_adj - total_base
print(f"  {'TOTAL':<6}  {'':20}  {'':>5}  {total_base:>5}  {total_adj:>5}  {grand_delta:>+7}")

print(f"\n  Prediction changes (neutral=True → neutral=False):")
match_df = pd.DataFrame(all_match_rows)
changed = match_df[match_df["pred_changed"]]
if changed.empty:
    print("  None — home advantage affected point totals but not scoreline predictions.")
else:
    print(f"  {len(changed)} predictions changed across all years:")
    for _, r in changed.iterrows():
        print(
            f"    {r['year']}  {r['home_team']} vs {r['away_team']:<25}  "
            f"{r['base_prediction']} → {r['adj_prediction']}  "
            f"({r['base_pts']:+} → {r['adj_pts']:+} pts)"
        )

# ─────────────────────────────────────────────────────────────
# Interpretation guidance
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("INTERPRETATION")
print(f"{'=' * 65}")
print(f"""
Grand delta = {grand_delta:+} points across {len(WC_YEARS)} WCs
({sum(s['host_matches'] for s in year_summaries)} host-team home matches affected)

If grand delta is +3 or more and positive in most years:
    The host advantage fix is real and worth implementing.
    Apply neutral=False for USA, Canada, Mexico home matches
    in predict_2026.py.

If grand delta is between -2 and +2:
    The effect is negligible. The host advantage is either
    already captured by the alpha=0.25 bookmaker blend (the
    match odds already price in home crowd/travel effects)
    or is too small to matter over 3-5 host matches per WC.
    You can still implement it for correctness but do not
    expect a measurable points improvement.

If grand delta is negative:
    The host advantage parameter overcorrects for WC context.
    The training home advantage was estimated on club/qualifier
    data where home advantage is larger than in WC knockout
    matches. Do NOT implement the neutral=False change.
""")

# ─────────────────────────────────────────────────────────────
# Save detail CSV
# ─────────────────────────────────────────────────────────────
out_path = PROJECT_ROOT / "data" / "odds" / "host_advantage_backtest.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
match_df.to_csv(out_path, index=False)
print(f"Detailed results saved: {out_path}")