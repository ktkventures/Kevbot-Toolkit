"""
KevBot Trade Analyzer - Proof of Concept
=========================================
Analyzes exported trade data to find optimal confluence combinations.

Key concept: A "confluence record" is an atomic unit combining:
  - Timeframe (e.g., "1M") - the chart interval
  - Evaluator (e.g., "EMA") - module that analyzes price/indicator data
  - State (e.g., "SML") - the current condition output

Example: "1M-EMA-SML" is one confluence record.

The tool finds which combinations of confluence records at entry
correlate with the best trading outcomes.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONFIGURATION
# =============================================================================

TRIGGERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
TF_SLOTS = ["TF1", "TF2", "TF3", "TF4", "TF5", "TF6"]
DEFAULT_TF_LABELS = ["1M", "5M", "15M", "1H", "4H", "1D"]

# Side module definitions
SIDE_MODULES = {
    "EMA": {
        "name": "EMA Stack",
        "states": {
            0: None,      # Unknown/NA - not a valid confluence
            1: "SML",     # Bull Stack
            2: "SLM",
            3: "MSL",
            4: "MLS",
            5: "LSM",
            6: "LMS",     # Bear Stack
        }
    },
    "MACD": {
        "name": "MACD Simple",
        "states": {
            0: None,      # Unknown/NA
            1: "M>S↑",    # Strengthening bullish
            2: "M>S↓",    # Weakening bullish
            3: "M<S↓",    # Strengthening bearish
            4: "M<S↑",    # Weakening bearish
        }
    },
}

ACTIVE_MODULES = ["EMA", "MACD"]


# =============================================================================
# MOCK DATA GENERATOR
# =============================================================================

def generate_mock_data(n_bars=5000, seed=42):
    """Generate mock TradingView-style export data."""
    np.random.seed(seed)

    start_price = 100.0
    returns = np.random.normal(0.0002, 0.002, n_bars)
    prices = start_price * np.cumprod(1 + returns)

    start_time = datetime(2026, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]

    df = pd.DataFrame({
        "time": timestamps,
        "open": prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        "high": prices * (1 + np.random.uniform(0, 0.003, n_bars)),
        "low": prices * (1 + np.random.uniform(-0.003, 0, n_bars)),
        "close": prices,
        "volume": np.random.randint(1000, 50000, n_bars),
    })

    # Evaluator states per timeframe
    for mod_key in ACTIVE_MODULES:
        mod_info = SIDE_MODULES[mod_key]
        num_states = len([s for s in mod_info["states"].values() if s is not None])

        for i, tf in enumerate(TF_SLOTS):
            change_prob = 0.05 / (i + 1)
            states = []
            current_state = np.random.randint(1, num_states + 1)
            for _ in range(n_bars):
                if np.random.random() < change_prob:
                    current_state = np.random.randint(1, num_states + 1)
                states.append(current_state)
            df[f"Export: {tf} {mod_key} State"] = states

    # Entry/exit signals
    entry_prob = 0.015
    df["Export: Pos Long Entry"] = (np.random.random(n_bars) < entry_prob).astype(int)
    df["Export: Pos Short Entry"] = (np.random.random(n_bars) < entry_prob).astype(int)
    df["Export: Pos Long Exit"] = 0
    df["Export: Pos Short Exit"] = 0

    df["Export: LE Trigger Idx"] = np.where(df["Export: Pos Long Entry"] == 1, np.random.randint(0, 4, n_bars), -1)
    df["Export: SE Trigger Idx"] = np.where(df["Export: Pos Short Entry"] == 1, np.random.randint(0, 4, n_bars), -1)
    df["Export: LX Trigger Idx"] = -1
    df["Export: SX Trigger Idx"] = -1

    for i in range(n_bars):
        if df.loc[i, "Export: Pos Long Entry"] == 1:
            exit_bar = min(i + np.random.randint(5, 50), n_bars - 1)
            df.loc[exit_bar, "Export: Pos Long Exit"] = 1
            df.loc[exit_bar, "Export: LX Trigger Idx"] = np.random.randint(0, 4)
        if df.loc[i, "Export: Pos Short Entry"] == 1:
            exit_bar = min(i + np.random.randint(5, 50), n_bars - 1)
            df.loc[exit_bar, "Export: Pos Short Exit"] = 1
            df.loc[exit_bar, "Export: SX Trigger Idx"] = np.random.randint(0, 4)

    df["Export: Long Risk (Price)"] = df["close"] * np.random.uniform(0.005, 0.02, n_bars)
    df["Export: Short Risk (Price)"] = df["close"] * np.random.uniform(0.005, 0.02, n_bars)

    return df


# =============================================================================
# CONFLUENCE RECORD EXTRACTION
# =============================================================================

def get_confluence_records(row, tf_labels):
    """
    Extract all confluence records present at a given bar.

    Returns a set of strings like {"1M-EMA-SML", "5M-MACD-M>S↑", ...}
    """
    records = set()

    for mod_key in ACTIVE_MODULES:
        mod_info = SIDE_MODULES[mod_key]
        for tf_slot in TF_SLOTS:
            col_name = f"Export: {tf_slot} {mod_key} State"
            if col_name in row.index:
                state_val = int(row[col_name])
                state_label = mod_info["states"].get(state_val)
                if state_label is not None:  # Valid state
                    tf_label = tf_labels.get(tf_slot, tf_slot)
                    record = f"{tf_label}-{mod_key}-{state_label}"
                    records.add(record)

    return records


def extract_trades(df, tf_labels, direction="long", entry_trigger=None, exit_trigger=None):
    """
    Extract trades with confluence records at entry.

    Each trade has a 'confluence_records' field containing the set of
    all confluence records that were true at entry time.
    """
    if direction == "long":
        entry_col = "Export: Pos Long Entry"
        exit_col = "Export: Pos Long Exit"
        entry_trigger_col = "Export: LE Trigger Idx"
        exit_trigger_col = "Export: LX Trigger Idx"
        risk_col = "Export: Long Risk (Price)"
    else:
        entry_col = "Export: Pos Short Entry"
        exit_col = "Export: Pos Short Exit"
        entry_trigger_col = "Export: SE Trigger Idx"
        exit_trigger_col = "Export: SX Trigger Idx"
        risk_col = "Export: Short Risk (Price)"

    trades = []
    entry_idx = None

    for i in range(len(df)):
        if entry_idx is None and df.iloc[i][entry_col] == 1:
            if entry_trigger is not None and df.iloc[i][entry_trigger_col] != entry_trigger:
                continue
            entry_idx = i

        elif entry_idx is not None and df.iloc[i][exit_col] == 1:
            exit_row = df.iloc[i]
            if exit_trigger is not None and exit_row[exit_trigger_col] != exit_trigger:
                continue

            entry_row = df.iloc[entry_idx]
            entry_price = entry_row["close"]
            exit_price = exit_row["close"]
            risk = entry_row[risk_col]

            pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
            r_multiple = pnl / risk if risk > 0 else 0

            # Get confluence records at entry
            confluence_records = get_confluence_records(entry_row, tf_labels)

            trade = {
                "entry_time": entry_row["time"],
                "exit_time": exit_row["time"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "risk": risk,
                "r_multiple": r_multiple,
                "win": pnl > 0,
                "entry_trigger": TRIGGERS[int(entry_row[entry_trigger_col])] if entry_row[entry_trigger_col] >= 0 else "?",
                "exit_trigger": TRIGGERS[int(exit_row[exit_trigger_col])] if exit_row[exit_trigger_col] >= 0 else "?",
                "confluence_records": confluence_records,
            }
            trades.append(trade)
            entry_idx = None

    return pd.DataFrame(trades)


# =============================================================================
# KPI CALCULATIONS
# =============================================================================

def calculate_kpis(trades_df, pnl_settings=None):
    """Calculate KPIs for a set of trades."""
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_r": 0, "daily_pnl": 0,
            "final_balance": 0, "total_pnl_dollars": 0, "daily_pnl_dollars": 0
        }

    wins = trades_df[trades_df["win"] == True]
    losses = trades_df[trades_df["win"] == False]
    gross_profit = wins["r_multiple"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["r_multiple"].sum()) if len(losses) > 0 else 0
    total_r = trades_df["r_multiple"].sum()

    # Calculate trading days (unique days with trades)
    if "exit_time" in trades_df.columns:
        trading_days = trades_df["exit_time"].dt.date.nunique()
    else:
        trading_days = 1
    trading_days = max(trading_days, 1)  # Avoid division by zero

    # Dollar-denominated P&L calculations
    final_balance = 0
    total_pnl_dollars = 0

    if pnl_settings:
        starting_balance = pnl_settings.get("starting_balance", 10000)
        mode = pnl_settings.get("mode", "Fixed")

        if mode == "Fixed":
            # Fixed risk: each R = risk_per_trade dollars
            risk_per_trade = pnl_settings.get("risk_per_trade", 100)
            total_pnl_dollars = total_r * risk_per_trade
            final_balance = starting_balance + total_pnl_dollars
        else:
            # Compounding: risk grows with balance
            risk_pct = pnl_settings.get("risk_pct", 2.0) / 100
            balance = starting_balance

            # Sort trades by exit time for proper compounding
            sorted_trades = trades_df.sort_values("exit_time")
            for _, trade in sorted_trades.iterrows():
                risk_amount = balance * risk_pct
                trade_pnl = trade["r_multiple"] * risk_amount
                balance += trade_pnl

            final_balance = balance
            total_pnl_dollars = final_balance - starting_balance

    daily_pnl_dollars = total_pnl_dollars / trading_days if trading_days > 0 else 0

    return {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        "avg_r": trades_df["r_multiple"].mean(),
        "total_r": total_r,
        "daily_pnl": total_r / trading_days,
        "final_balance": final_balance,
        "total_pnl_dollars": total_pnl_dollars,
        "daily_pnl_dollars": daily_pnl_dollars,
    }


# =============================================================================
# CONFLUENCE ANALYSIS
# =============================================================================

def analyze_single_factors(trades_df, min_trades=5):
    """
    Rank individual confluence records by their impact.

    For each confluence record, calculate KPIs for trades where
    that record was present at entry.
    """
    if len(trades_df) == 0:
        return pd.DataFrame()

    # Collect all unique confluence records
    all_records = set()
    for records in trades_df["confluence_records"]:
        all_records.update(records)

    results = []
    for record in all_records:
        # Filter trades where this record was present
        mask = trades_df["confluence_records"].apply(lambda r: record in r)
        subset = trades_df[mask]

        if len(subset) >= min_trades:
            kpis = calculate_kpis(subset)
            results.append({
                "confluence": record,
                **kpis
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("profit_factor", ascending=False)

    return results_df


def analyze_combinations(trades_df, required_records, min_trades=5):
    """
    Find what additional confluence records improve results when combined
    with the required_records.

    required_records: set of confluence records that must be present
    Returns: DataFrame ranking additional records by improvement
    """
    if len(trades_df) == 0:
        return pd.DataFrame()

    # Filter to trades that have ALL required records
    if required_records:
        mask = trades_df["confluence_records"].apply(
            lambda r: required_records.issubset(r)
        )
        base_trades = trades_df[mask]
    else:
        base_trades = trades_df

    if len(base_trades) < min_trades:
        return pd.DataFrame()

    base_kpis = calculate_kpis(base_trades)

    # Find all other records present in these trades
    other_records = set()
    for records in base_trades["confluence_records"]:
        other_records.update(records - required_records)

    results = []
    for record in other_records:
        # Filter to trades that also have this additional record
        mask = base_trades["confluence_records"].apply(lambda r: record in r)
        subset = base_trades[mask]

        if len(subset) >= min_trades:
            kpis = calculate_kpis(subset)

            # Calculate improvement over base
            pf_improvement = kpis["profit_factor"] - base_kpis["profit_factor"]
            wr_improvement = kpis["win_rate"] - base_kpis["win_rate"]

            results.append({
                "add_confluence": record,
                **kpis,
                "pf_change": pf_improvement,
                "wr_change": wr_improvement,
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("profit_factor", ascending=False)

    return results_df


def find_best_combinations(trades_df, max_depth=4, min_trades=5, top_n=10):
    """
    Automatically find the best N-factor combinations.

    Uses a greedy approach: start with best single factor, then find
    best pair, best triple, etc.
    """
    if len(trades_df) == 0:
        return []

    results = []

    # Get all unique records
    all_records = set()
    for records in trades_df["confluence_records"]:
        all_records.update(records)
    all_records = list(all_records)

    # Try combinations of increasing size
    for depth in range(1, min(max_depth + 1, len(all_records) + 1)):
        for combo in combinations(all_records, depth):
            combo_set = set(combo)

            # Filter trades with ALL these records
            mask = trades_df["confluence_records"].apply(
                lambda r: combo_set.issubset(r)
            )
            subset = trades_df[mask]

            if len(subset) >= min_trades:
                kpis = calculate_kpis(subset)
                results.append({
                    "combination": combo_set,
                    "combo_str": " + ".join(sorted(combo_set)),
                    "depth": depth,
                    **kpis
                })

    # Sort by profit factor and return top N
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values(
            ["profit_factor", "total_trades"],
            ascending=[False, False]
        ).head(top_n)

    return results_df


# =============================================================================
# STREAMLIT UI
# =============================================================================

# CSS for styling
CUSTOM_CSS = """
<style>
/* Tighten up vertical spacing globally */
div[data-testid="stVerticalBlock"] > div {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
div[data-testid="stHorizontalBlock"] {
    gap: 0.1rem !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* Reduce caption margins */
.stCaption, p {
    margin-bottom: 0 !important;
    margin-top: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}
/* Tighter checkbox spacing */
div[data-testid="stCheckbox"] {
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stCheckbox"] > label {
    padding: 0 !important;
    min-height: 0 !important;
}
/* Container padding */
div[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0.5rem !important;
}
/* Remove gap before scrollable container */
div[data-testid="stVerticalBlockBorderWrapper"] + div {
    margin-top: 0 !important;
}
/* Scrollable container internal spacing */
div[style*="overflow"] > div {
    padding-top: 0 !important;
}
</style>
"""


def main():
    st.set_page_config(page_title="KevBot Trade Analyzer", page_icon="📊", layout="wide")

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar - Data & Settings
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Data Source")
        data_source = st.radio("Source:", ["Mock Data", "Upload CSV"], index=0, horizontal=True)

        if data_source == "Mock Data":
            col1, col2 = st.columns(2)
            with col1:
                n_bars = st.number_input("Bars", value=5000, min_value=1000, max_value=10000)
            with col2:
                seed = st.number_input("Seed", value=42)
            if st.button("Generate", type="primary", use_container_width=True):
                st.session_state.df = generate_mock_data(n_bars, seed)
                st.session_state.selected_confluences = set()
        else:
            uploaded = st.file_uploader("CSV file", type="csv")
            if uploaded:
                st.session_state.df = pd.read_csv(uploaded)
                st.session_state.selected_confluences = set()

        st.divider()
        st.header("Timeframes")
        tf_labels = {}
        cols = st.columns(3)
        for i, tf_slot in enumerate(TF_SLOTS):
            with cols[i % 3]:
                tf_labels[tf_slot] = st.text_input(tf_slot, value=DEFAULT_TF_LABELS[i], max_chars=4, label_visibility="collapsed")
        st.session_state.tf_labels = tf_labels

        st.divider()
        st.header("Strategy")
        direction = st.selectbox("Direction", ["long", "short"])
        col1, col2 = st.columns(2)
        trigger_options = ["Any"] + TRIGGERS[:4]
        with col1:
            entry_trigger = st.selectbox("Entry", trigger_options, index=1)
        with col2:
            exit_trigger = st.selectbox("Exit", trigger_options, index=2)

        min_trades = st.slider("Min trades", 1, 30, 5)

        st.divider()
        st.header("P&L Settings")
        risk_mode = st.radio(
            "Risk Mode",
            ["Fixed", "Compounding"],
            index=0,
            horizontal=True,
            help="Fixed: same $ risk per trade. Compounding: risk grows with balance."
        )
        starting_balance = st.number_input(
            "Starting Balance ($)",
            value=10000,
            min_value=100,
            step=1000
        )
        if risk_mode == "Fixed":
            risk_per_trade = st.number_input(
                "Risk per Trade ($)",
                value=100,
                min_value=1,
                step=10
            )
            risk_pct = None
        else:
            risk_pct = st.slider(
                "Risk per Trade (%)",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5
            )
            risk_per_trade = None

        # Store in session state for calculations
        st.session_state.pnl_settings = {
            "mode": risk_mode,
            "starting_balance": starting_balance,
            "risk_per_trade": risk_per_trade,
            "risk_pct": risk_pct
        }

        st.divider()
        st.header("Export")
        if st.button("📤 Export TradingView Parameters", use_container_width=True):
            st.info("🚧 Coming soon! This will export your confluence settings to TradingView indicator parameters.")

    # -------------------------------------------------------------------------
    # Initialize Data & State
    # -------------------------------------------------------------------------
    if "df" not in st.session_state:
        st.session_state.df = generate_mock_data(5000, 42)
    if "selected_confluences" not in st.session_state:
        st.session_state.selected_confluences = set()

    df = st.session_state.df
    tf_labels = st.session_state.get("tf_labels", dict(zip(TF_SLOTS, DEFAULT_TF_LABELS)))

    # Extract all trades
    entry_filter = None if entry_trigger == "Any" else TRIGGERS.index(entry_trigger)
    exit_filter = None if exit_trigger == "Any" else TRIGGERS.index(exit_trigger)
    all_trades = extract_trades(df, tf_labels, direction, entry_filter, exit_filter)

    # Apply confluence filter (AND only)
    selected = st.session_state.selected_confluences
    has_filter = len(selected) > 0

    if has_filter and len(all_trades) > 0:
        mask = all_trades["confluence_records"].apply(lambda r: selected.issubset(r))
        filtered_trades = all_trades[mask]
    else:
        filtered_trades = all_trades

    # -------------------------------------------------------------------------
    # Header - Strategy & Current Filter
    # -------------------------------------------------------------------------
    st.markdown(f"### {direction.title()} | Entry: {entry_trigger} → Exit: {exit_trigger}")

    # Show current filters with individual remove buttons
    if has_filter:
        filter_cols = st.columns(min(len(selected) + 1, 8))
        for i, conf in enumerate(sorted(selected)):
            with filter_cols[i]:
                if st.button(f"✕ {conf}", key=f"rm_{conf}", help=f"Remove {conf}"):
                    st.session_state.selected_confluences.discard(conf)
                    st.rerun()
    else:
        st.caption("Click cards below to filter by confluence")

    # KPIs row
    pnl_settings = st.session_state.get("pnl_settings", None)
    kpis = calculate_kpis(filtered_trades, pnl_settings)
    cols = st.columns(6)
    cols[0].metric("Trades", kpis["total_trades"])
    cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
    cols[2].metric("PF", f"{kpis['profit_factor']:.2f}" if kpis['profit_factor'] != float('inf') else "∞")
    cols[3].metric("Daily P&L", f"{kpis['daily_pnl']:+.2f}R", f"${kpis['daily_pnl_dollars']:+,.0f}")
    cols[4].metric("Total P&L", f"{kpis['total_r']:+.1f}R", f"${kpis['total_pnl_dollars']:+,.0f}")
    cols[5].metric("Final Balance", f"${kpis['final_balance']:,.0f}")

    # -------------------------------------------------------------------------
    # Main Layout: Equity Curve (left) | Analysis Panel (right)
    # -------------------------------------------------------------------------

    # Shared container styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 12px; padding: 4px; margin-bottom: 1rem;">
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="medium")

    # --- LEFT: Equity Curve ---
    with left_col:
        with st.container(border=True):
            st.markdown("**📈 Equity Curve**")

            if len(filtered_trades) > 0:
                equity_df = filtered_trades[["exit_time", "r_multiple"]].sort_values("exit_time")
                equity_df["cumulative_r"] = equity_df["r_multiple"].cumsum()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df["exit_time"], y=equity_df["cumulative_r"],
                    mode="lines", name="Cumulative R",
                    line=dict(color="#2196F3", width=2),
                    fill="tozeroy", fillcolor="rgba(33, 150, 243, 0.1)"
                ))
                fig.add_trace(go.Scatter(
                    x=equity_df["exit_time"], y=equity_df["cumulative_r"].cummax(),
                    mode="lines", name="HWM",
                    line=dict(color="green", width=1, dash="dot")
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.update_layout(
                    height=340,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="",
                    yaxis_title="Cumulative R",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trades to display")

            # Clear filter button
            if has_filter:
                if st.button("🗑️ Clear All", key="clear_all", use_container_width=True):
                    st.session_state.selected_confluences = set()
                    st.rerun()

    # --- RIGHT: Analysis Panel ---
    with right_col:
        with st.container(border=True):
            # Mode toggle
            mode = st.radio("Mode:", ["Drill-Down", "Auto-Search"], horizontal=True, key="mode_toggle", label_visibility="collapsed")

            if mode == "Drill-Down":
                # Header with sort option
                header_cols = st.columns([2, 1])
                with header_cols[0]:
                    st.markdown("**🎯 Confluences**")
                with header_cols[1]:
                    sort_by = st.selectbox(
                        "Sort",
                        ["Daily P/L", "PF", "WR%", "Trades"],
                        index=0,
                        label_visibility="collapsed"
                    )

                sort_map = {
                    "Daily P/L": "daily_pnl",
                    "PF": "profit_factor",
                    "WR%": "win_rate",
                    "Trades": "total_trades"
                }
                sort_col = sort_map[sort_by]

                if len(all_trades) > 0:
                    single_factors = analyze_single_factors(all_trades, min_trades)

                    if has_filter:
                        combos = analyze_combinations(all_trades, selected, min_trades)
                        if len(combos) > 0:
                            display_data = combos
                            conf_col = "add_confluence"
                        else:
                            display_data = single_factors
                            conf_col = "confluence"
                    else:
                        display_data = single_factors
                        conf_col = "confluence"

                    # Sort by selected metric
                    if len(display_data) > 0 and sort_col in display_data.columns:
                        display_data = display_data.sort_values(sort_col, ascending=False).head(15)

                    if len(display_data) > 0:
                        # Header row (HTML for tight spacing)
                        st.markdown(
                            '<div style="display:flex; font-size:11px; color:#888; padding:0 0 4px 0; border-bottom:1px solid #eee; margin-bottom:0;">'
                            '<span style="width:8%"></span>'
                            '<span style="width:42%">Confluence</span>'
                            '<span style="width:12%">#</span>'
                            '<span style="width:10%">PF</span>'
                            '<span style="width:12%">WR</span>'
                            '<span style="width:16%">Daily</span>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        # Table rows in scrollable container
                        with st.container(height=275):
                            for idx, row in display_data.iterrows():
                                conf = row[conf_col]
                                is_selected = conf in selected

                                pf = row['profit_factor']
                                pf_str = f"{pf:.1f}" if pf != float('inf') else "∞"

                                cols = st.columns([0.4, 2.1, 0.6, 0.5, 0.6, 0.8])

                                with cols[0]:
                                    if st.checkbox("", value=is_selected, key=f"sel_{conf}", label_visibility="collapsed"):
                                        if not is_selected:
                                            st.session_state.selected_confluences.add(conf)
                                            st.rerun()
                                    elif is_selected:
                                        st.session_state.selected_confluences.discard(conf)
                                        st.rerun()

                                cols[1].markdown(f"**{conf}**" if is_selected else conf)
                                cols[2].caption(f"{row['total_trades']}")
                                cols[3].caption(f"{pf_str}")
                                cols[4].caption(f"{row['win_rate']:.0f}%")
                                cols[5].caption(f"{row['daily_pnl']:+.2f}")
                    else:
                        st.info(f"No records with {min_trades}+ trades")
                else:
                    st.warning("No trades found")

            else:  # Auto-Search mode
                # Header + filters + search button on same row
                btn_cols = st.columns([1, 1, 2])
                with btn_cols[0]:
                    show_filters = st.popover("⚙️ Filters")
                with btn_cols[1]:
                    search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)

                # Filters in popover
                with show_filters:
                    st.markdown("**Search Filters**")
                    max_depth = st.slider("Max factors", 1, 5, 3, key="auto_depth")
                    top_n = st.slider("Top results", 5, 30, 10, key="auto_topn")
                    auto_min_trades = st.slider("Min trades", 1, 50, min_trades, key="auto_min_trades")
                    auto_min_wr = st.slider("Min win rate %", 0, 80, 0, key="auto_min_wr")
                    auto_min_pf = st.slider("Min profit factor", 0.0, 3.0, 0.0, step=0.1, key="auto_min_pf")

                if search_clicked:
                    with st.spinner("Searching..."):
                        best_combos = find_best_combinations(all_trades, max_depth, auto_min_trades, top_n)

                        # Apply additional filters
                        if len(best_combos) > 0:
                            if auto_min_wr > 0:
                                best_combos = best_combos[best_combos["win_rate"] >= auto_min_wr]
                            if auto_min_pf > 0:
                                best_combos = best_combos[best_combos["profit_factor"] >= auto_min_pf]

                    if len(best_combos) > 0:
                        st.session_state.auto_results = best_combos

                # Show results in scrollable container
                if "auto_results" in st.session_state and len(st.session_state.auto_results) > 0:
                    # Column headers (compact)
                    st.markdown(
                        '<div style="display:flex; font-size:11px; color:#888; padding:0 0 4px 0; border-bottom:1px solid #eee; margin-bottom:4px;">'
                        '<span style="width:7%">N</span>'
                        '<span style="width:50%">Combination</span>'
                        '<span style="width:10%">#</span>'
                        '<span style="width:9%">PF</span>'
                        '<span style="width:10%">WR</span>'
                        '<span style="width:14%"></span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    results_container = st.container(height=280)

                    with results_container:
                        for idx, row in st.session_state.auto_results.iterrows():
                            combo_set = row["combination"]
                            pf = row['profit_factor']
                            pf_str = f"{pf:.1f}" if pf != float('inf') else "∞"
                            depth = row["depth"]

                            cols = st.columns([0.4, 2.8, 0.5, 0.4, 0.5, 0.8])
                            cols[0].caption(f"{depth}")
                            cols[1].markdown(f"**{row['combo_str']}**")
                            cols[2].caption(f"{row['total_trades']}")
                            cols[3].caption(f"{pf_str}")
                            cols[4].caption(f"{row['win_rate']:.0f}%")

                            if cols[5].button("Apply", key=f"apply_{idx}"):
                                st.session_state.selected_confluences = combo_set.copy()
                                st.rerun()
                else:
                    st.caption("Click 'Find Best Combinations' to search")

    # -------------------------------------------------------------------------
    # Expandable sections below the fold
    # -------------------------------------------------------------------------
    with st.expander("🔍 Trade List"):
        if len(filtered_trades) > 0:
            display_trades = filtered_trades.tail(25).copy()
            display_trades["time"] = display_trades["entry_time"].astype(str).str[11:16]
            display_trades["confluences"] = display_trades["confluence_records"].apply(
                lambda r: ", ".join(sorted(r)[:3]) + ("..." if len(r) > 3 else "")
            )
            display_trades["R"] = display_trades["r_multiple"].apply(lambda x: f"{x:+.2f}")

            st.dataframe(
                display_trades[["time", "entry_trigger", "exit_trigger", "R", "win", "confluences"]],
                use_container_width=True, hide_index=True, height=300
            )

    with st.expander("📊 R-Multiple Distribution"):
        if len(filtered_trades) > 0:
            fig = px.histogram(filtered_trades, x="r_multiple", nbins=25)
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔧 Raw Data"):
        st.dataframe(df.head(30), use_container_width=True)


if __name__ == "__main__":
    main()
