"""
RoR Trader - MVP Application
=============================

A Streamlit application for building and analyzing trading strategies
using the interpreter-based confluence system.

Run with: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lightweight_charts import renderLightweightCharts
import json
import os

from mock_data import generate_mock_bars, resample_bars
from interpreters import (
    INTERPRETERS,
    run_all_interpreters,
    get_confluence_records
)


# =============================================================================
# CONFIGURATION
# =============================================================================

AVAILABLE_SYMBOLS = ["SPY", "AAPL", "QQQ", "TSLA", "NVDA", "MSFT", "AMD", "META"]
TIMEFRAMES = ["1Min", "5Min", "15Min", "1Hour"]
DIRECTIONS = ["LONG", "SHORT"]

# Mock triggers (in real app, these would come from interpreter library)
ENTRY_TRIGGERS = [
    "UT Bot Alert",
    "MACD Cross",
    "EMA Cross",
    "Swing Break",
]

EXIT_TRIGGERS = [
    "Opposite Signal",
    "Fixed R Target (2R)",
    "Fixed R Target (3R)",
    "Trailing Stop",
    "Time Exit (50 bars)",
]

# Strategies storage path
STRATEGIES_FILE = "strategies.json"


# =============================================================================
# DATA & TRADE GENERATION
# =============================================================================

@st.cache_data(ttl=3600)
def load_market_data(symbol: str, days: int = 30, seed: int = 42):
    """Load market data for a symbol (cached)."""
    end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    bars = generate_mock_bars([symbol], start, end, "1Min", seed=seed)
    return bars.loc[symbol] if symbol in bars.index.get_level_values(0) else pd.DataFrame()


def generate_mock_trades(
    df: pd.DataFrame,
    direction: str,
    entry_trigger: str,
    exit_trigger: str,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate mock trades based on the data.

    In a real implementation, this would use the actual trigger logic.
    For MVP, we generate plausible trades at random intervals.
    """
    np.random.seed(seed + hash(entry_trigger) % 100)

    trades = []
    n_bars = len(df)

    # Generate entry points (roughly 2-5% of bars)
    entry_prob = 0.03
    i = 0

    while i < n_bars - 10:
        if np.random.random() < entry_prob:
            entry_idx = i
            entry_row = df.iloc[entry_idx]

            # Exit after 5-100 bars
            exit_offset = np.random.randint(5, min(100, n_bars - entry_idx - 1))
            exit_idx = entry_idx + exit_offset
            exit_row = df.iloc[exit_idx]

            # Calculate P&L
            entry_price = entry_row['close']
            exit_price = exit_row['close']

            if direction == "LONG":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            # Risk is based on entry bar's range
            risk = entry_row['high'] - entry_row['low']
            if risk < 0.01:
                risk = entry_price * 0.005  # Minimum 0.5% risk

            r_multiple = pnl / risk

            # Get confluence records at entry
            confluence = get_confluence_records(
                entry_row,
                "1M",
                list(INTERPRETERS.keys())
            )

            trades.append({
                'entry_time': entry_row.name,
                'exit_time': exit_row.name,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'risk': risk,
                'r_multiple': r_multiple,
                'win': pnl > 0,
                'entry_trigger': entry_trigger,
                'exit_trigger': exit_trigger,
                'confluence_records': confluence,
            })

            i = exit_idx + 1
        else:
            i += 1

    return pd.DataFrame(trades)


# =============================================================================
# KPI CALCULATIONS
# =============================================================================

def calculate_kpis(trades_df: pd.DataFrame, starting_balance: float = 10000, risk_per_trade: float = 100) -> dict:
    """Calculate strategy KPIs."""
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_r": 0, "daily_r": 0,
            "final_balance": starting_balance, "total_pnl": 0
        }

    wins = trades_df[trades_df["win"] == True]
    losses = trades_df[trades_df["win"] == False]

    gross_profit = wins["r_multiple"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["r_multiple"].sum()) if len(losses) > 0 else 0
    total_r = trades_df["r_multiple"].sum()

    # Trading days
    if "exit_time" in trades_df.columns:
        trading_days = trades_df["exit_time"].dt.date.nunique()
    else:
        trading_days = 1
    trading_days = max(trading_days, 1)

    # Dollar P&L
    total_pnl = total_r * risk_per_trade
    final_balance = starting_balance + total_pnl

    return {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        "avg_r": trades_df["r_multiple"].mean(),
        "total_r": total_r,
        "daily_r": total_r / trading_days,
        "final_balance": final_balance,
        "total_pnl": total_pnl,
    }


# =============================================================================
# CONFLUENCE ANALYSIS
# =============================================================================

def analyze_confluences(trades_df: pd.DataFrame, required: set = None, min_trades: int = 5) -> pd.DataFrame:
    """
    Analyze how different confluence conditions affect results.

    If required is provided, shows impact of adding additional confluences.
    Otherwise, shows impact of each single confluence.
    """
    if len(trades_df) == 0:
        return pd.DataFrame()

    # Get base trades (filtered by required confluences)
    if required and len(required) > 0:
        mask = trades_df["confluence_records"].apply(lambda r: required.issubset(r))
        base_trades = trades_df[mask]
    else:
        base_trades = trades_df

    if len(base_trades) < min_trades:
        return pd.DataFrame()

    base_kpis = calculate_kpis(base_trades)

    # Find all unique confluence records
    all_records = set()
    for records in base_trades["confluence_records"]:
        all_records.update(records)

    # Remove already-required records
    if required:
        all_records -= required

    results = []
    for record in all_records:
        # Filter to trades with this record
        mask = base_trades["confluence_records"].apply(lambda r: record in r)
        subset = base_trades[mask]

        if len(subset) >= min_trades:
            kpis = calculate_kpis(subset)

            results.append({
                "confluence": record,
                "total_trades": kpis["total_trades"],
                "win_rate": kpis["win_rate"],
                "profit_factor": kpis["profit_factor"],
                "avg_r": kpis["avg_r"],
                "total_r": kpis["total_r"],
                "daily_r": kpis["daily_r"],
                "pf_change": kpis["profit_factor"] - base_kpis["profit_factor"],
                "wr_change": kpis["win_rate"] - base_kpis["win_rate"],
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("profit_factor", ascending=False)

    return results_df


def find_best_combinations(trades_df: pd.DataFrame, max_depth: int = 3, min_trades: int = 5, top_n: int = 10) -> pd.DataFrame:
    """Find the best confluence combinations automatically."""
    if len(trades_df) == 0:
        return pd.DataFrame()

    # Get all unique records
    all_records = set()
    for records in trades_df["confluence_records"]:
        all_records.update(records)
    all_records = list(all_records)

    results = []

    for depth in range(1, min(max_depth + 1, len(all_records) + 1)):
        for combo in combinations(all_records, depth):
            combo_set = set(combo)

            mask = trades_df["confluence_records"].apply(lambda r: combo_set.issubset(r))
            subset = trades_df[mask]

            if len(subset) >= min_trades:
                kpis = calculate_kpis(subset)

                results.append({
                    "combination": combo_set,
                    "combo_str": " + ".join(sorted(combo_set)),
                    "depth": depth,
                    **kpis
                })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values(
            ["profit_factor", "total_trades"],
            ascending=[False, False]
        ).head(top_n)

    return results_df


# =============================================================================
# CHART RENDERING
# =============================================================================

def render_price_chart(df: pd.DataFrame, trades: pd.DataFrame, config: dict):
    """
    Render TradingView-style candlestick chart with trade markers.

    Args:
        df: DataFrame with OHLCV data (timestamp index)
        trades: DataFrame with trade entry/exit data
        config: Strategy config with 'direction' key
    """
    if len(df) == 0:
        st.info("No data available for chart")
        return

    # Prepare candlestick data
    candles = df.reset_index()

    # Handle both 'timestamp' and datetime index names
    time_col = candles.columns[0]  # First column after reset_index is the former index
    candles['time'] = pd.to_datetime(candles[time_col]).astype(int) // 10**9

    candle_data = candles[['time', 'open', 'high', 'low', 'close']].to_dict('records')

    # Create entry/exit markers from trades
    markers = []
    direction = config.get('direction', 'LONG')

    if len(trades) > 0:
        for _, trade in trades.iterrows():
            # Entry marker
            entry_time = int(pd.to_datetime(trade['entry_time']).timestamp())
            markers.append({
                'time': entry_time,
                'position': 'belowBar' if direction == 'LONG' else 'aboveBar',
                'color': '#2196F3',
                'shape': 'arrowUp' if direction == 'LONG' else 'arrowDown',
                'text': 'Entry'
            })

            # Exit marker
            exit_time = int(pd.to_datetime(trade['exit_time']).timestamp())
            is_win = trade.get('win', trade.get('pnl', 0) > 0)
            markers.append({
                'time': exit_time,
                'position': 'aboveBar' if direction == 'LONG' else 'belowBar',
                'color': '#4CAF50' if is_win else '#f44336',
                'shape': 'arrowDown' if direction == 'LONG' else 'arrowUp',
                'text': f"{trade['r_multiple']:+.1f}R"
            })

    # Chart configuration
    chart_options = {
        "layout": {
            "background": {"color": "#1E1E1E"},
            "textColor": "#DDD"
        },
        "grid": {
            "vertLines": {"color": "#2B2B2B"},
            "horzLines": {"color": "#2B2B2B"}
        },
        "crosshair": {
            "mode": 0
        },
        "timeScale": {
            "borderColor": "#2B2B2B",
            "timeVisible": True,
            "secondsVisible": False
        },
        "rightPriceScale": {
            "borderColor": "#2B2B2B"
        },
        "height": 500
    }

    # Candlestick series
    series = [{
        "type": "Candlestick",
        "data": candle_data,
        "options": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderUpColor": "#26a69a",
            "borderDownColor": "#ef5350",
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350"
        },
        "markers": markers
    }]

    # Render the chart
    renderLightweightCharts([{
        "chart": chart_options,
        "series": series
    }], key='price_chart')


# =============================================================================
# STRATEGY STORAGE
# =============================================================================

def load_strategies() -> list:
    """Load saved strategies from file."""
    if os.path.exists(STRATEGIES_FILE):
        with open(STRATEGIES_FILE, 'r') as f:
            return json.load(f)
    return []


def save_strategy(strategy: dict):
    """Save a strategy to file."""
    strategies = load_strategies()

    # Add timestamp and ID
    strategy['id'] = len(strategies) + 1
    strategy['created_at'] = datetime.now().isoformat()

    # Convert set to list for JSON serialization
    if 'confluence' in strategy and isinstance(strategy['confluence'], set):
        strategy['confluence'] = list(strategy['confluence'])

    strategies.append(strategy)

    with open(STRATEGIES_FILE, 'w') as f:
        json.dump(strategies, f, indent=2)


# =============================================================================
# STREAMLIT UI
# =============================================================================

# Custom CSS
CUSTOM_CSS = """
<style>
.step-indicator {
    display: flex;
    justify-content: center;
    margin-bottom: 1rem;
}
.step {
    padding: 0.5rem 1rem;
    margin: 0 0.5rem;
    border-radius: 20px;
    background: #f0f0f0;
}
.step.active {
    background: #2196F3;
    color: white;
}
.step.completed {
    background: #4CAF50;
    color: white;
}
</style>
"""


def main():
    st.set_page_config(
        page_title="RoR Trader",
        page_icon="📈",
        layout="wide"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'selected_confluences' not in st.session_state:
        st.session_state.selected_confluences = set()
    if 'strategy_config' not in st.session_state:
        st.session_state.strategy_config = {}

    # Sidebar navigation
    with st.sidebar:
        st.title("📈 RoR Trader")
        st.caption("Return on Risk Trader")

        st.divider()

        page = st.radio(
            "Navigation",
            ["Strategy Builder", "My Strategies", "Settings"],
            index=0
        )

        st.divider()

        # Data settings (always visible)
        st.subheader("Data Settings")
        data_days = st.slider("Historical Days", 7, 90, 30)
        data_seed = st.number_input("Data Seed", value=42, help="Change for different random data")

    # Main content
    if page == "Strategy Builder":
        render_strategy_builder(data_days, data_seed)
    elif page == "My Strategies":
        render_my_strategies()
    else:
        render_settings()


def render_strategy_builder(data_days: int, data_seed: int):
    """Render the 3-step strategy builder."""

    # Step indicator
    step = st.session_state.step
    steps = ["Setup", "Confluence", "Save"]

    cols = st.columns(5)
    cols[1].markdown(
        f"{'●' if step >= 1 else '○'} **Step 1: Setup**" +
        (" ✓" if step > 1 else ""),
        unsafe_allow_html=True
    )
    cols[2].markdown(
        f"{'●' if step >= 2 else '○'} **Step 2: Confluence**" +
        (" ✓" if step > 2 else "")
    )
    cols[3].markdown(
        f"{'●' if step >= 3 else '○'} **Step 3: Save**"
    )

    st.divider()

    # =========================================================================
    # STEP 1: SETUP
    # =========================================================================
    if step == 1:
        st.header("Step 1: Define Your Strategy")

        col1, col2 = st.columns(2)

        with col1:
            symbol = st.selectbox("Ticker", AVAILABLE_SYMBOLS, index=0)
            direction = st.radio("Direction", DIRECTIONS, horizontal=True)
            entry_trigger = st.selectbox("Entry Trigger", ENTRY_TRIGGERS)

        with col2:
            timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=0)
            st.write("")  # Spacer
            st.write("")
            exit_trigger = st.selectbox("Exit Trigger", EXIT_TRIGGERS)

        st.divider()

        if st.button("Next: Add Confluence →", type="primary", use_container_width=True):
            # Save config and move to step 2
            st.session_state.strategy_config = {
                'symbol': symbol,
                'direction': direction,
                'timeframe': timeframe,
                'entry_trigger': entry_trigger,
                'exit_trigger': exit_trigger,
            }
            st.session_state.step = 2
            st.session_state.selected_confluences = set()
            st.rerun()

    # =========================================================================
    # STEP 2: CONFLUENCE
    # =========================================================================
    elif step == 2:
        config = st.session_state.strategy_config

        # Header with strategy summary
        st.markdown(
            f"### {config['symbol']} | {config['direction']} | "
            f"{config['entry_trigger']} → {config['exit_trigger']}"
        )

        # Load data and run interpreters
        with st.spinner("Loading market data..."):
            df = load_market_data(config['symbol'], data_days, data_seed)

            if len(df) == 0:
                st.error("No data available")
                return

            # Run interpreters
            df = run_all_interpreters(df)

            # Generate mock trades
            trades = generate_mock_trades(
                df,
                config['direction'],
                config['entry_trigger'],
                config['exit_trigger'],
                seed=data_seed
            )

        # Apply confluence filter
        selected = st.session_state.selected_confluences
        if len(selected) > 0 and len(trades) > 0:
            mask = trades["confluence_records"].apply(lambda r: selected.issubset(r))
            filtered_trades = trades[mask]
        else:
            filtered_trades = trades

        # Active confluence tags
        if len(selected) > 0:
            st.caption("Active Confluence Filters:")
            tag_cols = st.columns(min(len(selected) + 1, 6))
            for i, conf in enumerate(sorted(selected)):
                with tag_cols[i % 5]:
                    if st.button(f"✕ {conf}", key=f"rm_{conf}"):
                        st.session_state.selected_confluences.discard(conf)
                        st.rerun()

            with tag_cols[-1]:
                if st.button("Clear All"):
                    st.session_state.selected_confluences = set()
                    st.rerun()
        else:
            st.caption("No confluence filters active. Add conditions below to refine your strategy.")

        # KPIs
        kpis = calculate_kpis(filtered_trades)

        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Trades", kpis["total_trades"])
        kpi_cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
        kpi_cols[2].metric("Profit Factor", f"{kpis['profit_factor']:.2f}" if kpis['profit_factor'] != float('inf') else "∞")
        kpi_cols[3].metric("Avg R", f"{kpis['avg_r']:+.2f}")
        kpi_cols[4].metric("Total R", f"{kpis['total_r']:+.1f}")
        kpi_cols[5].metric("Daily R", f"{kpis['daily_r']:+.2f}")

        # Main content: Chart/Equity (left) + Confluence panel (right)
        left_col, right_col = st.columns([1, 1])

        # -----------------------------------------------------------------
        # LEFT COLUMN: Tabbed view for Equity Curve / Price Chart
        # -----------------------------------------------------------------
        with left_col:
            chart_tab1, chart_tab2 = st.tabs(["📈 Equity Curve", "📊 Price Chart"])

            with chart_tab1:
                if len(filtered_trades) > 0:
                    equity_df = filtered_trades[["exit_time", "r_multiple"]].sort_values("exit_time")
                    equity_df["cumulative_r"] = equity_df["r_multiple"].cumsum()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df["exit_time"],
                        y=equity_df["cumulative_r"],
                        mode="lines",
                        name="Equity",
                        line=dict(color="#2196F3", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(33, 150, 243, 0.1)"
                    ))
                    fig.add_trace(go.Scatter(
                        x=equity_df["exit_time"],
                        y=equity_df["cumulative_r"].cummax(),
                        mode="lines",
                        name="High Water Mark",
                        line=dict(color="green", width=1, dash="dot")
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="",
                        yaxis_title="Cumulative R",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trades match current filters")

            with chart_tab2:
                if len(filtered_trades) > 0:
                    st.caption(f"{len(filtered_trades)} trades on {config['symbol']} ({config['direction']})")
                else:
                    st.caption(f"{config['symbol']} price data (no trades match filters)")

                # Render the TradingView-style chart
                render_price_chart(df, filtered_trades, config)

        # -----------------------------------------------------------------
        # RIGHT COLUMN: Confluence Drill-Down (always visible)
        # -----------------------------------------------------------------
        with right_col:
            st.subheader("🎯 Confluence Drill-Down")

            mode = st.radio("Mode", ["Drill-Down", "Auto-Search"], horizontal=True, label_visibility="collapsed")

            if mode == "Drill-Down":
                sort_by = st.selectbox("Sort by", ["Profit Factor", "Win Rate", "Daily R", "Trades"], index=0, label_visibility="collapsed")

                sort_map = {"Profit Factor": "profit_factor", "Win Rate": "win_rate", "Daily R": "daily_r", "Trades": "total_trades"}

                confluence_df = analyze_confluences(trades, selected, min_trades=3)

                if len(confluence_df) > 0:
                    confluence_df = confluence_df.sort_values(sort_map[sort_by], ascending=False).head(15)

                    # Display as interactive table
                    for _, row in confluence_df.iterrows():
                        conf = row['confluence']
                        is_selected = conf in selected

                        col1, col2, col3, col4, col5 = st.columns([0.5, 3, 1, 1, 1])

                        with col1:
                            if st.checkbox("", value=is_selected, key=f"sel_{conf}", label_visibility="collapsed"):
                                if not is_selected:
                                    st.session_state.selected_confluences.add(conf)
                                    st.rerun()
                            elif is_selected:
                                st.session_state.selected_confluences.discard(conf)
                                st.rerun()

                        with col2:
                            st.markdown(f"**{conf}**" if is_selected else conf)
                        with col3:
                            st.caption(f"{row['total_trades']} trades")
                        with col4:
                            pf = row['profit_factor']
                            st.caption(f"PF: {pf:.1f}" if pf != float('inf') else "PF: ∞")
                        with col5:
                            st.caption(f"WR: {row['win_rate']:.0f}%")
                else:
                    st.info("Not enough trades for analysis")

            else:  # Auto-Search
                search_cols = st.columns([1, 1, 2])
                with search_cols[0]:
                    max_depth = st.slider("Max factors", 1, 4, 2)
                with search_cols[1]:
                    min_trades = st.slider("Min trades", 1, 20, 5)

                if st.button("🔍 Find Best Combinations", type="primary"):
                    with st.spinner("Searching..."):
                        best = find_best_combinations(trades, max_depth, min_trades, top_n=10)

                    if len(best) > 0:
                        st.session_state.auto_results = best

                if 'auto_results' in st.session_state and len(st.session_state.auto_results) > 0:
                    for _, row in st.session_state.auto_results.iterrows():
                        col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1])

                        with col1:
                            st.caption(f"{row['depth']}")
                        with col2:
                            st.markdown(f"**{row['combo_str']}**")
                        with col3:
                            pf = row['profit_factor']
                            st.caption(f"PF: {pf:.1f}" if pf != float('inf') else "∞")
                        with col4:
                            if st.button("Apply", key=f"apply_{row['combo_str']}"):
                                st.session_state.selected_confluences = row['combination'].copy()
                                st.rerun()

        # Trade list (expandable) - full width below columns
        with st.expander("📋 Trade List"):
            if len(filtered_trades) > 0:
                display = filtered_trades.tail(20).copy()
                display['time'] = display['entry_time'].dt.strftime('%m/%d %H:%M')
                display['R'] = display['r_multiple'].apply(lambda x: f"{x:+.2f}")
                display['result'] = display['win'].apply(lambda x: "✓" if x else "✗")
                display['confluences'] = display['confluence_records'].apply(
                    lambda r: ", ".join(sorted(r)[:3]) + ("..." if len(r) > 3 else "")
                )

                st.dataframe(
                    display[['time', 'entry_trigger', 'exit_trigger', 'R', 'result', 'confluences']],
                    use_container_width=True,
                    hide_index=True
                )

        # Navigation
        st.divider()
        nav_cols = st.columns([1, 3, 1])

        with nav_cols[0]:
            if st.button("← Back to Setup"):
                st.session_state.step = 1
                st.rerun()

        with nav_cols[2]:
            if st.button("Save Strategy →", type="primary"):
                st.session_state.step = 3
                st.session_state.final_kpis = kpis
                st.rerun()

    # =========================================================================
    # STEP 3: SAVE
    # =========================================================================
    elif step == 3:
        config = st.session_state.strategy_config
        selected = st.session_state.selected_confluences
        kpis = st.session_state.get('final_kpis', {})

        st.header("Step 3: Save Your Strategy")

        # Strategy name
        default_name = f"{config['symbol']} {config['direction']} - {config['entry_trigger']}"
        strategy_name = st.text_input("Strategy Name", value=default_name)

        # Summary
        st.subheader("Strategy Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Configuration:**
            - Ticker: {config['symbol']}
            - Direction: {config['direction']}
            - Timeframe: {config['timeframe']}
            - Entry: {config['entry_trigger']}
            - Exit: {config['exit_trigger']}
            """)

        with col2:
            st.markdown("**Confluence Conditions:**")
            if len(selected) > 0:
                for conf in sorted(selected):
                    st.markdown(f"- {conf}")
            else:
                st.markdown("- *No confluence filters*")

        # KPIs
        st.subheader("Backtest Results")
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Trades", kpis.get("total_trades", 0))
        kpi_cols[1].metric("Win Rate", f"{kpis.get('win_rate', 0):.1f}%")
        kpi_cols[2].metric("Profit Factor", f"{kpis.get('profit_factor', 0):.2f}")
        kpi_cols[3].metric("Total R", f"{kpis.get('total_r', 0):+.1f}")
        kpi_cols[4].metric("Daily R", f"{kpis.get('daily_r', 0):+.2f}")

        # Options
        st.subheader("Options")
        enable_forward = st.checkbox("Enable Forward Testing", value=True)
        enable_alerts = st.checkbox("Enable Alerts", value=False)

        # Save
        st.divider()
        nav_cols = st.columns([1, 3, 1])

        with nav_cols[0]:
            if st.button("← Back"):
                st.session_state.step = 2
                st.rerun()

        with nav_cols[2]:
            if st.button("💾 Save Strategy", type="primary"):
                strategy = {
                    'name': strategy_name,
                    **config,
                    'confluence': list(selected),
                    'kpis': kpis,
                    'forward_testing': enable_forward,
                    'alerts': enable_alerts,
                }
                save_strategy(strategy)
                st.success(f"Strategy '{strategy_name}' saved!")
                st.balloons()

                # Reset for new strategy
                st.session_state.step = 1
                st.session_state.selected_confluences = set()
                st.session_state.strategy_config = {}


def render_my_strategies():
    """Render the My Strategies page."""
    st.header("📁 My Strategies")

    strategies = load_strategies()

    if len(strategies) == 0:
        st.info("No strategies saved yet. Create one in the Strategy Builder!")
        return

    for strat in strategies:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"### {strat['name']}")
                st.caption(f"{strat['symbol']} | {strat['direction']} | {strat['entry_trigger']} → {strat['exit_trigger']}")

            with col2:
                kpis = strat.get('kpis', {})
                st.metric("Win Rate", f"{kpis.get('win_rate', 0):.1f}%")

            with col3:
                st.metric("Profit Factor", f"{kpis.get('profit_factor', 0):.2f}")

            # Confluence
            confluence = strat.get('confluence', [])
            if len(confluence) > 0:
                st.caption(f"Confluence: {', '.join(confluence[:3])}" + ("..." if len(confluence) > 3 else ""))

            # Status
            status = "🟢 Forward Testing" if strat.get('forward_testing') else "⚪ Backtest Only"
            st.caption(status)


def render_settings():
    """Render the Settings page."""
    st.header("⚙️ Settings")

    st.subheader("Interpreter Library")
    st.caption("Enable interpreters to use their outputs as confluence conditions.")

    for key, config in INTERPRETERS.items():
        with st.container(border=True):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.checkbox("", value=True, key=f"interp_{key}", label_visibility="collapsed")
            with col2:
                st.markdown(f"**{config.name}**")
                st.caption(config.description)
                st.caption(f"Outputs: {', '.join(config.outputs)}")

    st.divider()

    st.subheader("Data Connection")
    st.info("Alpaca API connection not configured. Using mock data.")

    if st.button("Configure Alpaca API"):
        st.warning("Create a .env file with ALPACA_API_KEY and ALPACA_SECRET_KEY")


if __name__ == "__main__":
    main()
