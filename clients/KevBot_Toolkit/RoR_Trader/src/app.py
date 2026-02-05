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

from data_loader import load_market_data, get_data_source, is_alpaca_configured
from indicators import (
    INDICATORS,
    run_all_indicators,
    get_indicator_overlay_config,
    get_available_overlay_indicators,
    INDICATOR_COLORS
)
from interpreters import (
    INTERPRETERS,
    run_all_interpreters,
    detect_all_triggers,
    get_confluence_records,
    get_available_triggers
)
from triggers import (
    generate_trades,
    EXIT_TYPES,
)
from confluence_groups import (
    load_confluence_groups,
    save_confluence_groups,
    get_enabled_groups,
    get_group_by_id,
    get_group_triggers,
    get_all_triggers,
    get_entry_triggers as get_confluence_entry_triggers,
    get_exit_triggers as get_confluence_exit_triggers,
    duplicate_group,
    generate_unique_id,
    validate_group_id,
    TEMPLATES,
    get_template,
    get_parameter_schema,
    get_plot_schema,
    get_output_descriptions,
    ConfluenceGroup,
    PlotSettings,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

AVAILABLE_SYMBOLS = ["SPY", "AAPL", "QQQ", "TSLA", "NVDA", "MSFT", "AMD", "META"]
TIMEFRAMES = ["1Min", "5Min", "15Min", "1Hour"]
DIRECTIONS = ["LONG", "SHORT"]

# Strategies storage path
STRATEGIES_FILE = "strategies.json"


# =============================================================================
# TRIGGER MAPPING
# =============================================================================

def get_base_trigger_id(confluence_trigger_id: str) -> str:
    """
    Map a confluence group trigger ID to the base trigger ID used in DataFrame columns.

    Confluence group trigger IDs are formatted as: {group_id}_{base_trigger}
    e.g., "ema_stack_default_cross_bull" -> "ema_cross_bull"

    The base trigger IDs match the columns created by detect_all_triggers().
    Uses trigger_prefix from TEMPLATES to build the mapping dynamically.
    """
    all_triggers = get_all_triggers()

    if confluence_trigger_id in all_triggers:
        trigger_def = all_triggers[confluence_trigger_id]
        base_trigger = trigger_def.base_trigger

        # Find the group that owns this trigger to get its template's prefix
        enabled_groups = get_enabled_groups()
        for group in enabled_groups:
            if group.get_trigger_id(base_trigger) == confluence_trigger_id:
                template = TEMPLATES.get(group.base_template)
                if template and "trigger_prefix" in template:
                    return f"{template['trigger_prefix']}_{base_trigger}"
                break

    # Fallback: return as-is (might be a direct trigger ID)
    return confluence_trigger_id


# =============================================================================
# CONFLUENCE RECORD FORMATTING
# =============================================================================

def build_interpreter_to_template() -> dict:
    """Build mapping from interpreter keys to template IDs from TEMPLATES data."""
    mapping = {}
    for template_id, template in TEMPLATES.items():
        for interp_key in template.get("interpreters", []):
            mapping[interp_key] = template_id
    return mapping


INTERPRETER_TO_TEMPLATE = build_interpreter_to_template()


def format_confluence_record(record: str, groups: list = None) -> str:
    """
    Format a confluence record with the confluence group version name.

    Confluence records are formatted as: "{timeframe}-{INTERPRETER}-{state}"
    e.g., "1M-EMA_STACK-SML"

    Returns formatted string like: "EMA Stack (Default): SML"
    """
    if groups is None:
        groups = get_enabled_groups()

    parts = record.split("-")
    if len(parts) < 3:
        return record  # Can't parse, return as-is

    timeframe = parts[0]
    interpreter = parts[1]
    state = "-".join(parts[2:])  # Handle states that might contain dashes

    # Find the template ID for this interpreter
    template_id = INTERPRETER_TO_TEMPLATE.get(interpreter)
    if not template_id:
        return record  # Unknown interpreter

    # Find the first enabled group with this template
    for group in groups:
        if group.base_template == template_id:
            return f"{group.name}: {state}"

    # Fallback: just clean up the interpreter name
    interpreter_name = interpreter.replace("_", " ").title()
    return f"{interpreter_name}: {state}"


def format_confluence_set(records: set, groups: list = None) -> str:
    """Format a set of confluence records into a readable string."""
    if groups is None:
        groups = get_enabled_groups()

    formatted = [format_confluence_record(r, groups) for r in sorted(records)]
    return " + ".join(formatted)


def get_overlay_indicators_for_group(group: ConfluenceGroup) -> list:
    """
    Get the indicator column names for a confluence group.

    Returns list of column names that should be displayed for this group.
    """
    template = get_template(group.base_template)
    if not template:
        return []

    # Return the indicator columns defined in the template
    return template.get("indicator_columns", [])


def get_overlay_colors_for_group(group: ConfluenceGroup) -> dict:
    """
    Get the colors for a confluence group's indicators.

    Returns dict mapping column_name -> color
    """
    template = get_template(group.base_template)
    if not template:
        return {}

    # Map template indicator columns to their colors from plot_settings
    colors = {}
    indicator_cols = template.get("indicator_columns", [])
    plot_schema = template.get("plot_schema", {})

    # Build color mapping based on template type
    if group.base_template == "ema_stack":
        colors["ema_short"] = group.plot_settings.colors.get("short_color", "#22c55e")
        colors["ema_mid"] = group.plot_settings.colors.get("mid_color", "#eab308")
        colors["ema_long"] = group.plot_settings.colors.get("long_color", "#ef4444")
    elif group.base_template == "vwap":
        colors["vwap"] = group.plot_settings.colors.get("vwap_color", "#8b5cf6")
        colors["vwap_upper"] = group.plot_settings.colors.get("band_color", "#c4b5fd")
        colors["vwap_lower"] = group.plot_settings.colors.get("band_color", "#c4b5fd")
    elif group.base_template == "macd":
        colors["macd_line"] = group.plot_settings.colors.get("macd_color", "#2563eb")
        colors["macd_signal"] = group.plot_settings.colors.get("signal_color", "#f97316")
    elif group.base_template == "utbot":
        colors["utbot_stop"] = group.plot_settings.colors.get("trail_color", "#64748b")

    return colors


# =============================================================================
# DATA & TRADE GENERATION
# =============================================================================

@st.cache_data(ttl=3600)
def prepare_data_with_indicators(symbol: str, days: int = 30, seed: int = 42):
    """
    Load market data and run all indicators, interpreters, and trigger detection.

    Uses Alpaca API if configured, otherwise falls back to mock data.

    Returns DataFrame ready for trade generation and analysis.
    """
    # Load raw bars (Alpaca if configured, mock otherwise)
    df = load_market_data(symbol, days=days, seed=seed)

    if len(df) == 0:
        return df

    # Run indicators
    df = run_all_indicators(df)

    # Run interpreters
    df = run_all_interpreters(df)

    # Detect triggers
    df = detect_all_triggers(df)

    return df


# =============================================================================
# KPI CALCULATIONS
# =============================================================================

def safe_subtract(a: float, b: float) -> float:
    """Safely subtract two values that may be infinity.
    inf - inf = 0 (both infinite, no meaningful difference).
    """
    if a == float('inf') and b == float('inf'):
        return 0.0
    if a == float('-inf') and b == float('-inf'):
        return 0.0
    return a - b

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

def analyze_confluences(trades_df: pd.DataFrame, required: set = None, min_trades: int = 5,
                        starting_balance: float = 10000, risk_per_trade: float = 100) -> pd.DataFrame:
    """
    Analyze how different confluence conditions affect results.

    If required is provided, shows impact of adding additional confluences.
    Otherwise, shows impact of each single confluence.
    """
    if len(trades_df) == 0:
        return pd.DataFrame()

    # Get base trades (filtered by required confluences)
    if required and len(required) > 0:
        mask = trades_df["confluence_records"].apply(lambda r: isinstance(r, set) and required.issubset(r))
        base_trades = trades_df[mask]
    else:
        base_trades = trades_df

    if len(base_trades) < min_trades:
        return pd.DataFrame()

    base_kpis = calculate_kpis(base_trades, starting_balance=starting_balance, risk_per_trade=risk_per_trade)

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
        mask = base_trades["confluence_records"].apply(lambda r: isinstance(r, set) and record in r)
        subset = base_trades[mask]

        if len(subset) >= min_trades:
            kpis = calculate_kpis(subset, starting_balance=starting_balance, risk_per_trade=risk_per_trade)

            results.append({
                "confluence": record,
                "total_trades": kpis["total_trades"],
                "win_rate": kpis["win_rate"],
                "profit_factor": kpis["profit_factor"],
                "avg_r": kpis["avg_r"],
                "total_r": kpis["total_r"],
                "daily_r": kpis["daily_r"],
                "pf_change": safe_subtract(kpis["profit_factor"], base_kpis["profit_factor"]),
                "wr_change": kpis["win_rate"] - base_kpis["win_rate"],
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("profit_factor", ascending=False, na_position="last")

    return results_df


def find_best_combinations(trades_df: pd.DataFrame, max_depth: int = 3, min_trades: int = 5, top_n: int = 10,
                           starting_balance: float = 10000, risk_per_trade: float = 100) -> pd.DataFrame:
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

            mask = trades_df["confluence_records"].apply(lambda r: isinstance(r, set) and combo_set.issubset(r))
            subset = trades_df[mask]

            if len(subset) >= min_trades:
                kpis = calculate_kpis(subset, starting_balance=starting_balance, risk_per_trade=risk_per_trade)

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
            ascending=[False, False],
            na_position="last"
        ).head(top_n)

    return results_df


# =============================================================================
# CHART RENDERING
# =============================================================================

def render_price_chart(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    config: dict,
    show_indicators: list = None,
    indicator_colors: dict = None
):
    """
    Render TradingView-style candlestick chart with trade markers and indicator overlays.

    Args:
        df: DataFrame with OHLCV data and indicator columns (timestamp index)
        trades: DataFrame with trade entry/exit data
        config: Strategy config with 'direction' key
        show_indicators: List of indicator column names to overlay (e.g., ['ema_short', 'ema_mid'])
        indicator_colors: Dict mapping column names to colors (from confluence group settings)
    """
    if len(df) == 0:
        st.info("No data available for chart")
        return

    # Prepare candlestick data
    time_col = df.index.name if df.index.name else 'timestamp'
    candles = df.reset_index()

    # Verify the expected column exists; fall back to positional if not
    if time_col not in candles.columns:
        time_col = candles.columns[0]
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
        "height": 450
    }

    # Candlestick series with markers
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

    # Add indicator overlays
    if show_indicators:
        for ind_id in show_indicators:
            if ind_id in candles.columns:
                # Prepare indicator data
                ind_data = []
                for _, row in candles.iterrows():
                    if pd.notna(row.get(ind_id)):
                        ind_data.append({
                            "time": int(row['time']),
                            "value": float(row[ind_id])
                        })

                if ind_data:
                    # Get color for this indicator (prefer confluence group colors, fallback to defaults)
                    if indicator_colors and ind_id in indicator_colors:
                        color = indicator_colors[ind_id]
                    else:
                        color = INDICATOR_COLORS.get(ind_id, "#FFFFFF")

                    series.append({
                        "type": "Line",
                        "data": ind_data,
                        "options": {
                            "color": color,
                            "lineWidth": 2,
                            "priceLineVisible": False,
                            "crosshairMarkerVisible": True,
                            "title": ind_id.upper().replace("_", " ")
                        }
                    })

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

        # Data source indicator
        data_source = get_data_source()
        if is_alpaca_configured():
            st.success(f"📡 {data_source}")
        else:
            st.warning(f"🎲 {data_source}")

        st.divider()

        page = st.radio(
            "Navigation",
            ["Strategy Builder", "My Strategies", "Confluence Groups"],
            index=0
        )

        st.divider()

        # Data settings (always visible)
        st.subheader("Data Settings")
        data_days = st.slider("Historical Days", 7, 90, 30)
        if not is_alpaca_configured():
            data_seed = st.number_input("Data Seed", value=42, help="Change for different random data")
        else:
            data_seed = 42  # Not used with real data

    # Main content
    if page == "Strategy Builder":
        render_strategy_builder(data_days, data_seed)
    elif page == "My Strategies":
        render_my_strategies()
    else:
        render_confluence_groups()


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

            # Get entry triggers from enabled confluence groups
            # Returns dict mapping confluence_trigger_id -> display_name
            # e.g., "ema_stack_default_cross_bull" -> "EMA Stack (Default): Short > Mid Cross"
            enabled_groups = get_enabled_groups()
            entry_triggers = get_confluence_entry_triggers(direction, enabled_groups)

            if len(entry_triggers) == 0:
                st.warning("No entry triggers available. Enable confluence groups in the Confluence Groups page.")
                entry_trigger = None
                entry_trigger_name = None
            else:
                entry_trigger_options = list(entry_triggers.keys())
                entry_trigger_labels = list(entry_triggers.values())
                entry_trigger_idx = st.selectbox(
                    "Entry Trigger",
                    range(len(entry_trigger_options)),
                    format_func=lambda i: entry_trigger_labels[i],
                    help="Triggers from your enabled Confluence Groups"
                )
                entry_trigger = entry_trigger_options[entry_trigger_idx]
                entry_trigger_name = entry_triggers[entry_trigger]

        with col2:
            timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=0)
            st.write("")  # Spacer

            # Exit triggers use the same confluence group triggers
            # Get ALL triggers (not filtered by direction) - user decides how to use them
            all_triggers = get_all_triggers(enabled_groups)
            exit_triggers = {tid: tdef.name for tid, tdef in all_triggers.items()}

            if len(exit_triggers) == 0:
                st.warning("No exit triggers available. Enable confluence groups in the Confluence Groups page.")
                exit_trigger = None
                exit_trigger_name = None
            else:
                exit_trigger_options = list(exit_triggers.keys())
                exit_trigger_labels = list(exit_triggers.values())
                exit_trigger_idx = st.selectbox(
                    "Exit Trigger",
                    range(len(exit_trigger_options)),
                    format_func=lambda i: exit_trigger_labels[i],
                    help="Triggers from your enabled Confluence Groups"
                )
                exit_trigger = exit_trigger_options[exit_trigger_idx]
                exit_trigger_name = exit_triggers[exit_trigger]

        # Risk Settings
        st.divider()
        st.subheader("Risk Settings")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            risk_per_trade = st.number_input(
                "Risk Per Trade ($)",
                min_value=1.0,
                max_value=10000.0,
                value=100.0,
                step=25.0,
                help="Dollar amount risked per trade"
            )
        with risk_col2:
            stop_atr_mult = st.number_input(
                "Stop Loss (ATR x)",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Stop loss distance as a multiple of ATR"
            )
        with risk_col3:
            starting_balance = st.number_input(
                "Starting Balance ($)",
                min_value=100.0,
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                help="Starting account balance for P&L calculations"
            )

        st.divider()

        same_trigger = entry_trigger is not None and entry_trigger == exit_trigger
        if same_trigger:
            st.warning("Entry and exit triggers cannot be the same.")
        can_proceed = entry_trigger is not None and exit_trigger is not None and not same_trigger
        if st.button("Next: Add Confluence →", type="primary", use_container_width=True, disabled=not can_proceed):
            # Save config and move to step 2
            # Map the confluence trigger IDs to the base trigger IDs for trade generation
            base_entry_trigger_id = get_base_trigger_id(entry_trigger)
            base_exit_trigger_id = get_base_trigger_id(exit_trigger)

            st.session_state.strategy_config = {
                'symbol': symbol,
                'direction': direction,
                'timeframe': timeframe,
                'entry_trigger': base_entry_trigger_id,  # Base trigger ID for trade generation
                'entry_trigger_confluence_id': entry_trigger,  # Full confluence ID for display
                'exit_trigger': base_exit_trigger_id,  # Base trigger ID for trade generation
                'exit_trigger_confluence_id': exit_trigger,  # Full confluence ID for display
                'entry_trigger_name': entry_trigger_name,
                'exit_trigger_name': exit_trigger_name,
                'risk_per_trade': risk_per_trade,
                'stop_atr_mult': stop_atr_mult,
                'starting_balance': starting_balance,
                'data_days': data_days,
                'data_seed': data_seed,
            }
            st.session_state.step = 2
            st.session_state.selected_confluences = set()
            st.rerun()

    # =========================================================================
    # STEP 2: CONFLUENCE
    # =========================================================================
    elif step == 2:
        config = st.session_state.strategy_config

        # Load enabled confluence groups for formatting and overlays
        enabled_groups = get_enabled_groups()

        # Header with strategy summary
        entry_name = config.get('entry_trigger_name', config['entry_trigger'])
        exit_name = config.get('exit_trigger_name', config['exit_trigger'])
        st.markdown(
            f"### {config['symbol']} | {config['direction']} | "
            f"{entry_name} → {exit_name}"
        )

        # Load data with indicators, interpreters, and triggers
        with st.spinner("Loading market data and running analysis..."):
            df = prepare_data_with_indicators(config['symbol'], data_days, data_seed)

            if len(df) == 0:
                st.error("No data available")
                return

            # Generate trades using real trigger logic
            trades = generate_trades(
                df,
                direction=config['direction'],
                entry_trigger=config['entry_trigger'],
                exit_trigger=config['exit_trigger'],
                confluence_required=None,  # Will filter after generation
                risk_per_trade=config.get('risk_per_trade', 100.0),
                stop_atr_mult=config.get('stop_atr_mult', 1.5),
            )

        # Apply confluence filter
        selected = st.session_state.selected_confluences
        if len(selected) > 0 and len(trades) > 0:
            mask = trades["confluence_records"].apply(lambda r: isinstance(r, set) and selected.issubset(r))
            filtered_trades = trades[mask]
        else:
            filtered_trades = trades

        # Active confluence tags
        if len(selected) > 0:
            st.caption("Active Confluence Filters:")
            tag_cols = st.columns(min(len(selected) + 1, 6))
            for i, conf in enumerate(sorted(selected)):
                conf_display = format_confluence_record(conf, enabled_groups)
                with tag_cols[i % 5]:
                    if st.button(f"✕ {conf_display}", key=f"rm_{conf}"):
                        st.session_state.selected_confluences.discard(conf)
                        st.rerun()

            with tag_cols[-1]:
                if st.button("Clear All"):
                    st.session_state.selected_confluences = set()
                    st.rerun()
        else:
            st.caption("No confluence filters active. Add conditions below to refine your strategy.")

        # KPIs
        kpis = calculate_kpis(
            filtered_trades,
            starting_balance=config.get('starting_balance', 10000.0),
            risk_per_trade=config.get('risk_per_trade', 100.0),
        )

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
                # Confluence group selector for indicator overlays
                # Only show groups that have chart overlays (not MACD which is typically a separate pane)
                overlay_groups = [g for g in enabled_groups if g.base_template in ["ema_stack", "vwap", "utbot"]]

                if len(overlay_groups) > 0:
                    selected_overlay_groups = st.multiselect(
                        "Show Indicators",
                        options=[g.id for g in overlay_groups],
                        default=[overlay_groups[0].id] if len(overlay_groups) > 0 else [],
                        format_func=lambda gid: next((g.name for g in overlay_groups if g.id == gid), gid),
                        help="Select confluence groups to overlay on chart"
                    )

                    # Collect indicator columns and colors from selected groups
                    show_indicators = []
                    indicator_colors = {}
                    for gid in selected_overlay_groups:
                        group = get_group_by_id(gid, enabled_groups)
                        if group:
                            cols = get_overlay_indicators_for_group(group)
                            show_indicators.extend(cols)
                            indicator_colors.update(get_overlay_colors_for_group(group))
                else:
                    st.info("No overlay-compatible confluence groups enabled")
                    show_indicators = []
                    indicator_colors = {}

                if len(filtered_trades) > 0:
                    st.caption(f"{len(filtered_trades)} trades on {config['symbol']} ({config['direction']})")
                else:
                    st.caption(f"{config['symbol']} price data (no trades match filters)")

                # Render the TradingView-style chart with indicator overlays
                render_price_chart(df, filtered_trades, config, show_indicators=show_indicators, indicator_colors=indicator_colors)

        # -----------------------------------------------------------------
        # RIGHT COLUMN: Confluence Drill-Down (always visible)
        # -----------------------------------------------------------------
        with right_col:
            st.subheader("🎯 Confluence Drill-Down")

            mode = st.radio("Mode", ["Drill-Down", "Auto-Search"], horizontal=True, label_visibility="collapsed")

            if mode == "Drill-Down":
                sort_by = st.selectbox("Sort by", ["Profit Factor", "Win Rate", "Daily R", "Trades"], index=0, label_visibility="collapsed")

                sort_map = {"Profit Factor": "profit_factor", "Win Rate": "win_rate", "Daily R": "daily_r", "Trades": "total_trades"}

                confluence_df = analyze_confluences(
                    trades, selected, min_trades=3,
                    starting_balance=config.get('starting_balance', 10000.0),
                    risk_per_trade=config.get('risk_per_trade', 100.0),
                )

                if len(confluence_df) > 0:
                    confluence_df = confluence_df.sort_values(sort_map[sort_by], ascending=False, na_position="last").head(15)

                    # Display as interactive table
                    for _, row in confluence_df.iterrows():
                        conf = row['confluence']  # Raw record for selection logic
                        conf_display = format_confluence_record(conf, enabled_groups)  # Formatted for display
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
                            st.markdown(f"**{conf_display}**" if is_selected else conf_display)
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
                        best = find_best_combinations(
                            trades, max_depth, min_trades, top_n=10,
                            starting_balance=config.get('starting_balance', 10000.0),
                            risk_per_trade=config.get('risk_per_trade', 100.0),
                        )

                    if len(best) > 0:
                        st.session_state.auto_results = best

                if 'auto_results' in st.session_state and len(st.session_state.auto_results) > 0:
                    for _, row in st.session_state.auto_results.iterrows():
                        col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1])

                        # Format the combination for display
                        combo_display = format_confluence_set(row['combination'], enabled_groups)

                        with col1:
                            st.caption(f"{row['depth']}")
                        with col2:
                            st.markdown(f"**{combo_display}**")
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
                # Format confluence records with group version names
                display['confluences'] = display['confluence_records'].apply(
                    lambda r: ", ".join([format_confluence_record(rec, enabled_groups) for rec in sorted(r)[:3]]) + ("..." if len(r) > 3 else "")
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

        # Use the display name for default strategy name
        entry_display_name = config.get('entry_trigger_name', config['entry_trigger'])
        default_name = f"{config['symbol']} {config['direction']} - {entry_display_name}"
        strategy_name = st.text_input("Strategy Name", value=default_name)

        # Summary
        st.subheader("Strategy Summary")

        col1, col2 = st.columns(2)

        # Get display names
        exit_display_name = config.get('exit_trigger_name', config['exit_trigger'])

        with col1:
            st.markdown(f"""
            **Configuration:**
            - Ticker: {config['symbol']}
            - Direction: {config['direction']}
            - Timeframe: {config['timeframe']}
            - Entry: {entry_display_name}
            - Exit: {exit_display_name}
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


def render_confluence_groups():
    """Render the Confluence Groups management page."""
    st.header("Confluence Groups")
    st.caption("Configure indicators, interpreters, and triggers for your analysis.")

    # Initialize session state for editing
    if 'editing_group' not in st.session_state:
        st.session_state.editing_group = None
    if 'show_new_group' not in st.session_state:
        st.session_state.show_new_group = False

    # Load groups
    groups = load_confluence_groups()

    # Top action bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**{len(groups)} confluence groups** ({len(get_enabled_groups(groups))} enabled)")
    with col3:
        if st.button("+ New Group", use_container_width=True):
            st.session_state.show_new_group = True
            st.session_state.editing_group = None

    st.divider()

    # New group creation dialog
    if st.session_state.show_new_group:
        render_new_group_dialog(groups)
        st.divider()

    # Group details editing
    if st.session_state.editing_group:
        render_group_details(st.session_state.editing_group, groups)
        st.divider()

    # Group list organized by category
    categories = {}
    for group in groups:
        template = get_template(group.base_template)
        category = template["category"] if template else "Other"
        if category not in categories:
            categories[category] = []
        categories[category].append(group)

    for category, cat_groups in categories.items():
        st.subheader(category)

        for group in cat_groups:
            render_group_card(group, groups)

        st.markdown("")  # Spacing


def render_group_card(group: ConfluenceGroup, all_groups: list):
    """Render a single confluence group card."""
    template = get_template(group.base_template)

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([0.08, 0.52, 0.25, 0.15])

        # Enable/disable checkbox
        with col1:
            enabled = st.checkbox(
                "",
                value=group.enabled,
                key=f"enable_{group.id}",
                label_visibility="collapsed"
            )
            if enabled != group.enabled:
                group.enabled = enabled
                save_confluence_groups(all_groups)
                st.rerun()

        # Group info
        with col2:
            default_badge = " (default)" if group.is_default else ""
            st.markdown(f"**{group.name}**{default_badge}")

            # Show key parameters
            param_str = format_parameters(group.parameters, group.base_template)
            st.caption(param_str)

        # Outputs preview
        with col3:
            if template:
                outputs = template.get("outputs", [])
                st.caption(f"Outputs: {', '.join(outputs[:4])}" + ("..." if len(outputs) > 4 else ""))

        # Actions
        with col4:
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Details", key=f"details_{group.id}", use_container_width=True):
                    st.session_state.editing_group = group.id
                    st.session_state.show_new_group = False
                    st.rerun()
            with action_cols[1]:
                if st.button("Copy", key=f"copy_{group.id}", use_container_width=True):
                    new_id = generate_unique_id(group.base_template, all_groups)
                    new_group = duplicate_group(group, new_id, f"{group.version} Copy")
                    all_groups.append(new_group)
                    save_confluence_groups(all_groups)
                    st.session_state.editing_group = new_id
                    st.rerun()


def render_new_group_dialog(all_groups: list):
    """Render the new group creation dialog."""
    st.subheader("Create New Confluence Group")

    col1, col2 = st.columns(2)

    with col1:
        # Template selection
        template_options = list(TEMPLATES.keys())
        template_labels = [TEMPLATES[t]["name"] for t in template_options]
        template_idx = st.selectbox(
            "Base Template",
            range(len(template_options)),
            format_func=lambda i: template_labels[i]
        )
        selected_template = template_options[template_idx]

        template = TEMPLATES[selected_template]
        st.caption(template["description"])

    with col2:
        # Version name input (the part in parentheses)
        new_version = st.text_input("Version Name", value="Custom", help="e.g., 'Scalping', 'Aggressive', 'Custom'")
        st.caption(f"Full name will be: **{template['name']} ({new_version})**")

        # ID input (auto-generated but editable)
        suggested_id = generate_unique_id(selected_template, all_groups)
        new_id = st.text_input("Group ID", value=suggested_id, help="Unique identifier (lowercase, no spaces)")

    # Validation
    id_valid = validate_group_id(new_id, all_groups)
    version_valid = len(new_version.strip()) > 0

    if not id_valid:
        st.warning("ID must be unique and contain only letters, numbers, and underscores.")
    if not version_valid:
        st.warning("Version name cannot be empty.")

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Create", disabled=not (id_valid and version_valid), use_container_width=True):
            # Create new group with default parameters from template
            param_schema = get_parameter_schema(selected_template)
            default_params = {k: v["default"] for k, v in param_schema.items()}

            plot_schema = get_plot_schema(selected_template)
            default_colors = {k: v["default"] for k, v in plot_schema.items() if v["type"] == "color"}

            new_group = ConfluenceGroup(
                id=new_id,
                base_template=selected_template,
                version=new_version,
                description=f"Custom {template['name']} configuration",
                enabled=True,
                is_default=False,
                parameters=default_params,
                plot_settings=PlotSettings(colors=default_colors, line_width=1, visible=True),
            )

            all_groups.append(new_group)
            save_confluence_groups(all_groups)
            st.session_state.show_new_group = False
            st.session_state.editing_group = new_id
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_new_group = False
            st.rerun()


def render_group_details(group_id: str, all_groups: list):
    """Render the detailed view/edit panel for a confluence group."""
    group = get_group_by_id(group_id, all_groups)
    if not group:
        st.error(f"Group not found: {group_id}")
        return

    template = get_template(group.base_template)
    if not template:
        st.error(f"Template not found: {group.base_template}")
        return

    # Header with close button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"Edit: {group.name}")
        st.caption(f"Based on: {template['name']} | ID: {group.id}")
    with col2:
        if st.button("Close", use_container_width=True):
            st.session_state.editing_group = None
            st.rerun()

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Parameters", "Plot Settings", "Outputs & Triggers", "Danger Zone"])

    # TAB 1: Parameters
    with tab1:
        st.markdown("**Indicator Parameters**")

        param_schema = get_parameter_schema(group.base_template)
        updated_params = {}
        changed = False

        for param_key, schema in param_schema.items():
            current_value = group.parameters.get(param_key, schema["default"])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(schema["label"])

            with col2:
                if schema["type"] == "int":
                    new_value = st.number_input(
                        schema["label"],
                        min_value=schema.get("min", 1),
                        max_value=schema.get("max", 500),
                        value=int(current_value),
                        key=f"param_{group.id}_{param_key}",
                        label_visibility="collapsed"
                    )
                elif schema["type"] == "float":
                    new_value = st.number_input(
                        schema["label"],
                        min_value=float(schema.get("min", 0.0)),
                        max_value=float(schema.get("max", 100.0)),
                        value=float(current_value),
                        step=0.1,
                        key=f"param_{group.id}_{param_key}",
                        label_visibility="collapsed"
                    )
                else:
                    new_value = current_value

                updated_params[param_key] = new_value
                if new_value != current_value:
                    changed = True

        if changed:
            if st.button("Save Parameters", key="save_params"):
                group.parameters = updated_params
                save_confluence_groups(all_groups)
                st.success("Parameters saved!")
                st.rerun()

    # TAB 2: Plot Settings
    with tab2:
        st.markdown("**Chart Colors**")

        plot_schema = get_plot_schema(group.base_template)
        updated_colors = {}
        colors_changed = False

        for color_key, schema in plot_schema.items():
            if schema["type"] != "color":
                continue

            current_color = group.plot_settings.colors.get(color_key, schema["default"])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(schema["label"])
            with col2:
                new_color = st.color_picker(
                    schema["label"],
                    value=current_color,
                    key=f"color_{group.id}_{color_key}",
                    label_visibility="collapsed"
                )
                updated_colors[color_key] = new_color
                if new_color != current_color:
                    colors_changed = True

        st.markdown("**Line Settings**")
        new_line_width = st.slider(
            "Line Width",
            min_value=1,
            max_value=5,
            value=group.plot_settings.line_width,
            key=f"linewidth_{group.id}"
        )
        if new_line_width != group.plot_settings.line_width:
            colors_changed = True

        new_visible = st.checkbox(
            "Visible on Chart",
            value=group.plot_settings.visible,
            key=f"visible_{group.id}"
        )
        if new_visible != group.plot_settings.visible:
            colors_changed = True

        if colors_changed:
            if st.button("Save Plot Settings", key="save_plot"):
                group.plot_settings.colors = updated_colors
                group.plot_settings.line_width = new_line_width
                group.plot_settings.visible = new_visible
                save_confluence_groups(all_groups)
                st.success("Plot settings saved!")
                st.rerun()

    # TAB 3: Outputs & Triggers (read-only, from template)
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Interpreter Outputs**")
            output_descriptions = get_output_descriptions(group.base_template)
            for output, description in output_descriptions.items():
                st.markdown(f"- **{output}**: {description}")

        with col2:
            st.markdown("**Available Triggers**")
            triggers = get_group_triggers(group)
            for trigger in triggers:
                direction_icon = "LONG" if trigger.direction == "LONG" else "SHORT" if trigger.direction == "SHORT" else "BOTH"
                type_icon = "ENTRY" if trigger.trigger_type == "ENTRY" else "EXIT"
                st.markdown(f"- **{trigger.name}**")
                st.caption(f"  {direction_icon} {type_icon} | ID: `{trigger.id}`")

    # TAB 4: Danger Zone
    with tab4:
        st.markdown("**Rename Version**")
        st.caption(f"Template: {group.template_name}")
        new_version = st.text_input("Version Name", value=group.version, key=f"rename_{group.id}")
        st.caption(f"Full name will be: **{group.template_name} ({new_version})**")
        if new_version != group.version:
            if st.button("Rename", key="rename_btn"):
                group.version = new_version
                save_confluence_groups(all_groups)
                st.success("Renamed!")
                st.rerun()

        st.markdown("---")

        if group.is_default:
            st.info("Default groups cannot be deleted. You can disable them instead.")
        else:
            st.markdown("**Delete Group**")
            st.warning("This action cannot be undone.")
            if st.button("Delete Group", type="primary", key="delete_btn"):
                all_groups.remove(group)
                save_confluence_groups(all_groups)
                st.session_state.editing_group = None
                st.rerun()


def format_parameters(params: dict, template_id: str) -> str:
    """Format parameters as a readable string."""
    template = get_template(template_id)
    if not template:
        return str(params)

    schema = template.get("parameters_schema", {})

    parts = []
    for key, value in params.items():
        label = schema.get(key, {}).get("label", key)
        # Shorten common labels
        short_label = label.replace(" Period", "").replace("Threshold", "")
        parts.append(f"{short_label}: {value}")

    return " | ".join(parts)



if __name__ == "__main__":
    main()
