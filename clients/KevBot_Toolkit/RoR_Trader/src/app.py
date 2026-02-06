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
import copy

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
from portfolios import (
    load_portfolios,
    save_portfolio,
    get_portfolio_by_id,
    update_portfolio,
    delete_portfolio,
    duplicate_portfolio,
    get_portfolio_trades,
    calculate_portfolio_kpis,
    compute_drawdown_series,
    compute_strategy_correlation,
    evaluate_prop_firm_rules,
    PROP_FIRM_RULES,
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

# Strategies storage path (resolve relative to this script, not cwd)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_FILE = os.path.join(_SCRIPT_DIR, "strategies.json")


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
def prepare_data_with_indicators(symbol: str, days: int = 30, seed: int = 42,
                                  start_date=None, end_date=None):
    """
    Load market data and run all indicators, interpreters, and trigger detection.

    Uses Alpaca API if configured, otherwise falls back to mock data.

    Args:
        symbol: Stock symbol
        days: Number of days (used if start_date/end_date not provided)
        seed: Random seed for mock data
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)

    Returns DataFrame ready for trade generation and analysis.
    """
    # Load raw bars (Alpaca if configured, mock otherwise)
    df = load_market_data(symbol, days=days, seed=seed,
                          start_date=start_date, end_date=end_date)

    if len(df) == 0:
        return df

    # Run indicators
    df = run_all_indicators(df)

    # Run interpreters
    df = run_all_interpreters(df)

    # Detect triggers
    df = detect_all_triggers(df)

    return df


def prepare_forward_test_data(strat: dict):
    """
    Load continuous data from before forward_test_start to now,
    run the full pipeline, and split trades at the boundary.

    Returns (df, backtest_trades, forward_trades, forward_test_start_dt)
    """
    forward_test_start_dt = datetime.fromisoformat(strat['forward_test_start'])
    data_days = strat.get('data_days', 30)
    data_seed = strat.get('data_seed', 42)

    # Start before forward_test_start to have backtest data + indicator warmup
    start_date = forward_test_start_dt - timedelta(days=data_days * 2)
    # Round end_date to market close today for cache-friendly behavior
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

    df = prepare_data_with_indicators(
        strat['symbol'], seed=data_seed,
        start_date=start_date, end_date=end_date
    )

    if len(df) == 0:
        empty = pd.DataFrame()
        return df, empty, empty, forward_test_start_dt

    confluence_set = set(strat.get('confluence', [])) if strat.get('confluence') else None

    trades = generate_trades(
        df,
        direction=strat['direction'],
        entry_trigger=strat['entry_trigger'],
        exit_trigger=strat['exit_trigger'],
        confluence_required=confluence_set,
        risk_per_trade=strat.get('risk_per_trade', 100.0),
        stop_atr_mult=strat.get('stop_atr_mult', 1.5),
    )

    backtest_trades, forward_trades = split_trades_at_boundary(trades, forward_test_start_dt)
    return df, backtest_trades, forward_trades, forward_test_start_dt


def get_strategy_trades(strat: dict) -> pd.DataFrame:
    """
    Get trades for any modern strategy (backtest-only or forward-testing).
    Returns all trades as a single DataFrame. For legacy strategies, returns empty.
    """
    if 'entry_trigger_confluence_id' not in strat:
        return pd.DataFrame()

    if strat.get('forward_testing') and strat.get('forward_test_start'):
        _, bt, fw, _ = prepare_forward_test_data(strat)
        return pd.concat([bt, fw], ignore_index=True)
    else:
        data_days = strat.get('data_days', 30)
        data_seed = strat.get('data_seed', 42)
        df = prepare_data_with_indicators(strat['symbol'], data_days, data_seed)
        if len(df) == 0:
            return pd.DataFrame()
        confluence_set = set(strat.get('confluence', [])) if strat.get('confluence') else None
        return generate_trades(
            df,
            direction=strat['direction'],
            entry_trigger=strat['entry_trigger'],
            exit_trigger=strat['exit_trigger'],
            confluence_required=confluence_set,
            risk_per_trade=strat.get('risk_per_trade', 100.0),
            stop_atr_mult=strat.get('stop_atr_mult', 1.5),
        )


def render_mini_equity_curve(trades: pd.DataFrame, key: str, boundary_dt=None):
    """Render a small sparkline-style equity curve for a strategy card."""
    if len(trades) == 0:
        st.caption("No trades")
        return

    equity = trades[["exit_time", "r_multiple"]].sort_values("exit_time").reset_index(drop=True)
    equity["cumulative_r"] = equity["r_multiple"].cumsum()

    fig = go.Figure()

    if boundary_dt is not None:
        # Match timezone
        boundary_ts = boundary_dt
        if hasattr(equity["exit_time"].dtype, 'tz') and equity["exit_time"].dtype.tz is not None:
            boundary_ts = pd.Timestamp(boundary_dt).tz_localize(equity["exit_time"].dtype.tz)

        bt_mask = equity["exit_time"] < boundary_ts
        fw_mask = equity["exit_time"] >= boundary_ts

        bt_data = equity[bt_mask]
        fw_data = equity[fw_mask]

        if len(bt_data) > 0:
            fig.add_trace(go.Scatter(
                x=bt_data["exit_time"], y=bt_data["cumulative_r"],
                mode="lines", line=dict(color="#2196F3", width=1.5),
                fill="tozeroy", fillcolor="rgba(33, 150, 243, 0.08)",
                showlegend=False
            ))

        if len(fw_data) > 0:
            if len(bt_data) > 0:
                bridge = bt_data.iloc[[-1]]
                fw_plot = pd.concat([bridge, fw_data], ignore_index=True)
            else:
                fw_plot = fw_data
            fig.add_trace(go.Scatter(
                x=fw_plot["exit_time"], y=fw_plot["cumulative_r"],
                mode="lines", line=dict(color="#4CAF50", width=1.5),
                fill="tozeroy", fillcolor="rgba(76, 175, 80, 0.08)",
                showlegend=False
            ))
    else:
        final_r = equity["cumulative_r"].iloc[-1]
        color = "#4CAF50" if final_r >= 0 else "#f44336"
        fig.add_trace(go.Scatter(
            x=equity["exit_time"], y=equity["cumulative_r"],
            mode="lines", line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=f"rgba({'76, 175, 80' if final_r >= 0 else '244, 67, 54'}, 0.08)",
            showlegend=False
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig.update_layout(
        height=100, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def split_trades_at_boundary(trades_df: pd.DataFrame, boundary_dt: datetime):
    """Split trades into backtest (before boundary) and forward (at/after boundary)."""
    if len(trades_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Match timezone awareness of the entry_time column
    if hasattr(trades_df['entry_time'].dtype, 'tz') and trades_df['entry_time'].dtype.tz is not None:
        boundary_dt = pd.Timestamp(boundary_dt).tz_localize(trades_df['entry_time'].dtype.tz)

    backtest = trades_df[trades_df['entry_time'] < boundary_dt].copy()
    forward = trades_df[trades_df['entry_time'] >= boundary_dt].copy()
    return backtest, forward


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
    indicator_colors: dict = None,
    chart_key: str = 'price_chart'
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
    }], key=chart_key)


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

    # Add timestamp and ID (max+1 is safe after deletions)
    strategy['id'] = max((s.get('id', 0) for s in strategies), default=0) + 1
    strategy['created_at'] = datetime.now().isoformat()

    # Track forward test start date
    if strategy.get('forward_testing'):
        strategy['forward_test_start'] = strategy['created_at']

    # Convert set to list for JSON serialization
    if 'confluence' in strategy and isinstance(strategy['confluence'], set):
        strategy['confluence'] = list(strategy['confluence'])

    strategies.append(strategy)

    with open(STRATEGIES_FILE, 'w') as f:
        json.dump(strategies, f, indent=2)


def get_strategy_by_id(strategy_id: int) -> dict | None:
    """Get a single strategy by ID."""
    for s in load_strategies():
        if s.get('id') == strategy_id:
            return s
    return None


def update_strategy(strategy_id: int, updated_strategy: dict) -> bool:
    """
    Update an existing strategy in strategies.json.
    Preserves original id and created_at. Sets updated_at.
    Resets forward_test_start if forward testing is enabled.
    """
    strategies = load_strategies()

    for i, strat in enumerate(strategies):
        if strat.get('id') == strategy_id:
            updated_strategy['id'] = strategy_id
            updated_strategy['created_at'] = strat['created_at']
            updated_strategy['updated_at'] = datetime.now().isoformat()

            if updated_strategy.get('forward_testing'):
                updated_strategy['forward_test_start'] = datetime.now().isoformat()

            if 'confluence' in updated_strategy and isinstance(updated_strategy['confluence'], set):
                updated_strategy['confluence'] = list(updated_strategy['confluence'])

            strategies[i] = updated_strategy

            with open(STRATEGIES_FILE, 'w') as f:
                json.dump(strategies, f, indent=2)
            return True

    return False


def delete_strategy(strategy_id: int) -> bool:
    """Delete a strategy from strategies.json by ID."""
    strategies = load_strategies()
    original_len = len(strategies)
    strategies = [s for s in strategies if s.get('id') != strategy_id]

    if len(strategies) < original_len:
        with open(STRATEGIES_FILE, 'w') as f:
            json.dump(strategies, f, indent=2)
        return True
    return False


def duplicate_strategy(strategy_id: int) -> dict | None:
    """
    Duplicate a strategy. Creates a new copy with forward testing disabled.
    The original (including forward test data) is untouched.
    """
    strategies = load_strategies()
    source = None
    for s in strategies:
        if s.get('id') == strategy_id:
            source = s
            break

    if source is None:
        return None

    new_strategy = copy.deepcopy(source)
    new_strategy['id'] = max((s.get('id', 0) for s in strategies), default=0) + 1
    new_strategy['created_at'] = datetime.now().isoformat()
    new_strategy['name'] = source['name'] + " (Copy)"
    new_strategy['forward_testing'] = False
    new_strategy.pop('forward_test_start', None)
    new_strategy.pop('updated_at', None)

    strategies.append(new_strategy)

    with open(STRATEGIES_FILE, 'w') as f:
        json.dump(strategies, f, indent=2)

    return new_strategy


def get_trigger_display_name(strat: dict, trigger_key: str) -> str:
    """Get display name for a trigger, handling legacy strategies."""
    name_key = trigger_key + '_name'
    return strat.get(name_key, strat.get(trigger_key, 'Unknown'))


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
    if 'viewing_strategy_id' not in st.session_state:
        st.session_state.viewing_strategy_id = None
    if 'editing_strategy_id' not in st.session_state:
        st.session_state.editing_strategy_id = None
    if 'confirm_delete_id' not in st.session_state:
        st.session_state.confirm_delete_id = None
    if 'confirm_edit_id' not in st.session_state:
        st.session_state.confirm_edit_id = None
    if 'viewing_portfolio_id' not in st.session_state:
        st.session_state.viewing_portfolio_id = None
    if 'editing_portfolio_id' not in st.session_state:
        st.session_state.editing_portfolio_id = None
    if 'creating_portfolio' not in st.session_state:
        st.session_state.creating_portfolio = False
    if 'confirm_delete_portfolio_id' not in st.session_state:
        st.session_state.confirm_delete_portfolio_id = None

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
            ["Strategy Builder", "My Strategies", "Portfolios", "Confluence Groups"],
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
    elif page == "Portfolios":
        render_portfolios()
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

    # Editing banner
    editing_id = st.session_state.get('editing_strategy_id')
    if editing_id:
        editing_strat = get_strategy_by_id(editing_id)
        if editing_strat:
            st.info(f"Editing: {editing_strat['name']}")
            if st.button("Cancel Edit", key="cancel_edit_builder"):
                st.session_state.editing_strategy_id = None
                st.session_state.step = 1
                st.session_state.strategy_config = {}
                st.session_state.selected_confluences = set()
                st.rerun()

    # =========================================================================
    # STEP 1: SETUP
    # =========================================================================
    if step == 1:
        st.header("Step 1: Define Your Strategy")

        # Pre-populate defaults when editing
        edit_config = st.session_state.strategy_config if editing_id else {}

        col1, col2 = st.columns(2)

        with col1:
            symbol_idx = AVAILABLE_SYMBOLS.index(edit_config['symbol']) if edit_config.get('symbol') in AVAILABLE_SYMBOLS else 0
            symbol = st.selectbox("Ticker", AVAILABLE_SYMBOLS, index=symbol_idx)
            direction_idx = DIRECTIONS.index(edit_config['direction']) if edit_config.get('direction') in DIRECTIONS else 0
            direction = st.radio("Direction", DIRECTIONS, horizontal=True, index=direction_idx)

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
                # Pre-select saved entry trigger when editing
                saved_entry = edit_config.get('entry_trigger_confluence_id', '')
                entry_default_idx = entry_trigger_options.index(saved_entry) if saved_entry in entry_trigger_options else 0
                entry_trigger_idx = st.selectbox(
                    "Entry Trigger",
                    range(len(entry_trigger_options)),
                    index=entry_default_idx,
                    format_func=lambda i: entry_trigger_labels[i],
                    help="Triggers from your enabled Confluence Groups"
                )
                entry_trigger = entry_trigger_options[entry_trigger_idx]
                entry_trigger_name = entry_triggers[entry_trigger]

        with col2:
            tf_idx = TIMEFRAMES.index(edit_config['timeframe']) if edit_config.get('timeframe') in TIMEFRAMES else 0
            timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=tf_idx)
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
                # Pre-select saved exit trigger when editing
                saved_exit = edit_config.get('exit_trigger_confluence_id', '')
                exit_default_idx = exit_trigger_options.index(saved_exit) if saved_exit in exit_trigger_options else 0
                exit_trigger_idx = st.selectbox(
                    "Exit Trigger",
                    range(len(exit_trigger_options)),
                    index=exit_default_idx,
                    format_func=lambda i: exit_trigger_labels[i],
                    help="Triggers from your enabled Confluence Groups"
                )
                exit_trigger = exit_trigger_options[exit_trigger_idx]
                exit_trigger_name = exit_triggers[exit_trigger]

        # Stop Loss Settings
        st.divider()
        st.subheader("Stop Loss")
        stop_atr_mult = st.number_input(
            "Stop Loss (ATR x)",
            min_value=0.5,
            max_value=5.0,
            value=float(edit_config.get('stop_atr_mult', 1.5)),
            step=0.1,
            help="Stop loss distance as a multiple of ATR. Acts as an early exit if price moves against you."
        )

        # Dollar sizing deferred to portfolio level — use fixed defaults for R calculations
        risk_per_trade = 100.0
        starting_balance = 10000.0

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
            editing_id = st.session_state.get('editing_strategy_id')
            save_label = "💾 Update Strategy" if editing_id else "💾 Save Strategy"

            if st.button(save_label, type="primary"):
                strategy = {
                    'name': strategy_name,
                    **config,
                    'confluence': list(selected),
                    'kpis': kpis,
                    'forward_testing': enable_forward,
                    'alerts': enable_alerts,
                }

                if editing_id:
                    update_strategy(editing_id, strategy)
                    st.success(f"Strategy '{strategy_name}' updated!")
                else:
                    save_strategy(strategy)
                    st.success(f"Strategy '{strategy_name}' saved!")

                st.balloons()

                # Reset for new strategy
                st.session_state.step = 1
                st.session_state.selected_confluences = set()
                st.session_state.strategy_config = {}
                st.session_state.editing_strategy_id = None


def render_my_strategies():
    """Render the My Strategies page — routes to list or detail view."""
    if st.session_state.viewing_strategy_id is not None:
        render_strategy_detail(st.session_state.viewing_strategy_id)
        return
    render_strategy_list()


def render_strategy_list():
    """Render the strategy list view with sorting and filtering."""
    st.header("My Strategies")

    strategies = load_strategies()

    if len(strategies) == 0:
        st.info("No strategies saved yet. Create one in the Strategy Builder!")
        return

    # --- Filter & Sort Bar ---
    filter_cols = st.columns([1, 1, 1, 2])

    with filter_cols[0]:
        ticker_filter = st.selectbox("Ticker", ["All"] + AVAILABLE_SYMBOLS, key="strat_filter_ticker")
    with filter_cols[1]:
        direction_filter = st.selectbox("Direction", ["All", "LONG", "SHORT"], key="strat_filter_dir")
    with filter_cols[2]:
        status_filter = st.selectbox("Status", ["All", "Forward Testing", "Backtest Only"], key="strat_filter_status")
    with filter_cols[3]:
        sort_option = st.selectbox(
            "Sort By",
            ["Newest First", "Oldest First", "Name A-Z", "Win Rate (High)", "Profit Factor (High)", "Total R (High)"],
            key="strat_sort"
        )

    # Apply filters
    if ticker_filter != "All":
        strategies = [s for s in strategies if s.get('symbol') == ticker_filter]
    if direction_filter != "All":
        strategies = [s for s in strategies if s.get('direction') == direction_filter]
    if status_filter == "Forward Testing":
        strategies = [s for s in strategies if s.get('forward_testing')]
    elif status_filter == "Backtest Only":
        strategies = [s for s in strategies if not s.get('forward_testing')]

    # Apply sorting
    sort_keys = {
        "Newest First": (lambda s: s.get('created_at', ''), True),
        "Oldest First": (lambda s: s.get('created_at', ''), False),
        "Name A-Z": (lambda s: s.get('name', '').lower(), False),
        "Win Rate (High)": (lambda s: s.get('kpis', {}).get('win_rate', 0), True),
        "Profit Factor (High)": (lambda s: s.get('kpis', {}).get('profit_factor', 0), True),
        "Total R (High)": (lambda s: s.get('kpis', {}).get('total_r', 0), True),
    }
    key_fn, reverse = sort_keys[sort_option]
    strategies.sort(key=key_fn, reverse=reverse)

    if len(strategies) == 0:
        st.info("No strategies match the current filters.")
        return

    st.caption(f"{len(strategies)} strategies")

    # --- Strategy Cards ---
    enabled_groups = get_enabled_groups()

    for strat in strategies:
        sid = strat.get('id', 0)
        with st.container(border=True):
            # Main layout: info on left, mini equity curve on right
            info_col, chart_col = st.columns([3, 2])

            with info_col:
                # Name + metadata
                st.markdown(f"### {strat['name']}")
                entry_display = get_trigger_display_name(strat, 'entry_trigger')
                exit_display = get_trigger_display_name(strat, 'exit_trigger')
                st.caption(f"{strat['symbol']} | {strat['direction']} | {entry_display} → {exit_display}")

                # KPI metrics inline
                kpis = strat.get('kpis', {})
                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Win Rate", f"{kpis.get('win_rate', 0):.1f}%")
                pf = kpis.get('profit_factor', 0)
                kpi_cols[1].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
                kpi_cols[2].metric("Total R", f"{kpis.get('total_r', 0):+.1f}")

            with chart_col:
                # Status badge
                if strat.get('forward_testing') and strat.get('forward_test_start'):
                    ft_start = datetime.fromisoformat(strat['forward_test_start'])
                    ft_days = (datetime.now() - ft_start).days
                    st.caption(f"🟢 Forward Testing ({ft_days}d)")
                elif strat.get('forward_testing'):
                    st.caption("🟢 Forward Testing")
                else:
                    st.caption("⚪ Backtest Only")

                # Mini equity curve
                is_legacy = 'entry_trigger_confluence_id' not in strat
                if not is_legacy:
                    trades = get_strategy_trades(strat)
                    boundary = None
                    if strat.get('forward_testing') and strat.get('forward_test_start'):
                        boundary = datetime.fromisoformat(strat['forward_test_start'])
                    render_mini_equity_curve(trades, key=f"mini_eq_{sid}", boundary_dt=boundary)

            # Confluence tags
            confluence = strat.get('confluence', [])
            if len(confluence) > 0:
                formatted = [format_confluence_record(c, enabled_groups) for c in confluence[:3]]
                st.caption(f"Confluence: {', '.join(formatted)}" + ("..." if len(confluence) > 3 else ""))

            # Action buttons
            btn_cols = st.columns([1, 1, 1, 1, 4])
            with btn_cols[0]:
                if st.button("View", key=f"view_{sid}"):
                    st.session_state.viewing_strategy_id = sid
                    st.rerun()
            with btn_cols[1]:
                if st.button("Edit", key=f"edit_{sid}"):
                    initiate_edit(strat)
            with btn_cols[2]:
                if st.button("Clone", key=f"clone_{sid}"):
                    new = duplicate_strategy(sid)
                    if new:
                        st.toast(f"Cloned as '{new['name']}'")
                        st.rerun()
            with btn_cols[3]:
                if st.button("Delete", key=f"del_{sid}", type="secondary"):
                    st.session_state.confirm_delete_id = sid
                    st.rerun()

            # Inline delete confirmation
            if st.session_state.confirm_delete_id == sid:
                st.warning(f"Are you sure you want to delete '{strat['name']}'? This cannot be undone.")
                confirm_cols = st.columns([1, 1, 6])
                with confirm_cols[0]:
                    if st.button("Yes, Delete", key=f"confirm_del_{sid}", type="primary"):
                        delete_strategy(sid)
                        st.session_state.confirm_delete_id = None
                        st.rerun()
                with confirm_cols[1]:
                    if st.button("Cancel", key=f"cancel_del_{sid}"):
                        st.session_state.confirm_delete_id = None
                        st.rerun()

            # Inline edit confirmation (forward-tested strategies)
            if st.session_state.confirm_edit_id == sid:
                st.warning(
                    "This strategy has forward testing enabled. "
                    "Editing will reset the forward test start date. "
                    "You can also duplicate the strategy to preserve the original."
                )
                edit_cols = st.columns([1, 1, 1, 5])
                with edit_cols[0]:
                    if st.button("Edit Anyway", key=f"confirm_edit_{sid}", type="primary"):
                        st.session_state.confirm_edit_id = None
                        load_strategy_into_builder(strat)
                with edit_cols[1]:
                    if st.button("Duplicate Instead", key=f"dup_instead_{sid}"):
                        new = duplicate_strategy(sid)
                        if new:
                            st.session_state.confirm_edit_id = None
                            load_strategy_into_builder(new)
                with edit_cols[2]:
                    if st.button("Cancel", key=f"cancel_edit_{sid}"):
                        st.session_state.confirm_edit_id = None
                        st.rerun()


# =============================================================================
# STRATEGY DETAIL VIEW
# =============================================================================

def render_strategy_detail(strategy_id: int):
    """Render the full detail view for a single strategy."""
    strat = get_strategy_by_id(strategy_id)
    if strat is None:
        st.error("Strategy not found.")
        st.session_state.viewing_strategy_id = None
        st.rerun()
        return

    # Back button
    if st.button("← Back to Strategies"):
        st.session_state.viewing_strategy_id = None
        st.rerun()

    # Header
    st.header(strat['name'])

    meta_cols = st.columns(5)
    meta_cols[0].markdown(f"**Ticker:** {strat['symbol']}")
    meta_cols[1].markdown(f"**Direction:** {strat['direction']}")
    meta_cols[2].markdown(f"**Timeframe:** {strat.get('timeframe', '1Min')}")
    meta_cols[3].markdown(f"**Entry:** {get_trigger_display_name(strat, 'entry_trigger')}")
    meta_cols[4].markdown(f"**Exit:** {get_trigger_display_name(strat, 'exit_trigger')}")

    # Confluence conditions
    enabled_groups = get_enabled_groups()
    confluence = strat.get('confluence', [])
    if confluence:
        formatted = [format_confluence_record(c, enabled_groups) for c in confluence]
        st.caption("Confluence: " + " + ".join(formatted))

    # Action bar
    action_cols = st.columns([1, 1, 1, 1, 4])
    with action_cols[0]:
        if st.button("Edit Strategy", key="detail_edit"):
            initiate_edit(strat)
    with action_cols[1]:
        if st.button("Clone", key="detail_clone"):
            new = duplicate_strategy(strategy_id)
            if new:
                st.toast(f"Cloned as '{new['name']}'")
                st.rerun()
    with action_cols[2]:
        if st.button("Delete", key="detail_delete", type="secondary"):
            st.session_state.confirm_delete_id = strategy_id
            st.rerun()
    with action_cols[3]:
        if strat.get('forward_testing') and strat.get('forward_test_start'):
            ft_start = datetime.fromisoformat(strat['forward_test_start'])
            ft_days = (datetime.now() - ft_start).days
            status = f"🟢 Forward Testing ({ft_days}d)"
        elif strat.get('forward_testing'):
            status = "🟢 Forward Testing"
        else:
            status = "⚪ Backtest Only"
        st.markdown(f"**{status}**")

    # Inline delete confirmation
    if st.session_state.confirm_delete_id == strategy_id:
        st.warning(f"Are you sure you want to delete '{strat['name']}'? This cannot be undone.")
        confirm_cols = st.columns([1, 1, 6])
        with confirm_cols[0]:
            if st.button("Yes, Delete", key="detail_confirm_del", type="primary"):
                delete_strategy(strategy_id)
                st.session_state.confirm_delete_id = None
                st.session_state.viewing_strategy_id = None
                st.rerun()
        with confirm_cols[1]:
            if st.button("Cancel", key="detail_cancel_del"):
                st.session_state.confirm_delete_id = None
                st.rerun()

    # Inline edit confirmation (forward-tested strategies)
    if st.session_state.confirm_edit_id == strategy_id:
        st.warning(
            "This strategy has forward testing enabled. "
            "Editing will reset the forward test start date. "
            "You can also duplicate the strategy to preserve the original."
        )
        edit_cols = st.columns([1, 1, 1, 5])
        with edit_cols[0]:
            if st.button("Edit Anyway", key="detail_confirm_edit", type="primary"):
                st.session_state.confirm_edit_id = None
                load_strategy_into_builder(strat)
        with edit_cols[1]:
            if st.button("Duplicate Instead", key="detail_dup_instead"):
                new = duplicate_strategy(strategy_id)
                if new:
                    st.session_state.confirm_edit_id = None
                    load_strategy_into_builder(new)
        with edit_cols[2]:
            if st.button("Cancel", key="detail_cancel_edit"):
                st.session_state.confirm_edit_id = None
                st.rerun()

    st.divider()

    # Route to appropriate view based on strategy type
    is_legacy = 'entry_trigger_confluence_id' not in strat
    if is_legacy:
        st.info("This is a legacy strategy. Live re-backtest is not available. Showing saved KPIs.")
        render_saved_kpis(strat)
    elif strat.get('forward_testing') and strat.get('forward_test_start'):
        render_forward_test_view(strat)
    else:
        render_live_backtest(strat)


def render_saved_kpis(strat: dict):
    """Display saved KPIs for legacy strategies that cannot be re-backtested."""
    kpis = strat.get('kpis', {})

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Trades", kpis.get("total_trades", 0))
    kpi_cols[1].metric("Win Rate", f"{kpis.get('win_rate', 0):.1f}%")
    pf = kpis.get('profit_factor', 0)
    kpi_cols[2].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
    kpi_cols[3].metric("Avg R", f"{kpis.get('avg_r', 0):+.2f}")
    kpi_cols[4].metric("Total R", f"{kpis.get('total_r', 0):+.1f}")
    kpi_cols[5].metric("Daily R", f"{kpis.get('daily_r', 0):+.2f}")

    st.subheader("Strategy Configuration")
    stop = strat.get('stop_atr_mult')
    if stop:
        st.markdown(f"**Stop Loss:** {stop}x ATR")
    created = strat.get('created_at', 'Unknown')
    st.markdown(f"**Created:** {created[:10] if len(created) >= 10 else created}")


def render_live_backtest(strat: dict):
    """Re-run backtest with current data and display full results."""
    data_days = strat.get('data_days', 30)
    data_seed = strat.get('data_seed', 42)

    with st.spinner("Running backtest with current data..."):
        df = prepare_data_with_indicators(strat['symbol'], data_days, data_seed)

        if len(df) == 0:
            st.error("No data available for this symbol.")
            return

        confluence_set = set(strat.get('confluence', [])) if strat.get('confluence') else None

        trades = generate_trades(
            df,
            direction=strat['direction'],
            entry_trigger=strat['entry_trigger'],
            exit_trigger=strat['exit_trigger'],
            confluence_required=confluence_set,
            risk_per_trade=strat.get('risk_per_trade', 100.0),
            stop_atr_mult=strat.get('stop_atr_mult', 1.5),
        )

    if len(trades) == 0:
        st.warning("No trades generated. The entry trigger may reference a confluence group that is no longer enabled.")

    # Compute live KPIs
    kpis = calculate_kpis(
        trades,
        starting_balance=strat.get('starting_balance', 10000.0),
        risk_per_trade=strat.get('risk_per_trade', 100.0),
    )

    # KPI row (R-based metrics only — dollar sizing deferred to portfolio level)
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Trades", kpis["total_trades"])
    kpi_cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
    pf = kpis['profit_factor']
    kpi_cols[2].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
    kpi_cols[3].metric("Avg R", f"{kpis['avg_r']:+.2f}")
    kpi_cols[4].metric("Total R", f"{kpis['total_r']:+.1f}")
    kpi_cols[5].metric("Daily R", f"{kpis['daily_r']:+.2f}")

    # Tabbed content
    tab_backtest, tab_config = st.tabs(["Backtest Results", "Configuration"])

    with tab_backtest:
        # Charts side by side
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.subheader("Equity Curve")
            if len(trades) > 0:
                equity_df = trades[["exit_time", "r_multiple"]].sort_values("exit_time")
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
                st.info("No trades to display.")

        with chart_right:
            st.subheader("R-Multiple Distribution")
            if len(trades) > 0:
                fig_hist = px.histogram(
                    trades, x="r_multiple", nbins=20,
                    labels={"r_multiple": "R-Multiple"},
                    color_discrete_sequence=["#2196F3"]
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_hist.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No trades to display.")

        # Price chart (full width)
        st.subheader("Price Chart")
        enabled_groups = get_enabled_groups()
        overlay_groups = [g for g in enabled_groups if g.base_template in ["ema_stack", "vwap", "utbot"]]
        show_indicators = []
        indicator_colors = {}
        for group in overlay_groups:
            show_indicators.extend(get_overlay_indicators_for_group(group))
            indicator_colors.update(get_overlay_colors_for_group(group))

        render_price_chart(
            df, trades, strat,
            show_indicators=show_indicators,
            indicator_colors=indicator_colors,
            chart_key='detail_price_chart'
        )

        # Trade history table
        st.subheader("Trade History")
        if len(trades) > 0:
            display = trades.copy()
            display['entry'] = display['entry_time'].dt.strftime('%m/%d %H:%M')
            display['exit'] = display['exit_time'].dt.strftime('%m/%d %H:%M')
            display['R'] = display['r_multiple'].apply(lambda x: f"{x:+.2f}")
            display['result'] = display['win'].apply(lambda x: "Win" if x else "Loss")
            st.dataframe(
                display[['entry', 'exit', 'exit_reason', 'R', 'result']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'entry': 'Entry Time',
                    'exit': 'Exit Time',
                    'exit_reason': 'Exit Reason',
                    'R': 'R-Multiple',
                    'result': 'Result',
                }
            )
        else:
            st.info("No trades to display.")

    with tab_config:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strategy Setup**")
            st.markdown(f"- Ticker: {strat['symbol']}")
            st.markdown(f"- Direction: {strat['direction']}")
            st.markdown(f"- Timeframe: {strat.get('timeframe', '1Min')}")
            st.markdown(f"- Entry: {get_trigger_display_name(strat, 'entry_trigger')}")
            st.markdown(f"- Exit: {get_trigger_display_name(strat, 'exit_trigger')}")
        with col2:
            st.markdown("**Settings**")
            st.markdown(f"- Stop Loss: {strat.get('stop_atr_mult', 1.5):.1f}x ATR")
            st.markdown(f"- Data Days: {strat.get('data_days', 30)}")
            created = strat.get('created_at', 'Unknown')
            st.markdown(f"- Created: {created[:19] if len(created) >= 19 else created}")
            if strat.get('updated_at'):
                st.markdown(f"- Last Updated: {strat['updated_at'][:19]}")

        st.markdown("**Confluence Conditions**")
        confluence = strat.get('confluence', [])
        if confluence:
            enabled_groups = get_enabled_groups()
            for c in confluence:
                st.markdown(f"- {format_confluence_record(c, enabled_groups)}")
        else:
            st.caption("No confluence conditions")


# =============================================================================
# FORWARD TEST VIEW
# =============================================================================

def render_forward_test_view(strat: dict):
    """Render the forward test view for a strategy with forward testing enabled."""
    forward_start_str = strat.get('forward_test_start', '')
    forward_start_dt = datetime.fromisoformat(forward_start_str)
    duration_days = (datetime.now() - forward_start_dt).days

    st.markdown(
        f"**Forward Testing since {forward_start_str[:10]}** "
        f"({duration_days}d)"
    )

    with st.spinner("Loading forward test data..."):
        df, backtest_trades, forward_trades, boundary_dt = prepare_forward_test_data(strat)

    if len(df) == 0:
        st.error("No data available for this symbol.")
        return

    # KPI comparison
    render_kpi_comparison(backtest_trades, forward_trades)

    # Tabs
    tab_charts, tab_trades, tab_config = st.tabs(["Equity & Charts", "Trade History", "Configuration"])

    with tab_charts:
        # Combined equity curve
        all_trades = pd.concat([backtest_trades, forward_trades], ignore_index=True)
        render_combined_equity_curve(all_trades, boundary_dt)

        # R-distribution comparison
        render_r_distribution_comparison(backtest_trades, forward_trades)

        # Price chart
        st.subheader("Price Chart")
        enabled_groups = get_enabled_groups()
        overlay_groups = [g for g in enabled_groups if g.base_template in ["ema_stack", "vwap", "utbot"]]
        show_indicators = []
        indicator_colors = {}
        for group in overlay_groups:
            show_indicators.extend(get_overlay_indicators_for_group(group))
            indicator_colors.update(get_overlay_colors_for_group(group))

        render_price_chart(
            df, all_trades, strat,
            show_indicators=show_indicators,
            indicator_colors=indicator_colors,
            chart_key='forward_test_chart'
        )

    with tab_trades:
        render_split_trade_history(backtest_trades, forward_trades)

    with tab_config:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strategy Setup**")
            st.markdown(f"- Ticker: {strat['symbol']}")
            st.markdown(f"- Direction: {strat['direction']}")
            st.markdown(f"- Timeframe: {strat.get('timeframe', '1Min')}")
            st.markdown(f"- Entry: {get_trigger_display_name(strat, 'entry_trigger')}")
            st.markdown(f"- Exit: {get_trigger_display_name(strat, 'exit_trigger')}")
        with col2:
            st.markdown("**Settings**")
            st.markdown(f"- Stop Loss: {strat.get('stop_atr_mult', 1.5):.1f}x ATR")
            st.markdown(f"- Data Days: {strat.get('data_days', 30)}")
            created = strat.get('created_at', 'Unknown')
            st.markdown(f"- Created: {created[:19] if len(created) >= 19 else created}")
            st.markdown(f"- Forward Test Start: {forward_start_str[:19]}")
            if strat.get('updated_at'):
                st.markdown(f"- Last Updated: {strat['updated_at'][:19]}")

        st.markdown("**Confluence Conditions**")
        confluence = strat.get('confluence', [])
        if confluence:
            enabled_groups = get_enabled_groups()
            for c in confluence:
                st.markdown(f"- {format_confluence_record(c, enabled_groups)}")
        else:
            st.caption("No confluence conditions")


def render_kpi_comparison(backtest_trades: pd.DataFrame, forward_trades: pd.DataFrame):
    """Render side-by-side KPI comparison between backtest and forward test."""
    bt_kpis = calculate_kpis(backtest_trades)
    fw_kpis = calculate_kpis(forward_trades)

    col_bt, col_fw = st.columns(2)

    with col_bt:
        st.subheader("Backtest")
        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Trades", bt_kpis["total_trades"])
        kpi_cols[1].metric("Win Rate", f"{bt_kpis['win_rate']:.1f}%")
        pf = bt_kpis['profit_factor']
        kpi_cols[2].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
        kpi_cols[3].metric("Avg R", f"{bt_kpis['avg_r']:+.2f}")
        kpi_cols[4].metric("Total R", f"{bt_kpis['total_r']:+.1f}")
        kpi_cols[5].metric("Daily R", f"{bt_kpis['daily_r']:+.2f}")

    with col_fw:
        st.subheader("Forward Test")
        if len(forward_trades) == 0:
            st.info("No forward test trades yet.")
        else:
            kpi_cols = st.columns(6)

            # Trades (no delta — count isn't comparable)
            kpi_cols[0].metric("Trades", fw_kpis["total_trades"])

            # Win Rate with delta
            wr_delta = safe_subtract(fw_kpis['win_rate'], bt_kpis['win_rate'])
            kpi_cols[1].metric("Win Rate", f"{fw_kpis['win_rate']:.1f}%",
                               delta=f"{wr_delta:+.1f}%")

            # Profit Factor with delta
            fw_pf = fw_kpis['profit_factor']
            bt_pf = bt_kpis['profit_factor']
            pf_display = "Inf" if fw_pf == float('inf') else f"{fw_pf:.2f}"
            pf_delta = safe_subtract(fw_pf, bt_pf)
            if fw_pf == float('inf') or bt_pf == float('inf'):
                kpi_cols[2].metric("Profit Factor", pf_display)
            else:
                kpi_cols[2].metric("Profit Factor", pf_display, delta=f"{pf_delta:+.2f}")

            # Avg R with delta
            avg_r_delta = fw_kpis['avg_r'] - bt_kpis['avg_r']
            kpi_cols[3].metric("Avg R", f"{fw_kpis['avg_r']:+.2f}",
                               delta=f"{avg_r_delta:+.2f}")

            # Total R (no delta — different time periods)
            kpi_cols[4].metric("Total R", f"{fw_kpis['total_r']:+.1f}")

            # Daily R with delta
            daily_r_delta = fw_kpis['daily_r'] - bt_kpis['daily_r']
            kpi_cols[5].metric("Daily R", f"{fw_kpis['daily_r']:+.2f}",
                               delta=f"{daily_r_delta:+.2f}")


def render_combined_equity_curve(trades_df: pd.DataFrame, boundary_dt: datetime):
    """Render a combined equity curve with vertical line at forward test start."""
    st.subheader("Equity Curve")

    if len(trades_df) == 0:
        st.info("No trades to display.")
        return

    equity_df = trades_df[["exit_time", "r_multiple"]].sort_values("exit_time").reset_index(drop=True)
    equity_df["cumulative_r"] = equity_df["r_multiple"].cumsum()

    # Match timezone awareness for comparison
    boundary_ts = boundary_dt
    if hasattr(equity_df["exit_time"].dtype, 'tz') and equity_df["exit_time"].dtype.tz is not None:
        boundary_ts = pd.Timestamp(boundary_dt).tz_localize(equity_df["exit_time"].dtype.tz)

    # Split into backtest and forward portions
    bt_mask = equity_df["exit_time"] < boundary_ts
    fw_mask = equity_df["exit_time"] >= boundary_ts

    fig = go.Figure()

    # Backtest portion (blue)
    bt_data = equity_df[bt_mask]
    if len(bt_data) > 0:
        fig.add_trace(go.Scatter(
            x=bt_data["exit_time"],
            y=bt_data["cumulative_r"],
            mode="lines",
            name="Backtest",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33, 150, 243, 0.1)"
        ))

    # Forward portion (green) — connect to last backtest point
    fw_data = equity_df[fw_mask]
    if len(fw_data) > 0:
        # Include last backtest point for continuity
        if len(bt_data) > 0:
            bridge = bt_data.iloc[[-1]]
            fw_plot = pd.concat([bridge, fw_data], ignore_index=True)
        else:
            fw_plot = fw_data

        fig.add_trace(go.Scatter(
            x=fw_plot["exit_time"],
            y=fw_plot["cumulative_r"],
            mode="lines",
            name="Forward Test",
            line=dict(color="#4CAF50", width=2),
            fill="tozeroy",
            fillcolor="rgba(76, 175, 80, 0.1)"
        ))

    # High water mark
    fig.add_trace(go.Scatter(
        x=equity_df["exit_time"],
        y=equity_df["cumulative_r"].cummax(),
        mode="lines",
        name="High Water Mark",
        line=dict(color="green", width=1, dash="dot")
    ))

    # Vertical line at forward test start
    # Use shape + annotation instead of add_vline to avoid Plotly datetime arithmetic bug
    fig.add_shape(
        type="line", x0=boundary_ts, x1=boundary_ts, y0=0, y1=1,
        yref="paper", line=dict(color="orange", width=2, dash="dash")
    )
    fig.add_annotation(
        x=boundary_ts, y=1, yref="paper",
        text="Forward Test Start", showarrow=False,
        font=dict(color="orange"), xanchor="left", yanchor="bottom"
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="",
        yaxis_title="Cumulative R",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_r_distribution_comparison(backtest_trades: pd.DataFrame, forward_trades: pd.DataFrame):
    """Render side-by-side R-multiple distribution histograms."""
    col_bt, col_fw = st.columns(2)

    with col_bt:
        st.subheader("Backtest R-Distribution")
        if len(backtest_trades) > 0:
            fig = px.histogram(
                backtest_trades, x="r_multiple", nbins=20,
                labels={"r_multiple": "R-Multiple"},
                color_discrete_sequence=["#2196F3"]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No backtest trades.")

    with col_fw:
        st.subheader("Forward R-Distribution")
        if len(forward_trades) > 0:
            fig = px.histogram(
                forward_trades, x="r_multiple", nbins=20,
                labels={"r_multiple": "R-Multiple"},
                color_discrete_sequence=["#4CAF50"]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forward test trades yet.")


def render_split_trade_history(backtest_trades: pd.DataFrame, forward_trades: pd.DataFrame):
    """Render trade history split into forward test and backtest sections."""
    def format_trade_table(trades: pd.DataFrame) -> pd.DataFrame:
        display = trades.copy()
        display['entry'] = display['entry_time'].dt.strftime('%m/%d %H:%M')
        display['exit'] = display['exit_time'].dt.strftime('%m/%d %H:%M')
        display['R'] = display['r_multiple'].apply(lambda x: f"{x:+.2f}")
        display['result'] = display['win'].apply(lambda x: "Win" if x else "Loss")
        return display[['entry', 'exit', 'exit_reason', 'R', 'result']]

    trade_col_config = {
        'entry': 'Entry Time',
        'exit': 'Exit Time',
        'exit_reason': 'Exit Reason',
        'R': 'R-Multiple',
        'result': 'Result',
    }

    with st.expander(f"Forward Test Trades ({len(forward_trades)})", expanded=True):
        if len(forward_trades) > 0:
            display = format_trade_table(forward_trades.sort_values('entry_time', ascending=False))
            st.dataframe(display, use_container_width=True, hide_index=True,
                         column_config=trade_col_config)
        else:
            st.info("No forward test trades yet.")

    with st.expander(f"Backtest Trades ({len(backtest_trades)})", expanded=False):
        if len(backtest_trades) > 0:
            display = format_trade_table(backtest_trades.sort_values('entry_time', ascending=False))
            st.dataframe(display, use_container_width=True, hide_index=True,
                         column_config=trade_col_config)
        else:
            st.info("No backtest trades.")


# =============================================================================
# EDIT FLOW
# =============================================================================

def initiate_edit(strat: dict):
    """Start editing a strategy. Warns if forward testing is enabled."""
    if strat.get('forward_testing'):
        st.session_state.confirm_edit_id = strat.get('id')
        st.rerun()
    else:
        load_strategy_into_builder(strat)


def load_strategy_into_builder(strat: dict):
    """Load a saved strategy into the Strategy Builder for editing."""
    if 'entry_trigger_confluence_id' not in strat:
        st.error("Legacy strategies cannot be edited. Please create a new strategy in the Strategy Builder.")
        return

    st.session_state.editing_strategy_id = strat['id']

    st.session_state.strategy_config = {
        'symbol': strat['symbol'],
        'direction': strat['direction'],
        'timeframe': strat.get('timeframe', '1Min'),
        'entry_trigger': strat['entry_trigger'],
        'entry_trigger_confluence_id': strat.get('entry_trigger_confluence_id', ''),
        'exit_trigger': strat['exit_trigger'],
        'exit_trigger_confluence_id': strat.get('exit_trigger_confluence_id', ''),
        'entry_trigger_name': strat.get('entry_trigger_name', strat['entry_trigger']),
        'exit_trigger_name': strat.get('exit_trigger_name', strat['exit_trigger']),
        'risk_per_trade': strat.get('risk_per_trade', 100.0),
        'stop_atr_mult': strat.get('stop_atr_mult', 1.5),
        'starting_balance': strat.get('starting_balance', 10000.0),
        'data_days': strat.get('data_days', 30),
        'data_seed': strat.get('data_seed', 42),
    }

    st.session_state.selected_confluences = set(strat.get('confluence', []))
    st.session_state.step = 2
    st.session_state.viewing_strategy_id = None
    st.session_state.confirm_edit_id = None

    st.rerun()


# =============================================================================
# PORTFOLIOS
# =============================================================================

def render_portfolios():
    """Route to portfolio list, detail, or builder view."""
    if st.session_state.creating_portfolio or st.session_state.editing_portfolio_id is not None:
        render_portfolio_builder()
        return
    if st.session_state.viewing_portfolio_id is not None:
        render_portfolio_detail(st.session_state.viewing_portfolio_id)
        return
    render_portfolio_list()


def render_portfolio_list():
    """Render portfolio list with cards."""
    st.header("My Portfolios")

    col_header, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("+ New Portfolio", type="primary"):
            st.session_state.creating_portfolio = True
            st.rerun()

    portfolios = load_portfolios()

    if len(portfolios) == 0:
        st.info("No portfolios yet. Create one to combine your strategies!")
        return

    for port in portfolios:
        pid = port.get('id', 0)
        with st.container(border=True):
            info_col, chart_col = st.columns([3, 2])

            with info_col:
                st.markdown(f"### {port['name']}")
                n_strats = len(port.get('strategies', []))
                balance = port.get('starting_balance', 10000)
                compound = port.get('compound_rate', 0) * 100
                st.caption(
                    f"{n_strats} strategies | ${balance:,.0f} starting balance"
                    + (f" | {compound:.0f}% risk scaling" if compound > 0 else "")
                )

                # KPIs from cache
                kpis = port.get('cached_kpis', {})
                if kpis:
                    kpi_cols = st.columns(3)
                    kpi_cols[0].metric("Total P&L", f"${kpis.get('total_pnl', 0):+,.0f}")
                    max_dd = kpis.get('max_drawdown_pct', 0)
                    kpi_cols[1].metric("Max DD", f"{max_dd:.1f}%")
                    kpi_cols[2].metric("Win Rate", f"{kpis.get('win_rate', 0):.1f}%")

                # Prop firm badge
                firm_key = port.get('prop_firm')
                if firm_key and firm_key in PROP_FIRM_RULES:
                    st.caption(f"Prop Firm: {PROP_FIRM_RULES[firm_key]['name']}")

            with chart_col:
                # Strategy names
                strat_names = []
                for ps in port.get('strategies', [])[:4]:
                    s = get_strategy_by_id(ps['strategy_id'])
                    if s:
                        strat_names.append(f"{s['symbol']} {s['direction']}")
                if strat_names:
                    st.caption(", ".join(strat_names) + ("..." if n_strats > 4 else ""))

                # Mini equity curve from cached data
                if kpis and kpis.get('total_trades', 0) > 0:
                    try:
                        data = get_portfolio_trades(port, get_strategy_by_id, get_strategy_trades)
                        if len(data['combined_trades']) > 0:
                            trades = data['combined_trades']
                            eq = trades[['exit_time', 'cumulative_pnl']].copy()
                            fig = go.Figure()
                            final = eq['cumulative_pnl'].iloc[-1]
                            color = "#4CAF50" if final >= 0 else "#f44336"
                            fig.add_trace(go.Scatter(
                                x=eq['exit_time'], y=eq['cumulative_pnl'],
                                mode='lines', line=dict(color=color, width=1.5),
                                fill='tozeroy',
                                fillcolor=f"rgba({'76,175,80' if final >= 0 else '244,67,54'}, 0.08)",
                                showlegend=False
                            ))
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
                            fig.update_layout(
                                height=100, margin=dict(l=0, r=0, t=0, b=0),
                                xaxis=dict(visible=False), yaxis=dict(visible=False),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"port_eq_{pid}")
                    except Exception:
                        pass

            # Action buttons
            btn_cols = st.columns([1, 1, 1, 1, 4])
            with btn_cols[0]:
                if st.button("View", key=f"port_view_{pid}"):
                    st.session_state.viewing_portfolio_id = pid
                    st.rerun()
            with btn_cols[1]:
                if st.button("Edit", key=f"port_edit_{pid}"):
                    st.session_state.editing_portfolio_id = pid
                    st.rerun()
            with btn_cols[2]:
                if st.button("Clone", key=f"port_clone_{pid}"):
                    new = duplicate_portfolio(pid)
                    if new:
                        st.toast(f"Cloned as '{new['name']}'")
                        st.rerun()
            with btn_cols[3]:
                if st.button("Delete", key=f"port_del_{pid}", type="secondary"):
                    st.session_state.confirm_delete_portfolio_id = pid
                    st.rerun()

            # Inline delete confirmation
            if st.session_state.confirm_delete_portfolio_id == pid:
                st.warning(f"Delete '{port['name']}'? This cannot be undone.")
                dc = st.columns([1, 1, 6])
                with dc[0]:
                    if st.button("Yes, Delete", key=f"port_cdel_{pid}", type="primary"):
                        delete_portfolio(pid)
                        st.session_state.confirm_delete_portfolio_id = None
                        st.rerun()
                with dc[1]:
                    if st.button("Cancel", key=f"port_cancel_del_{pid}"):
                        st.session_state.confirm_delete_portfolio_id = None
                        st.rerun()


def render_portfolio_builder():
    """Render the portfolio create/edit form."""
    editing_id = st.session_state.editing_portfolio_id
    is_edit = editing_id is not None

    if is_edit:
        existing = get_portfolio_by_id(editing_id)
        if existing is None:
            st.error("Portfolio not found.")
            st.session_state.editing_portfolio_id = None
            st.rerun()
            return
        st.header(f"Edit Portfolio: {existing['name']}")
    else:
        existing = None
        st.header("New Portfolio")

    if st.button("Cancel"):
        st.session_state.creating_portfolio = False
        st.session_state.editing_portfolio_id = None
        st.rerun()

    # Portfolio settings
    st.subheader("Portfolio Settings")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Portfolio Name",
                              value=existing['name'] if existing else "My Portfolio")
        starting_balance = st.number_input(
            "Starting Balance ($)", min_value=1000.0, step=1000.0,
            value=existing.get('starting_balance', 25000.0) if existing else 25000.0)
    with col2:
        compound_pct = st.slider(
            "Risk Scaling (Compound Rate)",
            min_value=0, max_value=100, step=5,
            value=int((existing.get('compound_rate', 0.0) if existing else 0.0) * 100),
            help="0% = fixed risk per trade. 100% = risk scales 1:1 with account growth. "
                 "E.g., at 50%, if your account is up 20%, risk per trade increases by 10%."
        )
        compound_rate = compound_pct / 100.0

        prop_firm_options = {"": "None", "ttp": "Trade The Pool", "ftmo": "FTMO"}
        current_firm = existing.get('prop_firm', '') if existing else ''
        prop_firm = st.selectbox(
            "Prop Firm (for compliance check)",
            options=list(prop_firm_options.keys()),
            format_func=lambda k: prop_firm_options[k],
            index=list(prop_firm_options.keys()).index(current_firm) if current_firm in prop_firm_options else 0
        )

    # Strategy selection
    st.subheader("Select Strategies")

    all_strategies = load_strategies()
    modern_strategies = [s for s in all_strategies if 'entry_trigger_confluence_id' in s]

    if len(modern_strategies) == 0:
        st.warning("No strategies available. Create strategies in the Strategy Builder first.")
        return

    # Build strategy options
    strat_options = {s['id']: f"{s['name']} ({s['symbol']} {s['direction']})" for s in modern_strategies}

    # Pre-populate if editing
    if existing:
        default_ids = [ps['strategy_id'] for ps in existing.get('strategies', [])
                       if ps['strategy_id'] in strat_options]
        existing_risks = {ps['strategy_id']: ps['risk_per_trade']
                         for ps in existing.get('strategies', [])}
    else:
        default_ids = []
        existing_risks = {}

    selected_ids = st.multiselect(
        "Choose strategies to include",
        options=list(strat_options.keys()),
        default=default_ids,
        format_func=lambda sid: strat_options[sid]
    )

    # Risk per trade for each selected strategy
    strategy_configs = []
    if selected_ids:
        st.markdown("**Risk Per Trade**")
        for sid in selected_ids:
            strat = next((s for s in modern_strategies if s['id'] == sid), None)
            if strat is None:
                continue
            default_risk = existing_risks.get(sid, strat.get('risk_per_trade', 100.0))
            risk = st.number_input(
                f"{strat['name']}",
                min_value=1.0, step=10.0,
                value=float(default_risk),
                key=f"port_risk_{sid}"
            )
            strategy_configs.append({'strategy_id': sid, 'risk_per_trade': risk})

    # Preview
    if strategy_configs:
        with st.expander("Preview Combined KPIs"):
            preview_portfolio = {
                'starting_balance': starting_balance,
                'compound_rate': compound_rate,
                'strategies': strategy_configs,
            }
            with st.spinner("Computing..."):
                data = get_portfolio_trades(preview_portfolio, get_strategy_by_id, get_strategy_trades)
                kpis = calculate_portfolio_kpis(preview_portfolio, data['combined_trades'], data['daily_pnl'])

            kpi_cols = st.columns(6)
            kpi_cols[0].metric("Trades", kpis['total_trades'])
            kpi_cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
            pf = kpis['profit_factor']
            kpi_cols[2].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
            kpi_cols[3].metric("Total P&L", f"${kpis['total_pnl']:+,.0f}")
            kpi_cols[4].metric("Final Balance", f"${kpis['final_balance']:,.0f}")
            max_dd = kpis['max_drawdown_pct']
            kpi_cols[5].metric("Max Drawdown", f"{max_dd:.1f}%")

    st.divider()

    # Save button
    btn_label = "Update Portfolio" if is_edit else "Save Portfolio"
    if st.button(btn_label, type="primary", disabled=len(strategy_configs) == 0):
        # Build portfolio dict
        portfolio = {
            'name': name,
            'starting_balance': starting_balance,
            'compound_rate': compound_rate,
            'strategies': strategy_configs,
            'prop_firm': prop_firm if prop_firm else None,
            'custom_rules': existing.get('custom_rules', []) if existing else [],
        }

        # Compute and cache KPIs
        data = get_portfolio_trades(portfolio, get_strategy_by_id, get_strategy_trades)
        portfolio['cached_kpis'] = calculate_portfolio_kpis(portfolio, data['combined_trades'], data['daily_pnl'])

        if is_edit:
            update_portfolio(editing_id, portfolio)
            st.toast("Portfolio updated!")
        else:
            save_portfolio(portfolio)
            st.toast("Portfolio saved!")

        st.session_state.creating_portfolio = False
        st.session_state.editing_portfolio_id = None
        st.rerun()


def render_portfolio_detail(portfolio_id: int):
    """Render full portfolio detail with tabbed view."""
    port = get_portfolio_by_id(portfolio_id)
    if port is None:
        st.error("Portfolio not found.")
        st.session_state.viewing_portfolio_id = None
        st.rerun()
        return

    # Back button
    if st.button("← Back to Portfolios"):
        st.session_state.viewing_portfolio_id = None
        st.rerun()

    # Header
    st.header(port['name'])
    n_strats = len(port.get('strategies', []))
    balance = port.get('starting_balance', 10000)
    compound = port.get('compound_rate', 0) * 100
    meta = f"{n_strats} strategies | ${balance:,.0f} starting balance"
    if compound > 0:
        meta += f" | {compound:.0f}% risk scaling"
    st.caption(meta)

    # Action bar
    action_cols = st.columns([1, 1, 1, 5])
    with action_cols[0]:
        if st.button("Edit Portfolio", key="pdetail_edit"):
            st.session_state.editing_portfolio_id = portfolio_id
            st.session_state.viewing_portfolio_id = None
            st.rerun()
    with action_cols[1]:
        if st.button("Clone", key="pdetail_clone"):
            new = duplicate_portfolio(portfolio_id)
            if new:
                st.toast(f"Cloned as '{new['name']}'")
                st.rerun()
    with action_cols[2]:
        if st.button("Delete", key="pdetail_del", type="secondary"):
            st.session_state.confirm_delete_portfolio_id = portfolio_id
            st.rerun()

    # Inline delete confirmation
    if st.session_state.confirm_delete_portfolio_id == portfolio_id:
        st.warning(f"Delete '{port['name']}'? This cannot be undone.")
        dc = st.columns([1, 1, 6])
        with dc[0]:
            if st.button("Yes, Delete", key="pdetail_cdel", type="primary"):
                delete_portfolio(portfolio_id)
                st.session_state.confirm_delete_portfolio_id = None
                st.session_state.viewing_portfolio_id = None
                st.rerun()
        with dc[1]:
            if st.button("Cancel", key="pdetail_cancel_del"):
                st.session_state.confirm_delete_portfolio_id = None
                st.rerun()

    st.divider()

    # Compute portfolio data
    with st.spinner("Computing portfolio analytics..."):
        data = get_portfolio_trades(port, get_strategy_by_id, get_strategy_trades)
        kpis = calculate_portfolio_kpis(port, data['combined_trades'], data['daily_pnl'])

    if len(data['combined_trades']) == 0:
        st.warning("No trades generated. Some strategies may reference disabled confluence groups.")
        return

    drawdown = compute_drawdown_series(data['combined_trades'], port['starting_balance'])

    # Tabs
    tab_perf, tab_strats, tab_prop, tab_deploy = st.tabs(
        ["Performance", "Strategies", "Prop Firm Check", "Deploy"]
    )

    with tab_perf:
        render_portfolio_performance(port, kpis, data, drawdown)

    with tab_strats:
        render_portfolio_strategies(port, data)

    with tab_prop:
        render_portfolio_prop_firm(port, kpis, data['daily_pnl'])

    with tab_deploy:
        render_portfolio_deploy(port)


def render_portfolio_performance(port, kpis, data, drawdown):
    """Render the Performance tab."""
    # KPI row
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Trades", kpis['total_trades'])
    kpi_cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
    pf = kpis['profit_factor']
    kpi_cols[2].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
    kpi_cols[3].metric("Total P&L", f"${kpis['total_pnl']:+,.0f}")
    kpi_cols[4].metric("Final Balance", f"${kpis['final_balance']:,.0f}")
    kpi_cols[5].metric("Max Drawdown", f"{kpis['max_drawdown_pct']:.1f}%")

    # Combined equity curve (multi-line)
    st.subheader("Combined Equity Curve")
    combined = data['combined_trades']
    fig = go.Figure()

    # Per-strategy lines (dashed, lighter)
    colors = ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#EF5350", "#26C6DA", "#8D6E63", "#78909C"]
    for i, (sid, strat_trades) in enumerate(data['strategy_trades'].items()):
        if len(strat_trades) == 0:
            continue
        sorted_t = strat_trades.sort_values('exit_time')
        # Use base risk for per-strategy equity (no compounding isolation)
        ps_config = next((ps for ps in port['strategies'] if ps['strategy_id'] == sid), None)
        if ps_config:
            cum_pnl = (sorted_t['r_multiple'] * ps_config['risk_per_trade']).cumsum()
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=sorted_t['exit_time'], y=cum_pnl,
                mode='lines', name=strat_trades['strategy_name'].iloc[0],
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.6
            ))

    # Combined line (bold)
    fig.add_trace(go.Scatter(
        x=combined['exit_time'], y=combined['cumulative_pnl'],
        mode='lines', name='Combined Portfolio',
        line=dict(color='white', width=2.5)
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="", yaxis_title="Cumulative P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    st.subheader("Drawdown Analysis")
    if len(drawdown) > 0:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown['exit_time'], y=drawdown['drawdown_pct'],
            mode='lines', name='Drawdown',
            line=dict(color='#f44336', width=1.5),
            fill='tozeroy', fillcolor='rgba(244, 67, 54, 0.15)'
        ))

        # Prop firm limit line
        firm_key = port.get('prop_firm')
        if firm_key and firm_key in PROP_FIRM_RULES:
            for rule in PROP_FIRM_RULES[firm_key]['rules']:
                if rule['type'] == 'max_total_drawdown_pct':
                    fig_dd.add_hline(
                        y=-rule['value'], line_dash="dash", line_color="orange",
                        annotation_text=f"{PROP_FIRM_RULES[firm_key]['name']} Limit",
                        annotation_font_color="orange"
                    )

        fig_dd.add_hline(y=0, line_color="gray", opacity=0.3)
        fig_dd.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="", yaxis_title="Drawdown %",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        dd_cols = st.columns(3)
        dd_cols[0].caption(f"Max DD: {kpis['max_drawdown_pct']:.1f}% (${kpis['max_drawdown_dollars']:,.0f})")
        dd_cols[1].caption(f"Profitable Days: {kpis['profitable_days_pct']:.0f}% ({kpis['profitable_days_count']}/{kpis['total_trading_days']})")
        dd_cols[2].caption(f"Avg Daily P&L: ${kpis['avg_daily_pnl']:+,.0f} (Std: ${kpis['daily_pnl_std']:,.0f})")

    # Daily P&L distribution
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Daily P&L Distribution")
        daily = data['daily_pnl']
        if len(daily) > 0:
            fig_hist = px.histogram(
                daily, x='daily_pnl', nbins=20,
                labels={'daily_pnl': 'Daily P&L ($)'},
                color_discrete_sequence=['#2196F3']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    with chart_right:
        st.subheader("Strategy Correlation")
        corr = compute_strategy_correlation(data['strategy_daily_pnl'])
        if len(corr) >= 2:
            fig_corr = px.imshow(
                corr, text_auto='.2f',
                color_continuous_scale='RdYlGn', zmin=-1, zmax=1,
                aspect='auto'
            )
            fig_corr.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need 2+ strategies for correlation analysis.")


def render_portfolio_strategies(port, data):
    """Render the Strategies tab."""
    for ps in port.get('strategies', []):
        sid = ps['strategy_id']
        strat = get_strategy_by_id(sid)

        with st.container(border=True):
            if strat is None:
                st.warning(f"Strategy ID {sid} not found (may have been deleted).")
                continue

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**{strat['name']}**")
                st.caption(f"{strat['symbol']} | {strat['direction']} | Risk: ${ps['risk_per_trade']:.0f}/trade")

                # Per-strategy KPIs
                strat_trades = data['strategy_trades'].get(sid)
                if strat_trades is not None and len(strat_trades) > 0:
                    skpis = calculate_kpis(strat_trades)
                    kpi_cols = st.columns(4)
                    kpi_cols[0].metric("Trades", skpis['total_trades'])
                    kpi_cols[1].metric("Win Rate", f"{skpis['win_rate']:.1f}%")
                    kpi_cols[2].metric("Avg R", f"{skpis['avg_r']:+.2f}")
                    kpi_cols[3].metric("Total R", f"{skpis['total_r']:+.1f}")
                else:
                    st.caption("No trades generated.")

            with col2:
                if strat_trades is not None and len(strat_trades) > 0:
                    render_mini_equity_curve(strat_trades, key=f"port_strat_eq_{sid}")


def render_portfolio_prop_firm(port, kpis, daily_pnl):
    """Render the Prop Firm Check tab."""
    firm_options = list(PROP_FIRM_RULES.keys()) + ["custom"]
    firm_labels = {k: PROP_FIRM_RULES[k]['name'] for k in PROP_FIRM_RULES}
    firm_labels["custom"] = "Custom Rules"

    current_firm = port.get('prop_firm', '') or 'ttp'
    if current_firm not in firm_options:
        current_firm = 'ttp'

    firm_key = st.radio(
        "Select Prop Firm",
        firm_options,
        format_func=lambda k: firm_labels.get(k, k),
        horizontal=True,
        index=firm_options.index(current_firm)
    )

    st.divider()

    # Custom rules editor
    if firm_key == "custom":
        render_custom_rules_editor(port)
        st.divider()

    # Evaluate
    custom_rules = port.get('custom_rules', []) if firm_key == "custom" else None
    result = evaluate_prop_firm_rules(firm_key, port, kpis, daily_pnl, custom_rules=custom_rules)

    # Rules table
    st.subheader(f"{result['firm_name']} — Rules Compliance")

    if not result['rules']:
        st.info("No rules defined." if firm_key == "custom" else "No rules found for this firm.")
        return

    for r in result['rules']:
        cols = st.columns([3, 2, 2, 1])
        cols[0].markdown(f"**{r['name']}**")
        cols[1].markdown(f"Limit: {r['limit_display']}")
        cols[2].markdown(f"Yours: {r['value_display']}")
        if r['passed']:
            cols[3].markdown("**:green[PASS]**")
        else:
            cols[3].markdown("**:red[FAIL]**")

    st.divider()

    # Overall banner
    if result['overall_pass']:
        st.success(f"COMPLIANT — This portfolio passes all {result['firm_name']} rules.")
    else:
        failed = [r['name'] for r in result['rules'] if not r['passed']]
        st.error(f"NON-COMPLIANT — Failed: {', '.join(failed)}")

    # Margin of safety
    st.subheader("Margin of Safety")
    for r in result['rules']:
        if r['passed']:
            st.caption(f"{r['name']}: {r['margin']:+.1f} buffer")
        else:
            st.caption(f"{r['name']}: {r['margin']:+.1f} (needs improvement)")

    # Recommendations — check all firms
    st.subheader("Firm Compatibility")
    for fk, fdef in PROP_FIRM_RULES.items():
        other_result = evaluate_prop_firm_rules(fk, port, kpis, daily_pnl)
        if other_result['overall_pass']:
            st.markdown(f":green[Pass] — {fdef['name']}")
        else:
            failed = [r['name'] for r in other_result['rules'] if not r['passed']]
            st.markdown(f":red[Fail] — {fdef['name']} ({', '.join(failed)})")


def render_custom_rules_editor(port):
    """Render the custom prop firm rules editor."""
    custom_rules = port.get('custom_rules', [])

    st.subheader("Custom Rules")

    if custom_rules:
        for i, rule in enumerate(custom_rules):
            cols = st.columns([3, 2, 1])
            cols[0].markdown(f"**{rule['name']}** ({rule['type']})")
            cols[1].markdown(f"Value: {rule['value']}")
            if cols[2].button("Remove", key=f"rm_crule_{i}"):
                custom_rules.pop(i)
                port['custom_rules'] = custom_rules
                update_portfolio(port['id'], port)
                st.rerun()
    else:
        st.caption("No custom rules defined.")

    type_labels = {
        "min_profit_pct": "Minimum Profit %",
        "max_daily_loss_pct": "Maximum Daily Loss %",
        "max_total_drawdown_pct": "Maximum Total Drawdown %",
        "min_profitable_days": "Minimum Profitable Days",
        "min_trading_days": "Minimum Trading Days",
    }

    with st.expander("+ Add Rule"):
        rule_name = st.text_input("Rule Name", key="crule_name")
        rule_type = st.selectbox("Rule Type", list(type_labels.keys()),
                                  format_func=lambda t: type_labels[t], key="crule_type")
        rule_value = st.number_input("Value", min_value=0.0, step=0.5, key="crule_value")

        if st.button("Add Rule", key="crule_add"):
            if rule_name:
                new_rule = {'name': rule_name, 'type': rule_type, 'value': rule_value}
                if rule_type == 'min_profitable_days':
                    new_rule['threshold_pct'] = 0.5
                custom_rules.append(new_rule)
                port['custom_rules'] = custom_rules
                update_portfolio(port['id'], port)
                st.rerun()


def render_portfolio_deploy(port):
    """Render the Deploy tab placeholder."""
    st.info("Deploy functionality is coming soon.")
    st.caption("This tab will allow you to deploy your portfolio strategies as live alerts "
               "and connect to trading bots.")


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
