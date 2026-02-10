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
import subprocess
import signal as signal_module
import inspect

from data_loader import load_market_data, get_data_source, is_alpaca_configured
from indicators import (
    INDICATORS,
    run_all_indicators,
    run_indicators_for_group,
    get_indicator_overlay_config,
    get_available_overlay_indicators,
    INDICATOR_COLORS,
    calculate_ema,
    calculate_macd,
    calculate_vwap,
    calculate_volume_sma,
    calculate_atr,
)
from interpreters import (
    INTERPRETERS,
    run_all_interpreters,
    detect_all_triggers,
    get_confluence_records,
    get_available_triggers,
    interpret_ema_stack,
    interpret_macd_line,
    interpret_macd_histogram,
    interpret_vwap,
    interpret_rvol,
    detect_ema_triggers,
    detect_macd_triggers,
    detect_macd_hist_triggers,
    detect_vwap_triggers,
    detect_rvol_triggers,
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
    evaluate_requirement_set,
    load_requirements,
    get_requirement_set_by_id,
    save_requirement_set,
    update_requirement_set,
    delete_requirement_set,
    duplicate_requirement_set,
    compute_strategy_recommendations,
)
from alerts import (
    load_alert_config,
    save_alert_config,
    get_strategy_alert_config,
    set_strategy_alert_config,
    get_portfolio_alert_config,
    set_portfolio_alert_config,
    load_alerts,
    save_alert,
    acknowledge_alert,
    clear_alerts,
    get_alerts_for_strategy,
    get_alerts_for_portfolio,
    load_monitor_status,
    save_monitor_status,
    get_portfolio_webhooks,
    add_portfolio_webhook,
    update_portfolio_webhook,
    delete_portfolio_webhook,
    get_all_active_webhooks,
    get_active_alert_configs,
    get_webhook_delivery_log,
    build_placeholder_context,
    render_payload,
    send_webhook,
    PLACEHOLDER_CATALOG,
    load_webhook_templates,
    add_webhook_template,
    update_webhook_template,
    delete_webhook_template,
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
OVERLAY_COMPATIBLE_TEMPLATES = ["ema_stack", "vwap", "utbot"]

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
    Get the actual DataFrame column names for a confluence group's indicators.

    Resolves template abstract names (ema_short) to real column names (ema_9)
    based on the group's configured parameters.
    """
    template = get_template(group.base_template)
    if not template:
        return []

    if group.base_template == "ema_stack":
        short = group.parameters.get("short_period", 9)
        mid = group.parameters.get("mid_period", 21)
        long = group.parameters.get("long_period", 200)
        return [f"ema_{short}", f"ema_{mid}", f"ema_{long}"]

    # Other templates have indicator_columns that match actual DataFrame names
    return template.get("indicator_columns", [])


def get_overlay_colors_for_group(group: ConfluenceGroup) -> dict:
    """
    Get colors mapped to actual DataFrame column names for a confluence group.

    Returns dict mapping column_name -> color
    """
    template = get_template(group.base_template)
    if not template:
        return {}

    colors = {}

    if group.base_template == "ema_stack":
        short = group.parameters.get("short_period", 9)
        mid = group.parameters.get("mid_period", 21)
        long = group.parameters.get("long_period", 200)
        colors[f"ema_{short}"] = group.plot_settings.colors.get("short_color", "#22c55e")
        colors[f"ema_{mid}"] = group.plot_settings.colors.get("mid_color", "#eab308")
        colors[f"ema_{long}"] = group.plot_settings.colors.get("long_color", "#ef4444")
    elif group.base_template == "vwap":
        colors["vwap"] = group.plot_settings.colors.get("vwap_color", "#8b5cf6")
        colors["vwap_sd1_upper"] = group.plot_settings.colors.get("sd1_band_color", "#c4b5fd")
        colors["vwap_sd1_lower"] = group.plot_settings.colors.get("sd1_band_color", "#c4b5fd")
        colors["vwap_sd2_upper"] = group.plot_settings.colors.get("sd2_band_color", "#ddd6fe")
        colors["vwap_sd2_lower"] = group.plot_settings.colors.get("sd2_band_color", "#ddd6fe")
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

    # Run group-specific indicators for custom parameters (e.g., custom EMA periods)
    from confluence_groups import get_enabled_groups
    for group in get_enabled_groups(load_confluence_groups()):
        df = run_indicators_for_group(df, group)

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
    chart_key: str = 'price_chart',
    secondary_panes: list = None
):
    """
    Render TradingView-style candlestick chart with trade markers and indicator overlays.

    Supports synchronized secondary panes (e.g., MACD oscillator, RVOL) that share
    the same time axis with zoom/scroll sync, like TradingView.

    Args:
        df: DataFrame with OHLCV data and indicator columns (timestamp index)
        trades: DataFrame with trade entry/exit data
        config: Strategy config with 'direction' key
        show_indicators: List of indicator column names to overlay (e.g., ['ema_short', 'ema_mid'])
        indicator_colors: Dict mapping column names to colors (from confluence group settings)
        secondary_panes: List of lightweight-charts pane config dicts to render below the price chart
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
        "height": 350 if secondary_panes else 450
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

    # Build chart pane list (price chart + optional synced secondary panes)
    chart_panes = [{"chart": chart_options, "series": series}]
    if secondary_panes:
        chart_panes.extend(secondary_panes)

    renderLightweightCharts(chart_panes, key=chart_key)


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
    if 'portfolio_builder_strategies' not in st.session_state:
        st.session_state.portfolio_builder_strategies = []
    if 'builder_recommendations' not in st.session_state:
        st.session_state.builder_recommendations = None
    if 'strategy_trades_cache' not in st.session_state:
        st.session_state.strategy_trades_cache = {}
    if 'editing_requirement_id' not in st.session_state:
        st.session_state.editing_requirement_id = None
    if 'creating_requirement' not in st.session_state:
        st.session_state.creating_requirement = False
    if 'confirm_delete_requirement_id' not in st.session_state:
        st.session_state.confirm_delete_requirement_id = None
    if 'nav_target' not in st.session_state:
        st.session_state.nav_target = None

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

        nav_options = ["Dashboard", "Strategy Builder", "My Strategies", "Portfolios", "Portfolio Requirements", "Alerts & Signals", "Webhook Templates", "Confluence Groups"]
        default_idx = 0
        if st.session_state.nav_target and st.session_state.nav_target in nav_options:
            default_idx = nav_options.index(st.session_state.nav_target)
            st.session_state.nav_target = None

        page = st.radio(
            "Navigation",
            nav_options,
            index=default_idx
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
    if page == "Dashboard":
        render_dashboard()
    elif page == "Strategy Builder":
        render_strategy_builder(data_days, data_seed)
    elif page == "My Strategies":
        render_my_strategies()
    elif page == "Portfolios":
        render_portfolios()
    elif page == "Portfolio Requirements":
        render_requirements_page()
    elif page == "Alerts & Signals":
        render_alerts_page()
    elif page == "Webhook Templates":
        render_webhook_templates_page()
    else:
        render_confluence_groups()


def render_dashboard():
    """Render the Dashboard landing page."""
    st.header("Dashboard")

    strategies = load_strategies()
    portfolios = load_portfolios()

    # --- Empty State ---
    if not strategies:
        st.markdown("### Welcome to RoR Trader")
        st.markdown(
            "Build data-backed trading strategies without writing code. "
            "RoR Trader uses an **Indicator → Interpreter → Trigger** pipeline "
            "to eliminate subjective chart reading and quantify every trading decision."
        )
        st.markdown(
            "1. **Indicators** calculate values on price data (EMA, VWAP, RSI, etc.)\n"
            "2. **Interpreters** classify indicator states into clear conditions\n"
            "3. **Triggers** fire entry/exit signals when conditions align\n"
            "4. **Confluence** layers multiple conditions to filter for high-probability setups"
        )
        if st.button("Create Your First Strategy", type="primary"):
            st.session_state.nav_target = "Strategy Builder"
            st.rerun()
        return

    # --- Overview Cards ---
    forward_testing = [s for s in strategies if s.get('forward_testing')]
    alerts = load_alerts(limit=100)
    recent_alerts = [
        a for a in alerts
        if a.get('timestamp') and _is_within_days(a['timestamp'], 7)
    ]

    card_cols = st.columns(4)
    card_cols[0].metric("Strategies", len(strategies))
    card_cols[1].metric("Forward Testing", len(forward_testing))
    card_cols[2].metric("Portfolios", len(portfolios))
    card_cols[3].metric("Recent Alerts (7d)", len(recent_alerts))

    st.divider()

    # --- Two-Column Layout ---
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Best Performing Strategy (by Total R)
        best_strat = max(strategies, key=lambda s: s.get('kpis', {}).get('total_r', 0))
        best_kpis = best_strat.get('kpis', {})

        st.subheader("Top Strategy")
        st.markdown(f"**{best_strat['name']}**")
        st.caption(f"{best_strat.get('symbol', '?')} | {best_strat.get('direction', '?')}")

        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Win Rate", f"{best_kpis.get('win_rate', 0):.1f}%")
        pf = best_kpis.get('profit_factor', 0)
        kpi_cols[1].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
        kpi_cols[2].metric("Total R", f"{best_kpis.get('total_r', 0):+.1f}")

        # Mini equity curve for best strategy
        try:
            trades = get_strategy_trades(best_strat)
            if len(trades) > 0:
                boundary = None
                if best_strat.get('forward_testing') and best_strat.get('forward_test_start'):
                    boundary = datetime.fromisoformat(best_strat['forward_test_start'])
                render_mini_equity_curve(trades, key="dash_best_strat_eq", boundary_dt=boundary)
        except Exception:
            st.caption("Could not load equity curve")

        if st.button("View Strategy", key="dash_view_best_strat"):
            st.session_state.viewing_strategy_id = best_strat['id']
            st.session_state.nav_target = "My Strategies"
            st.rerun()

        # Best Performing Portfolio (by Total P&L)
        if portfolios:
            st.divider()
            best_port = max(portfolios, key=lambda p: p.get('cached_kpis', {}).get('total_pnl', 0))
            best_port_kpis = best_port.get('cached_kpis', {})

            st.subheader("Top Portfolio")
            st.markdown(f"**{best_port['name']}**")
            st.caption(f"{len(best_port.get('strategies', []))} strategies | ${best_port.get('starting_balance', 0):,.0f} starting balance")

            pkpi_cols = st.columns(3)
            pkpi_cols[0].metric("Total P&L", f"${best_port_kpis.get('total_pnl', 0):+,.0f}")
            pkpi_cols[1].metric("Max DD", f"{best_port_kpis.get('max_drawdown_pct', 0):.1f}%")
            pkpi_cols[2].metric("Win Rate", f"{best_port_kpis.get('win_rate', 0):.1f}%")

            if st.button("View Portfolio", key="dash_view_best_port"):
                st.session_state.viewing_portfolio_id = best_port['id']
                st.session_state.nav_target = "Portfolios"
                st.rerun()

    with right_col:
        # Recent Alerts
        st.subheader("Recent Alerts")
        last_alerts = load_alerts(limit=5)
        if last_alerts:
            for alert in last_alerts:
                _render_alert_row(alert, prefix="dash_")
            if st.button("View All Alerts", key="dash_view_alerts"):
                st.session_state.nav_target = "Alerts & Signals"
                st.rerun()
        else:
            st.caption("No alerts yet.")

        st.divider()

        # System Status
        st.subheader("System Status")

        # Data source
        if is_alpaca_configured():
            st.markdown(":green[Alpaca API] — Live market data")
        else:
            st.markdown(":orange[Mock Data] — Configure Alpaca for live data")

        # Alert monitor
        monitor = load_monitor_status()
        if monitor.get('running'):
            last_poll = monitor.get('last_poll', '')
            poll_time = ''
            if last_poll:
                try:
                    dt = datetime.fromisoformat(last_poll)
                    poll_time = f" (last poll: {dt.strftime('%H:%M:%S')})"
                except (ValueError, TypeError):
                    pass
            strats_mon = monitor.get('strategies_monitored', 0)
            st.markdown(f":green[Alert Monitor Running] — {strats_mon} strategies{poll_time}")
        else:
            st.markdown(":gray[Alert Monitor Stopped]")

        # Forward tests
        st.markdown(f"**{len(forward_testing)}** strategies in forward testing")

    # --- Quick Actions ---
    st.divider()
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("New Strategy", use_container_width=True):
            st.session_state.nav_target = "Strategy Builder"
            st.session_state.step = 1
            st.rerun()
    with action_cols[1]:
        if st.button("View Strategies", use_container_width=True):
            st.session_state.nav_target = "My Strategies"
            st.rerun()
    with action_cols[2]:
        if st.button("View Portfolios", use_container_width=True):
            st.session_state.nav_target = "Portfolios"
            st.rerun()


def _is_within_days(timestamp_str: str, days: int) -> bool:
    """Check if an ISO timestamp string is within the last N days."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return (datetime.now() - dt).days <= days
    except (ValueError, TypeError):
        return False


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
                overlay_groups = [g for g in enabled_groups if g.base_template in OVERLAY_COMPATIBLE_TEMPLATES]

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
    tab_backtest, tab_confluence, tab_config, tab_alerts = st.tabs(["Backtest Results", "Confluence Analysis", "Configuration", "Alerts"])

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
        overlay_groups = [g for g in enabled_groups if g.base_template in OVERLAY_COMPATIBLE_TEMPLATES]
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

    with tab_confluence:
        render_confluence_analysis_tab(df, strat)

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

    with tab_alerts:
        render_strategy_alerts_tab(strat)


# =============================================================================
# CONFLUENCE ANALYSIS TAB (shared by backtest + forward test detail views)
# =============================================================================

def render_confluence_analysis_tab(df: pd.DataFrame, strat: dict):
    """
    Render interpreter states and trigger events for each confluence group
    used by this strategy, organized as sub-tabs by group.

    Each sub-tab shows a relevant chart followed by interpreter state
    timeline and trigger event tables.
    """
    enabled_groups = get_enabled_groups()

    if not enabled_groups:
        st.info("No confluence groups are enabled.")
        return

    # Create sub-tabs, one per enabled confluence group
    group_tabs = st.tabs([g.name for g in enabled_groups])

    for group, gtab in zip(enabled_groups, group_tabs):
        with gtab:
            template = get_template(group.base_template)
            if not template:
                st.caption(f"Template '{group.base_template}' not found.")
                continue

            # Show group's active parameters
            param_schema = template.get("parameters_schema", {})
            param_parts = []
            for key, schema in param_schema.items():
                value = group.parameters.get(key, schema.get("default", "?"))
                param_parts.append(f"{schema.get('label', key)}: **{value}**")
            if param_parts:
                st.caption(" | ".join(param_parts))

            # Chart relevant to this confluence group
            # Overlay templates get indicator lines on the price chart;
            # oscillator templates get a synced secondary pane below.
            grp_indicators = []
            grp_colors = {}
            secondary_panes = []

            if group.base_template in OVERLAY_COMPATIBLE_TEMPLATES:
                grp_indicators = get_overlay_indicators_for_group(group)
                grp_colors = get_overlay_colors_for_group(group)
            elif group.base_template in ("macd_line", "macd_histogram"):
                if "macd_line" in df.columns:
                    secondary_panes.append(_build_macd_lwc_pane(df, group))
            elif group.base_template == "rvol":
                if "rvol" in df.columns:
                    secondary_panes.append(_build_rvol_lwc_pane(df, group))

            render_price_chart(
                df,
                pd.DataFrame(),  # no trade markers
                strat,
                show_indicators=grp_indicators,
                indicator_colors=grp_colors,
                chart_key=f"confluence_chart_{group.id}",
                secondary_panes=secondary_panes if secondary_panes else None
            )

            # Interpreter state timeline
            st.markdown("**Interpreter States**")
            _render_interpreter_timeline(df, group, template)

            # Trigger events
            st.markdown("**Trigger Events**")
            _render_trigger_events_table(df, group, template)


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
    tab_charts, tab_trades, tab_confluence_ft, tab_config, tab_alerts = st.tabs(
        ["Equity & Charts", "Trade History", "Confluence Analysis", "Configuration", "Alerts"]
    )

    with tab_charts:
        # Combined equity curve
        all_trades = pd.concat([backtest_trades, forward_trades], ignore_index=True)
        render_combined_equity_curve(all_trades, boundary_dt)

        # R-distribution comparison
        render_r_distribution_comparison(backtest_trades, forward_trades)

        # Price chart
        st.subheader("Price Chart")
        enabled_groups = get_enabled_groups()
        overlay_groups = [g for g in enabled_groups if g.base_template in OVERLAY_COMPATIBLE_TEMPLATES]
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
        render_price_chart(
            df, all_trades, strat,
            show_indicators=[],
            indicator_colors={},
            chart_key='trade_history_chart'
        )

        render_split_trade_history(backtest_trades, forward_trades)

    with tab_confluence_ft:
        render_confluence_analysis_tab(df, strat)

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

    with tab_alerts:
        render_strategy_alerts_tab(strat)


def render_strategy_alerts_tab(strat: dict):
    """Render the Alerts tab for a strategy detail view."""
    strategy_id = strat['id']

    if not strat.get('forward_testing'):
        st.info("Enable forward testing on this strategy to use alerts.")
        return

    strat_cfg = get_strategy_alert_config(strategy_id)

    st.subheader("Signal Detection")

    col1, col2 = st.columns(2)
    with col1:
        alerts_on = st.toggle(
            "Alerts Enabled",
            value=strat_cfg.get('alerts_enabled', False),
            key=f"detail_alert_enabled_{strategy_id}",
        )
        entry_on = st.toggle(
            "Alert on Entry Signals",
            value=strat_cfg.get('alert_on_entry', True),
            key=f"detail_alert_entry_{strategy_id}",
            disabled=not alerts_on,
        )
        exit_on = st.toggle(
            "Alert on Exit Signals",
            value=strat_cfg.get('alert_on_exit', True),
            key=f"detail_alert_exit_{strategy_id}",
            disabled=not alerts_on,
        )

    with col2:
        st.info("Webhook delivery is configured per-portfolio on the portfolio's Webhooks tab.")

    if st.button("Save Alert Settings", key=f"save_strat_alert_{strategy_id}"):
        new_cfg = {
            'alerts_enabled': alerts_on,
            'alert_on_entry': entry_on,
            'alert_on_exit': exit_on,
        }
        set_strategy_alert_config(strategy_id, new_cfg)
        st.toast("Alert settings saved")
        st.rerun()

    # Recent alerts for this strategy
    st.divider()
    st.subheader("Recent Alerts")
    alerts = get_alerts_for_strategy(strategy_id, limit=20)

    if not alerts:
        st.caption("No alerts for this strategy yet.")
    else:
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            if alert_type == 'entry_signal':
                badge = ":green[ENTRY]"
            elif alert_type == 'exit_signal':
                badge = ":red[EXIT]"
            else:
                badge = f":gray[{alert_type.upper()}]"

            col_t, col_b, col_d = st.columns([2, 1, 5])
            with col_t:
                ts = alert.get('timestamp', '')
                try:
                    dt = datetime.fromisoformat(ts)
                    st.caption(dt.strftime("%m/%d %H:%M"))
                except (ValueError, TypeError):
                    st.caption(ts[:16] if ts else '?')
            with col_b:
                st.markdown(badge)
            with col_d:
                price = alert.get('price')
                st.markdown(
                    f"@ ${price:.2f}" if price else "Signal detected"
                )


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
            st.session_state.portfolio_builder_strategies = []
            st.session_state.builder_recommendations = None
            st.session_state.pop('_builder_initialized', None)
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

                # Requirement set badge
                req_id = port.get('requirement_set_id')
                if req_id:
                    rs = get_requirement_set_by_id(req_id)
                    if rs:
                        st.caption(f"Requirements: {rs['name']}")

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
                    st.session_state.portfolio_builder_strategies = []
                    st.session_state.builder_recommendations = None
                    st.session_state.pop('_builder_initialized', None)
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


def get_cached_strategy_trades(strat):
    """Get strategy trades with session state caching."""
    sid = strat['id']
    if sid not in st.session_state.strategy_trades_cache:
        st.session_state.strategy_trades_cache[sid] = get_strategy_trades(strat)
    return st.session_state.strategy_trades_cache[sid]


def render_portfolio_builder():
    """Render the interactive portfolio builder with live metrics."""
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
        # Initialize builder strategies from existing portfolio (once)
        if not st.session_state.get('_builder_initialized'):
            st.session_state.portfolio_builder_strategies = copy.deepcopy(existing.get('strategies', []))
            st.session_state._builder_initialized = True
    else:
        existing = None
        st.header("New Portfolio")

    if st.button("Cancel"):
        st.session_state.creating_portfolio = False
        st.session_state.editing_portfolio_id = None
        st.session_state.portfolio_builder_strategies = []
        st.session_state.builder_recommendations = None
        st.session_state.pop('_builder_initialized', None)
        st.rerun()

    # --- Settings row ---
    st.subheader("Settings")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        name = st.text_input("Portfolio Name",
                              value=existing['name'] if existing else "My Portfolio")
    with s2:
        starting_balance = st.number_input(
            "Starting Balance ($)", min_value=1000.0, step=1000.0,
            value=existing.get('starting_balance', 25000.0) if existing else 25000.0)
    with s3:
        compound_pct = st.slider(
            "Risk Scaling (%)",
            min_value=0, max_value=100, step=5,
            value=int((existing.get('compound_rate', 0.0) if existing else 0.0) * 100),
            help="0% = fixed risk per trade. 100% = risk scales 1:1 with account growth."
        )
        compound_rate = compound_pct / 100.0
    with s4:
        req_sets = load_requirements()
        req_options = {0: "None"} | {r['id']: r['name'] for r in req_sets}
        current_req_id = existing.get('requirement_set_id', 0) if existing else 0
        if current_req_id is None:
            current_req_id = 0
        req_set_id = st.selectbox(
            "Requirement Set",
            options=list(req_options.keys()),
            format_func=lambda k: req_options[k],
            index=list(req_options.keys()).index(current_req_id) if current_req_id in req_options else 0
        )

    builder_strategies = st.session_state.portfolio_builder_strategies

    # --- Compute live metrics ---
    preview_portfolio = {
        'starting_balance': starting_balance,
        'compound_rate': compound_rate,
        'strategies': builder_strategies,
    }

    data = None
    kpis = None
    if builder_strategies:
        data = get_portfolio_trades(preview_portfolio, get_strategy_by_id, get_cached_strategy_trades)
        if len(data['combined_trades']) > 0:
            kpis = calculate_portfolio_kpis(preview_portfolio, data['combined_trades'], data['daily_pnl'])

    # --- Live KPI row ---
    st.subheader("Live Metrics")
    if kpis:
        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Trades", kpis['total_trades'])
        kpi_cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
        pf = kpis['profit_factor']
        kpi_cols[2].metric("Profit Factor", "Inf" if pf == float('inf') else f"{pf:.2f}")
        kpi_cols[3].metric("Total P&L", f"${kpis['total_pnl']:+,.0f}")
        kpi_cols[4].metric("Final Balance", f"${kpis['final_balance']:,.0f}")
        kpi_cols[5].metric("Max Drawdown", f"{kpis['max_drawdown_pct']:.1f}%")
    else:
        st.caption("Add strategies to see portfolio analytics.")

    # --- Charts (left) + Strategy Management (right) ---
    chart_col, mgmt_col = st.columns([3, 2])

    with chart_col:
        if data and kpis and len(data['combined_trades']) > 0:
            # Combined equity curve
            st.markdown("**Combined Equity Curve**")
            combined = data['combined_trades']
            fig = go.Figure()

            colors = ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#EF5350", "#26C6DA"]
            for i, (sid, strat_trades) in enumerate(data['strategy_trades'].items()):
                if len(strat_trades) == 0:
                    continue
                sorted_t = strat_trades.sort_values('exit_time')
                ps_config = next((ps for ps in builder_strategies if ps['strategy_id'] == sid), None)
                if ps_config:
                    cum_pnl = (sorted_t['r_multiple'] * ps_config['risk_per_trade']).cumsum()
                    fig.add_trace(go.Scatter(
                        x=sorted_t['exit_time'], y=cum_pnl,
                        mode='lines', name=strat_trades['strategy_name'].iloc[0],
                        line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                        opacity=0.6
                    ))

            fig.add_trace(go.Scatter(
                x=combined['exit_time'], y=combined['cumulative_pnl'],
                mode='lines', name='Combined Portfolio',
                line=dict(color='white', width=2.5)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="", yaxis_title="Cumulative P&L ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True, key="builder_equity")

            # Drawdown chart
            st.markdown("**Drawdown**")
            dd = compute_drawdown_series(combined, starting_balance)
            if len(dd) > 0:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd['exit_time'], y=dd['drawdown_pct'],
                    mode='lines', name='Drawdown',
                    line=dict(color='#f44336', width=1.5),
                    fill='tozeroy', fillcolor='rgba(244, 67, 54, 0.15)'
                ))
                fig_dd.add_hline(y=0, line_color="gray", opacity=0.3)
                fig_dd.update_layout(
                    height=200, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="", yaxis_title="Drawdown %",
                )
                st.plotly_chart(fig_dd, use_container_width=True, key="builder_dd")

    with mgmt_col:
        # --- Add Strategy ---
        st.markdown("**Add Strategy**")
        all_strategies = load_strategies()
        modern_strategies = [s for s in all_strategies if 'entry_trigger_confluence_id' in s]
        current_ids = {ps['strategy_id'] for ps in builder_strategies}
        available = [s for s in modern_strategies if s['id'] not in current_ids]

        if available:
            strat_options = {s['id']: f"{s['name']} ({s['symbol']} {s['direction']})" for s in available}
            selected_strat_id = st.selectbox(
                "Choose Strategy",
                options=list(strat_options.keys()),
                format_func=lambda sid: strat_options[sid],
                key="builder_add_strat"
            )
            add_risk = st.number_input("Risk Per Trade ($)", min_value=1.0, step=10.0,
                                       value=100.0, key="builder_add_risk")
            if st.button("Add Strategy", type="primary", key="builder_add_btn"):
                st.session_state.portfolio_builder_strategies.append({
                    'strategy_id': selected_strat_id,
                    'risk_per_trade': add_risk,
                })
                st.session_state.builder_recommendations = None
                st.rerun()
        elif len(modern_strategies) == 0:
            st.caption("No strategies available. Create strategies first.")
        else:
            st.caption("All available strategies are already added.")

        # --- Current Strategies ---
        st.markdown("**Current Strategies**")
        if not builder_strategies:
            st.caption("No strategies added yet.")
        for i, ps in enumerate(builder_strategies):
            strat = get_strategy_by_id(ps['strategy_id'])
            if strat is None:
                st.warning(f"Strategy {ps['strategy_id']} not found.")
                continue
            with st.container(border=True):
                rc1, rc2, rc3 = st.columns([3, 2, 0.5])
                rc1.markdown(f"**{strat['symbol']} {strat['direction']}**")
                rc1.caption(strat['name'])
                new_risk = rc2.number_input(
                    "$/trade", min_value=1.0, step=10.0,
                    value=float(ps['risk_per_trade']),
                    key=f"builder_risk_{ps['strategy_id']}",
                    label_visibility="collapsed"
                )
                if new_risk != ps['risk_per_trade']:
                    st.session_state.portfolio_builder_strategies[i]['risk_per_trade'] = new_risk
                if rc3.button("x", key=f"builder_rm_{i}"):
                    st.session_state.portfolio_builder_strategies.pop(i)
                    st.session_state.builder_recommendations = None
                    st.rerun()

        # --- Recommendations ---
        if builder_strategies and available:
            with st.expander("Strategy Recommendations"):
                if st.button("Analyze Recommendations", key="builder_rec_btn"):
                    with st.spinner("Evaluating candidates..."):
                        recs = compute_strategy_recommendations(
                            preview_portfolio, data if data else {'combined_trades': pd.DataFrame(), 'daily_pnl': pd.DataFrame(), 'strategy_daily_pnl': pd.DataFrame(), 'strategy_trades': {}, 'equity_curve': pd.Series(dtype=float)},
                            available, get_strategy_by_id, get_cached_strategy_trades, top_n=5,
                        )
                        st.session_state.builder_recommendations = recs

                recs = st.session_state.builder_recommendations
                if recs:
                    for rec in recs:
                        with st.container(border=True):
                            c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1])
                            c1.markdown(f"**{rec['strategy_name']}**")
                            c2.caption(f"P&L: {rec['pnl_change']:+,.0f}")
                            c3.caption(f"DD: {rec['dd_change']:+.1f}%")
                            if c4.button("Add", key=f"rec_add_{rec['strategy_id']}"):
                                s = get_strategy_by_id(rec['strategy_id'])
                                st.session_state.portfolio_builder_strategies.append({
                                    'strategy_id': rec['strategy_id'],
                                    'risk_per_trade': s.get('risk_per_trade', 100.0) if s else 100.0,
                                })
                                st.session_state.builder_recommendations = None
                                st.rerun()
                elif recs is not None:
                    st.caption("No recommendations available.")

    st.divider()

    # --- Save button ---
    btn_label = "Update Portfolio" if is_edit else "Save Portfolio"
    if st.button(btn_label, type="primary", disabled=len(builder_strategies) == 0):
        portfolio = {
            'name': name,
            'starting_balance': starting_balance,
            'compound_rate': compound_rate,
            'strategies': builder_strategies,
            'requirement_set_id': req_set_id if req_set_id else None,
        }

        # Compute and cache KPIs
        save_data = get_portfolio_trades(portfolio, get_strategy_by_id, get_strategy_trades)
        portfolio['cached_kpis'] = calculate_portfolio_kpis(portfolio, save_data['combined_trades'], save_data['daily_pnl'])

        if is_edit:
            update_portfolio(editing_id, portfolio)
            st.toast("Portfolio updated!")
        else:
            save_portfolio(portfolio)
            st.toast("Portfolio saved!")

        st.session_state.creating_portfolio = False
        st.session_state.editing_portfolio_id = None
        st.session_state.portfolio_builder_strategies = []
        st.session_state.builder_recommendations = None
        st.session_state.pop('_builder_initialized', None)
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
    tab_perf, tab_strats, tab_prop, tab_webhooks, tab_deploy = st.tabs(
        ["Performance", "Strategies", "Prop Firm Check", "Webhooks", "Deploy"]
    )

    with tab_perf:
        render_portfolio_performance(port, kpis, data, drawdown)

    with tab_strats:
        render_portfolio_strategies(port, data)

    with tab_prop:
        render_portfolio_prop_firm(port, kpis, data['daily_pnl'])

    with tab_webhooks:
        render_portfolio_webhooks(port)

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

        # Requirement set limit line
        req_id = port.get('requirement_set_id')
        if req_id:
            rs = get_requirement_set_by_id(req_id)
            if rs:
                for rule in rs.get('rules', []):
                    if rule['type'] == 'max_total_drawdown_pct':
                        fig_dd.add_hline(
                            y=-rule['value'], line_dash="dash", line_color="orange",
                            annotation_text=f"{rs['name']} Limit",
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
    """Render the Prop Firm Check tab using requirement sets."""
    req_sets = load_requirements()

    if not req_sets:
        st.info("No requirement sets available. Create one on the Portfolio Requirements page.")
        return

    # Build options
    req_options = {r['id']: r['name'] for r in req_sets}
    current_req_id = port.get('requirement_set_id')
    if current_req_id not in req_options:
        current_req_id = req_sets[0]['id']

    selected_req_id = st.selectbox(
        "Select Requirement Set",
        options=list(req_options.keys()),
        format_func=lambda k: req_options[k],
        index=list(req_options.keys()).index(current_req_id) if current_req_id in req_options else 0
    )

    st.divider()

    # Evaluate
    req_set = get_requirement_set_by_id(selected_req_id)
    if req_set is None:
        st.error("Requirement set not found.")
        return

    result = evaluate_requirement_set(req_set, port, kpis, daily_pnl)

    # Rules table
    st.subheader(f"{result['firm_name']} — Rules Compliance")

    if not result['rules']:
        st.info("No rules defined in this requirement set.")
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

    # Compatibility — check all requirement sets
    st.subheader("Requirement Set Compatibility")
    for rs in req_sets:
        other_result = evaluate_requirement_set(rs, port, kpis, daily_pnl)
        if other_result['overall_pass']:
            st.markdown(f":green[Pass] — {rs['name']}")
        else:
            failed = [r['name'] for r in other_result['rules'] if not r['passed']]
            st.markdown(f":red[Fail] — {rs['name']} ({', '.join(failed)})")


def render_portfolio_deploy(port):
    """Render the Deploy tab with monitor status and recent alerts."""
    portfolio_id = port['id']

    # Monitor status inline
    status = load_monitor_status()
    is_running = status.get('running', False)
    if is_running and status.get('pid'):
        try:
            os.kill(status['pid'], 0)
        except (OSError, ProcessLookupError):
            is_running = False

    if is_running:
        st.success("Alert Monitor: Running")
    else:
        st.warning("Alert Monitor: Stopped — Go to Alerts & Signals page to start it.")

    st.divider()

    # Requirement set info
    req_id = port.get('requirement_set_id')
    if req_id:
        req_set = get_requirement_set_by_id(req_id)
        if req_set:
            st.caption(f"Compliance monitored against: **{req_set['name']}**")

    st.info("Configure webhooks on the **Webhooks** tab.")

    # Recent alerts
    st.divider()
    st.subheader("Recent Portfolio Alerts")
    alerts = get_alerts_for_portfolio(portfolio_id, limit=20)

    if not alerts:
        st.caption("No alerts for this portfolio yet.")
    else:
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            if alert_type == 'entry_signal':
                badge = ":green[ENTRY]"
            elif alert_type == 'exit_signal':
                badge = ":red[EXIT]"
            elif alert_type == 'compliance_breach':
                badge = ":orange[COMPLIANCE]"
            else:
                badge = f":gray[{alert_type.upper()}]"

            col_t, col_b, col_d = st.columns([2, 1, 5])
            with col_t:
                ts = alert.get('timestamp', '')
                try:
                    dt = datetime.fromisoformat(ts)
                    st.caption(dt.strftime("%m/%d %H:%M"))
                except (ValueError, TypeError):
                    st.caption(ts[:16] if ts else '?')
            with col_b:
                st.markdown(badge)
            with col_d:
                strategy_name = alert.get('strategy_name', '')
                price = alert.get('price')
                if alert_type == 'compliance_breach':
                    st.markdown(f"Rule: {alert.get('rule_name', '?')}")
                elif price:
                    st.markdown(f"{strategy_name} @ ${price:.2f}")
                else:
                    st.markdown(strategy_name or "Alert")


# =============================================================================
# PORTFOLIO REQUIREMENTS PAGE
# =============================================================================

RULE_TYPE_LABELS = {
    "min_profit_pct": "Minimum Profit %",
    "max_daily_loss_pct": "Maximum Daily Loss %",
    "max_total_drawdown_pct": "Maximum Total Drawdown %",
    "min_profitable_days": "Minimum Profitable Days",
    "min_trading_days": "Minimum Trading Days",
}


def render_requirements_page():
    """Render the Portfolio Requirements management page."""
    # Route to editor if creating/editing
    if st.session_state.creating_requirement:
        render_requirement_set_editor()
        return
    if st.session_state.editing_requirement_id is not None:
        render_requirement_set_editor(st.session_state.editing_requirement_id)
        return

    st.header("Portfolio Requirements")
    st.caption("Manage prop firm rule sets and custom requirement sets for portfolio compliance checking.")

    col_header, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("+ New Requirement Set", type="primary"):
            st.session_state.creating_requirement = True
            st.rerun()

    req_sets = load_requirements()

    if len(req_sets) == 0:
        st.info("No requirement sets yet.")
        return

    for rs in req_sets:
        rid = rs.get('id', 0)
        with st.container(border=True):
            info_col, action_col = st.columns([4, 2])

            with info_col:
                name_parts = [f"### {rs['name']}"]
                if rs.get('built_in'):
                    name_parts.append("  `Built-in`")
                st.markdown("".join(name_parts))
                n_rules = len(rs.get('rules', []))
                st.caption(f"{n_rules} rule{'s' if n_rules != 1 else ''}")

                # Show rules summary
                for rule in rs.get('rules', [])[:4]:
                    st.caption(f"  - {rule['name']}: {RULE_TYPE_LABELS.get(rule['type'], rule['type'])} = {rule['value']}")
                if n_rules > 4:
                    st.caption(f"  ... and {n_rules - 4} more")

            with action_col:
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button("Edit", key=f"req_edit_{rid}"):
                        st.session_state.editing_requirement_id = rid
                        st.rerun()
                with btn_cols[1]:
                    if st.button("Clone", key=f"req_clone_{rid}"):
                        new = duplicate_requirement_set(rid)
                        if new:
                            st.toast(f"Cloned as '{new['name']}'")
                            st.rerun()
                with btn_cols[2]:
                    if rs.get('built_in'):
                        st.button("Delete", key=f"req_del_{rid}", disabled=True,
                                  help="Built-in sets cannot be deleted")
                    else:
                        if st.button("Delete", key=f"req_del_{rid}", type="secondary"):
                            st.session_state.confirm_delete_requirement_id = rid
                            st.rerun()

            # Inline delete confirmation
            if st.session_state.confirm_delete_requirement_id == rid:
                st.warning(f"Delete '{rs['name']}'? This cannot be undone.")
                dc = st.columns([1, 1, 6])
                with dc[0]:
                    if st.button("Yes, Delete", key=f"req_cdel_{rid}", type="primary"):
                        delete_requirement_set(rid)
                        st.session_state.confirm_delete_requirement_id = None
                        st.rerun()
                with dc[1]:
                    if st.button("Cancel", key=f"req_cancel_del_{rid}"):
                        st.session_state.confirm_delete_requirement_id = None
                        st.rerun()


def render_requirement_set_editor(req_id=None):
    """Render the requirement set create/edit form."""
    is_edit = req_id is not None

    if is_edit:
        existing = get_requirement_set_by_id(req_id)
        if existing is None:
            st.error("Requirement set not found.")
            st.session_state.editing_requirement_id = None
            st.rerun()
            return
        st.header(f"Edit: {existing['name']}")
        is_built_in = existing.get('built_in', False)
    else:
        existing = None
        is_built_in = False
        st.header("New Requirement Set")

    if st.button("Cancel"):
        st.session_state.creating_requirement = False
        st.session_state.editing_requirement_id = None
        st.rerun()

    name = st.text_input("Requirement Set Name",
                          value=existing['name'] if existing else "My Requirements",
                          disabled=is_built_in)

    # Rules editor
    st.subheader("Rules")

    # Use session state to track rules being edited
    rules_key = f"_req_rules_{req_id if req_id else 'new'}"
    if rules_key not in st.session_state:
        st.session_state[rules_key] = copy.deepcopy(existing.get('rules', [])) if existing else []

    rules = st.session_state[rules_key]

    if rules:
        for i, rule in enumerate(rules):
            cols = st.columns([3, 2, 1, 0.5])
            cols[0].markdown(f"**{rule['name']}**")
            cols[1].caption(f"{RULE_TYPE_LABELS.get(rule['type'], rule['type'])} = {rule['value']}")
            if rule.get('threshold_pct'):
                cols[2].caption(f"Threshold: {rule['threshold_pct']}%")
            if cols[3].button("x", key=f"req_rm_rule_{i}"):
                rules.pop(i)
                st.session_state[rules_key] = rules
                st.rerun()
    else:
        st.caption("No rules defined yet.")

    # Add rule form
    with st.expander("+ Add Rule"):
        r1, r2 = st.columns(2)
        with r1:
            new_rule_name = st.text_input("Rule Name", key="new_req_rule_name")
            new_rule_type = st.selectbox("Rule Type", list(RULE_TYPE_LABELS.keys()),
                                          format_func=lambda t: RULE_TYPE_LABELS[t],
                                          key="new_req_rule_type")
        with r2:
            new_rule_value = st.number_input("Value", min_value=0.0, step=0.5, key="new_req_rule_value")
            if new_rule_type == 'min_profitable_days':
                new_threshold = st.number_input("Threshold %", min_value=0.0, step=0.1,
                                                value=0.5, key="new_req_threshold")

        if st.button("Add Rule", key="req_add_rule_btn"):
            if new_rule_name:
                new_rule = {'name': new_rule_name, 'type': new_rule_type, 'value': new_rule_value}
                if new_rule_type == 'min_profitable_days':
                    new_rule['threshold_pct'] = new_threshold
                rules.append(new_rule)
                st.session_state[rules_key] = rules
                st.rerun()

    st.divider()

    # Save
    btn_label = "Update Requirement Set" if is_edit else "Save Requirement Set"
    if st.button(btn_label, type="primary"):
        req_set = {
            'name': name if not is_built_in else existing['name'],
            'rules': rules,
        }

        if is_edit:
            update_requirement_set(req_id, req_set)
            st.toast("Requirement set updated!")
        else:
            save_requirement_set(req_set)
            st.toast("Requirement set saved!")

        # Cleanup
        st.session_state.pop(rules_key, None)
        st.session_state.creating_requirement = False
        st.session_state.editing_requirement_id = None
        st.rerun()


# =============================================================================
# ALERTS & SIGNALS PAGE
# =============================================================================

def render_alerts_page():
    """Render the redesigned Alerts & Signals page."""
    st.header("Alerts & Signals")
    st.caption("Monitor signal detection, alert history, and webhook delivery.")

    config = load_alert_config()
    status = load_monitor_status()

    # Monitor Status Bar (kept)
    _render_monitor_status_bar(status, config)

    st.divider()

    # Active Alerts Management (collapsible)
    with st.expander("Manage Active Alerts & Webhooks", expanded=False):
        _render_active_alerts_management(config)

    st.divider()

    # Main content tabs
    tab_strat, tab_port, tab_outbound, tab_inbound = st.tabs([
        "Strategy Alerts", "Portfolio Alerts", "Outbound Webhooks", "Inbound Webhooks"
    ])

    with tab_strat:
        _render_strategy_alerts_tab()

    with tab_port:
        _render_portfolio_alerts_tab()

    with tab_outbound:
        _render_outbound_webhooks_tab()

    with tab_inbound:
        _render_inbound_webhooks_tab()


def _render_monitor_status_bar(status: dict, config: dict):
    """Render the monitor status bar with start/stop controls."""
    is_running = status.get('running', False)

    # Verify PID is actually alive if status says running
    if is_running and status.get('pid'):
        try:
            os.kill(status['pid'], 0)  # signal 0 = check if process exists
        except (OSError, ProcessLookupError):
            is_running = False
            status['running'] = False
            save_monitor_status(status)

    col_status, col_info, col_action = st.columns([2, 4, 2])

    with col_status:
        if is_running:
            st.success("Monitor: Running")
        else:
            st.error("Monitor: Stopped")

    with col_info:
        if is_running:
            last_poll = status.get('last_poll', 'Never')
            strats = status.get('strategies_monitored', 0)
            duration = status.get('last_poll_duration_ms')
            info_parts = [f"Last poll: {last_poll}", f"Strategies: {strats}"]
            if duration is not None:
                info_parts.append(f"Duration: {duration}ms")
            st.caption(" | ".join(info_parts))

            errors = status.get('errors', [])
            if errors:
                st.warning(f"{len(errors)} recent error(s). Latest: {errors[-1].get('error', '?')}")
        else:
            if not config.get('global', {}).get('enabled', False):
                st.caption("Enable alerts in Global Settings to start the monitor.")
            else:
                st.caption("Monitor is not running. Click Start to begin polling.")

    with col_action:
        if is_running:
            if st.button("Stop Monitor", type="secondary", use_container_width=True):
                pid = status.get('pid')
                if pid:
                    try:
                        os.kill(pid, signal_module.SIGTERM)
                    except (OSError, ProcessLookupError):
                        pass
                status['running'] = False
                save_monitor_status(status)
                st.rerun()
        else:
            can_start = config.get('global', {}).get('enabled', False)
            if st.button("Start Monitor", type="primary", disabled=not can_start,
                         use_container_width=True):
                # Launch alert_monitor.py as a background process
                monitor_script = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "alert_monitor.py"
                )
                subprocess.Popen(
                    ["python", monitor_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                st.toast("Monitor starting...")
                import time
                time.sleep(1)  # brief delay for status file to be written
                st.rerun()


def _render_active_alerts_management(config: dict):
    """Show only enabled strategies/portfolios with their alert configs. Allow deactivation."""
    active = get_active_alert_configs()
    strategies = load_strategies()
    portfolios = load_portfolios()

    # Strategy name lookup
    strat_names = {s['id']: s.get('name', f"Strategy {s['id']}") for s in strategies}
    port_names = {p['id']: p.get('name', f"Portfolio {p['id']}") for p in portfolios}

    st.markdown("**Active Strategy Alerts**")
    if not active["strategies"]:
        st.caption("No strategies have alerts enabled.")
    else:
        for s in active["strategies"]:
            sid = s["strategy_id"]
            name = strat_names.get(sid, f"Strategy {sid}")
            col_n, col_e, col_x, col_d = st.columns([4, 1, 1, 1])
            with col_n:
                st.markdown(f"{name}")
            with col_e:
                st.caption("Entry" if s.get("alert_on_entry") else "~~Entry~~")
            with col_x:
                st.caption("Exit" if s.get("alert_on_exit") else "~~Exit~~")
            with col_d:
                if st.button("Disable", key=f"disable_strat_{sid}", type="secondary"):
                    set_strategy_alert_config(sid, {**s, "alerts_enabled": False})
                    st.rerun()

    st.divider()

    st.markdown("**Active Portfolio Alerts & Webhooks**")
    if not active["portfolios"]:
        st.caption("No portfolios have alerts enabled.")
    else:
        for p in active["portfolios"]:
            pid = p["portfolio_id"]
            name = port_names.get(pid, f"Portfolio {pid}")
            wh_count = len(p.get("webhooks", []))
            col_n, col_w, col_d = st.columns([4, 2, 1])
            with col_n:
                st.markdown(f"{name}")
            with col_w:
                st.caption(f"{wh_count} webhook(s)")
            with col_d:
                if st.button("Disable", key=f"disable_port_{pid}", type="secondary"):
                    set_portfolio_alert_config(pid, {**p, "alerts_enabled": False})
                    st.rerun()

    st.divider()

    # Global Monitor Settings (compact)
    st.markdown("**Global Monitor Settings**")
    global_cfg = config.get('global', {})
    col1, col2 = st.columns(2)
    with col1:
        enabled = st.toggle(
            "Alerts Enabled",
            value=global_cfg.get('enabled', False),
            key="mgmt_global_alerts_enabled",
        )
        market_hours = st.toggle(
            "Market Hours Only",
            value=global_cfg.get('market_hours_only', True),
            key="mgmt_global_market_hours",
        )
    with col2:
        poll_interval = st.slider(
            "Poll Interval (seconds)",
            min_value=30,
            max_value=300,
            value=global_cfg.get('poll_interval_seconds', 60),
            step=10,
            key="mgmt_global_poll_interval",
        )

    if st.button("Save Global Settings", key="mgmt_save_global"):
        config['global'] = {
            'poll_interval_seconds': poll_interval,
            'market_hours_only': market_hours,
            'enabled': enabled,
        }
        save_alert_config(config)
        st.toast("Global settings saved")
        st.rerun()


def _render_alert_row(alert: dict, prefix: str = ""):
    """Shared helper to render a single alert row."""
    alert_type = alert.get('type', 'unknown')

    if alert_type == 'entry_signal':
        badge = ":green[ENTRY]"
    elif alert_type == 'exit_signal':
        badge = ":red[EXIT]"
    elif alert_type == 'compliance_breach':
        badge = ":orange[COMPLIANCE]"
    else:
        badge = f":gray[{alert_type.upper()}]"

    with st.container():
        col_time, col_badge, col_detail = st.columns([2, 1, 6])

        with col_time:
            ts = alert.get('timestamp', '')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    st.caption(dt.strftime("%m/%d %H:%M"))
                except (ValueError, TypeError):
                    st.caption(ts[:16])

        with col_badge:
            st.markdown(badge)

        with col_detail:
            symbol = alert.get('symbol', '')
            direction = alert.get('direction', '')
            strategy_name = alert.get('strategy_name', '')
            price = alert.get('price')

            detail_parts = []
            if strategy_name:
                detail_parts.append(f"**{strategy_name}**")
            if symbol and direction:
                detail_parts.append(f"{symbol} {direction}")
            if price:
                detail_parts.append(f"@ ${price:.2f}")

            if alert_type == 'compliance_breach':
                port_name = alert.get('portfolio_name', '')
                rule_name = alert.get('rule_name', '')
                detail_parts = [f"**{port_name}**", f"Rule: {rule_name}"]

            st.markdown(" | ".join(detail_parts) if detail_parts else "Alert")

            webhook_sent = alert.get('webhook_sent')
            if webhook_sent:
                st.caption("Webhook sent")


def _filter_alerts_by_date(alerts: list, key_prefix: str) -> list:
    """Add date range filter controls and return filtered alerts."""
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From",
            value=datetime.now().date() - timedelta(days=30),
            key=f"{key_prefix}_start_date",
        )
    with col2:
        end_date = st.date_input(
            "To",
            value=datetime.now().date(),
            key=f"{key_prefix}_end_date",
        )

    filtered = []
    for a in alerts:
        ts = a.get('timestamp', '')
        if ts:
            try:
                alert_date = datetime.fromisoformat(ts).date()
                if start_date <= alert_date <= end_date:
                    filtered.append(a)
            except (ValueError, TypeError):
                filtered.append(a)  # include if we can't parse
        else:
            filtered.append(a)
    return filtered


def _render_strategy_alerts_tab():
    """Render the Strategy Alerts tab showing recent strategy-level alerts."""
    alerts = load_alerts(limit=500)
    # Only show alerts where strategy-level visibility is true (or old alerts without the flag)
    strategy_alerts = [a for a in alerts
                       if (a.get('type') in ('entry_signal', 'exit_signal'))
                       and a.get('strategy_alerts_visible', True)]

    # Date range filter
    strategy_alerts = _filter_alerts_by_date(strategy_alerts, "strat_alerts")

    st.caption(f"{len(strategy_alerts)} alert(s) in range")

    if not strategy_alerts:
        st.info("No strategy alerts in this date range.")
        return

    for alert in strategy_alerts:
        _render_alert_row(alert, prefix="strat_")


def _render_portfolio_alerts_tab():
    """Render the Portfolio Alerts tab showing portfolio-enriched signals and compliance breaches."""
    alerts = load_alerts(limit=500)

    # Collect all portfolio-related alerts (signals with context + compliance)
    portfolio_alerts = [a for a in alerts
                        if (a.get('type') in ('entry_signal', 'exit_signal') and a.get('portfolio_context'))
                        or a.get('type') == 'compliance_breach']

    # Date range filter
    portfolio_alerts = _filter_alerts_by_date(portfolio_alerts, "port_alerts")
    st.caption(f"{len(portfolio_alerts)} alert(s) in range")

    if not portfolio_alerts:
        st.info("No portfolio alerts in this date range.")
        return

    # Signal alerts expanded per portfolio
    signal_alerts = [a for a in portfolio_alerts if a.get('type') in ('entry_signal', 'exit_signal')]
    if signal_alerts:
        st.markdown("**Signal Alerts (by Portfolio)**")
        for alert in signal_alerts:
            for ctx in alert.get('portfolio_context', []):
                alert_type = alert.get('type', 'unknown')
                if alert_type == 'entry_signal':
                    badge = ":green[ENTRY]"
                elif alert_type == 'exit_signal':
                    badge = ":red[EXIT]"
                else:
                    badge = f":gray[{alert_type.upper()}]"

                with st.container():
                    col_t, col_b, col_d = st.columns([2, 1, 6])
                    with col_t:
                        ts = alert.get('timestamp', '')
                        try:
                            dt = datetime.fromisoformat(ts)
                            st.caption(dt.strftime("%m/%d %H:%M"))
                        except (ValueError, TypeError):
                            st.caption(ts[:16] if ts else '?')
                    with col_b:
                        st.markdown(badge)
                    with col_d:
                        strategy_name = alert.get('strategy_name', '')
                        price = alert.get('price')
                        port_name = ctx.get('portfolio_name', '')
                        risk = ctx.get('position_risk', '')
                        parts = [f"**{port_name}**"]
                        if strategy_name:
                            parts.append(strategy_name)
                        if price:
                            parts.append(f"@ ${price:.2f}")
                        if risk:
                            parts.append(f"Risk: ${risk:.0f}")
                        st.markdown(" | ".join(parts))

    # Compliance breaches
    compliance_alerts = [a for a in portfolio_alerts if a.get('type') == 'compliance_breach']
    if compliance_alerts:
        st.divider()
        st.markdown("**Compliance Breaches**")
        for alert in compliance_alerts:
            _render_alert_row(alert, prefix="port_comp_")


def _render_outbound_webhooks_tab():
    """Render the Outbound Webhooks tab showing webhook delivery log."""
    deliveries = get_webhook_delivery_log(limit=500)

    # Date range filter (reuse same pattern but filter on sent_at)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From",
            value=datetime.now().date() - timedelta(days=30),
            key="outbound_wh_start_date",
        )
    with col2:
        end_date = st.date_input(
            "To",
            value=datetime.now().date(),
            key="outbound_wh_end_date",
        )

    filtered = []
    for d in deliveries:
        ts = d.get('sent_at', '')
        if ts:
            try:
                d_date = datetime.fromisoformat(ts).date()
                if start_date <= d_date <= end_date:
                    filtered.append(d)
            except (ValueError, TypeError):
                filtered.append(d)
        else:
            filtered.append(d)
    deliveries = filtered

    st.caption(f"{len(deliveries)} delivery(ies) in range")

    if not deliveries:
        st.info("No webhook deliveries in this date range.")
        return

    for d in deliveries:
        success = d.get("success", False)
        status_badge = ":green[OK]" if success else ":red[FAIL]"

        with st.container():
            col_time, col_status, col_name, col_detail = st.columns([2, 1, 2, 4])

            with col_time:
                ts = d.get("sent_at", "")
                try:
                    dt = datetime.fromisoformat(ts)
                    st.caption(dt.strftime("%m/%d %H:%M:%S"))
                except (ValueError, TypeError):
                    st.caption(ts[:19] if ts else "?")

            with col_status:
                st.markdown(status_badge)

            with col_name:
                st.markdown(f"**{d.get('webhook_name', 'Unknown')}**")
                status_code = d.get("status_code")
                if status_code:
                    st.caption(f"HTTP {status_code}")

            with col_detail:
                parts = []
                alert_type = d.get("alert_type", "")
                if alert_type:
                    parts.append(alert_type.replace("_", " ").title())
                symbol = d.get("symbol", "")
                if symbol:
                    parts.append(symbol)
                strategy_name = d.get("strategy_name", "")
                if strategy_name:
                    parts.append(strategy_name)
                st.markdown(" | ".join(parts) if parts else "Delivery")

                error = d.get("error")
                if error and not success:
                    st.caption(f"Error: {error}")

                payload = d.get("payload_sent", "")
                if payload:
                    with st.expander("View Payload", expanded=False):
                        st.code(payload, language="json")


def _render_inbound_webhooks_tab():
    """Placeholder for future inbound webhooks feature."""
    st.info("Inbound Webhooks — Coming Soon. This will allow external systems to send signals into RoR Trader.")


# =============================================================================
# PORTFOLIO WEBHOOKS (on portfolio detail page)
# =============================================================================

def render_portfolio_webhooks(port: dict):
    """Render the webhook configuration builder for a portfolio."""
    portfolio_id = port['id']
    port_cfg = get_portfolio_alert_config(portfolio_id)
    webhooks = port_cfg.get('webhooks', [])

    st.subheader("Outbound Webhooks")
    st.caption(f"{len(webhooks)} webhook(s) configured for this portfolio.")

    # Alert config toggles
    col1, col2 = st.columns(2)
    with col1:
        alerts_on = st.toggle(
            "Portfolio Alerts Enabled",
            value=port_cfg.get('alerts_enabled', False),
            key=f"wh_alert_enabled_{portfolio_id}",
        )
    with col2:
        compliance_on = st.toggle(
            "Compliance Breach Alerts",
            value=port_cfg.get('alert_on_compliance_breach', True),
            key=f"wh_compliance_{portfolio_id}",
            disabled=not alerts_on,
        )

    if st.button("Save Alert Settings", key=f"wh_save_alert_{portfolio_id}"):
        new_cfg = dict(port_cfg)
        new_cfg['alerts_enabled'] = alerts_on
        new_cfg['alert_on_compliance_breach'] = compliance_on
        set_portfolio_alert_config(portfolio_id, new_cfg)
        st.toast("Alert settings saved")

    st.divider()

    # Existing webhooks
    for i, wh in enumerate(webhooks):
        _render_webhook_editor(portfolio_id, port, wh, i)

    # Add new webhook
    if st.button("+ Add Webhook", key=f"add_wh_{portfolio_id}"):
        new_wh = {
            "name": "New Webhook",
            "url": "",
            "enabled": True,
            "events": {
                "entry_long": True,
                "entry_short": True,
                "exit_long": True,
                "exit_short": True,
                "compliance_breach": True,
            },
            "payload_template": "",
        }
        add_portfolio_webhook(portfolio_id, new_wh)
        st.toast("Webhook added")
        st.rerun()

    # Recent alerts for this portfolio
    st.divider()
    st.subheader("Recent Portfolio Alerts")
    alerts = get_alerts_for_portfolio(portfolio_id, limit=20)
    if not alerts:
        st.caption("No alerts for this portfolio yet.")
    else:
        for alert in alerts:
            _render_alert_row(alert, prefix=f"pwh_{portfolio_id}_")


def _render_webhook_editor(portfolio_id: int, port: dict, wh: dict, index: int):
    """Render an individual webhook configuration card."""
    wh_id = wh.get('id', '')
    wh_name = wh.get('name', f'Webhook {index + 1}')
    enabled = wh.get('enabled', True)

    status_icon = "ON" if enabled else "OFF"
    with st.expander(f"{wh_name} [{status_icon}]", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = st.text_input("Name", value=wh_name, key=f"wh_name_{wh_id}")
            url = st.text_input("URL", value=wh.get('url', ''),
                                placeholder="https://hook.us1.make.com/...",
                                key=f"wh_url_{wh_id}")
        with col2:
            wh_enabled = st.toggle("Enabled", value=enabled, key=f"wh_enabled_{wh_id}")

        # Event type checkboxes
        st.markdown("**Events**")
        events = wh.get('events', {})
        ev_cols = st.columns(5)
        entry_long = ev_cols[0].checkbox("Entry Long", value=events.get('entry_long', True), key=f"ev_el_{wh_id}")
        entry_short = ev_cols[1].checkbox("Entry Short", value=events.get('entry_short', True), key=f"ev_es_{wh_id}")
        exit_long = ev_cols[2].checkbox("Exit Long", value=events.get('exit_long', True), key=f"ev_xl_{wh_id}")
        exit_short = ev_cols[3].checkbox("Exit Short", value=events.get('exit_short', True), key=f"ev_xs_{wh_id}")
        compliance = ev_cols[4].checkbox("Compliance", value=events.get('compliance_breach', True), key=f"ev_cb_{wh_id}")

        # Payload template
        st.markdown("**Payload (JSON)**")
        st.caption("Leave empty for default Discord/Slack format. Use {{placeholder}} tokens for dynamic values.")

        # Insert helpers side-by-side
        helper_cols = st.columns(2)
        with helper_cols[0]:
            ph_options = ["-- Insert Placeholder --"] + list(PLACEHOLDER_CATALOG.keys())
            selected_ph = st.selectbox("Insert placeholder", ph_options, key=f"ph_select_{wh_id}",
                                       label_visibility="collapsed")
        with helper_cols[1]:
            templates = load_webhook_templates()
            tpl_options = ["-- Insert Template --"] + [t['name'] for t in templates]
            selected_tpl = st.selectbox("Insert template", tpl_options, key=f"tpl_select_{wh_id}",
                                        label_visibility="collapsed")

        # Show placeholder hint
        if selected_ph != "-- Insert Placeholder --":
            desc = PLACEHOLDER_CATALOG.get(selected_ph, "")
            st.caption(f"Copy: `{{{{{selected_ph}}}}}` — {desc}")

        # Handle template selection via session state
        tpl_key = f"_tpl_payload_{wh_id}"
        current_template = wh.get('payload_template', '')

        if selected_tpl != "-- Insert Template --":
            # Find the template and use its payload
            for t in templates:
                if t['name'] == selected_tpl:
                    current_template = t.get('payload_template', '')
                    break

        template = st.text_area(
            "Payload Template",
            value=current_template,
            height=150,
            key=f"wh_template_{wh_id}",
            placeholder='{\n  "content": "{{event_type}}: {{symbol}} {{direction}} @ ${{order_price}}"\n}',
        )

        # Save / Delete / Test buttons
        bcol1, bcol2, bcol3 = st.columns([2, 2, 4])
        with bcol1:
            if st.button("Save", key=f"save_wh_{wh_id}", type="primary"):
                updates = {
                    "name": name,
                    "url": url,
                    "enabled": wh_enabled,
                    "events": {
                        "entry_long": entry_long,
                        "entry_short": entry_short,
                        "exit_long": exit_long,
                        "exit_short": exit_short,
                        "compliance_breach": compliance,
                    },
                    "payload_template": template,
                }
                update_portfolio_webhook(portfolio_id, wh_id, updates)
                st.toast(f"Webhook '{name}' saved")
        with bcol2:
            if st.button("Delete", key=f"del_wh_{wh_id}", type="secondary"):
                delete_portfolio_webhook(portfolio_id, wh_id)
                st.toast(f"Webhook '{wh_name}' deleted")
                st.rerun()
        with bcol3:
            if st.button("Test", key=f"test_wh_{wh_id}"):
                test_alert = {
                    "type": "entry_signal", "symbol": "TEST", "direction": "LONG",
                    "strategy_name": "Test Strategy", "strategy_id": 0,
                    "price": 100.0, "stop_price": 98.5, "atr": 1.5,
                    "trigger": "test_trigger", "confluence_met": ["TEST-CONDITION"],
                    "risk_per_trade": 100.0, "timeframe": "1Min",
                    "timestamp": datetime.now().isoformat(),
                }
                ctx = build_placeholder_context(test_alert, {
                    "portfolio_id": portfolio_id,
                    "portfolio_name": port.get('name', ''),
                    "position_risk": 100.0,
                })
                payload = render_payload(template, ctx) if template else None
                result = send_webhook(url, test_alert, payload)
                if result["success"]:
                    st.success("Test webhook sent successfully!")
                else:
                    st.error(f"Test failed: {result.get('error', 'Unknown error')}")


# =============================================================================
# WEBHOOK TEMPLATES PAGE
# =============================================================================

def render_webhook_templates_page():
    """Render the Webhook Templates management page."""
    st.header("Webhook Templates")
    st.caption("Create and manage reusable payload templates for webhooks.")

    templates = load_webhook_templates()

    # Group by category
    categories = {}
    for t in templates:
        cat = t.get('category', 'Custom')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(t)

    # Show templates by category
    for cat, cat_templates in categories.items():
        st.subheader(cat)
        for t in cat_templates:
            is_default = t.get('is_default', False)
            tpl_id = t.get('id', '')

            with st.expander(f"{t['name']}" + (" (default)" if is_default else ""), expanded=False):
                if is_default:
                    # Read-only display for defaults
                    st.code(t.get('payload_template', ''), language="json")
                    if st.button("Duplicate", key=f"dup_tpl_{tpl_id}"):
                        new_tpl = {
                            "name": f"{t['name']} (Copy)",
                            "category": "Custom",
                            "payload_template": t.get('payload_template', ''),
                        }
                        add_webhook_template(new_tpl)
                        st.toast("Template duplicated")
                        st.rerun()
                else:
                    # Editable for user-created
                    tpl_name = st.text_input("Name", value=t.get('name', ''), key=f"tpl_name_{tpl_id}")
                    tpl_cat = st.text_input("Category", value=t.get('category', 'Custom'), key=f"tpl_cat_{tpl_id}")

                    # Insert Placeholder dropdown
                    ph_options = ["-- Insert Placeholder --"] + list(PLACEHOLDER_CATALOG.keys())
                    selected = st.selectbox("Insert", ph_options, key=f"tpl_ph_{tpl_id}",
                                            label_visibility="collapsed")
                    if selected != "-- Insert Placeholder --":
                        desc = PLACEHOLDER_CATALOG.get(selected, "")
                        st.caption(f"Copy: `{{{{{selected}}}}}` — {desc}")

                    tpl_payload = st.text_area("Payload", value=t.get('payload_template', ''),
                                               height=120, key=f"tpl_payload_{tpl_id}")

                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        if st.button("Save", key=f"save_tpl_{tpl_id}", type="primary"):
                            update_webhook_template(tpl_id, {
                                "name": tpl_name,
                                "category": tpl_cat,
                                "payload_template": tpl_payload,
                            })
                            st.toast("Template saved")
                            st.rerun()
                    with bcol2:
                        if st.button("Delete", key=f"del_tpl_{tpl_id}", type="secondary"):
                            delete_webhook_template(tpl_id)
                            st.toast("Template deleted")
                            st.rerun()

    # Create new template
    st.divider()
    st.subheader("Create New Template")
    new_name = st.text_input("Template Name", key="new_tpl_name", placeholder="My Custom Template")
    new_cat = st.text_input("Category", key="new_tpl_cat", value="Custom")

    ph_options = ["-- Insert Placeholder --"] + list(PLACEHOLDER_CATALOG.keys())
    new_selected = st.selectbox("Insert Placeholder", ph_options, key="new_tpl_ph",
                                label_visibility="collapsed")
    if new_selected != "-- Insert Placeholder --":
        desc = PLACEHOLDER_CATALOG.get(new_selected, "")
        st.caption(f"Copy: `{{{{{new_selected}}}}}` — {desc}")

    new_payload = st.text_area("Payload Template", key="new_tpl_payload", height=120,
                               placeholder='{\n  "symbol": "{{symbol}}",\n  "action": "{{order_action}}"\n}')

    if st.button("Create Template", key="create_tpl", type="primary",
                 disabled=not new_name.strip()):
        add_webhook_template({
            "name": new_name.strip(),
            "category": new_cat.strip() or "Custom",
            "payload_template": new_payload,
        })
        st.toast("Template created")
        st.rerun()


# =============================================================================
# CONFLUENCE GROUPS PAGE
# =============================================================================

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


# =============================================================================
# CONFLUENCE GROUP: CODE & PREVIEW TABS
# =============================================================================

# Template -> function mapping for the Code tab
TEMPLATE_FUNCTIONS = {
    "ema_stack": {
        "Indicator": [calculate_ema],
        "Interpreter": [interpret_ema_stack],
        "Triggers": [detect_ema_triggers],
    },
    "macd_line": {
        "Indicator": [calculate_macd],
        "Interpreter": [interpret_macd_line],
        "Triggers": [detect_macd_triggers],
    },
    "macd_histogram": {
        "Indicator": [calculate_macd],
        "Interpreter": [interpret_macd_histogram],
        "Triggers": [detect_macd_hist_triggers],
    },
    "vwap": {
        "Indicator": [calculate_vwap],
        "Interpreter": [interpret_vwap],
        "Triggers": [detect_vwap_triggers],
    },
    "rvol": {
        "Indicator": [calculate_volume_sma],
        "Interpreter": [interpret_rvol],
        "Triggers": [detect_rvol_triggers],
    },
    "utbot": {
        "Indicator": [],
        "Interpreter": [],
        "Triggers": [],
    },
}


def render_code_tab(group: ConfluenceGroup):
    """Render the Code tab showing source code for indicator, interpreter, and trigger functions."""
    funcs = TEMPLATE_FUNCTIONS.get(group.base_template, {})

    if not funcs or all(len(v) == 0 for v in funcs.values()):
        st.info(f"No source code available for template '{group.base_template}'. Implementation pending.")
        return

    # Show this group's effective parameters
    st.markdown("**Active Parameters for this Group**")
    template = get_template(group.base_template)
    param_schema = template.get("parameters_schema", {}) if template else {}
    param_items = []
    for key, schema in param_schema.items():
        value = group.parameters.get(key, schema.get("default", "?"))
        param_items.append(f"`{schema.get('label', key)}` = **{value}**")
    if param_items:
        st.markdown(" | ".join(param_items))
    else:
        st.caption("No parameters")

    st.divider()
    st.caption("Source code for this confluence group's indicator, interpreter, and trigger logic.")

    for section_name, func_list in funcs.items():
        if not func_list:
            continue
        with st.expander(f"{section_name}", expanded=True):
            for func in func_list:
                try:
                    source = inspect.getsource(func)
                    st.code(source, language="python")
                except (OSError, TypeError):
                    st.warning(f"Could not retrieve source for {func.__name__}")


def render_preview_tab(group: ConfluenceGroup):
    """
    Render the Preview tab with live indicator visualization on sample data.

    Shows:
    - Price chart with indicator overlays (for overlay-compatible templates)
    - Secondary chart for non-overlay indicators (MACD, RVOL)
    - Interpreter state timeline
    - Trigger event markers
    """
    from mock_data import generate_mock_bars

    template = get_template(group.base_template)
    if not template:
        st.error("Template not found.")
        return

    st.caption("Live preview using sample data to verify indicator, interpreter, and trigger behavior.")

    # Preview controls
    preview_symbol = st.selectbox(
        "Preview Symbol",
        AVAILABLE_SYMBOLS,
        index=0,
        key=f"preview_symbol_{group.id}"
    )

    # Generate sample data
    with st.spinner("Generating preview data..."):
        end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=3)

        bars = generate_mock_bars([preview_symbol], start, end, "1Min", seed=42)

        if hasattr(bars.index, 'get_level_values') and preview_symbol in bars.index.get_level_values(0):
            df = bars.loc[preview_symbol]
        elif len(bars) > 0:
            df = bars
        else:
            st.error("Could not generate sample data.")
            return

        if len(df) == 0:
            st.error("No sample data generated.")
            return

        # Run the full indicator pipeline
        df = run_all_indicators(df)

        # Run group-specific indicators for custom parameters
        df = run_indicators_for_group(df, group)

        # Run interpreters and triggers
        df = run_all_interpreters(df)
        df = detect_all_triggers(df)

    # --- Section 1: Chart ---
    # Overlay templates show indicator lines on the price chart.
    # Oscillator templates show a synced secondary pane (MACD/RVOL) below the price chart.
    show_indicators = []
    indicator_colors_map = {}
    secondary_panes = []

    if group.base_template in OVERLAY_COMPATIBLE_TEMPLATES:
        st.markdown("**Price Chart with Indicator Overlay**")
        show_indicators = get_overlay_indicators_for_group(group)
        indicator_colors_map = get_overlay_colors_for_group(group)
    elif group.base_template in ("macd_line", "macd_histogram"):
        st.markdown("**Price Chart + MACD**")
        if "macd_line" in df.columns:
            secondary_panes.append(_build_macd_lwc_pane(df, group))
    elif group.base_template == "rvol":
        st.markdown("**Price Chart + Relative Volume**")
        if "rvol" in df.columns:
            secondary_panes.append(_build_rvol_lwc_pane(df, group))

    render_price_chart(
        df,
        pd.DataFrame(),  # no trade markers on preview
        {"direction": "LONG"},
        show_indicators=show_indicators,
        indicator_colors=indicator_colors_map,
        chart_key=f"preview_chart_{group.id}",
        secondary_panes=secondary_panes if secondary_panes else None
    )

    # --- Section 2: Interpreter State Timeline ---
    st.markdown("**Interpreter States**")
    _render_interpreter_timeline(df, group, template)

    # --- Section 3: Trigger Events ---
    st.markdown("**Trigger Events**")
    _render_trigger_events_table(df, group, template)


def _build_macd_lwc_pane(df: pd.DataFrame, group: ConfluenceGroup) -> dict:
    """
    Build a lightweight-charts pane config for MACD histogram + lines.

    Returns a pane dict that can be passed to render_price_chart's secondary_panes
    for synchronized zoom/scroll with the price chart above.
    """
    macd_color = group.plot_settings.colors.get("macd_color", "#2563eb")
    signal_color = group.plot_settings.colors.get("signal_color", "#f97316")
    hist_pos_color = group.plot_settings.colors.get("hist_pos_color", "#22c55e")
    hist_neg_color = group.plot_settings.colors.get("hist_neg_color", "#ef4444")

    plot_df = df.reset_index()
    time_col = plot_df.columns[0]
    plot_df['_time'] = pd.to_datetime(plot_df[time_col]).astype(int) // 10**9

    hist_data = []
    macd_data = []
    signal_data = []

    for _, row in plot_df.iterrows():
        t = int(row['_time'])
        if pd.notna(row.get('macd_hist')):
            hist_data.append({
                "time": t,
                "value": float(row['macd_hist']),
                "color": hist_pos_color if row['macd_hist'] >= 0 else hist_neg_color
            })
        if pd.notna(row.get('macd_line')):
            macd_data.append({"time": t, "value": float(row['macd_line'])})
        if pd.notna(row.get('macd_signal')):
            signal_data.append({"time": t, "value": float(row['macd_signal'])})

    series = []
    if hist_data:
        series.append({
            "type": "Histogram",
            "data": hist_data,
            "options": {
                "priceLineVisible": False,
                "title": "Hist",
            }
        })
    if macd_data:
        series.append({
            "type": "Line",
            "data": macd_data,
            "options": {
                "color": macd_color,
                "lineWidth": 1,
                "priceLineVisible": False,
                "title": "MACD",
            }
        })
    if signal_data:
        series.append({
            "type": "Line",
            "data": signal_data,
            "options": {
                "color": signal_color,
                "lineWidth": 1,
                "priceLineVisible": False,
                "title": "Signal",
            }
        })

    return {
        "chart": {
            "layout": {
                "background": {"color": "#1E1E1E"},
                "textColor": "#DDD"
            },
            "grid": {
                "vertLines": {"color": "#2B2B2B"},
                "horzLines": {"color": "#2B2B2B"}
            },
            "crosshair": {"mode": 0},
            "timeScale": {
                "borderColor": "#2B2B2B",
                "timeVisible": True,
                "secondsVisible": False,
            },
            "rightPriceScale": {"borderColor": "#2B2B2B"},
            "height": 200,
        },
        "series": series
    }


def _build_rvol_lwc_pane(df: pd.DataFrame, group: ConfluenceGroup) -> dict:
    """
    Build a lightweight-charts pane config for relative volume histogram.

    Returns a pane dict that can be passed to render_price_chart's secondary_panes
    for synchronized zoom/scroll with the price chart above.
    """
    bar_color = group.plot_settings.colors.get("bar_color", "#64748b")
    high_color = group.plot_settings.colors.get("high_color", "#f59e0b")
    extreme_color = group.plot_settings.colors.get("extreme_color", "#ef4444")

    high_thresh = group.parameters.get("high_threshold", 1.5)
    extreme_thresh = group.parameters.get("extreme_threshold", 3.0)

    plot_df = df.reset_index()
    time_col = plot_df.columns[0]
    plot_df['_time'] = pd.to_datetime(plot_df[time_col]).astype(int) // 10**9

    rvol_data = []
    for _, row in plot_df.iterrows():
        if pd.notna(row.get('rvol')):
            val = float(row['rvol'])
            if val >= extreme_thresh:
                color = extreme_color
            elif val >= high_thresh:
                color = high_color
            else:
                color = bar_color
            rvol_data.append({
                "time": int(row['_time']),
                "value": val,
                "color": color
            })

    series = []
    if rvol_data:
        series.append({
            "type": "Histogram",
            "data": rvol_data,
            "options": {
                "priceLineVisible": False,
                "title": "RVOL",
            }
        })

    return {
        "chart": {
            "layout": {
                "background": {"color": "#1E1E1E"},
                "textColor": "#DDD"
            },
            "grid": {
                "vertLines": {"color": "#2B2B2B"},
                "horzLines": {"color": "#2B2B2B"}
            },
            "crosshair": {"mode": 0},
            "timeScale": {
                "borderColor": "#2B2B2B",
                "timeVisible": True,
                "secondsVisible": False,
            },
            "rightPriceScale": {"borderColor": "#2B2B2B"},
            "height": 180,
        },
        "series": series
    }


def _render_interpreter_timeline(df: pd.DataFrame, group: ConfluenceGroup, template: dict):
    """Render a table showing interpreter state changes over time."""
    interpreter_keys = template.get("interpreters", [])

    if not interpreter_keys:
        st.caption("No interpreters defined for this template.")
        return

    for interp_key in interpreter_keys:
        if interp_key not in df.columns:
            st.caption(f"Interpreter '{interp_key}' not in data.")
            continue

        states = df[interp_key].dropna()
        if len(states) == 0:
            st.caption(f"No state data for '{interp_key}'.")
            continue

        # Detect state changes
        changes = states[states != states.shift(1)]

        if len(changes) == 0:
            st.caption(f"No state changes detected for '{interp_key}'.")
            continue

        # Build summary table (last 25 state changes)
        change_records = []
        for idx, state in changes.tail(25).items():
            change_records.append({
                "Time": str(idx),
                "State": state,
            })

        st.caption(f"**{interp_key}** — Last {len(change_records)} state changes (of {len(changes)} total):")
        st.dataframe(
            pd.DataFrame(change_records),
            use_container_width=True,
            hide_index=True,
        )


def _render_trigger_events_table(df: pd.DataFrame, group: ConfluenceGroup, template: dict):
    """Render a table of trigger firings for this group."""
    trigger_defs = template.get("triggers", [])
    if not trigger_defs:
        st.caption("No triggers defined for this template.")
        return

    events = []
    for trig_def in trigger_defs:
        # Triggers in the DataFrame use the base trigger names (not group-prefixed)
        base = trig_def["base"]
        # Try group-prefixed trigger column first, then base trigger columns
        possible_cols = [
            f"trig_{group.id}_{base}",
            f"trig_{template.get('trigger_prefix', '')}_{base}",
        ]

        for col in possible_cols:
            if col in df.columns:
                fired = df[df[col] == True]
                for idx in fired.index:
                    events.append({
                        "Time": str(idx),
                        "Trigger": trig_def["name"],
                        "Direction": trig_def["direction"],
                        "Type": trig_def["type"],
                        "Price": f"${df.loc[idx, 'close']:.2f}" if 'close' in df.columns else "N/A",
                    })
                break

    if not events:
        st.caption("No triggers fired in the sample data period.")
        return

    events_df = pd.DataFrame(events).sort_values("Time", ascending=False).head(30)
    st.dataframe(events_df, use_container_width=True, hide_index=True)
    st.caption(f"{len(events)} total trigger events in sample data")


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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Parameters", "Plot Settings", "Outputs & Triggers", "Preview", "Code", "Danger Zone"
    ])

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

    # TAB 4: Preview
    with tab4:
        render_preview_tab(group)

    # TAB 5: Code
    with tab5:
        render_code_tab(group)

    # TAB 6: Danger Zone
    with tab6:
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
