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
from datetime import datetime, timedelta, date
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

from data_loader import load_market_data, get_data_source, is_alpaca_configured, estimate_bar_count, days_from_bar_count
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
import general_packs as gp_module
import risk_management_packs as rmp_module


# =============================================================================
# CONFIGURATION
# =============================================================================

AVAILABLE_SYMBOLS = ["SPY", "AAPL", "QQQ", "TSLA", "NVDA", "MSFT", "AMD", "META"]
TIMEFRAMES = [
    "1Min", "2Min", "3Min", "5Min", "10Min", "15Min", "30Min",
    "1Hour", "2Hour", "4Hour",
    "1Day", "1Week", "1Month",
]
DIRECTIONS = ["LONG", "SHORT"]

TIMEFRAME_GUIDANCE = {
    "1Min":   "~390 bars/day \u00b7 recommended \u226490 days",
    "2Min":   "~195 bars/day \u00b7 recommended \u22646 months",
    "3Min":   "~130 bars/day \u00b7 recommended \u22646 months",
    "5Min":   "~78 bars/day \u00b7 recommended \u22641 year",
    "10Min":  "~39 bars/day \u00b7 recommended \u22642 years",
    "15Min":  "~26 bars/day \u00b7 recommended \u22642 years",
    "30Min":  "~13 bars/day \u00b7 recommended \u22643 years",
    "1Hour":  "~7 bars/day \u00b7 recommended \u22645 years",
    "2Hour":  "~4 bars/day \u00b7 recommended \u22645 years",
    "4Hour":  "~2 bars/day \u00b7 recommended \u22645 years",
    "1Day":   "1 bar/day \u00b7 recommended \u22645 years",
    "1Week":  "1 bar/week \u00b7 recommended \u226410 years",
    "1Month": "~1 bar/month \u00b7 recommended \u226410 years",
}

ALPACA_DATA_FLOOR = date(2016, 1, 1)

LOOKBACK_MODES = ["Days", "Bars/Candles", "Date Range"]
OVERLAY_COMPATIBLE_TEMPLATES = ["ema_stack", "vwap", "utbot"]

# Strategies storage path (resolve relative to this script, not cwd)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_FILE = os.path.join(_SCRIPT_DIR, "strategies.json")

# Settings storage path (in config/ dir, like confluence_groups and general_packs)
SETTINGS_FILE = os.path.join(_SCRIPT_DIR, "..", "config", "settings.json")

SETTINGS_DEFAULTS = {
    "chart_visible_candles": 200,
    "default_extended_data_days": 365,
    "default_entry_trigger": "",
    "default_exit_trigger": "",
    "default_stop_config": {"method": "atr", "atr_mult": 1.5},
    "default_target_config": None,
    "global_data_seed": 42,
}


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
    General records are formatted as: "GEN-{PACK_ID}-{state}"
    e.g., "GEN-TOD_NY_OPEN-IN_WINDOW"

    Returns formatted string like: "EMA Stack (Default): SML"
    """
    # Handle GEN- prefixed records (General Pack conditions)
    if record.startswith("GEN-"):
        parts = record.split("-", 2)  # ["GEN", "PACK_ID", "STATE"]
        if len(parts) >= 3:
            pack_id_upper = parts[1]
            state = parts[2]
            gen_packs = gp_module.load_general_packs()
            for gpack in gen_packs:
                if gpack.id.upper() == pack_id_upper:
                    return f"{gpack.name}: {state}"
            return f"General ({pack_id_upper}): {state}"
        return record

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
                                  start_date=None, end_date=None,
                                  timeframe: str = "1Min"):
    """
    Load market data and run all indicators, interpreters, and trigger detection.

    Uses Alpaca API if configured, otherwise falls back to mock data.

    Args:
        symbol: Stock symbol
        days: Number of days (used if start_date/end_date not provided)
        seed: Random seed for mock data
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)
        timeframe: Bar timeframe (e.g., "1Min", "5Min", "1Hour")

    Returns DataFrame ready for trade generation and analysis.
    """
    # Load raw bars (Alpaca if configured, mock otherwise)
    df = load_market_data(symbol, days=days, seed=seed,
                          start_date=start_date, end_date=end_date,
                          timeframe=timeframe)

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

    # Evaluate enabled general packs (condition columns for trade tagging)
    gen_packs = gp_module.load_general_packs()
    enabled_gen = gp_module.get_enabled_general_packs(gen_packs)
    for gpack in enabled_gen:
        col_name = gpack.get_condition_column()
        df[col_name] = gp_module.evaluate_condition(df, gpack)

    return df


def prepare_forward_test_data(strat: dict, data_days_override: int = None):
    """
    Load continuous data from before forward_test_start to now,
    run the full pipeline, and split trades at the boundary.

    Args:
        strat: Strategy config dict
        data_days_override: If provided, use this instead of strat's data_days
            (used for the extended data view)

    Returns (df, backtest_trades, forward_trades, forward_test_start_dt)
    """
    forward_test_start_dt = datetime.fromisoformat(strat['forward_test_start'])
    data_days = data_days_override if data_days_override is not None else strat.get('data_days', 30)
    data_seed = strat.get('data_seed', 42)

    # Start before forward_test_start to have backtest data + indicator warmup
    start_date = forward_test_start_dt - timedelta(days=data_days * 2)
    # Round end_date to market close today for cache-friendly behavior
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

    strat_timeframe = strat.get('timeframe', '1Min')
    df = prepare_data_with_indicators(
        strat['symbol'], seed=data_seed,
        start_date=start_date, end_date=end_date,
        timeframe=strat_timeframe
    )

    if len(df) == 0:
        empty = pd.DataFrame()
        return df, empty, empty, forward_test_start_dt

    confluence_set = set(strat.get('confluence', [])) | set(strat.get('general_confluences', []))
    confluence_set = confluence_set if confluence_set else None
    general_cols = [c for c in df.columns if c.startswith("GP_")]

    trades = generate_trades(
        df,
        direction=strat['direction'],
        entry_trigger=strat['entry_trigger'],
        exit_trigger=strat.get('exit_trigger'),
        exit_triggers=strat.get('exit_triggers'),
        confluence_required=confluence_set,
        risk_per_trade=strat.get('risk_per_trade', 100.0),
        stop_atr_mult=strat.get('stop_atr_mult', 1.5),
        stop_config=strat.get('stop_config'),
        target_config=strat.get('target_config'),
        bar_count_exit=strat.get('bar_count_exit'),
        general_columns=general_cols,
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
        gst_start = None
        gst_end = None
        if strat.get('lookback_mode') == 'Date Range' and strat.get('lookback_start_date'):
            gst_start = datetime.fromisoformat(strat['lookback_start_date'])
            gst_end = datetime.fromisoformat(strat['lookback_end_date'])
        df = prepare_data_with_indicators(strat['symbol'], data_days, data_seed,
                                          start_date=gst_start, end_date=gst_end,
                                          timeframe=strat.get('timeframe', '1Min'))
        if len(df) == 0:
            return pd.DataFrame()
        confluence_set = set(strat.get('confluence', [])) | set(strat.get('general_confluences', []))
        confluence_set = confluence_set if confluence_set else None
        general_cols = [c for c in df.columns if c.startswith("GP_")]
        return generate_trades(
            df,
            direction=strat['direction'],
            entry_trigger=strat['entry_trigger'],
            exit_trigger=strat.get('exit_trigger'),
            exit_triggers=strat.get('exit_triggers'),
            confluence_required=confluence_set,
            risk_per_trade=strat.get('risk_per_trade', 100.0),
            stop_atr_mult=strat.get('stop_atr_mult', 1.5),
            stop_config=strat.get('stop_config'),
            target_config=strat.get('target_config'),
            bar_count_exit=strat.get('bar_count_exit'),
            general_columns=general_cols,
        )


def _generate_incremental_trades(strat: dict, since_dt) -> pd.DataFrame:
    """Load a small data window and generate trades for the recent period only.

    Args:
        strat: Strategy config dict
        since_dt: Only return trades with entry_time after this timestamp

    Returns:
        DataFrame of new trades (may be empty)
    """
    import math
    from data_loader import BARS_PER_DAY

    timeframe = strat.get('timeframe', '1Min')
    data_seed = strat.get('data_seed', 42)
    bpd = BARS_PER_DAY.get(timeframe, 390)

    # Warmup: enough bars for longest indicator (EMA-50) + safety margin
    warmup_bars = 100
    warmup_days = max(1, math.ceil(warmup_bars / bpd * 365 / 252))

    since_as_dt = pd.Timestamp(since_dt).to_pydatetime()
    if since_as_dt.tzinfo is not None:
        since_as_dt = since_as_dt.replace(tzinfo=None)

    start_date = since_as_dt - timedelta(days=warmup_days)
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

    df = prepare_data_with_indicators(
        strat['symbol'], seed=data_seed,
        start_date=start_date, end_date=end_date,
        timeframe=timeframe,
    )

    if len(df) == 0:
        return pd.DataFrame()

    confluence_set = set(strat.get('confluence', []))
    confluence_set |= set(strat.get('general_confluences', []))
    confluence_set = confluence_set if confluence_set else None
    general_cols = [c for c in df.columns if c.startswith("GP_")]

    trades = generate_trades(
        df,
        direction=strat['direction'],
        entry_trigger=strat['entry_trigger'],
        exit_trigger=strat.get('exit_trigger'),
        exit_triggers=strat.get('exit_triggers'),
        confluence_required=confluence_set,
        risk_per_trade=strat.get('risk_per_trade', 100.0),
        stop_atr_mult=strat.get('stop_atr_mult', 1.5),
        stop_config=strat.get('stop_config'),
        target_config=strat.get('target_config'),
        bar_count_exit=strat.get('bar_count_exit'),
        general_columns=general_cols,
    )

    if len(trades) == 0:
        return pd.DataFrame()

    # Filter to only truly new trades (entered after the cutoff)
    since_ts = pd.Timestamp(since_dt)
    if trades['entry_time'].dt.tz is not None and since_ts.tz is None:
        since_ts = since_ts.tz_localize('UTC')
    return trades[trades['entry_time'] > since_ts]


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


def extract_equity_curve_data(trades: pd.DataFrame, boundary_dt=None) -> dict:
    """Extract minimal equity curve data for persistent storage.

    Returns a JSON-serializable dict with exit_times, cumulative_r,
    and boundary_index (for forward-testing strategies).
    """
    if len(trades) == 0:
        return {"exit_times": [], "cumulative_r": [], "boundary_index": None}

    equity = trades[["exit_time", "r_multiple"]].sort_values("exit_time").reset_index(drop=True)
    equity["cumulative_r"] = equity["r_multiple"].cumsum()

    exit_times = [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in equity["exit_time"]]
    cumulative_r = [round(float(v), 4) for v in equity["cumulative_r"].values]

    boundary_index = None
    if boundary_dt is not None:
        boundary_ts = pd.Timestamp(boundary_dt)
        if hasattr(equity["exit_time"].dtype, 'tz') and equity["exit_time"].dtype.tz is not None:
            boundary_ts = boundary_ts.tz_localize(equity["exit_time"].dtype.tz)
        fw_mask = equity["exit_time"] >= boundary_ts
        if fw_mask.any():
            boundary_index = int(fw_mask.idxmax())

    return {
        "exit_times": exit_times,
        "cumulative_r": cumulative_r,
        "boundary_index": boundary_index,
    }


def _extract_minimal_trades(trades: pd.DataFrame) -> list:
    """Extract minimal trade records for persistent storage.

    Only stores the 4 fields needed for KPI and equity curve computation:
    entry_time, exit_time, r_multiple, win.
    """
    if len(trades) == 0:
        return []
    records = []
    for _, row in trades.iterrows():
        et = row["entry_time"]
        xt = row["exit_time"]
        records.append({
            "entry_time": et.isoformat() if hasattr(et, 'isoformat') else str(et),
            "exit_time": xt.isoformat() if hasattr(xt, 'isoformat') else str(xt),
            "r_multiple": round(float(row["r_multiple"]), 4),
            "win": bool(row["win"]),
        })
    return records


def _trades_df_from_stored(stored_trades: list) -> pd.DataFrame:
    """Reconstruct a trades DataFrame from stored minimal records."""
    if not stored_trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "r_multiple", "win"])
    df = pd.DataFrame(stored_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def extract_portfolio_equity_curve_data(combined_trades: pd.DataFrame) -> dict:
    """Extract minimal equity curve data for portfolio list page persistence."""
    if len(combined_trades) == 0:
        return {"exit_times": [], "cumulative_pnl": []}

    eq = combined_trades[['exit_time', 'cumulative_pnl']].copy()
    exit_times = [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in eq["exit_time"]]
    cumulative_pnl = [round(float(v), 2) for v in eq["cumulative_pnl"].values]

    return {
        "exit_times": exit_times,
        "cumulative_pnl": cumulative_pnl,
    }


def render_mini_equity_curve_from_data(eq_data: dict, key: str):
    """Render mini equity curve from persisted equity curve data dict."""
    exit_times = eq_data.get('exit_times', [])
    cumulative_r = eq_data.get('cumulative_r', [])
    boundary_index = eq_data.get('boundary_index')

    if not exit_times:
        st.caption("No trades")
        return

    times = pd.to_datetime(exit_times)

    fig = go.Figure()

    if boundary_index is not None and boundary_index < len(times):
        # Backtest portion
        if boundary_index > 0:
            fig.add_trace(go.Scatter(
                x=times[:boundary_index], y=cumulative_r[:boundary_index],
                mode="lines", line=dict(color="#2196F3", width=1.5),
                fill="tozeroy", fillcolor="rgba(33, 150, 243, 0.08)",
                showlegend=False
            ))
        # Forward portion (with bridge point from backtest)
        fw_start = max(0, boundary_index - 1)
        fig.add_trace(go.Scatter(
            x=times[fw_start:], y=cumulative_r[fw_start:],
            mode="lines", line=dict(color="#4CAF50", width=1.5),
            fill="tozeroy", fillcolor="rgba(76, 175, 80, 0.08)",
            showlegend=False
        ))
    else:
        final_r = cumulative_r[-1]
        color = "#4CAF50" if final_r >= 0 else "#f44336"
        fill = f"rgba({'76, 175, 80' if final_r >= 0 else '244, 67, 54'}, 0.08)"
        fig.add_trace(go.Scatter(
            x=times, y=cumulative_r,
            mode="lines", line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=fill,
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

# =============================================================================
# STRATEGY DISPLAY HELPERS
# =============================================================================

def format_stop_display(strat: dict) -> str:
    """Format stop loss config for human-readable display."""
    sc = strat.get('stop_config')
    if sc is None:
        return f"{strat.get('stop_atr_mult', 1.5):.1f}x ATR"
    method = sc.get('method', 'atr')
    if method == 'atr':
        return f"{sc.get('atr_mult', 1.5):.1f}x ATR"
    elif method == 'fixed_dollar':
        return f"${sc.get('dollar_amount', 1.0):.2f} Fixed"
    elif method == 'percentage':
        return f"{sc.get('percentage', 0.5):.2f}%"
    elif method == 'swing':
        return f"Swing ({sc.get('lookback', 5)} bars, ${sc.get('padding', 0.05):.2f} pad)"
    return f"{strat.get('stop_atr_mult', 1.5):.1f}x ATR"


def format_target_display(strat: dict) -> str:
    """Format target config for human-readable display."""
    tc = strat.get('target_config')
    if tc is None:
        return "None (signal exit only)"
    method = tc.get('method')
    if method is None:
        return "None (signal exit only)"
    if method == 'risk_reward':
        return f"{tc.get('rr_ratio', 2.0):.1f}R"
    elif method == 'atr':
        return f"{tc.get('atr_mult', 2.0):.1f}x ATR"
    elif method == 'fixed_dollar':
        return f"${tc.get('dollar_amount', 2.0):.2f} Fixed"
    elif method == 'percentage':
        return f"{tc.get('percentage', 1.0):.2f}%"
    elif method == 'swing':
        return f"Swing ({tc.get('lookback', 5)} bars, ${tc.get('padding', 0.05):.2f} pad)"
    return "None"


def format_exit_triggers_display(strat: dict) -> str:
    """Format exit triggers for human-readable display."""
    names = strat.get('exit_trigger_names')
    if names:
        return ", ".join(names)
    name = strat.get('exit_trigger_name')
    if name:
        return name
    return strat.get('exit_trigger', 'Unknown')


def count_trading_days(df: pd.DataFrame) -> int:
    """Count unique trading days in a DataFrame with a DatetimeIndex."""
    return max(df.index.normalize().nunique(), 1)


def calculate_kpis(trades_df: pd.DataFrame, starting_balance: float = 10000,
                   risk_per_trade: float = 100, total_trading_days: int = None) -> dict:
    """Calculate strategy KPIs.

    Args:
        total_trading_days: Total trading days in the data period (all market days,
            not just days with trades). When provided, Daily R = total_r / total_trading_days.
            When None, falls back to counting unique exit dates.
    """
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
            "max_r_drawdown": 0, "final_balance": starting_balance, "total_pnl": 0
        }

    wins = trades_df[trades_df["win"] == True]
    losses = trades_df[trades_df["win"] == False]

    gross_profit = wins["r_multiple"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["r_multiple"].sum()) if len(losses) > 0 else 0
    total_r = trades_df["r_multiple"].sum()

    # Trading days — prefer total period days over just days with exits
    if total_trading_days is not None and total_trading_days > 0:
        trading_days = total_trading_days
    elif "exit_time" in trades_df.columns:
        trading_days = trades_df["exit_time"].dt.date.nunique()
    else:
        trading_days = 1
    trading_days = max(trading_days, 1)

    # Cumulative R curve (used for R-squared and Max R Drawdown)
    if len(trades_df) >= 2:
        cumulative_r = trades_df["r_multiple"].cumsum().values
        # R-squared of equity curve (smoothness: 1.0 = perfectly linear growth)
        x = np.arange(len(cumulative_r))
        correlation = np.corrcoef(x, cumulative_r)[0, 1]
        r_squared = round(correlation ** 2, 4) if not np.isnan(correlation) else 0.0
        # Max R Drawdown (peak-to-trough in cumulative R space)
        running_max = np.maximum.accumulate(cumulative_r)
        drawdown = cumulative_r - running_max
        max_r_drawdown = round(float(drawdown.min()), 2)
    else:
        r_squared = 0.0
        max_r_drawdown = 0.0

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
        "r_squared": r_squared,
        "max_r_drawdown": max_r_drawdown,
        "final_balance": final_balance,
        "total_pnl": total_pnl,
    }


def calculate_secondary_kpis(trades_df: pd.DataFrame, kpis: dict) -> dict:
    """Calculate secondary/extended KPIs from trade data (always computed live, not saved)."""
    if len(trades_df) == 0:
        return {
            "win_count": 0, "loss_count": 0,
            "best_trade_r": 0, "worst_trade_r": 0,
            "avg_win_r": 0, "avg_loss_r": 0,
            "max_consec_wins": 0, "max_consec_losses": 0,
            "payoff_ratio": 0, "recovery_factor": 0,
            "longest_dd_trades": 0,
        }

    wins_mask = trades_df["win"].values
    r_mult = trades_df["r_multiple"].values

    # Win/loss counts
    win_count = int(wins_mask.sum())
    loss_count = len(wins_mask) - win_count

    # Best/worst trade
    best_trade_r = float(r_mult.max())
    worst_trade_r = float(r_mult.min())

    # Avg win / avg loss
    avg_win_r = float(r_mult[wins_mask].mean()) if win_count > 0 else 0
    avg_loss_r = float(r_mult[~wins_mask].mean()) if loss_count > 0 else 0

    # Max consecutive wins/losses
    max_consec_wins = max_consec_losses = 0
    current_wins = current_losses = 0
    for w in wins_mask:
        if w:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Payoff ratio (avg win R / abs(avg loss R))
    abs_avg_loss = abs(avg_loss_r)
    payoff_ratio = avg_win_r / abs_avg_loss if abs_avg_loss > 0 else float('inf')

    # Recovery factor (Total R / abs(Max R DD))
    max_r_dd_abs = abs(kpis.get("max_r_drawdown", 0))
    recovery_factor = kpis["total_r"] / max_r_dd_abs if max_r_dd_abs > 0 else float('inf')

    # Longest drawdown in trades (consecutive trades from peak to recovery)
    if len(r_mult) >= 2:
        cumulative = np.cumsum(r_mult)
        running_max = np.maximum.accumulate(cumulative)
        in_dd = cumulative < running_max
        longest_dd = 0
        current_dd = 0
        for d in in_dd:
            if d:
                current_dd += 1
                longest_dd = max(longest_dd, current_dd)
            else:
                current_dd = 0
    else:
        longest_dd = 0

    return {
        "win_count": win_count,
        "loss_count": loss_count,
        "best_trade_r": round(best_trade_r, 2),
        "worst_trade_r": round(worst_trade_r, 2),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float('inf') else float('inf'),
        "recovery_factor": round(recovery_factor, 1) if recovery_factor != float('inf') else float('inf'),
        "longest_dd_trades": longest_dd,
    }


# =============================================================================
# CONFLUENCE ANALYSIS
# =============================================================================

def analyze_confluences(trades_df: pd.DataFrame, required: set = None, min_trades: int = 5,
                        starting_balance: float = 10000, risk_per_trade: float = 100,
                        total_trading_days: int = None) -> pd.DataFrame:
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

    base_kpis = calculate_kpis(base_trades, starting_balance=starting_balance,
                               risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)

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
            kpis = calculate_kpis(subset, starting_balance=starting_balance,
                                  risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)

            results.append({
                "confluence": record,
                "total_trades": kpis["total_trades"],
                "win_rate": kpis["win_rate"],
                "profit_factor": kpis["profit_factor"],
                "avg_r": kpis["avg_r"],
                "total_r": kpis["total_r"],
                "daily_r": kpis["daily_r"],
                "r_squared": kpis["r_squared"],
                "pf_change": safe_subtract(kpis["profit_factor"], base_kpis["profit_factor"]),
                "wr_change": kpis["win_rate"] - base_kpis["win_rate"],
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("profit_factor", ascending=False, na_position="last")

    return results_df


def find_best_combinations(trades_df: pd.DataFrame, max_depth: int = 3, min_trades: int = 5, top_n: int = 10,
                           starting_balance: float = 10000, risk_per_trade: float = 100,
                           total_trading_days: int = None, exclude_prefix: str = None) -> pd.DataFrame:
    """Find the best confluence combinations automatically."""
    if len(trades_df) == 0:
        return pd.DataFrame()

    # Get all unique records
    all_records = set()
    for records in trades_df["confluence_records"]:
        all_records.update(records)
    if exclude_prefix:
        all_records = {r for r in all_records if not r.startswith(exclude_prefix)}
    all_records = list(all_records)

    results = []

    for depth in range(1, min(max_depth + 1, len(all_records) + 1)):
        for combo in combinations(all_records, depth):
            combo_set = set(combo)

            mask = trades_df["confluence_records"].apply(lambda r: isinstance(r, set) and combo_set.issubset(r))
            subset = trades_df[mask]

            if len(subset) >= min_trades:
                kpis = calculate_kpis(subset, starting_balance=starting_balance,
                                      risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)

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


def analyze_entry_triggers(
    df: pd.DataFrame, direction: str, groups: list,
    exit_triggers: list = None, bar_count_exit: int = None,
    risk_per_trade: float = 100.0, stop_config: dict = None,
    target_config: dict = None, confluence_required: set = None,
    starting_balance: float = 10000.0, total_trading_days: int = None,
    general_columns: list = None,
) -> pd.DataFrame:
    """For each available entry trigger, generate trades with current strategy config and compute KPIs."""
    entry_triggers = get_confluence_entry_triggers(direction, groups)
    all_trigger_defs = get_all_triggers(groups)

    # Ensure at least one exit mechanism exists
    effective_exit_triggers = exit_triggers or []
    effective_bar_count = bar_count_exit
    if not effective_exit_triggers and effective_bar_count is None:
        effective_bar_count = 4  # fallback default

    results = []
    for trig_cid, trig_name in entry_triggers.items():
        base_id = get_base_trigger_id(trig_cid)
        tdef = all_trigger_defs.get(trig_cid)
        trades = generate_trades(
            df, direction=direction, entry_trigger=base_id,
            exit_triggers=effective_exit_triggers,
            bar_count_exit=effective_bar_count,
            confluence_required=confluence_required,
            risk_per_trade=risk_per_trade, stop_config=stop_config,
            target_config=target_config,
            general_columns=general_columns,
        )
        if len(trades) == 0:
            continue
        kpis = calculate_kpis(trades, starting_balance=starting_balance,
                              risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)
        results.append({
            'trigger_id': trig_cid,
            'trigger_name': trig_name,
            'execution': tdef.execution if tdef else 'bar_close',
            'total_trades': kpis['total_trades'],
            'win_rate': kpis['win_rate'],
            'profit_factor': kpis['profit_factor'],
            'avg_r': kpis['avg_r'],
            'daily_r': kpis['daily_r'],
            'r_squared': kpis['r_squared'],
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('profit_factor', ascending=False, na_position='last')
    return results_df


def analyze_exit_triggers(
    df: pd.DataFrame, direction: str, entry_trigger_confluence_id: str,
    groups: list, risk_per_trade: float = 100.0,
    stop_config: dict = None, target_config: dict = None,
    starting_balance: float = 10000.0, total_trading_days: int = None,
    general_columns: list = None,
) -> pd.DataFrame:
    """For each available exit trigger, generate trades with current entry and compute KPIs."""
    exit_triggers = get_confluence_exit_triggers(groups)
    all_trigger_defs = get_all_triggers(groups)
    base_entry = get_base_trigger_id(entry_trigger_confluence_id)
    results = []

    for trig_cid, trig_name in exit_triggers.items():
        tdef = all_trigger_defs.get(trig_cid)

        # Detect bar_count exits
        is_bar_count = False
        bar_count_val = None
        for g in groups:
            if g.get_trigger_id("exit") == trig_cid and g.base_template == "bar_count":
                bar_count_val = g.parameters.get("candle_count", 4)
                is_bar_count = True
                break

        if is_bar_count:
            trades = generate_trades(
                df, direction=direction, entry_trigger=base_entry,
                exit_triggers=[], bar_count_exit=bar_count_val,
                risk_per_trade=risk_per_trade, stop_config=stop_config,
                target_config=target_config,
                general_columns=general_columns,
            )
        else:
            base_exit = get_base_trigger_id(trig_cid)
            trades = generate_trades(
                df, direction=direction, entry_trigger=base_entry,
                exit_triggers=[base_exit], bar_count_exit=None,
                risk_per_trade=risk_per_trade, stop_config=stop_config,
                target_config=target_config,
                general_columns=general_columns,
            )

        if len(trades) == 0:
            continue
        kpis = calculate_kpis(trades, starting_balance=starting_balance,
                              risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)
        results.append({
            'trigger_id': trig_cid,
            'trigger_name': trig_name,
            'execution': tdef.execution if tdef else 'bar_close',
            'total_trades': kpis['total_trades'],
            'win_rate': kpis['win_rate'],
            'profit_factor': kpis['profit_factor'],
            'avg_r': kpis['avg_r'],
            'daily_r': kpis['daily_r'],
            'r_squared': kpis['r_squared'],
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('profit_factor', ascending=False, na_position='last')
    return results_df


def find_best_exit_combinations(
    df: pd.DataFrame, direction: str, entry_trigger_confluence_id: str,
    groups: list, max_depth: int = 3, min_trades: int = 5, top_n: int = 50,
    risk_per_trade: float = 100.0, stop_config: dict = None,
    target_config: dict = None, starting_balance: float = 10000.0,
    total_trading_days: int = None, general_columns: list = None,
) -> pd.DataFrame:
    """Find the best exit trigger combinations (1-3 triggers) automatically."""
    exit_triggers = get_confluence_exit_triggers(groups)
    base_entry = get_base_trigger_id(entry_trigger_confluence_id)
    all_cids = list(exit_triggers.keys())

    if len(all_cids) == 0:
        return pd.DataFrame()

    # Pre-classify each exit trigger as bar_count or signal
    exit_info = {}
    for cid in all_cids:
        is_bar_count = False
        bar_count_val = None
        for g in groups:
            if g.get_trigger_id("exit") == cid and g.base_template == "bar_count":
                bar_count_val = g.parameters.get("candle_count", 4)
                is_bar_count = True
                break
        exit_info[cid] = {'is_bar_count': is_bar_count, 'bar_count_val': bar_count_val,
                          'name': exit_triggers[cid]}

    results = []
    for depth in range(1, min(max_depth + 1, len(all_cids) + 1)):
        for combo in combinations(all_cids, depth):
            # Separate bar_count from signal exits; allow at most one bar_count per combo
            bar_count_exits = [c for c in combo if exit_info[c]['is_bar_count']]
            signal_exits = [c for c in combo if not exit_info[c]['is_bar_count']]

            if len(bar_count_exits) > 1:
                continue  # skip combos with multiple bar_count exits

            bar_count_exit_val = exit_info[bar_count_exits[0]]['bar_count_val'] if bar_count_exits else None
            signal_base_ids = [get_base_trigger_id(c) for c in signal_exits]

            trades = generate_trades(
                df, direction=direction, entry_trigger=base_entry,
                exit_triggers=signal_base_ids, bar_count_exit=bar_count_exit_val,
                risk_per_trade=risk_per_trade, stop_config=stop_config,
                target_config=target_config, general_columns=general_columns,
            )

            if len(trades) < min_trades:
                continue

            kpis = calculate_kpis(trades, starting_balance=starting_balance,
                                  risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)

            combo_names = [exit_info[c]['name'] for c in combo]
            results.append({
                'combination': set(combo),
                'combo_str': " + ".join(sorted(combo_names)),
                'depth': depth,
                **kpis,
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values(
            ['profit_factor', 'total_trades'], ascending=[False, False], na_position='last'
        ).head(top_n)
    return results_df


def analyze_risk_management(
    df: pd.DataFrame, direction: str, entry_trigger: str,
    exit_triggers: list, bar_count_exit: int,
    groups: list, risk_per_trade: float = 100.0,
    confluence_required: set = None,
    starting_balance: float = 10000.0, total_trading_days: int = None,
    mode: str = "stop",
    base_stop_config: dict = None, base_target_config: dict = None,
    general_columns: list = None,
) -> pd.DataFrame:
    """
    For each enabled Risk Management Pack, generate trades varying either
    stop_config or target_config and compute KPIs.

    mode="stop"  → vary stop_config across packs, hold target_config fixed
    mode="target" → vary target_config across packs, hold stop_config fixed
    """
    rm_packs = rmp_module.load_risk_management_packs()
    enabled_packs = rmp_module.get_enabled_risk_management_packs(rm_packs)
    base_entry = get_base_trigger_id(entry_trigger) if entry_trigger else None
    if base_entry is None:
        return pd.DataFrame()

    # Build exit trigger list
    base_exits = []
    bar_count_val = None
    if exit_triggers:
        for ecid in exit_triggers:
            is_bc = False
            for g in groups:
                if g.get_trigger_id("exit") == ecid and g.base_template == "bar_count":
                    bar_count_val = g.parameters.get("candle_count", 4)
                    is_bc = True
                    break
            if not is_bc:
                base_exits.append(get_base_trigger_id(ecid))
    if bar_count_exit and bar_count_val is None:
        bar_count_val = bar_count_exit

    results = []

    for pack in enabled_packs:
        if mode == "stop":
            sc = pack.get_stop_config()
            tc = base_target_config
        else:
            sc = base_stop_config
            tc = pack.get_target_config()

        trades = generate_trades(
            df, direction=direction, entry_trigger=base_entry,
            exit_triggers=base_exits if base_exits else None,
            bar_count_exit=bar_count_val,
            risk_per_trade=risk_per_trade, stop_config=sc,
            target_config=tc, confluence_required=confluence_required,
            general_columns=general_columns,
        )
        if len(trades) == 0:
            continue
        kpis = calculate_kpis(trades, starting_balance=starting_balance,
                              risk_per_trade=risk_per_trade, total_trading_days=total_trading_days)

        stop_summary = rmp_module.format_stop_summary(pack)
        target_summary = rmp_module.format_target_summary(pack)

        results.append({
            'pack_id': pack.id,
            'pack_name': pack.name,
            'stop_summary': stop_summary,
            'target_summary': target_summary,
            'stop_config': sc,
            'target_config': tc,
            'total_trades': kpis['total_trades'],
            'win_rate': kpis['win_rate'],
            'profit_factor': kpis['profit_factor'],
            'avg_r': kpis['avg_r'],
            'daily_r': kpis['daily_r'],
            'r_squared': kpis['r_squared'],
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('profit_factor', ascending=False, na_position='last')
    return results_df


# =============================================================================
# CONFLUENCE FILTER DIALOG & HELPERS
# =============================================================================

@st.dialog("Filter & Sort")
def confluence_filter_dialog(show_auto_search_options: bool = False):
    """Lightbox dialog for confluence drill-down filter and sort settings."""
    filters = st.session_state.confluence_filters

    st.markdown("**Sort**")
    sort_options = ["Profit Factor", "Win Rate", "Daily R", "R² Smoothness", "Trades", "Avg R"]
    sort_map = {
        "Profit Factor": "profit_factor", "Win Rate": "win_rate",
        "Daily R": "daily_r", "R² Smoothness": "r_squared",
        "Trades": "total_trades", "Avg R": "avg_r"
    }
    reverse_sort_map = {v: k for k, v in sort_map.items()}
    current_sort_label = reverse_sort_map.get(filters['sort_by'], "Profit Factor")
    sort_by = st.selectbox("Sort by", sort_options, index=sort_options.index(current_sort_label))
    sort_dir = st.radio("Direction", ["Descending", "Ascending"], horizontal=True,
                        index=0 if not filters['sort_ascending'] else 1)

    st.markdown("**Minimum Thresholds**")
    min_trades = st.number_input("Min Trades", min_value=1, max_value=100,
                                  value=filters.get('min_trades', 3))
    min_wr = st.number_input("Min Win Rate (%)", min_value=0.0, max_value=100.0,
                              value=filters.get('min_win_rate') or 0.0, step=5.0)
    min_pf = st.number_input("Min Profit Factor", min_value=0.0, max_value=100.0,
                              value=filters.get('min_profit_factor') or 0.0, step=0.1)
    min_dr = st.number_input("Min Daily R", min_value=-10.0, max_value=10.0,
                              value=filters.get('min_daily_r') or -10.0, step=0.05)
    min_r2 = st.number_input("Min R²", min_value=0.0, max_value=1.0,
                              value=filters.get('min_r_squared') or 0.0, step=0.05)

    if show_auto_search_options:
        st.markdown("**Auto-Search Settings**")
        max_depth = st.slider("Max factors", 1, 4, filters.get('max_depth', 2))
    else:
        max_depth = filters.get('max_depth', 2)

    if st.button("Apply Filters", type="primary", use_container_width=True):
        st.session_state.confluence_filters = {
            'sort_by': sort_map[sort_by],
            'sort_ascending': sort_dir == "Ascending",
            'min_trades': int(min_trades),
            'min_win_rate': min_wr if min_wr > 0 else None,
            'min_profit_factor': min_pf if min_pf > 0 else None,
            'min_daily_r': min_dr if min_dr > -10 else None,
            'min_r_squared': min_r2 if min_r2 > 0 else None,
            'max_depth': max_depth,
            'search_query': st.session_state.confluence_filters.get('search_query', ''),
        }
        st.rerun()


def apply_confluence_filters(df: pd.DataFrame, filters: dict, search_query: str, groups: list) -> pd.DataFrame:
    """Apply search, KPI filters, and sorting to confluence results."""
    if len(df) == 0:
        return df

    # Text search on formatted display name
    if search_query.strip():
        query_lower = search_query.strip().lower()
        if 'confluence' in df.columns:
            df = df[df['confluence'].apply(
                lambda c: query_lower in format_confluence_record(c, groups).lower()
            )]
        elif 'combo_str' in df.columns:
            df = df[df['combination'].apply(
                lambda combo: any(query_lower in format_confluence_record(c, groups).lower() for c in combo)
            )]
        elif 'trigger_name' in df.columns:
            df = df[df['trigger_name'].str.lower().str.contains(query_lower, na=False)]
        elif 'pack_name' in df.columns:
            df = df[df['pack_name'].str.lower().str.contains(query_lower, na=False)]

    # KPI filters
    if filters.get('min_win_rate') is not None:
        df = df[df['win_rate'] >= filters['min_win_rate']]
    if filters.get('min_profit_factor') is not None:
        df = df[df['profit_factor'] >= filters['min_profit_factor']]
    if filters.get('min_daily_r') is not None:
        df = df[df['daily_r'] >= filters['min_daily_r']]
    if filters.get('min_r_squared') is not None:
        df = df[df['r_squared'] >= filters['min_r_squared']]

    # Sort
    sort_col = filters.get('sort_by', 'profit_factor')
    ascending = filters.get('sort_ascending', False)
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    return df


# =============================================================================
# CHART RENDERING
# =============================================================================

CANDLE_PRESET_OPTIONS = ["Default", "50", "100", "200", "400", "All"]

def render_candle_selector(chart_key: str) -> int:
    """Render a compact visible-candles selector and return the selected value.

    Returns the number of visible candles (0 = All).
    "Default" uses the global sidebar preset (returns None → caller passes None to render_price_chart).
    """
    _, right = st.columns([5, 1])
    with right:
        choice = st.selectbox(
            "Candles",
            CANDLE_PRESET_OPTIONS,
            index=0,
            key=f"candle_sel_{chart_key}",
            label_visibility="collapsed",
        )
    if choice == "Default":
        return None
    if choice == "All":
        return 0
    return int(choice)


@st.fragment
def render_chart_with_candle_selector(
    df, trades, config, show_indicators=None, indicator_colors=None,
    chart_key='price_chart', secondary_panes=None, extra_markers=None
):
    """Render a price chart with a per-chart candle count selector.

    Wrapped in @st.fragment so changing the selector only reruns the chart,
    not the full page (which would reset the active tab).
    """
    vc = render_candle_selector(chart_key)
    render_price_chart(
        df, trades, config,
        show_indicators=show_indicators,
        indicator_colors=indicator_colors,
        chart_key=chart_key,
        secondary_panes=secondary_panes,
        visible_candles=vc,
        extra_markers=extra_markers,
    )


def render_price_chart(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    config: dict,
    show_indicators: list = None,
    indicator_colors: dict = None,
    chart_key: str = 'price_chart',
    secondary_panes: list = None,
    visible_candles: int = None,
    extra_markers: list = None,
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
        visible_candles: Override for number of visible candles (None = use global default from session state)
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

    # Apply visible candles preset — trim to last N candles.
    # The lightweight-charts component always calls fitContent() on render,
    # so barSpacing cannot control the initial zoom. Instead we limit the
    # data to the last N candles so fitContent() fits only those.
    if visible_candles is None:
        visible_candles = st.session_state.get('chart_visible_candles', 200)
    if visible_candles > 0 and len(candles) > visible_candles:
        candles = candles.tail(visible_candles).reset_index(drop=True)

    candle_data = candles[['time', 'open', 'high', 'low', 'close']].to_dict('records')

    # Time window for filtering markers and secondary pane data
    min_time = candles['time'].min() if len(candles) > 0 else 0

    # Create entry/exit markers from trades (only within visible window)
    markers = []
    direction = config.get('direction', 'LONG')

    if len(trades) > 0:
        for _, trade in trades.iterrows():
            entry_time = int(pd.to_datetime(trade['entry_time']).timestamp())
            exit_time = int(pd.to_datetime(trade['exit_time']).timestamp())

            # Skip trades entirely outside the visible window
            if exit_time < min_time:
                continue

            # Entry marker
            if entry_time >= min_time:
                markers.append({
                    'time': entry_time,
                    'position': 'belowBar' if direction == 'LONG' else 'aboveBar',
                    'color': '#2196F3',
                    'shape': 'arrowUp' if direction == 'LONG' else 'arrowDown',
                    'text': 'Entry'
                })

            # Exit marker
            is_win = trade.get('win', trade.get('pnl', 0) > 0)
            markers.append({
                'time': exit_time,
                'position': 'aboveBar' if direction == 'LONG' else 'belowBar',
                'color': '#4CAF50' if is_win else '#f44336',
                'shape': 'arrowDown' if direction == 'LONG' else 'arrowUp',
                'text': f"{trade['r_multiple']:+.1f}R"
            })

    # Append any extra markers (e.g., condition state changes)
    if extra_markers:
        for em in extra_markers:
            em_time = em.get('time', 0)
            if em_time >= min_time:
                markers.append(em)

    time_scale_opts = {
        "borderColor": "#2B2B2B",
        "timeVisible": True,
        "secondsVisible": False,
    }

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
        "timeScale": time_scale_opts,
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
        # Trim secondary pane data to match the visible candle window
        for pane in secondary_panes:
            if visible_candles > 0 and min_time > 0:
                for s in pane.get("series", []):
                    s["data"] = [d for d in s["data"] if d["time"] >= min_time]
        chart_panes.extend(secondary_panes)

    renderLightweightCharts(chart_panes, key=chart_key)


# =============================================================================
# SETTINGS STORAGE
# =============================================================================

def load_settings() -> dict:
    """Load user settings from config/settings.json, falling back to defaults."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
            return {**SETTINGS_DEFAULTS, **saved}
        except (json.JSONDecodeError, Exception):
            pass
    return dict(SETTINGS_DEFAULTS)


def save_settings(settings: dict) -> bool:
    """Save user settings to config/settings.json."""
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False


# =============================================================================
# STRATEGY STORAGE
# =============================================================================

def load_strategies() -> list:
    """Load saved strategies from file."""
    if os.path.exists(STRATEGIES_FILE):
        try:
            with open(STRATEGIES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return []
    return []


def save_strategy(strategy: dict):
    """Save a strategy to file."""
    strategies = load_strategies()

    # Add timestamp and ID (max+1 is safe after deletions)
    strategy['id'] = max((s.get('id', 0) for s in strategies), default=0) + 1
    strategy['created_at'] = datetime.now().isoformat()

    # Forward testing is always on
    strategy['forward_testing'] = True
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

            # Forward testing is always on; reset boundary on edit
            updated_strategy['forward_testing'] = True
            updated_strategy['forward_test_start'] = datetime.now().isoformat()

            if 'confluence' in updated_strategy and isinstance(updated_strategy['confluence'], set):
                updated_strategy['confluence'] = list(updated_strategy['confluence'])

            strategies[i] = updated_strategy

            with open(STRATEGIES_FILE, 'w') as f:
                json.dump(strategies, f, indent=2)
            return True

    return False


def refresh_strategy_data(strategy_id: int) -> bool:
    """Incrementally refresh strategy data.

    If stored_trades exist: only processes new forward test data
    (small data window) and appends new trades.
    If not (migration): does a full refresh and populates stored_trades.

    Does NOT modify forward_test_start or any configuration fields.
    """
    strategies = load_strategies()

    for i, strat in enumerate(strategies):
        if strat.get('id') != strategy_id:
            continue
        if 'entry_trigger_confluence_id' not in strat:
            return False  # legacy strategy

        existing_stored = strat.get('stored_trades')

        if existing_stored:
            # --- INCREMENTAL PATH ---
            # Find the last known trade entry time
            last_entry_dt = max(
                pd.Timestamp(t['entry_time']) for t in existing_stored
            )

            # Generate only new trades since last known entry
            new_trades = _generate_incremental_trades(strat, last_entry_dt)

            if len(new_trades) > 0:
                new_records = _extract_minimal_trades(new_trades)
                existing_stored.extend(new_records)

            # Recompute KPIs + equity curve from all stored trades
            all_trades_df = _trades_df_from_stored(existing_stored)

            boundary_dt = None
            if strat.get('forward_test_start'):
                boundary_dt = datetime.fromisoformat(
                    strat['forward_test_start'])

            kpis = calculate_kpis(
                all_trades_df,
                starting_balance=strat.get('starting_balance', 10000.0),
                risk_per_trade=strat.get('risk_per_trade', 100.0),
            )
            eq_data = extract_equity_curve_data(
                all_trades_df, boundary_dt=boundary_dt)

            strat['stored_trades'] = existing_stored
            strat['kpis'] = kpis
            strat['equity_curve_data'] = eq_data
        else:
            # --- COLD START (migration) ---
            trades = get_strategy_trades(strat)

            boundary_dt = None
            if strat.get('forward_test_start'):
                boundary_dt = datetime.fromisoformat(
                    strat['forward_test_start'])

            total_days = None
            if strat.get('forward_testing') and strat.get('forward_test_start'):
                df, _, _, _ = prepare_forward_test_data(strat)
                if len(df) > 0:
                    total_days = count_trading_days(df)

            kpis = calculate_kpis(
                trades,
                starting_balance=strat.get('starting_balance', 10000.0),
                risk_per_trade=strat.get('risk_per_trade', 100.0),
                total_trading_days=total_days,
            )
            eq_data = extract_equity_curve_data(
                trades, boundary_dt=boundary_dt)

            strat['stored_trades'] = _extract_minimal_trades(trades)
            strat['kpis'] = kpis
            strat['equity_curve_data'] = eq_data

        strat['data_refreshed_at'] = datetime.now().isoformat()
        strategies[i] = strat

        with open(STRATEGIES_FILE, 'w') as f:
            json.dump(strategies, f, indent=2)
        return True

    return False


def bulk_refresh_all_strategies(progress_callback=None) -> dict:
    """Refresh data for all non-legacy strategies.

    Args:
        progress_callback: Optional function(current, total, strategy_name)

    Returns dict with 'success_count', 'skipped_count', 'failed_ids',
    'total_processed'.
    """
    strategies = load_strategies()
    processable = [s for s in strategies if 'entry_trigger_confluence_id' in s]
    skipped = len(strategies) - len(processable)
    success = 0
    failed = []

    for idx, strat in enumerate(processable):
        sid = strat.get('id')
        if progress_callback:
            progress_callback(idx + 1, len(processable),
                              strat.get('name', f'Strategy {sid}'))
        try:
            if refresh_strategy_data(sid):
                success += 1
            else:
                failed.append(sid)
        except Exception:
            failed.append(sid)

    return {
        'success_count': success,
        'skipped_count': skipped,
        'failed_ids': failed,
        'total_processed': len(processable),
    }


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
    # Clear forward test boundary from cloned equity curve
    if 'equity_curve_data' in new_strategy and new_strategy['equity_curve_data']:
        new_strategy['equity_curve_data']['boundary_index'] = None

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
    if 'builder_data_loaded' not in st.session_state:
        st.session_state.builder_data_loaded = False
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
    # Load persistent settings (falls back to SETTINGS_DEFAULTS for any missing keys)
    if 'chart_visible_candles' not in st.session_state:
        for key, value in load_settings().items():
            if key not in st.session_state:
                st.session_state[key] = value
    if 'confluence_filters' not in st.session_state:
        st.session_state.confluence_filters = {
            'sort_by': 'profit_factor',
            'sort_ascending': False,
            'min_trades': 3,
            'min_win_rate': None,
            'min_profit_factor': None,
            'min_daily_r': None,
            'min_r_squared': None,
            'max_depth': 2,
            'search_query': '',
        }
    if 'entry_trigger_results' not in st.session_state:
        st.session_state.entry_trigger_results = None
    if 'exit_trigger_results' not in st.session_state:
        st.session_state.exit_trigger_results = None
    if 'auto_exit_results' not in st.session_state:
        st.session_state.auto_exit_results = None
    if 'sl_results' not in st.session_state:
        st.session_state.sl_results = None
    if 'tp_results' not in st.session_state:
        st.session_state.tp_results = None

    # --- Top-level navigation ---
    SECTIONS = ["Dashboard", "Confluence Packs", "Strategies", "Portfolios", "Alerts", "Settings"]
    SECTION_SUB_PAGES = {
        "Confluence Packs": ["TF Confluence", "General", "Risk Management"],
        "Strategies": ["Strategy Builder", "My Strategies"],
        "Portfolios": ["My Portfolios", "Portfolio Requirements"],
        "Alerts": ["Alerts & Signals", "Webhook Templates"],
    }
    NAV_TARGET_MAP = {
        "Dashboard": ("Dashboard", None),
        "Strategy Builder": ("Strategies", "Strategy Builder"),
        "My Strategies": ("Strategies", "My Strategies"),
        "Portfolios": ("Portfolios", "My Portfolios"),
        "Portfolio Requirements": ("Portfolios", "Portfolio Requirements"),
        "Alerts & Signals": ("Alerts", "Alerts & Signals"),
        "Webhook Templates": ("Alerts", "Webhook Templates"),
        "Confluence Packs": ("Confluence Packs", "TF Confluence"),
        "Settings": ("Settings", None),
    }

    # Process nav_target — write directly to widget keys for programmatic navigation
    if st.session_state.nav_target and st.session_state.nav_target in NAV_TARGET_MAP:
        target_section, target_sub = NAV_TARGET_MAP[st.session_state.nav_target]
        st.session_state["main_nav"] = target_section
        if target_sub:
            st.session_state[f"sub_nav_{target_section.lower()}"] = target_sub
        st.session_state.nav_target = None

    # Sidebar — minimal base (app title + data source)
    with st.sidebar:
        st.title("RoR Trader")
        st.caption("Return on Risk Trader")

        data_source = get_data_source()
        if is_alpaca_configured():
            st.success(f"{data_source}")
            st.caption("Free plan: IEX data \u00b7 Paid plan: SIP (all exchanges)")
        else:
            st.warning(f"{data_source}")

    # Top navigation bar
    section = st.radio(
        "Navigation",
        SECTIONS,
        key="main_nav",
        horizontal=True,
        label_visibility="collapsed",
    )

    # Helper for sub-navigation within a section
    def render_sub_nav(section_name):
        sub_pages = SECTION_SUB_PAGES[section_name]
        sub_key = f"sub_nav_{section_name.lower()}"
        return st.radio(
            f"{section_name} pages",
            sub_pages,
            key=sub_key,
            horizontal=True,
            label_visibility="collapsed",
        )

    # Main content dispatch
    if section == "Dashboard":
        render_dashboard()
    elif section == "Confluence Packs":
        sub = render_sub_nav("Confluence Packs")
        if sub == "TF Confluence":
            render_confluence_groups()
        elif sub == "General":
            render_general_packs()
        elif sub == "Risk Management":
            render_risk_management_packs()
    elif section == "Strategies":
        sub = render_sub_nav("Strategies")
        if sub == "Strategy Builder":
            render_strategy_builder()
        else:
            render_my_strategies()
    elif section == "Portfolios":
        sub = render_sub_nav("Portfolios")
        if sub == "My Portfolios":
            render_portfolios()
        else:
            render_requirements_page()
    elif section == "Alerts":
        sub = render_sub_nav("Alerts")
        if sub == "Alerts & Signals":
            render_alerts_page()
        else:
            render_webhook_templates_page()
    elif section == "Settings":
        render_settings()


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

        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Win Rate", f"{best_kpis.get('win_rate', 0):.1f}%")
        pf = best_kpis.get('profit_factor', 0)
        kpi_cols[1].metric("Profit Factor", "∞" if pf == float('inf') else f"{pf:.2f}")
        kpi_cols[2].metric("Daily R", f"{best_kpis.get('daily_r', 0):+.2f}")
        kpi_cols[3].metric("Trades", best_kpis.get('total_trades', 0))
        max_rdd = best_kpis.get('max_r_drawdown', 0)
        kpi_cols[4].metric("Max R DD", f"{max_rdd:+.1f}R")

        # Mini equity curve for best strategy (prefer persisted data)
        eq_data = best_strat.get('equity_curve_data')
        if eq_data and len(eq_data.get('exit_times', [])) > 0:
            render_mini_equity_curve_from_data(eq_data, key="dash_best_strat_eq")
        elif 'entry_trigger_confluence_id' in best_strat:
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
            st.caption(f"{len(best_port.get('strategies', []))} strategies | \\${best_port.get('starting_balance', 0):,.0f} starting balance")

            pkpi_cols = st.columns(4)
            pkpi_cols[0].metric("Total P&L", f"${best_port_kpis.get('total_pnl', 0):+,.0f}")
            pkpi_cols[1].metric("Max DD", f"{best_port_kpis.get('max_drawdown_pct', 0):.1f}%")
            pkpi_cols[2].metric("Win Rate", f"{best_port_kpis.get('win_rate', 0):.1f}%")
            pkpi_cols[3].metric("Avg Daily P&L", f"${best_port_kpis.get('avg_daily_pnl', 0):+,.0f}")

            if st.button("View Portfolio", key="dash_view_best_port"):
                st.session_state.viewing_portfolio_id = best_port['id']
                st.session_state.nav_target = "Portfolios"
                st.rerun()

    with right_col:
        # Recent Alerts
        st.subheader("Recent Alerts")
        last_alerts = alerts[:5]
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
            st.session_state.builder_data_loaded = False
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


def render_strategy_builder():
    """Render the single-page strategy builder with sidebar config panel."""

    editing_id = st.session_state.get('editing_strategy_id')
    edit_config = st.session_state.get('strategy_config', {})

    # =========================================================================
    # SHARED DATA: triggers & groups (needed by both rows and sidebar)
    # =========================================================================
    enabled_groups = get_enabled_groups()
    all_trigger_defs = get_all_triggers(enabled_groups)
    entry_triggers = get_confluence_entry_triggers(
        edit_config.get('direction', 'LONG'), enabled_groups)
    all_trigger_map = {tid: tdef for tid, tdef in all_trigger_defs.items()}
    exit_trigger_display = {
        tid: f"{tdef.name} [{'C' if tdef.execution == 'bar_close' else 'I'}]"
        for tid, tdef in all_trigger_defs.items()
    }

    # =========================================================================
    # EDITING BANNER
    # =========================================================================
    if editing_id:
        editing_strat = get_strategy_by_id(editing_id)
        if editing_strat:
            _eb1, _eb2 = st.columns([5, 1])
            _eb1.info(f"Editing: **{editing_strat['name']}**")
            if _eb2.button("Cancel Edit", key="cancel_edit_builder"):
                st.session_state.editing_strategy_id = None
                st.session_state.builder_data_loaded = False
                st.session_state.strategy_config = {}
                st.session_state.selected_confluences = set()
                st.session_state.pop('sb_additional_exits', None)
                st.rerun()

    # =========================================================================
    # ROW 1: Method | Ticker | TF | Dir | Lookback | Params | Name | FT/AL | Load
    # =========================================================================
    r1c0, r1c1, r1c2, r1c3, r1c4, r1c5, r1c6, r1c7, r1c8 = st.columns(
        [0.6, 0.8, 0.8, 0.55, 0.7, 1.2, 1.4, 0.45, 0.5])

    with r1c0:
        strategy_origin = st.selectbox(
            "Method", ["Standard"], index=0,
            help="Strategy methodology (more methods coming soon)",
        )

    with r1c1:
        symbol_idx = AVAILABLE_SYMBOLS.index(edit_config['symbol']) if edit_config.get('symbol') in AVAILABLE_SYMBOLS else 0
        symbol = st.selectbox("Ticker", AVAILABLE_SYMBOLS, index=symbol_idx, key="sb_symbol")

    with r1c2:
        tf_idx = TIMEFRAMES.index(edit_config['timeframe']) if edit_config.get('timeframe') in TIMEFRAMES else 0
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=tf_idx, key="sb_timeframe")

    with r1c3:
        direction_idx = DIRECTIONS.index(edit_config['direction']) if edit_config.get('direction') in DIRECTIONS else 0
        direction = st.selectbox("Direction", DIRECTIONS, index=direction_idx, key="sb_direction")

    # Re-fetch entry triggers with actual selected direction
    entry_triggers = get_confluence_entry_triggers(direction, enabled_groups)

    with r1c4:
        saved_lookback_mode = edit_config.get('lookback_mode', 'Days')
        lb_idx = LOOKBACK_MODES.index(saved_lookback_mode) if saved_lookback_mode in LOOKBACK_MODES else 0
        lookback_mode = st.selectbox("Lookback", LOOKBACK_MODES, index=lb_idx, key="sb_lookback_mode")

    start_date = None
    end_date = None
    bar_count = None

    with r1c5:
        if lookback_mode == "Days":
            saved_data_days = edit_config.get('data_days', 30)
            data_days = st.number_input("Days", min_value=7, max_value=1825,
                                         value=saved_data_days, step=7, key="sb_data_days")
        elif lookback_mode == "Bars/Candles":
            saved_bar_count = edit_config.get('bar_count', 1000)
            bar_count = st.number_input("Bars", min_value=100, max_value=500000,
                                         value=saved_bar_count, step=100, key="sb_bar_count")
            data_days = days_from_bar_count(bar_count, timeframe)
        elif lookback_mode == "Date Range":
            from datetime import time as dtime
            saved_start = edit_config.get('lookback_start_date')
            saved_end = edit_config.get('lookback_end_date')
            default_start = datetime.fromisoformat(saved_start).date() if saved_start else date(2025, 1, 1)
            default_end = datetime.fromisoformat(saved_end).date() if saved_end else date.today()
            dr_c1, dr_c2 = st.columns(2)
            with dr_c1:
                start_date_input = st.date_input("Start", value=default_start,
                                                  min_value=date(2016, 1, 1), key="sb_start_date")
            with dr_c2:
                end_date_input = st.date_input("End", value=default_end,
                                                min_value=date(2016, 1, 1), key="sb_end_date")
            if start_date_input >= end_date_input:
                st.error("Start must be before end.")
            start_date = datetime.combine(start_date_input, dtime(9, 30))
            end_date = datetime.combine(end_date_input, dtime(16, 0))
            data_days = (end_date_input - start_date_input).days

    with r1c6:
        _existing = load_strategies()
        _next_id = max((s.get('id', 0) for s in _existing), default=0) + 1
        default_name = edit_config.get('name', f"{symbol} {direction} - {_next_id}")
        if editing_id:
            editing_s = get_strategy_by_id(editing_id)
            default_name = editing_s.get('name', default_name) if editing_s else default_name
        strategy_name = st.text_input("Name", value=default_name, key="sb_name")

    with r1c7:
        # Forward testing is always on — alerts fire when strategy is in a
        # portfolio with active webhooks
        enable_forward = True
        enable_alerts = True

    with r1c8:
        st.write("")  # vertical spacer to align button
        load_clicked = st.button("Load Data", type="primary", use_container_width=True)

    # Bar estimate (computed now, rendered after validation via placeholder)
    est_bars = estimate_bar_count(data_days, timeframe)
    _status_placeholder = st.empty()

    # =========================================================================
    # ROW 2 (collapsible): Entry Trigger | Exit Trigger | Stop Loss | Target
    # =========================================================================

    # --- Pending overrides (process before rendering widgets) ---
    if 'pending_stop_config' in st.session_state:
        pending_sc = st.session_state.pop('pending_stop_config')
        edit_config['stop_config'] = pending_sc
        st.session_state.strategy_config = dict(edit_config)
        for k in ['sb_stop_method', 'sb_stop_atr', 'sb_stop_dollar',
                  'sb_stop_pct', 'sb_stop_lookback', 'sb_stop_padding']:
            st.session_state.pop(k, None)
    if 'pending_target_config' in st.session_state:
        pending_tc = st.session_state.pop('pending_target_config')
        edit_config['target_config'] = pending_tc
        st.session_state.strategy_config = dict(edit_config)
        for k in ['sb_target_method', 'sb_target_rr', 'sb_target_atr',
                  'sb_target_dollar', 'sb_target_pct', 'sb_target_lookback',
                  'sb_target_padding']:
            st.session_state.pop(k, None)
    if 'pending_entry_trigger' in st.session_state:
        _pending_entry_idx = st.session_state.pop('pending_entry_trigger')
    else:
        _pending_entry_idx = None
    if 'pending_remove_target' in st.session_state:
        st.session_state.pop('pending_remove_target')
        _force_no_target = True
    else:
        _force_no_target = False

    with st.expander("Strategy Config", expanded=False):
        r2c1, r2c2, r2c3, r2c4 = st.columns([1.3, 1.3, 1.3, 1.3])

        # --- Entry Trigger ---
        with r2c1:
            if len(entry_triggers) == 0:
                st.warning("No entry triggers")
                entry_trigger = None
                entry_trigger_name = None
            else:
                entry_trigger_options = list(entry_triggers.keys())
                entry_trigger_labels = []
                for tid in entry_trigger_options:
                    name = entry_triggers[tid]
                    tdef = all_trigger_defs.get(tid)
                    exec_tag = "C" if not tdef or tdef.execution == "bar_close" else "I"
                    entry_trigger_labels.append(f"{name} [{exec_tag}]")
                saved_entry = edit_config.get('entry_trigger_confluence_id', '')
                if saved_entry in entry_trigger_options:
                    entry_default_idx = entry_trigger_options.index(saved_entry)
                else:
                    settings_default = st.session_state.get('default_entry_trigger', '')
                    if settings_default in entry_trigger_options:
                        entry_default_idx = entry_trigger_options.index(settings_default)
                    else:
                        entry_default_idx = 0
                if _pending_entry_idx is not None and 0 <= _pending_entry_idx < len(entry_trigger_options):
                    entry_default_idx = _pending_entry_idx
                entry_trigger_idx = st.selectbox(
                    "Entry Trigger",
                    range(len(entry_trigger_options)),
                    index=entry_default_idx,
                    format_func=lambda i: entry_trigger_labels[i],
                    key="sb_entry_trigger",
                )
                entry_trigger = entry_trigger_options[entry_trigger_idx]
                entry_trigger_name = entry_triggers[entry_trigger]

        # --- Exit Trigger (primary) ---
        with r2c2:
            exit_options = list(exit_trigger_display.keys())
            exit_labels = list(exit_trigger_display.values())
            has_exit_triggers = len(exit_options) > 0

            if not has_exit_triggers:
                st.warning("No exit triggers")
            else:
                saved_exit_cids = edit_config.get('exit_trigger_confluence_ids', [])
                if not saved_exit_cids and edit_config.get('exit_trigger_confluence_id'):
                    saved_exit_cids = [edit_config['exit_trigger_confluence_id']]
                saved_primary = saved_exit_cids[0] if saved_exit_cids else ''
                if saved_primary in exit_options:
                    exit_default_idx = exit_options.index(saved_primary)
                else:
                    settings_exit_default = st.session_state.get('default_exit_trigger', '')
                    if settings_exit_default in exit_options:
                        exit_default_idx = exit_options.index(settings_exit_default)
                    else:
                        exit_default_idx = 0
                primary_exit_idx = st.selectbox(
                    "Exit Trigger",
                    range(len(exit_options)),
                    index=exit_default_idx,
                    format_func=lambda i, _labels=exit_labels: _labels[i],
                    key="sb_exit_trigger_0",
                )
                primary_exit_cid = exit_options[primary_exit_idx]
                primary_exit_name = all_trigger_map[primary_exit_cid].name if primary_exit_cid in all_trigger_map else ""

        # --- Stop Loss ---
        with r2c3:
            stop_methods = ["ATR", "Fixed $", "Pct %", "Swing"]
            stop_method_keys = ["atr", "fixed_dollar", "percentage", "swing"]
            if 'stop_config' in edit_config:
                saved_stop = edit_config['stop_config'] or {}
            else:
                saved_stop = st.session_state.get('default_stop_config') or {}
            saved_stop_method = saved_stop.get('method', 'atr')
            default_stop_idx = stop_method_keys.index(saved_stop_method) if saved_stop_method in stop_method_keys else 0

            stop_method_idx = st.selectbox(
                "Stop Loss",
                range(len(stop_methods)),
                index=default_stop_idx,
                format_func=lambda i: stop_methods[i],
                key="sb_stop_method",
            )
            stop_method = stop_method_keys[stop_method_idx]
            stop_config_dict = {"method": stop_method}

            if stop_method == "atr":
                stop_config_dict["atr_mult"] = st.number_input(
                    "ATR×", min_value=0.5, max_value=5.0,
                    value=float(saved_stop.get('atr_mult', edit_config.get('stop_atr_mult', 1.5))),
                    step=0.1, key="sb_stop_atr",
                )
            elif stop_method == "fixed_dollar":
                stop_config_dict["dollar_amount"] = st.number_input(
                    "$", min_value=0.01, max_value=100.0,
                    value=float(saved_stop.get('dollar_amount', 1.0)),
                    step=0.1, key="sb_stop_dollar",
                )
            elif stop_method == "percentage":
                stop_config_dict["percentage"] = st.number_input(
                    "%", min_value=0.01, max_value=10.0,
                    value=float(saved_stop.get('percentage', 0.5)),
                    step=0.05, key="sb_stop_pct",
                )
            elif stop_method == "swing":
                stop_config_dict["lookback"] = st.number_input(
                    "Lookback", min_value=2, max_value=50,
                    value=int(saved_stop.get('lookback', 5)),
                    step=1, key="sb_stop_lookback",
                )
                stop_config_dict["padding"] = st.number_input(
                    "Pad $", min_value=0.0, max_value=10.0,
                    value=float(saved_stop.get('padding', 0.05)),
                    step=0.01, key="sb_stop_padding",
                )

        stop_atr_mult = stop_config_dict.get('atr_mult', 1.5) if stop_method == 'atr' else 1.5

        # --- Target ---
        with r2c4:
            target_methods = ["None", "R:R", "ATR", "Fixed $", "Pct %", "Swing"]
            target_method_keys = [None, "risk_reward", "atr", "fixed_dollar", "percentage", "swing"]
            if 'target_config' in edit_config:
                saved_target = edit_config['target_config'] or {}
            else:
                saved_target = st.session_state.get('default_target_config') or {}
            saved_t_method = saved_target.get('method') if saved_target else None
            default_target_idx = target_method_keys.index(saved_t_method) if saved_t_method in target_method_keys else 0
            if _force_no_target:
                default_target_idx = 0

            target_method_idx = st.selectbox(
                "Target",
                range(len(target_methods)),
                index=default_target_idx,
                format_func=lambda i: target_methods[i],
                key="sb_target_method",
            )
            target_method = target_method_keys[target_method_idx]

            if target_method is None:
                target_config_dict = None
            else:
                target_config_dict = {"method": target_method}
                if target_method == "risk_reward":
                    target_config_dict["rr_ratio"] = st.number_input(
                        "R:R", min_value=0.5, max_value=10.0,
                        value=float(saved_target.get('rr_ratio', 2.0)),
                        step=0.5, key="sb_target_rr",
                    )
                elif target_method == "atr":
                    target_config_dict["atr_mult"] = st.number_input(
                        "ATR×", min_value=0.5, max_value=10.0,
                        value=float(saved_target.get('atr_mult', 2.0)),
                        step=0.1, key="sb_target_atr",
                    )
                elif target_method == "fixed_dollar":
                    target_config_dict["dollar_amount"] = st.number_input(
                        "$", min_value=0.01, max_value=100.0,
                        value=float(saved_target.get('dollar_amount', 2.0)),
                        step=0.1, key="sb_target_dollar",
                    )
                elif target_method == "percentage":
                    target_config_dict["percentage"] = st.number_input(
                        "%", min_value=0.01, max_value=20.0,
                        value=float(saved_target.get('percentage', 1.0)),
                        step=0.05, key="sb_target_pct",
                    )
                elif target_method == "swing":
                    target_config_dict["lookback"] = st.number_input(
                        "Lookback", min_value=2, max_value=50,
                        value=int(saved_target.get('lookback', 5)),
                        step=1, key="sb_target_lookback",
                    )
                    target_config_dict["padding"] = st.number_input(
                        "Pad $", min_value=0.0, max_value=10.0,
                        value=float(saved_target.get('padding', 0.05)),
                        step=0.01, key="sb_target_padding",
                    )

    risk_per_trade = float(edit_config.get('risk_per_trade', 100.0))
    starting_balance = float(edit_config.get('starting_balance', 10000.0))

    # =========================================================================
    # ADDITIONAL EXITS (from drill-down, stored in session state)
    # =========================================================================
    exit_trigger_selections = []
    if has_exit_triggers:
        exit_trigger_selections.append((primary_exit_cid, primary_exit_name))

    # Initialize additional exits from saved strategy
    if 'sb_additional_exits' not in st.session_state:
        saved_exit_cids = edit_config.get('exit_trigger_confluence_ids', [])
        if not saved_exit_cids and edit_config.get('exit_trigger_confluence_id'):
            saved_exit_cids = [edit_config['exit_trigger_confluence_id']]
        st.session_state.sb_additional_exits = saved_exit_cids[1:] if len(saved_exit_cids) > 1 else []

    # Process pending exit operations from drill-down
    if 'pending_add_exit' in st.session_state:
        add_cid = st.session_state.pop('pending_add_exit')
        if add_cid in exit_options and len(st.session_state.sb_additional_exits) < 2:
            st.session_state.sb_additional_exits.append(add_cid)
    if 'pending_replace_exits' in st.session_state:
        replace_cids = st.session_state.pop('pending_replace_exits')
        valid_cids = [c for c in replace_cids if c in exit_options]
        if valid_cids:
            st.session_state.sb_additional_exits = valid_cids[1:] if len(valid_cids) > 1 else []
    if 'pending_remove_exit_idx' in st.session_state:
        rm_idx = st.session_state.pop('pending_remove_exit_idx')
        addl = st.session_state.sb_additional_exits
        adj_idx = rm_idx - 1  # index 0 is primary
        if 0 <= adj_idx < len(addl):
            addl.pop(adj_idx)

    # Add additional exits to selections
    for cid in st.session_state.get('sb_additional_exits', []):
        if cid in exit_options:
            name = all_trigger_map[cid].name if cid in all_trigger_map else ""
            exit_trigger_selections.append((cid, name))

    # =========================================================================
    # VALIDATION + STATUS LINE
    # =========================================================================
    exit_cids = [cid for cid, _ in exit_trigger_selections] if exit_trigger_selections else []
    has_duplicate_exits = len(exit_cids) != len(set(exit_cids))
    entry_in_exits = entry_trigger is not None and entry_trigger in exit_cids
    bar_count_count = sum(
        1 for cid, _ in exit_trigger_selections
        if any(g.get_trigger_id("exit") == cid and g.base_template == "bar_count" for g in enabled_groups)
    )
    has_multiple_bar_count = bar_count_count > 1
    can_save = (
        entry_trigger is not None
        and has_exit_triggers
        and len(exit_trigger_selections) > 0
        and not has_duplicate_exits
        and not entry_in_exits
        and not has_multiple_bar_count
        and st.session_state.get('builder_data_loaded', False)
    )

    # Fill status line (bar estimate + validation errors)
    est_parts = [f"~{est_bars:,} bars", TIMEFRAME_GUIDANCE.get(timeframe, "")]
    if est_bars > 200_000:
        est_parts.append(":red[**Very large dataset — may be slow**]")
    elif est_bars > 50_000:
        est_parts.append(":orange[Large dataset]")
    # Warn if lookback extends before Alpaca's data floor (Days/Bars modes only)
    if lookback_mode != "Date Range":
        implied_start = date.today() - timedelta(days=data_days)
        if implied_start < ALPACA_DATA_FLOOR:
            est_parts.append(":orange[Lookback extends before 2016 — Alpaca data may be unavailable]")
    if has_duplicate_exits:
        est_parts.append(":red[Duplicate exit triggers]")
    if entry_in_exits:
        est_parts.append(":red[Entry trigger in exits]")
    if has_multiple_bar_count:
        est_parts.append(":red[Only one bar count exit]")
    _status_placeholder.caption(" · ".join(p for p in est_parts if p))

    # =========================================================================
    # BUILD CONFIG FROM WIDGETS (always, for live updates)
    # =========================================================================
    data_seed = st.session_state.get('global_data_seed', 42)
    extended_data_days = st.session_state.get('default_extended_data_days', 365)
    base_entry_trigger_id = get_base_trigger_id(entry_trigger) if entry_trigger else None

    # Separate bar_count exits from signal-based exits
    bar_count_exit_value = None
    signal_exit_base_ids = []
    signal_exit_confluence_ids = []
    signal_exit_names = []
    for cid, name in exit_trigger_selections:
        is_bar_count = False
        for g in enabled_groups:
            if g.get_trigger_id("exit") == cid and g.base_template == "bar_count":
                bar_count_exit_value = g.parameters.get("candle_count", 4)
                is_bar_count = True
                break
        if not is_bar_count:
            signal_exit_base_ids.append(get_base_trigger_id(cid))
            signal_exit_confluence_ids.append(cid)
            signal_exit_names.append(name)

    config = {
        'symbol': symbol,
        'direction': direction,
        'timeframe': timeframe,
        'entry_trigger': base_entry_trigger_id,
        'entry_trigger_confluence_id': entry_trigger,
        'exit_triggers': signal_exit_base_ids,
        'exit_trigger_confluence_ids': signal_exit_confluence_ids,
        'exit_trigger_names': signal_exit_names,
        'exit_trigger': signal_exit_base_ids[0] if signal_exit_base_ids else None,
        'exit_trigger_confluence_id': signal_exit_confluence_ids[0] if signal_exit_confluence_ids else None,
        'entry_trigger_name': entry_trigger_name,
        'exit_trigger_name': signal_exit_names[0] if signal_exit_names else None,
        'bar_count_exit': bar_count_exit_value,
        'risk_per_trade': risk_per_trade,
        'stop_atr_mult': stop_atr_mult,
        'stop_config': stop_config_dict,
        'target_config': target_config_dict,
        'starting_balance': starting_balance,
        'data_days': data_days,
        'extended_data_days': extended_data_days,
        'data_seed': data_seed,
        'lookback_mode': lookback_mode,
        'bar_count': bar_count if lookback_mode == "Bars/Candles" else None,
        'lookback_start_date': start_date.isoformat() if start_date else None,
        'lookback_end_date': end_date.isoformat() if end_date else None,
        'strategy_origin': strategy_origin.lower(),
    }

    # Handle Load Data
    if load_clicked:
        st.session_state.builder_data_loaded = True
        st.session_state.strategy_config = config
        st.session_state.entry_trigger_results = None
        st.session_state.exit_trigger_results = None
        st.session_state.auto_exit_results = None
        st.session_state.sl_results = None
        st.session_state.tp_results = None
        st.rerun()

    # =========================================================================
    # MAIN AREA
    # =========================================================================
    if not st.session_state.get('builder_data_loaded', False):
        st.info("Select your settings above, then click **Load Data** to begin analysis.")
        return

    # Keep strategy_config in sync with sidebar for the edit flow
    st.session_state.strategy_config = config

    # Header with strategy name
    entry_name = entry_trigger_name or (entry_trigger if entry_trigger else "?")
    exit_parts = list(signal_exit_names)
    if config.get('bar_count_exit'):
        exit_parts.append(f"Exit @ {config['bar_count_exit']} bars")
    exit_str = " / ".join(exit_parts) if exit_parts else "?"
    st.markdown(f"### {strategy_name}")
    st.caption(f"{symbol} | {direction} | {entry_name} → {exit_str}")

    # Load data and generate trades
    with st.spinner("Loading market data and running analysis..."):
        df = prepare_data_with_indicators(symbol, data_days, data_seed,
                                          start_date=start_date, end_date=end_date,
                                          timeframe=timeframe)

        if len(df) == 0:
            st.error("No data available")
            return

        general_cols = [c for c in df.columns if c.startswith("GP_")]
        trades = generate_trades(
            df,
            direction=direction,
            entry_trigger=config['entry_trigger'],
            exit_trigger=config.get('exit_trigger'),
            exit_triggers=config.get('exit_triggers'),
            confluence_required=None,
            risk_per_trade=risk_per_trade,
            stop_atr_mult=stop_atr_mult,
            stop_config=stop_config_dict,
            target_config=target_config_dict,
            bar_count_exit=config.get('bar_count_exit'),
            general_columns=general_cols,
        )

    # Apply confluence filter
    selected = st.session_state.selected_confluences
    if len(selected) > 0 and len(trades) > 0:
        mask = trades["confluence_records"].apply(lambda r: isinstance(r, set) and selected.issubset(r))
        filtered_trades = trades[mask]
    else:
        filtered_trades = trades

    # KPIs
    period_trading_days = count_trading_days(df)
    kpis = calculate_kpis(
        filtered_trades,
        starting_balance=starting_balance,
        risk_per_trade=risk_per_trade,
        total_trading_days=period_trading_days,
    )

    kpi_cols = st.columns(8)
    kpi_cols[0].metric("Trades", kpis["total_trades"])
    kpi_cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
    kpi_cols[2].metric("Profit Factor", "∞" if kpis['profit_factor'] == float('inf') else f"{kpis['profit_factor']:.2f}")
    kpi_cols[3].metric("Avg R", f"{kpis['avg_r']:+.2f}")
    kpi_cols[4].metric("Total R", f"{kpis['total_r']:+.1f}")
    kpi_cols[5].metric("Daily R", f"{kpis['daily_r']:+.2f}")
    kpi_cols[6].metric("R²", f"{kpis['r_squared']:.2f}")
    kpi_cols[7].metric("Max R DD", f"{kpis['max_r_drawdown']:+.1f}R")

    render_secondary_kpis(filtered_trades, kpis, key_prefix="builder")

    # Optimizable Variables
    with st.expander("Optimizable Variables"):
        var_cols = st.columns(6)

        # 1. Entry Trigger
        with var_cols[0]:
            st.caption("**Entry**")
            e_name = config.get('entry_trigger_name') or '?'
            st.markdown(f"_{e_name}_")

        # 2. Exit Trigger(s)
        with var_cols[1]:
            st.caption("**Exit(s)**")
            ov_exit_names = list(config.get('exit_trigger_names', []))
            ov_exit_cids = list(config.get('exit_trigger_confluence_ids', []))
            if config.get('bar_count_exit'):
                ov_exit_names.append(f"{config['bar_count_exit']}-bar exit")
                for g in enabled_groups:
                    if g.base_template == "bar_count":
                        ov_exit_cids.append(g.get_trigger_id("exit"))
                        break
                else:
                    ov_exit_cids.append(None)
            for idx_e, (ename, ecid) in enumerate(zip(ov_exit_names, ov_exit_cids)):
                e_col1, e_col2 = st.columns([4, 1])
                with e_col1:
                    st.markdown(f"_{ename}_")
                with e_col2:
                    if len(ov_exit_names) > 1:
                        if st.button("✕", key=f"var_rm_exit_{idx_e}"):
                            actual_idx = idx_e if idx_e < len(config.get('exit_trigger_confluence_ids', [])) else None
                            if actual_idx is not None:
                                st.session_state.pending_remove_exit_idx = actual_idx
                            else:
                                remaining_cids = [c for c in config.get('exit_trigger_confluence_ids', [])]
                                if remaining_cids:
                                    st.session_state.pending_replace_exits = remaining_cids
                            st.rerun()
            if not ov_exit_names:
                st.markdown("_None_")

        # 3. TF Conditions (exclude GEN- records)
        with var_cols[2]:
            st.caption("**TF Conditions**")
            tf_confs = sorted(c for c in selected if not c.startswith("GEN-"))
            if len(tf_confs) > 0:
                for conf in tf_confs:
                    c_col1, c_col2 = st.columns([4, 1])
                    with c_col1:
                        st.markdown(f"_{format_confluence_record(conf, enabled_groups)}_")
                    with c_col2:
                        if st.button("✕", key=f"var_rm_conf_{conf}"):
                            st.session_state.selected_confluences.discard(conf)
                            st.rerun()
            else:
                st.markdown("_None_")

        # 4. General Conditions
        with var_cols[3]:
            st.caption("**General**")
            gen_confs = sorted(c for c in selected if c.startswith("GEN-"))
            if len(gen_confs) > 0:
                for conf in gen_confs:
                    c_col1, c_col2 = st.columns([4, 1])
                    with c_col1:
                        st.markdown(f"_{format_confluence_record(conf, enabled_groups)}_")
                    with c_col2:
                        if st.button("✕", key=f"var_rm_gen_{conf}"):
                            st.session_state.selected_confluences.discard(conf)
                            st.rerun()
            else:
                st.markdown("_None_")

        # 5. Stop Loss
        with var_cols[4]:
            st.caption("**Stop Loss**")
            st.markdown(f"_{format_stop_display(config)}_")

        # 6. Take Profit
        with var_cols[5]:
            st.caption("**Take Profit**")
            tp_display = format_target_display(config)
            if tp_display and tp_display != "None (signal exit only)":
                tp_col1, tp_col2 = st.columns([4, 1])
                with tp_col1:
                    st.markdown(f"_{tp_display}_")
                with tp_col2:
                    if st.button("✕", key="var_rm_target"):
                        st.session_state.pending_remove_target = True
                        st.rerun()
            else:
                st.markdown("_None_")

    # Main content: Chart/Equity (left) + Confluence panel (right)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        chart_tab1, chart_tab2 = st.tabs(["Equity Curve", "Price Chart"])

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
            overlay_groups = [g for g in enabled_groups if g.base_template in OVERLAY_COMPATIBLE_TEMPLATES]

            if len(overlay_groups) > 0:
                selected_overlay_groups = st.multiselect(
                    "Show Indicators",
                    options=[g.id for g in overlay_groups],
                    default=[overlay_groups[0].id] if len(overlay_groups) > 0 else [],
                    format_func=lambda gid: next((g.name for g in overlay_groups if g.id == gid), gid),
                    help="Select confluence packs to overlay on chart"
                )

                show_indicators = []
                indicator_colors = {}
                for gid in selected_overlay_groups:
                    group = get_group_by_id(gid, enabled_groups)
                    if group:
                        cols = get_overlay_indicators_for_group(group)
                        show_indicators.extend(cols)
                        indicator_colors.update(get_overlay_colors_for_group(group))
            else:
                st.info("No overlay-compatible confluence packs enabled")
                show_indicators = []
                indicator_colors = {}

            if len(filtered_trades) > 0:
                st.caption(f"{len(filtered_trades)} trades on {symbol} ({direction})")
            else:
                st.caption(f"{symbol} price data (no trades match filters)")

            osc_panes = build_secondary_panes(df, enabled_groups)
            render_chart_with_candle_selector(df, filtered_trades, config, show_indicators=show_indicators, indicator_colors=indicator_colors,
                               secondary_panes=osc_panes if osc_panes else None)

    with right_col:
        tab_entry, tab_exit, tab_tf, tab_gen, tab_sl, tab_tp = st.tabs([
            "Entry", "Exit", "TF Conditions",
            "General", "Stop Loss", "Take Profit"
        ])

        # =================================================================
        # Tab 1: Entry Trigger
        # =================================================================
        with tab_entry:
            entry_filters = st.session_state.confluence_filters
            e_search_col, e_action_col, e_filter_col = st.columns([3, 1.5, 0.5])
            with e_search_col:
                entry_search = st.text_input(
                    "Search", placeholder="Search entry triggers...",
                    key="entry_trig_search", label_visibility="collapsed"
                )
            with e_action_col:
                entry_analyze_clicked = st.button("Analyze", type="primary",
                                                  use_container_width=True, key="analyze_entry_btn")
            with e_filter_col:
                if st.button("⚙", use_container_width=True, key="entry_filter_btn"):
                    confluence_filter_dialog(show_auto_search_options=False)

            # Active entry tag
            entry_tag_name = config.get('entry_trigger_name') or '?'
            st.caption(f"Current: **{entry_tag_name}**")

            # Build confluence set for filtering
            confluence_set = selected if len(selected) > 0 else None

            if entry_analyze_clicked:
                with st.spinner("Generating trades for each entry trigger..."):
                    entry_results = analyze_entry_triggers(
                        df, direction, enabled_groups,
                        exit_triggers=config.get('exit_triggers'),
                        bar_count_exit=config.get('bar_count_exit'),
                        risk_per_trade=risk_per_trade,
                        stop_config=stop_config_dict,
                        target_config=target_config_dict,
                        confluence_required=confluence_set,
                        starting_balance=starting_balance,
                        total_trading_days=period_trading_days,
                        general_columns=general_cols,
                    )
                st.session_state.entry_trigger_results = entry_results

            if st.session_state.entry_trigger_results is not None and len(st.session_state.entry_trigger_results) > 0:
                display_df = apply_confluence_filters(
                    st.session_state.entry_trigger_results, entry_filters, entry_search, enabled_groups
                )
                display_df = display_df.head(20)
                current_entry_cid = config.get('entry_trigger_confluence_id', '')

                for _, row in display_df.iterrows():
                    is_current = (row['trigger_id'] == current_entry_cid)
                    with st.container(border=True):
                        t1, t2, t3 = st.columns([3.5, 0.8, 0.7])
                        with t1:
                            label = f"**{row['trigger_name']}**" if is_current else row['trigger_name']
                            if is_current:
                                label += " _(current)_"
                            st.markdown(label)
                        with t2:
                            exec_tag = "Close" if row.get('execution', 'bar_close') == 'bar_close' else "Intra"
                            st.caption(exec_tag)
                        with t3:
                            if not is_current and row['trigger_id'] in entry_trigger_options:
                                if st.button("Replace", key=f"rep_entry_{row['trigger_id']}"):
                                    st.session_state.pending_entry_trigger = entry_trigger_options.index(row['trigger_id'])
                                    st.session_state.entry_trigger_results = None
                                    st.rerun()

                        k1, k2, k3, k4, k5, k6 = st.columns(6)
                        k1.caption(f"Trades: {row['total_trades']}")
                        pf = row['profit_factor']
                        k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                        k3.caption(f"WR: {row['win_rate']:.1f}%")
                        k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                        k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                        k6.caption(f"R²: {row['r_squared']:.2f}")
            else:
                st.info("Click **Analyze** to compare all available entry triggers using the current strategy config.")

        # =================================================================
        # Tab 2: Exit Triggers
        # =================================================================
        with tab_exit:
            exit_mode = st.radio("Mode", ["Drill-Down", "Auto-Search"], horizontal=True,
                                 label_visibility="collapsed", key="exit_mode_radio")

            exit_filters = st.session_state.confluence_filters
            x_search_col, x_action_col, x_filter_col = st.columns([3, 1.5, 0.5])
            with x_search_col:
                exit_search_key = "exit_dd_search" if exit_mode == "Drill-Down" else "exit_as_search"
                exit_search_ph = "Search exit triggers..." if exit_mode == "Drill-Down" else "Search combinations..."
                exit_search = st.text_input(
                    "Search", placeholder=exit_search_ph,
                    key=exit_search_key, label_visibility="collapsed"
                )
            with x_action_col:
                if exit_mode == "Drill-Down":
                    exit_action_label = "Analyze"
                    exit_action_key = "analyze_exit_btn"
                else:
                    exit_action_label = "Search"
                    exit_action_key = "find_exit_combos_btn"
                exit_action_clicked = st.button(exit_action_label, type="primary",
                                                use_container_width=True, key=exit_action_key)
            with x_filter_col:
                if st.button("⚙", use_container_width=True, key="exit_filter_btn"):
                    confluence_filter_dialog(show_auto_search_options=(exit_mode == "Auto-Search"))

            # Active exit tags
            exit_tag_names = list(config.get('exit_trigger_names', []))
            exit_tag_cids = list(config.get('exit_trigger_confluence_ids', []))
            if config.get('bar_count_exit'):
                exit_tag_names.append(f"{config['bar_count_exit']}-bar exit")
                exit_tag_cids.append(None)
            if exit_tag_names:
                tag_count = len(exit_tag_names)
                exit_tag_cols = st.columns(min(tag_count, 4))
                for i_et, (et_name, et_cid) in enumerate(zip(exit_tag_names, exit_tag_cids)):
                    with exit_tag_cols[i_et % 4]:
                        if tag_count > 1 and et_cid is not None:
                            actual_idx = i_et if i_et < len(config.get('exit_trigger_confluence_ids', [])) else None
                            if actual_idx is not None and st.button(f"✕ {et_name}", key=f"ext_rm_{et_cid}"):
                                st.session_state.pending_remove_exit_idx = actual_idx
                                st.rerun()
                        else:
                            st.caption(et_name)

            if not config.get('entry_trigger_confluence_id'):
                st.warning("Select an entry trigger first.")
            elif exit_mode == "Drill-Down":
                if exit_action_clicked:
                    with st.spinner("Generating trades for each exit trigger..."):
                        exit_results = analyze_exit_triggers(
                            df, direction,
                            entry_trigger_confluence_id=config['entry_trigger_confluence_id'],
                            groups=enabled_groups,
                            risk_per_trade=risk_per_trade,
                            stop_config=stop_config_dict,
                            target_config=target_config_dict,
                            starting_balance=starting_balance,
                            total_trading_days=period_trading_days,
                            general_columns=general_cols,
                        )
                    st.session_state.exit_trigger_results = exit_results

                if st.session_state.exit_trigger_results is not None and len(st.session_state.exit_trigger_results) > 0:
                    display_df = apply_confluence_filters(
                        st.session_state.exit_trigger_results, exit_filters, exit_search, enabled_groups
                    )
                    display_df = display_df.head(20)
                    current_exit_cids = set(config.get('exit_trigger_confluence_ids', []))
                    if config.get('bar_count_exit'):
                        for g in enabled_groups:
                            if g.base_template == "bar_count":
                                current_exit_cids.add(g.get_trigger_id("exit"))

                    for _, row in display_df.iterrows():
                        is_current = (row['trigger_id'] in current_exit_cids)
                        with st.container(border=True):
                            t1, t2, t3 = st.columns([3.5, 0.8, 0.7])
                            with t1:
                                label = f"**{row['trigger_name']}**" if is_current else row['trigger_name']
                                if is_current:
                                    label += " _(current)_"
                                st.markdown(label)
                            with t2:
                                exec_tag = "Close" if row.get('execution', 'bar_close') == 'bar_close' else "Intra"
                                st.caption(exec_tag)
                            with t3:
                                if not is_current and has_exit_triggers and len(st.session_state.get('sb_additional_exits', [])) < 2:
                                    if row['trigger_id'] in exit_options:
                                        if st.button("Add", key=f"add_exit_{row['trigger_id']}"):
                                            st.session_state.pending_add_exit = row['trigger_id']
                                            st.session_state.exit_trigger_results = None
                                            st.rerun()

                            k1, k2, k3, k4, k5, k6 = st.columns(6)
                            k1.caption(f"Trades: {row['total_trades']}")
                            pf = row['profit_factor']
                            k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                            k3.caption(f"WR: {row['win_rate']:.1f}%")
                            k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                            k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                            k6.caption(f"R²: {row['r_squared']:.2f}")
                else:
                    st.info("Click **Analyze** to compare individual exit triggers against the current entry.")

            else:  # Auto-Search
                if exit_action_clicked:
                    with st.spinner("Searching exit combinations..."):
                        best_exits = find_best_exit_combinations(
                            df, direction,
                            entry_trigger_confluence_id=config['entry_trigger_confluence_id'],
                            groups=enabled_groups,
                            max_depth=exit_filters.get('max_depth', 3),
                            min_trades=exit_filters.get('min_trades', 5),
                            top_n=50,
                            risk_per_trade=risk_per_trade,
                            stop_config=stop_config_dict,
                            target_config=target_config_dict,
                            starting_balance=starting_balance,
                            total_trading_days=period_trading_days,
                            general_columns=general_cols,
                        )
                    if len(best_exits) > 0:
                        st.session_state.auto_exit_results = best_exits

                if st.session_state.auto_exit_results is not None and len(st.session_state.auto_exit_results) > 0:
                    exit_results = apply_confluence_filters(
                        st.session_state.auto_exit_results, exit_filters, exit_search, enabled_groups
                    )
                    exit_results = exit_results.head(20)

                    for _, row in exit_results.iterrows():
                        with st.container(border=True):
                            t1, t2, t3 = st.columns([0.3, 3.7, 0.8])
                            with t1:
                                st.caption(f"D{row['depth']}")
                            with t2:
                                st.markdown(f"**{row['combo_str']}**")
                            with t3:
                                if has_exit_triggers:
                                    combo_cids = sorted(row['combination'])
                                    all_in_options = all(c in exit_options for c in combo_cids)
                                    if all_in_options:
                                        if st.button("Replace", key=f"rep_exit_{row['combo_str']}"):
                                            st.session_state.pending_replace_exits = combo_cids
                                            st.session_state.auto_exit_results = None
                                            st.rerun()

                            k1, k2, k3, k4, k5, k6 = st.columns(6)
                            k1.caption(f"Trades: {row['total_trades']}")
                            pf = row['profit_factor']
                            k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                            k3.caption(f"WR: {row['win_rate']:.1f}%")
                            k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                            k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                            k6.caption(f"R²: {row['r_squared']:.2f}")
                else:
                    st.info("Click **Search** to find optimal exit trigger combos (up to 3).")

        # =================================================================
        # Tab 3: TF Conditions (existing Drill-Down / Auto-Search)
        # =================================================================
        with tab_tf:
            mode = st.radio("Mode", ["Drill-Down", "Auto-Search"], horizontal=True, label_visibility="collapsed")

            filters = st.session_state.confluence_filters
            if mode == "Auto-Search":
                search_col, action_col, filter_col = st.columns([3, 1.5, 0.5])
            else:
                search_col, filter_col = st.columns([4, 1])
            with search_col:
                search_key = "dd_search" if mode == "Drill-Down" else "as_search"
                search_placeholder = "Search indicators..." if mode == "Drill-Down" else "Search combinations..."
                search_query = st.text_input("Search", placeholder=search_placeholder,
                                              value=filters.get('search_query', ''),
                                              key=search_key, label_visibility="collapsed")
                st.session_state.confluence_filters['search_query'] = search_query
            if mode == "Auto-Search":
                with action_col:
                    tf_search_clicked = st.button("Search", type="primary",
                                                  use_container_width=True, key="tf_auto_search_btn")
            with filter_col:
                if st.button("⚙" if mode == "Auto-Search" else "⚙ Filter",
                             use_container_width=True, key="tf_filter_btn"):
                    confluence_filter_dialog(show_auto_search_options=(mode == "Auto-Search"))

            # Active TF condition tags (exclude GEN- records)
            tf_selected = sorted(c for c in selected if not c.startswith("GEN-"))
            if len(tf_selected) > 0:
                tf_tag_cols = st.columns(min(len(tf_selected) + 1, 5))
                for i_tf, conf in enumerate(tf_selected):
                    with tf_tag_cols[i_tf % (len(tf_tag_cols) - 1)]:
                        if st.button(f"✕ {format_confluence_record(conf, enabled_groups)}", key=f"tftag_rm_{conf}"):
                            st.session_state.selected_confluences.discard(conf)
                            st.rerun()
                with tf_tag_cols[-1]:
                    if st.button("Clear TF", key="tf_clear_all"):
                        gen_keep = {c for c in st.session_state.selected_confluences if c.startswith("GEN-")}
                        st.session_state.selected_confluences = gen_keep
                        st.rerun()

            if mode == "Drill-Down":
                confluence_df = analyze_confluences(
                    trades, selected, min_trades=filters.get('min_trades', 3),
                    starting_balance=starting_balance,
                    risk_per_trade=risk_per_trade,
                    total_trading_days=period_trading_days,
                )

                # Filter out GEN- records (those belong in the General tab)
                if len(confluence_df) > 0:
                    confluence_df = confluence_df[~confluence_df['confluence'].str.startswith('GEN-')]

                if len(confluence_df) > 0:
                    confluence_df = apply_confluence_filters(confluence_df, filters, search_query, enabled_groups)
                    confluence_df = confluence_df.head(20)

                    for _, row in confluence_df.iterrows():
                        conf = row['confluence']
                        conf_display = format_confluence_record(conf, enabled_groups)
                        is_selected = conf in selected

                        with st.container(border=True):
                            top1, top2 = st.columns([4, 0.7])
                            with top1:
                                label = f"**{conf_display}**" if is_selected else conf_display
                                if is_selected:
                                    label += " _(active)_"
                                st.markdown(label)
                            with top2:
                                if not is_selected:
                                    if st.button("Add", key=f"add_{conf}"):
                                        st.session_state.selected_confluences.add(conf)
                                        st.rerun()

                            k1, k2, k3, k4, k5, k6 = st.columns(6)
                            k1.caption(f"Trades: {row['total_trades']}")
                            pf = row['profit_factor']
                            k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                            k3.caption(f"WR: {row['win_rate']:.1f}%")
                            k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                            k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                            k6.caption(f"R²: {row['r_squared']:.2f}")
                else:
                    st.info("Not enough trades for analysis")

            else:  # Auto-Search
                if tf_search_clicked:
                    with st.spinner("Searching..."):
                        best = find_best_combinations(
                            trades, filters.get('max_depth', 2), filters.get('min_trades', 5), top_n=50,
                            starting_balance=starting_balance,
                            risk_per_trade=risk_per_trade,
                            total_trading_days=period_trading_days,
                            exclude_prefix="GEN-",
                        )
                    if len(best) > 0:
                        st.session_state.auto_results = best

                if 'auto_results' in st.session_state and len(st.session_state.auto_results) > 0:
                    results = apply_confluence_filters(
                        st.session_state.auto_results, filters, search_query, enabled_groups
                    )
                    results = results.head(20)

                    for _, row in results.iterrows():
                        combo_display = format_confluence_set(row['combination'], enabled_groups)

                        with st.container(border=True):
                            t1, t2, t3 = st.columns([0.3, 3.5, 0.8])
                            with t1:
                                st.caption(f"D{row['depth']}")
                            with t2:
                                st.markdown(f"**{combo_display}**")
                            with t3:
                                if st.button("Replace", key=f"rep_tf_{row['combo_str']}"):
                                    st.session_state.selected_confluences = row['combination'].copy()
                                    st.rerun()

                            k1, k2, k3, k4, k5, k6 = st.columns(6)
                            k1.caption(f"Trades: {row['total_trades']}")
                            pf = row['profit_factor']
                            k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                            k3.caption(f"WR: {row['win_rate']:.1f}%")
                            k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                            k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                            k6.caption(f"R²: {row['r_squared']:.2f}")

        # =================================================================
        # Tab 4: General Conditions Drill-Down
        # =================================================================
        with tab_gen:
            gen_packs = gp_module.load_general_packs()
            enabled_gen = gp_module.get_enabled_general_packs(gen_packs)

            if len(enabled_gen) == 0:
                st.info("No general packs enabled. Configure them on the **General** sub-page under Confluence Packs.")
            else:
                gen_search_col, gen_filter_col = st.columns([4, 1])
                with gen_search_col:
                    gen_search = st.text_input("Search", placeholder="Search general conditions...",
                                               key="gen_search", label_visibility="collapsed")
                with gen_filter_col:
                    if st.button("⚙ Filter", use_container_width=True, key="gen_filter_btn"):
                        confluence_filter_dialog(show_auto_search_options=False)

                # Active General condition tags
                gen_selected = sorted(c for c in selected if c.startswith("GEN-"))
                if len(gen_selected) > 0:
                    gen_tag_cols = st.columns(min(len(gen_selected) + 1, 5))
                    for i_gen, conf in enumerate(gen_selected):
                        with gen_tag_cols[i_gen % (len(gen_tag_cols) - 1)]:
                            if st.button(f"✕ {format_confluence_record(conf, enabled_groups)}", key=f"gentag_rm_{conf}"):
                                st.session_state.selected_confluences.discard(conf)
                                st.rerun()
                    with gen_tag_cols[-1]:
                        if st.button("Clear Gen", key="gen_clear_all"):
                            tf_keep = {c for c in st.session_state.selected_confluences if not c.startswith("GEN-")}
                            st.session_state.selected_confluences = tf_keep
                            st.rerun()

                # Analyze — reuse the same analyze_confluences, then filter to GEN- only
                gen_filters = st.session_state.confluence_filters
                gen_confluence_df = analyze_confluences(
                    trades, selected, min_trades=gen_filters.get('min_trades', 3),
                    starting_balance=starting_balance,
                    risk_per_trade=risk_per_trade,
                    total_trading_days=period_trading_days,
                )

                if len(gen_confluence_df) > 0:
                    gen_confluence_df = gen_confluence_df[gen_confluence_df['confluence'].str.startswith('GEN-')]

                if len(gen_confluence_df) > 0:
                    gen_confluence_df = apply_confluence_filters(gen_confluence_df, gen_filters, gen_search, enabled_groups)
                    gen_confluence_df = gen_confluence_df.head(20)

                    for _, row in gen_confluence_df.iterrows():
                        conf = row['confluence']
                        conf_display = format_confluence_record(conf, enabled_groups)
                        is_selected = conf in selected

                        with st.container(border=True):
                            top1, top2 = st.columns([4, 0.7])
                            with top1:
                                label = f"**{conf_display}**" if is_selected else conf_display
                                if is_selected:
                                    label += " _(active)_"
                                st.markdown(label)
                            with top2:
                                if not is_selected:
                                    if st.button("Add", key=f"gen_add_{conf}"):
                                        st.session_state.selected_confluences.add(conf)
                                        st.rerun()

                            k1, k2, k3, k4, k5, k6 = st.columns(6)
                            k1.caption(f"Trades: {row['total_trades']}")
                            pf = row['profit_factor']
                            k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                            k3.caption(f"WR: {row['win_rate']:.1f}%")
                            k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                            k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                            k6.caption(f"R²: {row['r_squared']:.2f}")
                else:
                    st.info("No general condition data available. Make sure general packs are enabled and trades have been generated.")

        # =================================================================
        # Tab 5: Stop Loss Optimization
        # =================================================================
        with tab_sl:
            sl_filters = st.session_state.confluence_filters
            sl_search_col, sl_action_col, sl_filter_col = st.columns([3, 1.5, 0.5])
            with sl_search_col:
                sl_search = st.text_input("Search", placeholder="Search stop configs...",
                                          key="sl_search", label_visibility="collapsed")
            with sl_action_col:
                sl_analyze_clicked = st.button("Analyze", type="primary",
                                               use_container_width=True, key="analyze_sl_btn")
            with sl_filter_col:
                if st.button("⚙", use_container_width=True, key="sl_filter_btn"):
                    confluence_filter_dialog(show_auto_search_options=False)

            # Current stop tag
            stop_display = format_stop_display(config)
            st.caption(f"Current: **{stop_display}**")

            # Build exit CIDs list for the helper
            exit_cids = [cid for cid, _ in exit_trigger_selections] if exit_trigger_selections else []
            confluence_set = selected if len(selected) > 0 else None

            if sl_analyze_clicked:
                with st.spinner("Comparing stop-loss configurations..."):
                    sl_results = analyze_risk_management(
                        df, direction, entry_trigger, exit_cids,
                        config.get('bar_count_exit'), enabled_groups,
                        risk_per_trade=risk_per_trade,
                        confluence_required=confluence_set,
                        starting_balance=starting_balance,
                        total_trading_days=period_trading_days,
                        mode="stop",
                        base_stop_config=stop_config_dict,
                        base_target_config=target_config_dict,
                        general_columns=general_cols,
                    )
                st.session_state.sl_results = sl_results

            if st.session_state.sl_results is not None and len(st.session_state.sl_results) > 0:
                sl_display_df = apply_confluence_filters(
                    st.session_state.sl_results, sl_filters, sl_search, enabled_groups
                )
                sl_display_df = sl_display_df.head(20)

                for _, row in sl_display_df.iterrows():
                    is_current = str(row.get('stop_config')) == str(stop_config_dict)
                    with st.container(border=True):
                        t1, t2, t3 = st.columns([3.5, 0.8, 0.7])
                        with t1:
                            label = f"**{row['pack_name']}**"
                            if is_current:
                                label += " _(current)_"
                            st.markdown(label)
                        with t2:
                            st.caption(f"Stop: {row['stop_summary']}")
                        with t3:
                            if not is_current:
                                if st.button("Replace", key=f"rep_sl_{row['pack_id']}"):
                                    st.session_state.pending_stop_config = row['stop_config']
                                    st.session_state.sl_results = None
                                    st.rerun()

                        k1, k2, k3, k4, k5, k6 = st.columns(6)
                        k1.caption(f"Trades: {row['total_trades']}")
                        pf = row['profit_factor']
                        k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                        k3.caption(f"WR: {row['win_rate']:.1f}%")
                        k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                        k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                        k6.caption(f"R²: {row['r_squared']:.2f}")
            else:
                st.info("Click **Analyze** to compare stop-loss configurations from your enabled Risk Management Packs.")

        # =================================================================
        # Tab 6: Take Profit Optimization
        # =================================================================
        with tab_tp:
            tp_filters = st.session_state.confluence_filters
            tp_search_col, tp_action_col, tp_filter_col = st.columns([3, 1.5, 0.5])
            with tp_search_col:
                tp_search = st.text_input("Search", placeholder="Search target configs...",
                                          key="tp_search", label_visibility="collapsed")
            with tp_action_col:
                tp_analyze_clicked = st.button("Analyze", type="primary",
                                               use_container_width=True, key="analyze_tp_btn")
            with tp_filter_col:
                if st.button("⚙", use_container_width=True, key="tp_filter_btn"):
                    confluence_filter_dialog(show_auto_search_options=False)

            # Current target tag
            tp_display = format_target_display(config)
            st.caption(f"Current: **{tp_display}**")

            exit_cids_tp = [cid for cid, _ in exit_trigger_selections] if exit_trigger_selections else []
            confluence_set_tp = selected if len(selected) > 0 else None

            if tp_analyze_clicked:
                with st.spinner("Comparing take-profit configurations..."):
                    tp_results = analyze_risk_management(
                        df, direction, entry_trigger, exit_cids_tp,
                        config.get('bar_count_exit'), enabled_groups,
                        risk_per_trade=risk_per_trade,
                        confluence_required=confluence_set_tp,
                        starting_balance=starting_balance,
                        total_trading_days=period_trading_days,
                        mode="target",
                        base_stop_config=stop_config_dict,
                        base_target_config=target_config_dict,
                        general_columns=general_cols,
                    )
                st.session_state.tp_results = tp_results

            if st.session_state.tp_results is not None and len(st.session_state.tp_results) > 0:
                tp_display_df = apply_confluence_filters(
                    st.session_state.tp_results, tp_filters, tp_search, enabled_groups
                )
                tp_display_df = tp_display_df.head(20)

                for _, row in tp_display_df.iterrows():
                    is_current = str(row.get('target_config')) == str(target_config_dict)
                    with st.container(border=True):
                        t1, t2, t3 = st.columns([3.5, 0.8, 0.7])
                        with t1:
                            label = f"**{row['pack_name']}**"
                            if is_current:
                                label += " _(current)_"
                            st.markdown(label)
                        with t2:
                            st.caption(f"Target: {row['target_summary']}")
                        with t3:
                            if not is_current:
                                if st.button("Replace", key=f"rep_tp_{row['pack_id']}"):
                                    st.session_state.pending_target_config = row['target_config']
                                    st.session_state.tp_results = None
                                    st.rerun()

                        k1, k2, k3, k4, k5, k6 = st.columns(6)
                        k1.caption(f"Trades: {row['total_trades']}")
                        pf = row['profit_factor']
                        k2.caption(f"PF: {'∞' if pf == float('inf') else f'{pf:.1f}'}")
                        k3.caption(f"WR: {row['win_rate']:.1f}%")
                        k4.caption(f"Avg R: {row['avg_r']:+.2f}")
                        k5.caption(f"Daily R: {row['daily_r']:+.2f}")
                        k6.caption(f"R²: {row['r_squared']:.2f}")
            else:
                st.info("Click **Analyze** to compare take-profit targets from your enabled Risk Management Packs.")

    # Trade list (expandable)
    with st.expander("Trade List"):
        if len(filtered_trades) > 0:
            display = filtered_trades.tail(20).copy()
            display['time'] = display['entry_time'].dt.strftime('%m/%d %H:%M')
            display['R'] = display['r_multiple'].apply(lambda x: f"{x:+.2f}")
            display['result'] = display['win'].apply(lambda x: "✓" if x else "✗")
            display['confluences'] = display['confluence_records'].apply(
                lambda r: ", ".join([format_confluence_record(rec, enabled_groups) for rec in sorted(r)[:3]]) + ("..." if len(r) > 3 else "")
            )
            st.dataframe(
                display[['time', 'entry_trigger', 'exit_reason', 'R', 'result', 'confluences']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'time': 'Time',
                    'entry_trigger': 'Entry',
                    'exit_reason': 'Exit Reason',
                    'R': 'R-Multiple',
                    'result': 'Result',
                    'confluences': 'Confluences',
                }
            )

    # =========================================================================
    # SAVE BUTTON (bottom of page)
    # =========================================================================
    st.divider()
    save_label = "Update Strategy" if editing_id else "Save Strategy"
    _save_col1, _save_col2, _save_col3 = st.columns([3, 1, 3])
    with _save_col2:
        save_clicked = st.button(save_label, type="primary", use_container_width=True, disabled=not can_save)

    if save_clicked:
        # Extract equity curve data for list-page rendering (no backtest needed on list)
        ec_boundary = datetime.now() if enable_forward else None
        eq_data = extract_equity_curve_data(filtered_trades, boundary_dt=ec_boundary)
        stored_trades = _extract_minimal_trades(filtered_trades)

        strategy = {
            'name': strategy_name,
            **config,
            'confluence': [c for c in selected if not c.startswith("GEN-")],
            'general_confluences': [c for c in selected if c.startswith("GEN-")],
            'kpis': kpis,
            'equity_curve_data': eq_data,
            'stored_trades': stored_trades,
            'forward_testing': enable_forward,
            'alerts': enable_alerts,
        }

        if editing_id:
            update_strategy(editing_id, strategy)
            saved_id = editing_id
        else:
            save_strategy(strategy)
            saved_id = strategy['id']

        # Invalidate session caches for this strategy
        st.session_state.strategy_trades_cache.pop(saved_id, None)
        st.session_state.pop(f"bt_trades_{saved_id}", None)
        st.session_state.pop(f"ft_data_{saved_id}", None)
        for _k in [k for k in st.session_state if k.startswith(f"bt_ext_{saved_id}_") or k.startswith(f"ft_ext_{saved_id}_")]:
            st.session_state.pop(_k, None)
        for _port in load_portfolios():
            if any(ps['strategy_id'] == saved_id for ps in _port.get('strategies', [])):
                st.session_state.pop(f"port_data_{_port['id']}", None)

        st.session_state.builder_data_loaded = False
        st.session_state.selected_confluences = set()
        st.session_state.strategy_config = {}
        st.session_state.editing_strategy_id = None
        st.session_state.pop('sb_additional_exits', None)
        st.session_state.viewing_strategy_id = saved_id
        st.session_state.nav_target = "My Strategies"
        st.rerun()


def render_my_strategies():
    """Render the My Strategies page — routes to list or detail view."""
    if st.session_state.viewing_strategy_id is not None:
        render_strategy_detail(st.session_state.viewing_strategy_id)
        return
    render_strategy_list()


def render_strategy_list():
    """Render the strategy list view with sorting and filtering."""
    col_header, col_update, col_new = st.columns([3, 1, 1])
    with col_header:
        st.header("My Strategies")
    with col_update:
        st.write("")  # vertical spacing to align with header
        if st.button("Update Data",
                      help="Refresh KPIs and equity curves for all strategies"):
            st.session_state._trigger_bulk_update = True
            st.rerun()
    with col_new:
        st.write("")  # vertical spacing to align with header
        if st.button("+ New Strategy", type="primary"):
            st.session_state.nav_target = "Strategy Builder"
            st.session_state.builder_data_loaded = False
            st.session_state.strategy_config = {}
            st.session_state.selected_confluences = set()
            st.session_state.editing_strategy_id = None
            st.session_state.pop('sb_additional_exits', None)
            st.rerun()

    # --- Bulk data refresh handler ---
    if st.session_state.get('_trigger_bulk_update', False):
        st.session_state._trigger_bulk_update = False
        _upd_strategies = load_strategies()
        _processable = [s for s in _upd_strategies
                        if 'entry_trigger_confluence_id' in s]

        if not _processable:
            st.warning("No strategies to update.")
        else:
            _progress = st.progress(0.0, text="Initializing...")

            def _on_progress(current, total, name):
                _progress.progress(current / total,
                                   text=f"Updating {current}/{total}: {name}")

            _result = bulk_refresh_all_strategies(
                progress_callback=_on_progress)
            _progress.progress(1.0, text="Complete!")

            # Clear session caches so detail pages use fresh data
            for _s in _processable:
                _sid = _s['id']
                st.session_state.strategy_trades_cache.pop(_sid, None)
                st.session_state.pop(f"bt_trades_{_sid}", None)
                st.session_state.pop(f"ft_data_{_sid}", None)
                for _k in [k for k in st.session_state
                           if k.startswith(f"bt_ext_{_sid}_")
                           or k.startswith(f"ft_ext_{_sid}_")]:
                    st.session_state.pop(_k, None)
            # Invalidate portfolio caches that depend on strategy data
            for _port in load_portfolios():
                st.session_state.pop(f"port_data_{_port['id']}", None)

            _msg = f"Updated {_result['success_count']} strategies"
            if _result['skipped_count']:
                _msg += f" ({_result['skipped_count']} legacy skipped)"
            if _result['failed_ids']:
                st.error(
                    f"Failed: strategy IDs {_result['failed_ids']}")
            st.success(_msg)
            import time as _time_mod
            _time_mod.sleep(1.5)
            st.rerun()

    strategies = load_strategies()

    if len(strategies) == 0:
        st.info("No strategies saved yet. Create one in the Strategy Builder!")
        return

    # --- Filter & Sort Bar ---
    filter_cols = st.columns([1, 1, 2])

    with filter_cols[0]:
        ticker_filter = st.selectbox("Ticker", ["All"] + AVAILABLE_SYMBOLS, key="strat_filter_ticker")
    with filter_cols[1]:
        direction_filter = st.selectbox("Direction", ["All", "LONG", "SHORT"], key="strat_filter_dir")
    with filter_cols[2]:
        sort_option = st.selectbox(
            "Sort By",
            ["Newest First", "Oldest First", "Name A-Z", "Win Rate (High)", "Profit Factor (High)", "Total R (High)", "Daily R (High)", "Max R DD (Best)"],
            key="strat_sort"
        )

    # Apply filters
    if ticker_filter != "All":
        strategies = [s for s in strategies if s.get('symbol') == ticker_filter]
    if direction_filter != "All":
        strategies = [s for s in strategies if s.get('direction') == direction_filter]

    # Apply sorting
    sort_keys = {
        "Newest First": (lambda s: s.get('created_at', ''), True),
        "Oldest First": (lambda s: s.get('created_at', ''), False),
        "Name A-Z": (lambda s: s.get('name', '').lower(), False),
        "Win Rate (High)": (lambda s: s.get('kpis', {}).get('win_rate', 0), True),
        "Profit Factor (High)": (lambda s: s.get('kpis', {}).get('profit_factor', 0), True),
        "Total R (High)": (lambda s: s.get('kpis', {}).get('total_r', 0), True),
        "Daily R (High)": (lambda s: s.get('kpis', {}).get('daily_r', 0), True),
        "Max R DD (Best)": (lambda s: s.get('kpis', {}).get('max_r_drawdown', 0), True),
    }
    key_fn, reverse = sort_keys[sort_option]
    strategies.sort(key=key_fn, reverse=reverse)

    if len(strategies) == 0:
        st.info("No strategies match the current filters.")
        return

    st.caption(f"{len(strategies)} strategies")

    # --- Strategy Cards ---
    enabled_groups = get_enabled_groups()

    # Pre-compute set of strategy IDs being monitored (in webhook-enabled portfolios)
    _alert_config = load_alert_config()
    _all_portfolios = load_portfolios()
    _monitored_ids = set()
    for _p in _all_portfolios:
        _pid = str(_p['id'])
        _pcfg = _alert_config.get('portfolios', {}).get(_pid, {})
        if any(wh.get('enabled', True) for wh in _pcfg.get('webhooks', [])):
            for _alloc in _p.get('strategies', []):
                _monitored_ids.add(_alloc.get('strategy_id'))

    for i, strat in enumerate(strategies):
        sid = strat.get('id', 0)
        is_legacy = 'entry_trigger_confluence_id' not in strat

        # Prefer persisted equity curve data (no backtest needed)
        eq_data = strat.get('equity_curve_data') if not is_legacy else None

        if eq_data is None and not is_legacy:
            # Migration fallback: compute + persist for next load
            trades = get_strategy_trades(strat)
            if len(trades) > 0:
                boundary = None
                if strat.get('forward_testing') and strat.get('forward_test_start'):
                    boundary = datetime.fromisoformat(strat['forward_test_start'])
                eq_data = extract_equity_curve_data(trades, boundary_dt=boundary)
                strat['equity_curve_data'] = eq_data
                # Backfill missing KPIs
                sk = strat.setdefault('kpis', {})
                if 'max_r_drawdown' not in sk:
                    cr = trades["r_multiple"].cumsum().values
                    sk['max_r_drawdown'] = round(float((cr - np.maximum.accumulate(cr)).min()), 2)
                if 'r_squared' not in sk and len(trades) >= 2:
                    cr = trades["r_multiple"].cumsum().values
                    x_vals = np.arange(len(cr))
                    corr = np.corrcoef(x_vals, cr)[0, 1]
                    sk['r_squared'] = round(corr ** 2, 4) if not np.isnan(corr) else 0.0
                update_strategy(sid, strat)

        # Backfill max_r_drawdown from persisted curve if still missing
        kpis = strat.get('kpis', {})
        if 'max_r_drawdown' not in kpis and eq_data and len(eq_data.get('cumulative_r', [])) >= 2:
            cr = np.array(eq_data['cumulative_r'])
            kpis['max_r_drawdown'] = round(float((cr - np.maximum.accumulate(cr)).min()), 2)

        # 2-column grid: new row every 2 cards
        if i % 2 == 0:
            grid_cols = st.columns(2)

        with grid_cols[i % 2]:
            with st.container(border=True):
                # Name
                st.markdown(f"#### {strat['name']}")

                # Symbol / Direction / Status / Monitoring
                if strat.get('forward_test_start'):
                    ft_start = datetime.fromisoformat(strat['forward_test_start'])
                    ft_days = (datetime.now() - ft_start).days
                    status_text = f":green[Fwd ({ft_days}d)]"
                else:
                    status_text = ":green[Fwd]"
                monitor_badge = " | :orange[Monitored]" if sid in _monitored_ids else ""
                st.caption(f"{strat['symbol']} {strat['direction']} | {status_text}{monitor_badge}")

                # Mini equity curve (full card width)
                if not is_legacy and eq_data and len(eq_data.get('exit_times', [])) > 0:
                    render_mini_equity_curve_from_data(eq_data, key=f"mini_eq_{sid}")

                # KPI metrics
                kpi_cols = st.columns(5)
                kpi_cols[0].metric("WR", f"{kpis.get('win_rate', 0):.1f}%")
                pf = kpis.get('profit_factor', 0)
                kpi_cols[1].metric("PF", "∞" if pf == float('inf') else f"{pf:.2f}")
                kpi_cols[2].metric("Daily R", f"{kpis.get('daily_r', 0):+.2f}")
                kpi_cols[3].metric("Trades", kpis.get('total_trades', 0))
                max_rdd = kpis.get('max_r_drawdown', 0)
                kpi_cols[4].metric("Max DD", f"{max_rdd:+.1f}R")

                # Trigger badges
                entry_display = get_trigger_display_name(strat, 'entry_trigger')
                exit_display = format_exit_triggers_display(strat)
                stop_display = format_stop_display(strat)
                target_display = format_target_display(strat)
                st.caption(f"Entry: {entry_display} | Exit: {exit_display}")
                st.caption(f"Stop: {stop_display} | Target: {target_display}")

                # Confluence tags (always shown for uniform card height)
                confluence = strat.get('confluence', [])
                if len(confluence) > 0:
                    formatted = [format_confluence_record(c, enabled_groups) for c in confluence[:3]]
                    st.caption(f"Confluence: {', '.join(formatted)}" + ("..." if len(confluence) > 3 else ""))
                else:
                    st.caption("Confluence: None")

                # Action buttons
                btn_cols = st.columns(4)
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
                    st.warning(f"Delete '{strat['name']}'? This cannot be undone.")
                    confirm_cols = st.columns(2)
                    with confirm_cols[0]:
                        if st.button("Yes, Delete", key=f"confirm_del_{sid}", type="primary"):
                            delete_strategy(sid)
                            st.session_state.strategy_trades_cache.pop(sid, None)
                            st.session_state.pop(f"bt_trades_{sid}", None)
                            st.session_state.pop(f"ft_data_{sid}", None)
                            for _k in [k for k in st.session_state if k.startswith(f"bt_ext_{sid}_") or k.startswith(f"ft_ext_{sid}_")]:
                                st.session_state.pop(_k, None)
                            st.session_state.confirm_delete_id = None
                            st.rerun()
                    with confirm_cols[1]:
                        if st.button("Cancel", key=f"cancel_del_{sid}"):
                            st.session_state.confirm_delete_id = None
                            st.rerun()

                # Inline edit confirmation (forward-tested strategies)
                if st.session_state.confirm_edit_id == sid:
                    st.warning("Editing resets forward test. Duplicate to preserve the original.")
                    edit_cols = st.columns(3)
                    with edit_cols[0]:
                        if st.button("Edit Anyway", key=f"confirm_edit_{sid}", type="primary"):
                            st.session_state.confirm_edit_id = None
                            load_strategy_into_builder(strat)
                    with edit_cols[1]:
                        if st.button("Duplicate", key=f"dup_instead_{sid}"):
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

    enabled_groups = get_enabled_groups()

    meta_row1 = st.columns(6)
    meta_row1[0].markdown(f"**Ticker:** {strat['symbol']}")
    meta_row1[1].markdown(f"**Direction:** {strat['direction']}")
    meta_row1[2].markdown(f"**Timeframe:** {strat.get('timeframe', '1Min')}")
    meta_row1[3].markdown(f"**Entry:** {get_trigger_display_name(strat, 'entry_trigger')}")
    meta_row1[4].markdown(f"**Exit:** {format_exit_triggers_display(strat)}")
    meta_row1[5].markdown(f"**Stop:** {format_stop_display(strat)} · **Target:** {format_target_display(strat)}")

    # Confluence conditions (TF + General)
    confluence = strat.get('confluence', [])
    general_confluence = strat.get('general_confluences', [])
    conf_parts = []
    if confluence:
        formatted_tf = [format_confluence_record(c, enabled_groups) for c in confluence]
        conf_parts.append("TF: " + " + ".join(formatted_tf))
    if general_confluence:
        formatted_gen = [format_confluence_record(c, enabled_groups) for c in general_confluence]
        conf_parts.append("General: " + " + ".join(formatted_gen))
    if conf_parts:
        st.caption(" · ".join(conf_parts))
    elif not confluence and not general_confluence:
        st.caption("No confluence conditions")

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
                st.session_state.strategy_trades_cache.pop(strategy_id, None)
                st.session_state.pop(f"bt_trades_{strategy_id}", None)
                st.session_state.pop(f"ft_data_{strategy_id}", None)
                for _k in [k for k in st.session_state if k.startswith(f"bt_ext_{strategy_id}_") or k.startswith(f"ft_ext_{strategy_id}_")]:
                    st.session_state.pop(_k, None)
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


def render_secondary_kpis(trades_df: pd.DataFrame, kpis: dict, key_prefix: str = ""):
    """Render secondary KPIs in an expander below primary KPIs."""
    if len(trades_df) == 0:
        return

    sec = calculate_secondary_kpis(trades_df, kpis)

    with st.expander("Extended KPIs"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.caption("**Trade Distribution**")
            st.markdown(f"Wins: **{sec['win_count']}** &nbsp;/&nbsp; Losses: **{sec['loss_count']}**")

        with c2:
            st.caption("**Best / Worst**")
            st.markdown(f"Best Trade: **{sec['best_trade_r']:+.2f}R**")
            st.markdown(f"Worst Trade: **{sec['worst_trade_r']:+.2f}R**")
            st.markdown(f"Avg Win: **{sec['avg_win_r']:+.2f}R**")
            st.markdown(f"Avg Loss: **{sec['avg_loss_r']:+.2f}R**")

        with c3:
            st.caption("**Streaks**")
            st.markdown(f"Max Consec. Wins: **{sec['max_consec_wins']}**")
            st.markdown(f"Max Consec. Losses: **{sec['max_consec_losses']}**")
            st.markdown(f"Longest DD: **{sec['longest_dd_trades']} trades**")

        with c4:
            st.caption("**Risk / Reward**")
            pr = sec['payoff_ratio']
            st.markdown(f"Payoff Ratio: **{'∞' if pr == float('inf') else f'{pr:.2f}'}**")
            rf = sec['recovery_factor']
            st.markdown(f"Recovery Factor: **{'∞' if rf == float('inf') else f'{rf:.1f}'}**")
            st.markdown(f"Max R DD: **{kpis.get('max_r_drawdown', 0):+.1f}R**")


def render_saved_kpis(strat: dict):
    """Display saved KPIs for legacy strategies that cannot be re-backtested."""
    kpis = strat.get('kpis', {})

    kpi_cols = st.columns(8)
    kpi_cols[0].metric("Trades", kpis.get("total_trades", 0))
    kpi_cols[1].metric("Win Rate", f"{kpis.get('win_rate', 0):.1f}%")
    pf = kpis.get('profit_factor', 0)
    kpi_cols[2].metric("Profit Factor", "∞" if pf == float('inf') else f"{pf:.2f}")
    kpi_cols[3].metric("Avg R", f"{kpis.get('avg_r', 0):+.2f}")
    kpi_cols[4].metric("Total R", f"{kpis.get('total_r', 0):+.1f}")
    kpi_cols[5].metric("Daily R", f"{kpis.get('daily_r', 0):+.2f}")
    kpi_cols[6].metric("R²", f"{kpis.get('r_squared', 0):.2f}")
    kpi_cols[7].metric("Max R DD", f"{kpis.get('max_r_drawdown', 0):+.1f}R")

    st.subheader("Strategy Configuration")
    st.markdown(f"**Stop Loss:** {format_stop_display(strat)}")
    st.markdown(f"**Target:** {format_target_display(strat)}")
    st.markdown(f"**Exit Triggers:** {format_exit_triggers_display(strat)}")
    created = strat.get('created_at', 'Unknown')
    st.markdown(f"**Created:** {created[:10] if len(created) >= 10 else created}")


def render_live_backtest(strat: dict):
    """Re-run backtest with current data and display full results."""
    data_days = strat.get('data_days', 30)
    data_seed = strat.get('data_seed', 42)
    strat_timeframe = strat.get('timeframe', '1Min')
    strat_start = None
    strat_end = None
    if strat.get('lookback_mode') == 'Date Range' and strat.get('lookback_start_date'):
        strat_start = datetime.fromisoformat(strat['lookback_start_date'])
        strat_end = datetime.fromisoformat(strat['lookback_end_date'])

    # Data loading (cached via @st.cache_data, 1hr TTL)
    df = prepare_data_with_indicators(strat['symbol'], data_days, data_seed,
                                      start_date=strat_start, end_date=strat_end,
                                      timeframe=strat_timeframe)

    if len(df) == 0:
        st.error("No data available for this symbol.")
        return

    # Trade generation (cached in session state per strategy)
    bt_cache_key = f"bt_trades_{strat['id']}"
    if bt_cache_key not in st.session_state:
        with st.spinner("Running backtest with current data..."):
            confluence_set = set(strat.get('confluence', [])) | set(strat.get('general_confluences', []))
            confluence_set = confluence_set if confluence_set else None
            general_cols = [c for c in df.columns if c.startswith("GP_")]

            st.session_state[bt_cache_key] = generate_trades(
                df,
                direction=strat['direction'],
                entry_trigger=strat['entry_trigger'],
                exit_trigger=strat.get('exit_trigger'),
                exit_triggers=strat.get('exit_triggers'),
                confluence_required=confluence_set,
                risk_per_trade=strat.get('risk_per_trade', 100.0),
                stop_atr_mult=strat.get('stop_atr_mult', 1.5),
                stop_config=strat.get('stop_config'),
                target_config=strat.get('target_config'),
                bar_count_exit=strat.get('bar_count_exit'),
                general_columns=general_cols,
            )
    trades = st.session_state[bt_cache_key]

    if len(trades) == 0:
        st.warning("No trades generated. The entry trigger may reference a confluence pack that is no longer enabled.")

    confluence_set = set(strat.get('confluence', [])) | set(strat.get('general_confluences', []))
    confluence_set = confluence_set if confluence_set else None
    extended_data_days = strat.get('extended_data_days', 365)

    # 7-tab layout
    tab_kpi, tab_kpi_ext, tab_price, tab_trades, tab_confluence, tab_config, tab_alerts = st.tabs([
        "Equity & KPIs", "Equity & KPIs (Extended)", "Price Chart",
        "Trade History", "Confluence Analysis", "Configuration", "Alerts"
    ])

    # --- Tab 1: Equity & KPIs (standard data_days range) ---
    with tab_kpi:
        kpis = calculate_kpis(
            trades,
            starting_balance=strat.get('starting_balance', 10000.0),
            risk_per_trade=strat.get('risk_per_trade', 100.0),
            total_trading_days=count_trading_days(df),
        )

        kpi_cols = st.columns(8)
        kpi_cols[0].metric("Trades", kpis["total_trades"])
        kpi_cols[1].metric("Win Rate", f"{kpis['win_rate']:.1f}%")
        pf = kpis['profit_factor']
        kpi_cols[2].metric("Profit Factor", "∞" if pf == float('inf') else f"{pf:.2f}")
        kpi_cols[3].metric("Avg R", f"{kpis['avg_r']:+.2f}")
        kpi_cols[4].metric("Total R", f"{kpis['total_r']:+.1f}")
        kpi_cols[5].metric("Daily R", f"{kpis['daily_r']:+.2f}")
        kpi_cols[6].metric("R²", f"{kpis['r_squared']:.2f}")
        kpi_cols[7].metric("Max R DD", f"{kpis['max_r_drawdown']:+.1f}R")

        render_secondary_kpis(trades, kpis, key_prefix="detail_bt")

        chart_left, chart_right = st.columns(2)
        with chart_left:
            render_backtest_equity_curve(trades)
        with chart_right:
            render_backtest_r_distribution(trades)

    # --- Tab 2: Equity & KPIs (Extended) ---
    with tab_kpi_ext:
        ext_lc1, ext_lc2, ext_lc3 = st.columns([1, 2, 4])
        with ext_lc1:
            ext_lookback_mode = st.selectbox(
                "Lookback", LOOKBACK_MODES, key="bt_ext_lookback_mode")
        ext_start_date = None
        ext_end_date = None
        with ext_lc2:
            if ext_lookback_mode == "Days":
                extended_data_days = st.number_input(
                    "Days", min_value=7, max_value=1825,
                    value=strat.get('extended_data_days', 365),
                    step=7, key="bt_ext_days_slider")
            elif ext_lookback_mode == "Bars/Candles":
                ext_bar_count = st.number_input(
                    "Bars", min_value=100, max_value=500000, value=5000,
                    step=500, key="bt_ext_bar_count")
                extended_data_days = days_from_bar_count(ext_bar_count, strat_timeframe)
            elif ext_lookback_mode == "Date Range":
                from datetime import time as dtime
                dr1, dr2 = st.columns(2)
                with dr1:
                    ext_start_input = st.date_input(
                        "Start", value=date(2024, 1, 1),
                        min_value=date(2016, 1, 1), key="bt_ext_start")
                with dr2:
                    ext_end_input = st.date_input(
                        "End", value=date.today(),
                        min_value=date(2016, 1, 1), key="bt_ext_end")
                if ext_start_input >= ext_end_input:
                    st.error("Start must be before end.")
                ext_start_date = datetime.combine(ext_start_input, dtime(9, 30))
                ext_end_date = datetime.combine(ext_end_input, dtime(16, 0))
                extended_data_days = (ext_end_input - ext_start_input).days
        with ext_lc3:
            ext_est = estimate_bar_count(extended_data_days, strat_timeframe)
            st.caption(f"~{ext_est:,} bars · {TIMEFRAME_GUIDANCE.get(strat_timeframe, '')}")

        bt_ext_key = f"bt_ext_{strat['id']}_{extended_data_days}_{ext_start_date}_{ext_end_date}"

        # Lazy load: only compute when user clicks the button
        if bt_ext_key not in st.session_state:
            if st.button("Load Extended Data", key="bt_ext_load_btn",
                         type="primary"):
                with st.spinner(f"Loading extended backtest ({extended_data_days} days)..."):
                    _ext_df = prepare_data_with_indicators(strat['symbol'], extended_data_days, data_seed,
                                                          start_date=ext_start_date, end_date=ext_end_date,
                                                          timeframe=strat_timeframe)
                    if len(_ext_df) == 0:
                        st.session_state[bt_ext_key] = (None, None)
                    else:
                        _ext_gc = [c for c in _ext_df.columns if c.startswith("GP_")]
                        _ext_trades = generate_trades(
                            _ext_df,
                            direction=strat['direction'],
                            entry_trigger=strat['entry_trigger'],
                            exit_trigger=strat.get('exit_trigger'),
                            exit_triggers=strat.get('exit_triggers'),
                            confluence_required=confluence_set,
                            risk_per_trade=strat.get('risk_per_trade', 100.0),
                            stop_atr_mult=strat.get('stop_atr_mult', 1.5),
                            stop_config=strat.get('stop_config'),
                            target_config=strat.get('target_config'),
                            bar_count_exit=strat.get('bar_count_exit'),
                            general_columns=_ext_gc,
                        )
                        _ext_kpis = calculate_kpis(
                            _ext_trades,
                            starting_balance=strat.get('starting_balance', 10000.0),
                            risk_per_trade=strat.get('risk_per_trade', 100.0),
                            total_trading_days=count_trading_days(_ext_df),
                        )
                        st.session_state[bt_ext_key] = (_ext_trades, _ext_kpis)
                st.rerun()
            else:
                st.info("Click **Load Extended Data** to run the extended backtest.")

        if bt_ext_key in st.session_state:
            ext_trades, ext_kpis = st.session_state[bt_ext_key]

            if ext_trades is None:
                st.warning("No data available for extended period.")
            else:
                kpi_cols = st.columns(8)
                kpi_cols[0].metric("Trades", ext_kpis["total_trades"])
                kpi_cols[1].metric("Win Rate", f"{ext_kpis['win_rate']:.1f}%")
                ext_pf = ext_kpis['profit_factor']
                kpi_cols[2].metric("Profit Factor", "∞" if ext_pf == float('inf') else f"{ext_pf:.2f}")
                kpi_cols[3].metric("Avg R", f"{ext_kpis['avg_r']:+.2f}")
                kpi_cols[4].metric("Total R", f"{ext_kpis['total_r']:+.1f}")
                kpi_cols[5].metric("Daily R", f"{ext_kpis['daily_r']:+.2f}")
                kpi_cols[6].metric("R²", f"{ext_kpis['r_squared']:.2f}")
                kpi_cols[7].metric("Max R DD", f"{ext_kpis['max_r_drawdown']:+.1f}R")

                render_secondary_kpis(ext_trades, ext_kpis, key_prefix="detail_bt_ext")

                chart_left, chart_right = st.columns(2)
                with chart_left:
                    render_backtest_equity_curve(ext_trades, key_suffix="ext")
                with chart_right:
                    render_backtest_r_distribution(ext_trades, key_suffix="ext")

    # --- Tab 3: Price Chart (with indicators) ---
    with tab_price:
        enabled_groups = get_enabled_groups()
        overlay_groups = [g for g in enabled_groups if g.base_template in OVERLAY_COMPATIBLE_TEMPLATES]
        show_indicators = []
        indicator_colors = {}
        for group in overlay_groups:
            show_indicators.extend(get_overlay_indicators_for_group(group))
            indicator_colors.update(get_overlay_colors_for_group(group))

        osc_panes = build_secondary_panes(df, enabled_groups)
        render_chart_with_candle_selector(
            df, trades, strat,
            show_indicators=show_indicators,
            indicator_colors=indicator_colors,
            chart_key='detail_price_chart',
            secondary_panes=osc_panes if osc_panes else None
        )

        render_backtest_trade_table(trades)

    # --- Tab 4: Trade History (clean chart, no indicators) ---
    with tab_trades:
        render_chart_with_candle_selector(
            df, trades, strat,
            show_indicators=[],
            indicator_colors={},
            chart_key='bt_trade_history_chart'
        )

        render_backtest_trade_table(trades)

    # --- Tab 5: Confluence Analysis ---
    with tab_confluence:
        render_confluence_analysis_tab(df, strat, trades)

    # --- Tab 6: Configuration ---
    with tab_config:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strategy Setup**")
            st.markdown(f"- Ticker: {strat['symbol']}")
            st.markdown(f"- Direction: {strat['direction']}")
            st.markdown(f"- Timeframe: {strat.get('timeframe', '1Min')}")
            st.markdown(f"- Entry: {get_trigger_display_name(strat, 'entry_trigger')}")
            st.markdown(f"- Exit: {format_exit_triggers_display(strat)}")
        with col2:
            st.markdown("**Settings**")
            st.markdown(f"- Stop Loss: {format_stop_display(strat)}")
            st.markdown(f"- Target: {format_target_display(strat)}")
            lb_mode = strat.get('lookback_mode', 'Days')
            if lb_mode == "Bars/Candles" and strat.get('bar_count'):
                st.markdown(f"- Lookback: {strat['bar_count']:,} bars ({strat.get('data_days', 30)} days)")
            elif lb_mode == "Date Range" and strat.get('lookback_start_date'):
                st.markdown(f"- Lookback: {strat['lookback_start_date'][:10]} to {strat['lookback_end_date'][:10]}")
            else:
                st.markdown(f"- Lookback: {strat.get('data_days', 30)} days")
            st.markdown(f"- Extended Data Days: {strat.get('extended_data_days', 365)}")
            created = strat.get('created_at', 'Unknown')
            st.markdown(f"- Created: {created[:19] if len(created) >= 19 else created}")
            if strat.get('updated_at'):
                st.markdown(f"- Last Updated: {strat['updated_at'][:19]}")

        st.markdown("**Confluence Conditions**")
        confluence = strat.get('confluence', [])
        general_confs = strat.get('general_confluences', [])
        if confluence or general_confs:
            enabled_groups = get_enabled_groups()
            if confluence:
                st.markdown("_TF Conditions:_")
                for c in confluence:
                    st.markdown(f"- {format_confluence_record(c, enabled_groups)}")
            if general_confs:
                st.markdown("_General Conditions:_")
                for c in general_confs:
                    st.markdown(f"- {format_confluence_record(c, enabled_groups)}")
        else:
            st.caption("No confluence conditions")

    # --- Tab 7: Alerts ---
    with tab_alerts:
        render_strategy_alerts_tab(strat)


# =============================================================================
# CONFLUENCE ANALYSIS TAB (shared by backtest + forward test detail views)
# =============================================================================

def _get_strategy_relevant_groups(strat: dict) -> list:
    """
    Get only the confluence groups that are actively used by this strategy.

    A group is "used" if:
    - Its trigger ID matches the strategy's entry or exit trigger confluence ID
    - Its interpreter key appears in any of the strategy's confluence conditions
    """
    enabled_groups = get_enabled_groups()
    if not enabled_groups:
        return []

    entry_conf_id = strat.get('entry_trigger_confluence_id', '')
    exit_conf_id = strat.get('exit_trigger_confluence_id', '')
    confluence_records = strat.get('confluence', [])

    # Extract interpreter keys from confluence records (format: "1M-MACD_LINE-M>S+")
    confluence_interpreters = set()
    for record in confluence_records:
        parts = record.split("-")
        if len(parts) >= 2:
            confluence_interpreters.add(parts[1])

    relevant = []
    for group in enabled_groups:
        # Check if group owns the entry or exit trigger
        if entry_conf_id.startswith(group.id + "_") or exit_conf_id.startswith(group.id + "_"):
            relevant.append(group)
            continue

        # Check if group's interpreter appears in confluence conditions
        template = get_template(group.base_template)
        if template:
            for interp_key in template.get("interpreters", []):
                if interp_key in confluence_interpreters:
                    relevant.append(group)
                    break

    return relevant


def render_confluence_analysis_tab(df: pd.DataFrame, strat: dict, trades: pd.DataFrame = None):
    """
    Render interpreter states and trigger events for each confluence group
    used by this strategy, organized as sub-tabs by group.

    Only shows groups that are actively used by the strategy (as triggers
    or confluence conditions). Each sub-tab shows a relevant chart with
    trade entry/exit markers, followed by interpreter state timeline and
    trigger event tables.

    Args:
        df: DataFrame with OHLCV + indicator/interpreter/trigger columns
        strat: Strategy config dict
        trades: DataFrame with trade data for entry/exit markers on charts
    """
    relevant_groups = _get_strategy_relevant_groups(strat)

    if not relevant_groups:
        st.info("No confluence packs are used by this strategy.")
        return

    if trades is None:
        trades = pd.DataFrame()

    # Create sub-tabs, one per relevant confluence group
    group_tabs = st.tabs([g.name for g in relevant_groups])

    for group, gtab in zip(relevant_groups, group_tabs):
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
            # All charts show trade entry/exit markers for reference.
            grp_indicators = []
            grp_colors = {}

            if group.base_template in OVERLAY_COMPATIBLE_TEMPLATES:
                grp_indicators = get_overlay_indicators_for_group(group)
                grp_colors = get_overlay_colors_for_group(group)

            secondary_panes = build_secondary_panes(df, [group])

            render_chart_with_candle_selector(
                df,
                trades,
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

    # Cache forward test data in session state (compute once per session)
    ft_cache_key = f"ft_data_{strat['id']}"
    if ft_cache_key not in st.session_state:
        with st.spinner("Loading forward test data..."):
            st.session_state[ft_cache_key] = prepare_forward_test_data(strat)
    df, backtest_trades, forward_trades, boundary_dt = st.session_state[ft_cache_key]

    if len(df) == 0:
        st.error("No data available for this symbol.")
        return

    all_trades = pd.concat([backtest_trades, forward_trades], ignore_index=True)
    extended_data_days = strat.get('extended_data_days', 365)

    # Compute trading days for KPI comparison
    boundary_ts = boundary_dt
    if df.index.tz is not None and boundary_ts.tzinfo is None:
        boundary_ts = pd.Timestamp(boundary_dt).tz_localize(df.index.tz)
    bt_trading_days = count_trading_days(df.loc[df.index < boundary_ts])
    fw_trading_days = count_trading_days(df.loc[df.index >= boundary_ts])

    st.caption(
        f"BT: {len(backtest_trades)} trades ({bt_trading_days}d) · "
        f"FW: {len(forward_trades)} trades ({fw_trading_days}d) · "
        f"Boundary: {boundary_dt.strftime('%Y-%m-%d')}"
    )

    # 7-tab layout
    tab_kpi, tab_kpi_ext, tab_price, tab_trades, tab_confluence_ft, tab_config, tab_alerts = st.tabs([
        "Equity & KPIs", "Equity & KPIs (Extended)", "Price Chart",
        "Trade History", "Confluence Analysis", "Configuration", "Alerts"
    ])

    # --- Tab 1: Equity & KPIs (standard data_days range) ---
    with tab_kpi:
        render_kpi_comparison(backtest_trades, forward_trades, bt_trading_days, fw_trading_days)
        render_combined_equity_curve(all_trades, boundary_dt)
        render_r_distribution_comparison(backtest_trades, forward_trades)

    # --- Tab 2: Equity & KPIs (Extended) ---
    with tab_kpi_ext:
        fw_ext_lc1, fw_ext_lc2, fw_ext_lc3 = st.columns([1, 2, 4])
        strat_timeframe_fw = strat.get('timeframe', '1Min')
        with fw_ext_lc1:
            fw_ext_mode = st.selectbox(
                "Lookback", LOOKBACK_MODES, key="fw_ext_lookback_mode")
        with fw_ext_lc2:
            if fw_ext_mode == "Days":
                extended_data_days = st.number_input(
                    "Days", min_value=7, max_value=1825,
                    value=strat.get('extended_data_days', 365),
                    step=7, key="fw_ext_days_slider")
            elif fw_ext_mode == "Bars/Candles":
                fw_ext_bars = st.number_input(
                    "Bars", min_value=100, max_value=500000, value=5000,
                    step=500, key="fw_ext_bar_count")
                extended_data_days = days_from_bar_count(fw_ext_bars, strat_timeframe_fw)
            elif fw_ext_mode == "Date Range":
                from datetime import time as dtime
                fw_dr1, fw_dr2 = st.columns(2)
                with fw_dr1:
                    fw_ext_start = st.date_input(
                        "Start", value=date(2024, 1, 1),
                        min_value=date(2016, 1, 1), key="fw_ext_start")
                with fw_dr2:
                    fw_ext_end = st.date_input(
                        "End", value=date.today(),
                        min_value=date(2016, 1, 1), key="fw_ext_end")
                if fw_ext_start >= fw_ext_end:
                    st.error("Start must be before end.")
                extended_data_days = (fw_ext_end - fw_ext_start).days
        with fw_ext_lc3:
            fw_ext_est = estimate_bar_count(extended_data_days, strat_timeframe_fw)
            st.caption(f"~{fw_ext_est:,} bars · {TIMEFRAME_GUIDANCE.get(strat_timeframe_fw, '')}")

        ft_ext_key = f"ft_ext_{strat['id']}_{extended_data_days}"

        # Lazy load: only compute when user clicks the button
        if ft_ext_key not in st.session_state:
            if st.button("Load Extended Data", key="ft_ext_load_btn",
                         type="primary"):
                with st.spinner(f"Loading extended data ({extended_data_days} days)..."):
                    st.session_state[ft_ext_key] = prepare_forward_test_data(
                        strat, data_days_override=extended_data_days
                    )
                st.rerun()
            else:
                st.info("Click **Load Extended Data** to run the extended backtest.")

        if ft_ext_key in st.session_state:
            ext_df, ext_bt, ext_fw, ext_boundary = st.session_state[ft_ext_key]

            if len(ext_df) == 0:
                st.warning("No data available for extended period.")
            else:
                ext_boundary_ts = ext_boundary
                if ext_df.index.tz is not None and ext_boundary_ts.tzinfo is None:
                    ext_boundary_ts = pd.Timestamp(ext_boundary).tz_localize(ext_df.index.tz)
                ext_bt_days = count_trading_days(ext_df.loc[ext_df.index < ext_boundary_ts])
                ext_fw_days = count_trading_days(ext_df.loc[ext_df.index >= ext_boundary_ts])

                render_kpi_comparison(ext_bt, ext_fw, ext_bt_days, ext_fw_days)

                ext_all = pd.concat([ext_bt, ext_fw], ignore_index=True)
                render_combined_equity_curve(ext_all, ext_boundary, key_suffix="ext")
                render_r_distribution_comparison(ext_bt, ext_fw, key_suffix="ext")

    # --- Tab 3: Price Chart (with indicators) ---
    with tab_price:
        enabled_groups = get_enabled_groups()
        overlay_groups = [g for g in enabled_groups if g.base_template in OVERLAY_COMPATIBLE_TEMPLATES]
        show_indicators = []
        indicator_colors = {}
        for group in overlay_groups:
            show_indicators.extend(get_overlay_indicators_for_group(group))
            indicator_colors.update(get_overlay_colors_for_group(group))

        osc_panes = build_secondary_panes(df, enabled_groups)
        render_chart_with_candle_selector(
            df, all_trades, strat,
            show_indicators=show_indicators,
            indicator_colors=indicator_colors,
            chart_key='forward_test_chart',
            secondary_panes=osc_panes if osc_panes else None
        )

        render_split_trade_history(backtest_trades, forward_trades)

    # --- Tab 4: Trade History (clean chart, no indicators) ---
    with tab_trades:
        render_chart_with_candle_selector(
            df, all_trades, strat,
            show_indicators=[],
            indicator_colors={},
            chart_key='trade_history_chart'
        )

        render_split_trade_history(backtest_trades, forward_trades)

    # --- Tab 5: Confluence Analysis ---
    with tab_confluence_ft:
        render_confluence_analysis_tab(df, strat, all_trades)

    # --- Tab 6: Configuration ---
    with tab_config:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strategy Setup**")
            st.markdown(f"- Ticker: {strat['symbol']}")
            st.markdown(f"- Direction: {strat['direction']}")
            st.markdown(f"- Timeframe: {strat.get('timeframe', '1Min')}")
            st.markdown(f"- Entry: {get_trigger_display_name(strat, 'entry_trigger')}")
            st.markdown(f"- Exit: {format_exit_triggers_display(strat)}")
        with col2:
            st.markdown("**Settings**")
            st.markdown(f"- Stop Loss: {format_stop_display(strat)}")
            st.markdown(f"- Target: {format_target_display(strat)}")
            lb_mode = strat.get('lookback_mode', 'Days')
            if lb_mode == "Bars/Candles" and strat.get('bar_count'):
                st.markdown(f"- Lookback: {strat['bar_count']:,} bars ({strat.get('data_days', 30)} days)")
            elif lb_mode == "Date Range" and strat.get('lookback_start_date'):
                st.markdown(f"- Lookback: {strat['lookback_start_date'][:10]} to {strat['lookback_end_date'][:10]}")
            else:
                st.markdown(f"- Lookback: {strat.get('data_days', 30)} days")
            st.markdown(f"- Extended Data Days: {strat.get('extended_data_days', 365)}")
            created = strat.get('created_at', 'Unknown')
            st.markdown(f"- Created: {created[:19] if len(created) >= 19 else created}")
            st.markdown(f"- Forward Test Start: {forward_start_str[:19]}")
            if strat.get('updated_at'):
                st.markdown(f"- Last Updated: {strat['updated_at'][:19]}")

        st.markdown("**Confluence Conditions**")
        confluence = strat.get('confluence', [])
        general_confs = strat.get('general_confluences', [])
        if confluence or general_confs:
            enabled_groups = get_enabled_groups()
            if confluence:
                st.markdown("_TF Conditions:_")
                for c in confluence:
                    st.markdown(f"- {format_confluence_record(c, enabled_groups)}")
            if general_confs:
                st.markdown("_General Conditions:_")
                for c in general_confs:
                    st.markdown(f"- {format_confluence_record(c, enabled_groups)}")
        else:
            st.caption("No confluence conditions")

    # --- Tab 7: Alerts ---
    with tab_alerts:
        render_strategy_alerts_tab(strat)


def render_strategy_alerts_tab(strat: dict):
    """Render the Alerts tab for a strategy detail view."""
    strategy_id = strat['id']

    st.subheader("Signal Detection")

    # Show which portfolios have active webhooks for this strategy
    _portfolios = load_portfolios()
    _alert_cfg = load_alert_config()
    _linked = []
    for _p in _portfolios:
        for _alloc in _p.get('strategies', []):
            if _alloc.get('strategy_id') == strategy_id:
                _pid = str(_p['id'])
                _pcfg = _alert_cfg.get('portfolios', {}).get(_pid, {})
                _whs = [wh for wh in _pcfg.get('webhooks', []) if wh.get('enabled', True)]
                _linked.append((_p['name'], len(_whs)))
                break

    if _linked:
        _parts = []
        for _pname, _wcount in _linked:
            if _wcount > 0:
                _parts.append(f"**{_pname}** ({_wcount} webhook{'s' if _wcount != 1 else ''})")
            else:
                _parts.append(f"{_pname} (no webhooks)")
        st.info(f"Alerts are delivered automatically via portfolio webhooks: {', '.join(_parts)}")
    else:
        st.info("Add this strategy to a portfolio with webhooks to enable alert delivery.")

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


def _fmt_pf(val):
    """Format profit factor, handling infinity."""
    return "∞" if val == float('inf') else f"{val:.2f}"


def render_kpi_comparison(backtest_trades: pd.DataFrame, forward_trades: pd.DataFrame,
                          bt_trading_days: int = None, fw_trading_days: int = None):
    """Render KPI comparison between backtest and forward test as tables."""
    bt_kpis = calculate_kpis(backtest_trades, total_trading_days=bt_trading_days)
    fw_kpis = calculate_kpis(forward_trades, total_trading_days=fw_trading_days)
    has_fw = len(forward_trades) > 0

    # --- Primary KPIs Table ---
    st.subheader("Primary KPIs")

    def _delta(fw_val, bt_val):
        """Compute delta string, handling inf."""
        if fw_val == float('inf') or bt_val == float('inf'):
            return "—"
        d = fw_val - bt_val
        return f"{d:+.2f}"

    primary_data = {
        "": ["Backtest"] + (["Forward Test", "Delta"] if has_fw else []),
        "Trades": [bt_kpis["total_trades"]] + ([fw_kpis["total_trades"], "—"] if has_fw else []),
        "Win Rate": [f"{bt_kpis['win_rate']:.1f}%"] + ([f"{fw_kpis['win_rate']:.1f}%", f"{fw_kpis['win_rate'] - bt_kpis['win_rate']:+.1f}%"] if has_fw else []),
        "Profit Factor": [_fmt_pf(bt_kpis['profit_factor'])] + ([_fmt_pf(fw_kpis['profit_factor']), _delta(fw_kpis['profit_factor'], bt_kpis['profit_factor'])] if has_fw else []),
        "Avg R": [f"{bt_kpis['avg_r']:+.2f}"] + ([f"{fw_kpis['avg_r']:+.2f}", f"{fw_kpis['avg_r'] - bt_kpis['avg_r']:+.2f}"] if has_fw else []),
        "Total R": [f"{bt_kpis['total_r']:+.1f}"] + ([f"{fw_kpis['total_r']:+.1f}", "—"] if has_fw else []),
        "Daily R": [f"{bt_kpis['daily_r']:+.2f}"] + ([f"{fw_kpis['daily_r']:+.2f}", f"{fw_kpis['daily_r'] - bt_kpis['daily_r']:+.2f}"] if has_fw else []),
        "R²": [f"{bt_kpis['r_squared']:.2f}"] + ([f"{fw_kpis['r_squared']:.2f}", f"{fw_kpis['r_squared'] - bt_kpis['r_squared']:+.2f}"] if has_fw else []),
        "Max R DD": [f"{bt_kpis['max_r_drawdown']:+.1f}R"] + ([f"{fw_kpis['max_r_drawdown']:+.1f}R", f"{fw_kpis['max_r_drawdown'] - bt_kpis['max_r_drawdown']:+.1f}R"] if has_fw else []),
    }

    st.dataframe(
        pd.DataFrame(primary_data),
        use_container_width=True,
        hide_index=True,
    )

    if not has_fw:
        st.info("No forward test trades yet.")

    # --- Extended KPIs Table ---
    bt_sec = calculate_secondary_kpis(backtest_trades, bt_kpis)
    fw_sec = calculate_secondary_kpis(forward_trades, fw_kpis) if has_fw else None

    def _sec_delta(fw_val, bt_val):
        if isinstance(fw_val, float) and fw_val == float('inf'):
            return "—"
        if isinstance(bt_val, float) and bt_val == float('inf'):
            return "—"
        d = fw_val - bt_val
        if isinstance(d, float):
            return f"{d:+.2f}"
        return f"{d:+d}"

    def _fmt_sec(val, suffix=""):
        if isinstance(val, float) and val == float('inf'):
            return "∞"
        if isinstance(val, float):
            return f"{val:+.2f}{suffix}" if suffix else f"{val:.2f}"
        return str(val)

    with st.expander("Extended KPIs"):
        ext_data = {
            "": ["Backtest"] + (["Forward Test", "Delta"] if has_fw else []),
            "Wins": [bt_sec['win_count']] + ([fw_sec['win_count'], _sec_delta(fw_sec['win_count'], bt_sec['win_count'])] if has_fw else []),
            "Losses": [bt_sec['loss_count']] + ([fw_sec['loss_count'], _sec_delta(fw_sec['loss_count'], bt_sec['loss_count'])] if has_fw else []),
            "Best Trade": [f"{bt_sec['best_trade_r']:+.2f}R"] + ([f"{fw_sec['best_trade_r']:+.2f}R", f"{fw_sec['best_trade_r'] - bt_sec['best_trade_r']:+.2f}R"] if has_fw else []),
            "Worst Trade": [f"{bt_sec['worst_trade_r']:+.2f}R"] + ([f"{fw_sec['worst_trade_r']:+.2f}R", f"{fw_sec['worst_trade_r'] - bt_sec['worst_trade_r']:+.2f}R"] if has_fw else []),
            "Avg Win": [f"{bt_sec['avg_win_r']:+.2f}R"] + ([f"{fw_sec['avg_win_r']:+.2f}R", f"{fw_sec['avg_win_r'] - bt_sec['avg_win_r']:+.2f}R"] if has_fw else []),
            "Avg Loss": [f"{bt_sec['avg_loss_r']:+.2f}R"] + ([f"{fw_sec['avg_loss_r']:+.2f}R", f"{fw_sec['avg_loss_r'] - bt_sec['avg_loss_r']:+.2f}R"] if has_fw else []),
            "Max Consec Wins": [bt_sec['max_consec_wins']] + ([fw_sec['max_consec_wins'], _sec_delta(fw_sec['max_consec_wins'], bt_sec['max_consec_wins'])] if has_fw else []),
            "Max Consec Losses": [bt_sec['max_consec_losses']] + ([fw_sec['max_consec_losses'], _sec_delta(fw_sec['max_consec_losses'], bt_sec['max_consec_losses'])] if has_fw else []),
            "Payoff Ratio": [_fmt_sec(bt_sec['payoff_ratio'])] + ([_fmt_sec(fw_sec['payoff_ratio']), _sec_delta(fw_sec['payoff_ratio'], bt_sec['payoff_ratio'])] if has_fw else []),
            "Recovery Factor": [_fmt_sec(bt_sec['recovery_factor'])] + ([_fmt_sec(fw_sec['recovery_factor']), _sec_delta(fw_sec['recovery_factor'], bt_sec['recovery_factor'])] if has_fw else []),
            "Longest DD": [f"{bt_sec['longest_dd_trades']} trades"] + ([f"{fw_sec['longest_dd_trades']} trades", _sec_delta(fw_sec['longest_dd_trades'], bt_sec['longest_dd_trades'])] if has_fw else []),
        }

        st.dataframe(
            pd.DataFrame(ext_data),
            use_container_width=True,
            hide_index=True,
        )


def render_backtest_equity_curve(trades: pd.DataFrame, key_suffix: str = ""):
    """Render equity curve for backtest-only view (no forward test boundary)."""
    st.subheader("Equity Curve")
    if len(trades) == 0:
        st.info("No trades to display.")
        return

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
    st.plotly_chart(fig, use_container_width=True, key=f"bt_equity_{key_suffix}" if key_suffix else None)


def render_backtest_r_distribution(trades: pd.DataFrame, key_suffix: str = ""):
    """Render R-distribution histogram for backtest-only view."""
    st.subheader("R-Multiple Distribution")
    if len(trades) == 0:
        st.info("No trades to display.")
        return

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
    st.plotly_chart(fig_hist, use_container_width=True, key=f"bt_r_dist_{key_suffix}" if key_suffix else None)


def render_backtest_trade_table(trades: pd.DataFrame):
    """Render trade history table for backtest-only view."""
    st.subheader("Trade History")
    if len(trades) == 0:
        st.info("No trades to display.")
        return

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


def render_combined_equity_curve(trades_df: pd.DataFrame, boundary_dt: datetime, key_suffix: str = ""):
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
    st.plotly_chart(fig, use_container_width=True, key=f"combined_eq_{key_suffix}" if key_suffix else None)


def render_r_distribution_comparison(backtest_trades: pd.DataFrame, forward_trades: pd.DataFrame, key_suffix: str = ""):
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
            st.plotly_chart(fig, use_container_width=True, key=f"r_dist_bt_{key_suffix}" if key_suffix else None)
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
            st.plotly_chart(fig, use_container_width=True, key=f"r_dist_fw_{key_suffix}" if key_suffix else None)
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

    # Build exit trigger arrays from new or legacy format
    exit_triggers_list = strat.get('exit_triggers')
    exit_cids = strat.get('exit_trigger_confluence_ids')
    exit_names = strat.get('exit_trigger_names')
    if not exit_cids:
        # Legacy single-exit format — wrap in arrays
        legacy_cid = strat.get('exit_trigger_confluence_id', '')
        legacy_name = strat.get('exit_trigger_name', strat.get('exit_trigger', ''))
        legacy_base = strat.get('exit_trigger', '')
        exit_cids = [legacy_cid] if legacy_cid else []
        exit_names = [legacy_name] if legacy_name else []
        exit_triggers_list = [legacy_base] if legacy_base else []

    st.session_state.strategy_config = {
        'symbol': strat['symbol'],
        'direction': strat['direction'],
        'timeframe': strat.get('timeframe', '1Min'),
        'entry_trigger': strat['entry_trigger'],
        'entry_trigger_confluence_id': strat.get('entry_trigger_confluence_id', ''),
        # New multi-exit format
        'exit_triggers': exit_triggers_list,
        'exit_trigger_confluence_ids': exit_cids,
        'exit_trigger_names': exit_names,
        # Legacy single-exit fields
        'exit_trigger': strat.get('exit_trigger', ''),
        'exit_trigger_confluence_id': strat.get('exit_trigger_confluence_id', ''),
        'entry_trigger_name': strat.get('entry_trigger_name', strat['entry_trigger']),
        'exit_trigger_name': strat.get('exit_trigger_name', strat.get('exit_trigger', '')),
        'risk_per_trade': strat.get('risk_per_trade', 100.0),
        'stop_atr_mult': strat.get('stop_atr_mult', 1.5),
        'stop_config': strat.get('stop_config'),
        'target_config': strat.get('target_config'),
        'starting_balance': strat.get('starting_balance', 10000.0),
        'data_days': strat.get('data_days', 30),
        'extended_data_days': strat.get('extended_data_days', 365),
        'data_seed': strat.get('data_seed', 42),
        'lookback_mode': strat.get('lookback_mode', 'Days'),
        'bar_count': strat.get('bar_count'),
        'lookback_start_date': strat.get('lookback_start_date'),
        'lookback_end_date': strat.get('lookback_end_date'),
        'strategy_origin': strat.get('strategy_origin', 'standard'),
    }

    # Set additional exits for UI (skip the primary)
    st.session_state.sb_additional_exits = exit_cids[1:] if len(exit_cids) > 1 else []

    tf_confs = set(strat.get('confluence', []))
    gen_confs = set(strat.get('general_confluences', []))
    st.session_state.selected_confluences = tf_confs | gen_confs
    st.session_state.builder_data_loaded = True
    st.session_state.viewing_strategy_id = None
    st.session_state.confirm_edit_id = None
    st.session_state.nav_target = "Strategy Builder"

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

    # Build lookup dict for strategy name resolution (avoids per-card file reads)
    _all_strats = load_strategies()
    _strat_by_id = {s['id']: s for s in _all_strats}

    # Load alert config once for webhook count display
    _port_alert_config = load_alert_config()

    for i, port in enumerate(portfolios):
        pid = port.get('id', 0)
        kpis = port.get('cached_kpis', {})
        n_strats = len(port.get('strategies', []))

        # Prefer persisted equity curve data (no backtest needed)
        eq_data = port.get('equity_curve_data')
        port_data = None  # only computed if needed for compliance check

        if eq_data is None and kpis and kpis.get('total_trades', 0) > 0:
            # Migration fallback: compute + persist for next load
            try:
                port_data = get_portfolio_trades(port, _strat_by_id.get, get_strategy_trades)
                eq_data = extract_portfolio_equity_curve_data(port_data['combined_trades'])
                port['equity_curve_data'] = eq_data
                update_portfolio(pid, port)
            except Exception:
                pass

        # 2-column grid: new row every 2 cards
        if i % 2 == 0:
            grid_cols = st.columns(2)

        with grid_cols[i % 2]:
            with st.container(border=True):
                # Name
                st.markdown(f"#### {port['name']}")

                # Metadata caption (all fields preserved)
                balance = port.get('starting_balance', 10000)
                compound = port.get('compound_rate', 0) * 100
                port_strats = port.get('strategies', [])
                avg_risk = sum(ps.get('risk_per_trade', 100) for ps in port_strats) / max(len(port_strats), 1)
                total_days = kpis.get('total_trading_days', 0)
                total_trades_count = kpis.get('total_trades', 0)
                avg_trades_day = total_trades_count / max(total_days, 1) if total_days > 0 else 0

                # Webhook status
                _pcfg = _port_alert_config.get('portfolios', {}).get(str(pid), {})
                _active_whs = [wh for wh in _pcfg.get('webhooks', []) if wh.get('enabled', True)]

                meta_parts = [f"{n_strats} strategies", f"\\${balance:,.0f} balance"]
                if compound > 0:
                    meta_parts.append(f"{compound:.0f}% scaling")
                meta_parts.append(f"\\${avg_risk:,.0f} avg risk/trade")
                if avg_trades_day > 0:
                    meta_parts.append(f"{avg_trades_day:.1f} trades/day")
                if _active_whs:
                    meta_parts.append(f"{len(_active_whs)} webhook(s)")
                st.caption(" | ".join(meta_parts))

                # Strategy names
                strat_names = []
                for ps in port.get('strategies', [])[:4]:
                    s = _strat_by_id.get(ps['strategy_id'])
                    if s:
                        strat_names.append(f"{s['symbol']} {s['direction']}")
                if strat_names:
                    st.caption(", ".join(strat_names) + ("..." if n_strats > 4 else ""))

                # Mini equity curve (full card width) — from persisted data
                if eq_data and len(eq_data.get('exit_times', [])) > 0:
                    try:
                        times = pd.to_datetime(eq_data['exit_times'])
                        cum_pnl = eq_data['cumulative_pnl']
                        fig = go.Figure()
                        final = cum_pnl[-1]
                        color = "#4CAF50" if final >= 0 else "#f44336"
                        fig.add_trace(go.Scatter(
                            x=times, y=cum_pnl,
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

                # KPIs from cache
                if kpis:
                    kpi_cols = st.columns(4)
                    kpi_cols[0].metric("P&L", f"\\${kpis.get('total_pnl', 0):+,.0f}")
                    max_dd = kpis.get('max_drawdown_pct', 0)
                    kpi_cols[1].metric("Max DD", f"{max_dd:.1f}%")
                    kpi_cols[2].metric("WR", f"{kpis.get('win_rate', 0):.1f}%")
                    kpi_cols[3].metric("Avg Daily", f"\\${kpis.get('avg_daily_pnl', 0):+,.0f}")

                # Requirement set badge (compact summary)
                req_id = port.get('requirement_set_id')
                if req_id:
                    rs = get_requirement_set_by_id(req_id)
                    if rs and kpis:
                        try:
                            # Lazy-load portfolio trades only for compliance check
                            if port_data is None:
                                port_data = get_portfolio_trades(port, get_strategy_by_id, get_strategy_trades)
                            eval_result = evaluate_requirement_set(rs, port, kpis, port_data['daily_pnl'])
                            passed = sum(1 for r in eval_result['rules'] if r['passed'])
                            total = len(eval_result['rules'])
                            st.caption(f"Reqs: {rs['name']} ({passed}/{total} pass)")
                        except Exception:
                            st.caption(f"Reqs: {rs['name']}")
                    elif rs:
                        st.caption(f"Reqs: {rs['name']}")

                # Action buttons
                btn_cols = st.columns(4)
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
                    dc = st.columns(2)
                    with dc[0]:
                        if st.button("Yes, Delete", key=f"port_cdel_{pid}", type="primary"):
                            delete_portfolio(pid)
                            st.session_state.pop(f"port_data_{pid}", None)
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


def get_cached_forward_test_data(strat):
    """Cache forward test data in session state (compute once per session)."""
    cache_key = f"ft_data_{strat['id']}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = prepare_forward_test_data(strat)
    return st.session_state[cache_key]


def get_cached_portfolio_data(port):
    """Cache portfolio trade data in session state (compute once per session)."""
    cache_key = f"port_data_{port['id']}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = get_portfolio_trades(
            port, get_strategy_by_id, get_cached_strategy_trades
        )
    return st.session_state[cache_key]


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
        kpi_cols[2].metric("Profit Factor", "∞" if pf == float('inf') else f"{pf:.2f}")
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

        # Compute and cache KPIs + equity curve
        save_data = get_portfolio_trades(portfolio, get_strategy_by_id, get_strategy_trades)
        portfolio['cached_kpis'] = calculate_portfolio_kpis(portfolio, save_data['combined_trades'], save_data['daily_pnl'])
        portfolio['equity_curve_data'] = extract_portfolio_equity_curve_data(save_data['combined_trades'])

        if is_edit:
            update_portfolio(editing_id, portfolio)
            st.session_state.pop(f"port_data_{editing_id}", None)
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
                st.session_state.pop(f"port_data_{portfolio_id}", None)
                st.session_state.confirm_delete_portfolio_id = None
                st.session_state.viewing_portfolio_id = None
                st.rerun()
        with dc[1]:
            if st.button("Cancel", key="pdetail_cancel_del"):
                st.session_state.confirm_delete_portfolio_id = None
                st.rerun()

    st.divider()

    # Compute portfolio data (cached in session state per portfolio)
    pd_cache_key = f"port_data_{port['id']}"
    if pd_cache_key not in st.session_state:
        with st.spinner("Computing portfolio analytics..."):
            st.session_state[pd_cache_key] = get_portfolio_trades(
                port, get_strategy_by_id, get_cached_strategy_trades
            )
    data = st.session_state[pd_cache_key]
    kpis = calculate_portfolio_kpis(port, data['combined_trades'], data['daily_pnl'])

    if len(data['combined_trades']) == 0:
        st.warning("No trades generated. Some strategies may reference disabled confluence packs.")
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
    kpi_cols[2].metric("Profit Factor", "∞" if pf == float('inf') else f"{pf:.2f}")
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
                st.caption(f"{strat['symbol']} | {strat['direction']} | Risk: \\${ps['risk_per_trade']:.0f}/trade")

                # Per-strategy KPIs
                strat_trades = data['strategy_trades'].get(sid)
                if strat_trades is not None and len(strat_trades) > 0:
                    skpis = calculate_kpis(strat_trades)
                    kpi_cols = st.columns(5)
                    kpi_cols[0].metric("Win Rate", f"{skpis['win_rate']:.1f}%")
                    spf = skpis['profit_factor']
                    kpi_cols[1].metric("Profit Factor", "∞" if spf == float('inf') else f"{spf:.2f}")
                    kpi_cols[2].metric("Daily R", f"{skpis['daily_r']:+.2f}")
                    kpi_cols[3].metric("Trades", skpis['total_trades'])
                    kpi_cols[4].metric("Max R DD", f"{skpis['max_r_drawdown']:+.1f}R")
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
                    st.markdown(f"{strategy_name} @ \\${price:.2f}")
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

    # Send Test Alert
    with st.expander("E2E Test", expanded=False):
        _render_send_test_alert(config)

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


def _render_send_test_alert(config: dict):
    """Render a Send Test Alert button that fires through all active webhooks."""
    active_webhooks = get_all_active_webhooks()
    if not active_webhooks:
        st.caption("No active webhooks configured. Add webhooks to a portfolio to test.")
        return

    if st.button("Send Test Alert", use_container_width=True):
        test_alert = {
            "type": "entry_signal",
            "level": "strategy",
            "symbol": "TEST",
            "direction": "LONG",
            "strategy_name": "Test Alert (E2E Verification)",
            "strategy_id": 0,
            "price": 100.00,
            "stop_price": 98.50,
            "atr": 1.50,
            "trigger": "test_trigger",
            "confluence_met": ["TEST-CONDITION"],
            "risk_per_trade": 100.0,
            "timeframe": "1Min",
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
            "webhook_sent": False,
            "portfolio_context": [],
            "webhook_deliveries": [],
        }

        # Save the test alert to history
        save_alert(test_alert)

        # Deliver to all active webhooks
        successes = 0
        failures = []
        deliveries = []
        for wh in active_webhooks:
            port_ctx = {"portfolio_id": wh["portfolio_id"], "portfolio_name": "", "position_risk": 100.0}
            placeholder_ctx = build_placeholder_context(test_alert, port_ctx)
            template = wh.get("payload_template", "")
            custom_payload = render_payload(template, placeholder_ctx) if template else None
            result = send_webhook(wh.get("url", ""), test_alert, custom_payload)
            deliveries.append({
                "webhook_id": wh.get("id", ""),
                "webhook_name": wh.get("name", ""),
                "portfolio_id": wh["portfolio_id"],
                "sent_at": datetime.now().isoformat(),
                "success": result["success"],
                "status_code": result.get("status_code"),
                "payload_sent": result.get("payload_sent", ""),
                "error": result.get("error", ""),
            })
            if result["success"]:
                successes += 1
            else:
                failures.append(f"{wh.get('name', '?')}: {result.get('error', 'Unknown')}")

        if successes == len(active_webhooks):
            st.toast(f"Test alert sent to {successes} webhook(s)")
        elif successes > 0:
            st.warning(f"{successes}/{len(active_webhooks)} succeeded. Failures: {'; '.join(failures)}")
        else:
            st.error(f"All {len(active_webhooks)} webhook(s) failed: {'; '.join(failures)}")


def _render_active_alerts_management(config: dict):
    """Show monitored strategies and portfolios with active webhooks."""
    strategies = load_strategies()
    portfolios = load_portfolios()

    strat_names = {s['id']: s.get('name', f"Strategy {s['id']}") for s in strategies}

    # Build set of strategies in webhook-enabled portfolios
    monitored_strategy_ids = set()
    active_ports = []  # (port, webhook_count)
    for port in portfolios:
        pid = str(port['id'])
        pcfg = config.get('portfolios', {}).get(pid, {})
        active_whs = [wh for wh in pcfg.get('webhooks', []) if wh.get('enabled', True)]
        if active_whs:
            active_ports.append((port, len(active_whs)))
            for alloc in port.get('strategies', []):
                monitored_strategy_ids.add(alloc.get('strategy_id'))

    st.markdown("**Monitored Strategies**")
    st.caption("Strategies in portfolios with active webhooks are monitored automatically.")
    if not monitored_strategy_ids:
        st.caption("No strategies are currently monitored. Add webhooks to a portfolio to enable monitoring.")
    else:
        for sid in sorted(monitored_strategy_ids):
            name = strat_names.get(sid, f"Strategy {sid}")
            st.markdown(f"- {name}")

    st.divider()

    st.markdown("**Portfolios with Active Webhooks**")
    if not active_ports:
        st.caption("No portfolios have active webhooks.")
    else:
        for port, wh_count in active_ports:
            n_strats = len(port.get('strategies', []))
            st.markdown(f"- **{port['name']}** — {wh_count} webhook(s), {n_strats} strategies")

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
    compliance_on = st.toggle(
        "Compliance Breach Alerts",
        value=port_cfg.get('alert_on_compliance_breach', True),
        key=f"wh_compliance_{portfolio_id}",
        help="Send alerts when portfolio requirement rules are breached",
    )
    st.caption("Alerts are delivered automatically to all enabled webhooks below.")

    if st.button("Save Alert Settings", key=f"wh_save_alert_{portfolio_id}"):
        new_cfg = dict(port_cfg)
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

        # Payload template + test results side-by-side
        st.markdown("**Payload (JSON)**")
        st.caption("Leave empty for default Discord/Slack format. Use {{placeholder}} tokens for dynamic values.")

        # Insert helpers
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

        # Handle placeholder selection — auto-append to text area
        _ph_state_key = f"_last_ph_sel_{wh_id}"
        if selected_ph != "-- Insert Placeholder --":
            if st.session_state.get(_ph_state_key) != selected_ph:
                current_text = st.session_state.get(f"wh_template_{wh_id}", wh.get('payload_template', ''))
                token = '{{' + selected_ph + '}}'
                st.session_state[f"wh_template_{wh_id}"] = current_text + token
                st.session_state[_ph_state_key] = selected_ph
                st.rerun()
            desc = PLACEHOLDER_CATALOG.get(selected_ph, "")
            st.caption(f"`{{{{{selected_ph}}}}}` — {desc}")

        # Handle template selection — replace text area content
        _tpl_state_key = f"_last_tpl_sel_{wh_id}"
        if selected_tpl != "-- Insert Template --":
            if st.session_state.get(_tpl_state_key) != selected_tpl:
                for t in templates:
                    if t['name'] == selected_tpl:
                        st.session_state[f"wh_template_{wh_id}"] = t.get('payload_template', '')
                        break
                st.session_state[_tpl_state_key] = selected_tpl
                st.rerun()

        # Two-column layout: template editor (left) + test panel (right)
        tpl_col, test_col = st.columns([3, 2])

        with tpl_col:
            template = st.text_area(
                "Payload Template",
                value=wh.get('payload_template', ''),
                height=200,
                key=f"wh_template_{wh_id}",
                placeholder='{\n  "content": "{{event_type}}: {{symbol}} {{direction}} @ ${{order_price}}"\n}',
            )

        with test_col:
            if st.button("Test Webhook", key=f"test_wh_{wh_id}"):
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
                resolved = render_payload(template, ctx) if template else None
                result = send_webhook(url, test_alert, resolved)
                # Persist result in session state so it survives rerun
                st.session_state[f"_wh_test_{wh_id}"] = {
                    "success": result["success"],
                    "error": result.get("error"),
                    "payload": resolved or result.get("payload_sent", ""),
                }

            # Display persisted test result (survives rerun / expander toggle)
            _test_result = st.session_state.get(f"_wh_test_{wh_id}")
            if _test_result:
                if _test_result["success"]:
                    st.success("Sent successfully")
                else:
                    st.error(f"Failed: {_test_result.get('error', 'Unknown')}")
                if _test_result.get("payload"):
                    st.caption("Resolved payload:")
                    st.code(_test_result["payload"], language="json")

        # Save / Delete buttons
        bcol1, bcol2 = st.columns(2)
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
                st.session_state.pop(f"_wh_test_{wh_id}", None)  # clear stale test result
                st.toast(f"Webhook '{name}' saved")
        with bcol2:
            if st.button("Delete", key=f"del_wh_{wh_id}", type="secondary"):
                delete_portfolio_webhook(portfolio_id, wh_id)
                st.session_state.pop(f"_wh_test_{wh_id}", None)
                st.toast(f"Webhook '{wh_name}' deleted")
                st.rerun()


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
# SETTINGS PAGE
# =============================================================================

def render_settings():
    """Render the Settings page — global app preferences."""
    st.header("Settings")

    # --- Chart Defaults ---
    st.subheader("Chart Defaults")
    candle_presets = {
        "Tight (50)": 50,
        "Close (100)": 100,
        "Default (200)": 200,
        "Wide (400)": 400,
        "Full (All)": 0,
    }
    current_val = st.session_state.get('chart_visible_candles', 200)
    preset_values = list(candle_presets.values())
    current_idx = preset_values.index(current_val) if current_val in preset_values else 2
    preset_label = st.selectbox(
        "Default Visible Candles",
        list(candle_presets.keys()),
        index=current_idx,
        key="settings_visible_candles",
        help="Number of candles visible when a chart first loads. You can still zoom/scroll manually.",
    )
    st.session_state['chart_visible_candles'] = candle_presets[preset_label]
    st.caption("Individual charts have a per-chart override dropdown.")

    # --- Backtest Defaults ---
    st.divider()
    st.subheader("Backtest Defaults")

    ext_days = st.number_input(
        "Extended Data Days", min_value=7, max_value=1825,
        value=st.session_state.get('default_extended_data_days', 365),
        step=7, key="settings_extended_data_days",
        help="Default lookback for the Extended Equity & KPIs tab (up to 5 years)",
    )
    st.session_state['default_extended_data_days'] = ext_days

    # --- Default Triggers ---
    st.divider()
    st.subheader("Default Triggers")
    st.caption("Pre-select entry and exit triggers for new strategies. "
               "You can always change them in the Strategy Builder.")

    enabled_groups = get_enabled_groups()
    all_triggers = get_all_triggers(enabled_groups)
    trigger_ids = list(all_triggers.keys())
    trigger_labels = [f"{tdef.name} [{'C' if tdef.execution == 'bar_close' else 'I'}]"
                      for tdef in all_triggers.values()]

    if trigger_ids:
        saved_entry_default = st.session_state.get('default_entry_trigger', '')
        entry_idx = trigger_ids.index(saved_entry_default) if saved_entry_default in trigger_ids else 0
        sel_entry = st.selectbox(
            "Default Entry Trigger",
            range(len(trigger_ids)),
            index=entry_idx,
            format_func=lambda i: trigger_labels[i],
            key="settings_default_entry",
        )
        st.session_state['default_entry_trigger'] = trigger_ids[sel_entry]

        saved_exit_default = st.session_state.get('default_exit_trigger', '')
        exit_idx = trigger_ids.index(saved_exit_default) if saved_exit_default in trigger_ids else 0
        sel_exit = st.selectbox(
            "Default Exit Trigger",
            range(len(trigger_ids)),
            index=exit_idx,
            format_func=lambda i: trigger_labels[i],
            key="settings_default_exit",
        )
        st.session_state['default_exit_trigger'] = trigger_ids[sel_exit]
    else:
        st.info("Enable confluence packs first to configure default triggers.")

    # --- Default Risk Management ---
    st.divider()
    st.subheader("Default Risk Management")
    st.caption("Pre-select stop loss and target methods for new strategies.")

    rm_col1, rm_col2 = st.columns(2)

    # Stop Loss defaults
    with rm_col1:
        stop_methods_settings = ["ATR", "Fixed $", "Pct %", "Swing"]
        stop_keys_settings = ["atr", "fixed_dollar", "percentage", "swing"]
        saved_def_stop = st.session_state.get('default_stop_config') or {"method": "atr"}
        saved_def_stop_method = saved_def_stop.get('method', 'atr')
        def_stop_idx = stop_keys_settings.index(saved_def_stop_method) if saved_def_stop_method in stop_keys_settings else 0

        def_stop_method_idx = st.selectbox(
            "Default Stop Loss",
            range(len(stop_methods_settings)),
            index=def_stop_idx,
            format_func=lambda i: stop_methods_settings[i],
            key="settings_default_stop_method",
        )
        def_stop_method = stop_keys_settings[def_stop_method_idx]
        def_stop_config = {"method": def_stop_method}

        if def_stop_method == "atr":
            def_stop_config["atr_mult"] = st.number_input(
                "ATR Multiplier", min_value=0.5, max_value=5.0,
                value=float(saved_def_stop.get('atr_mult', 1.5)),
                step=0.1, key="settings_stop_atr",
            )
        elif def_stop_method == "fixed_dollar":
            def_stop_config["dollar_amount"] = st.number_input(
                "Dollar Amount", min_value=0.01, max_value=100.0,
                value=float(saved_def_stop.get('dollar_amount', 1.0)),
                step=0.1, key="settings_stop_dollar",
            )
        elif def_stop_method == "percentage":
            def_stop_config["percentage"] = st.number_input(
                "Percentage", min_value=0.01, max_value=10.0,
                value=float(saved_def_stop.get('percentage', 0.5)),
                step=0.05, key="settings_stop_pct",
            )
        elif def_stop_method == "swing":
            def_stop_config["lookback"] = st.number_input(
                "Lookback Bars", min_value=2, max_value=50,
                value=int(saved_def_stop.get('lookback', 5)),
                step=1, key="settings_stop_lookback",
            )
            def_stop_config["padding"] = st.number_input(
                "Padding $", min_value=0.0, max_value=10.0,
                value=float(saved_def_stop.get('padding', 0.05)),
                step=0.01, key="settings_stop_padding",
            )

        st.session_state['default_stop_config'] = def_stop_config

    # Target defaults
    with rm_col2:
        target_methods_settings = ["None", "R:R", "ATR", "Fixed $", "Pct %", "Swing"]
        target_keys_settings = [None, "risk_reward", "atr", "fixed_dollar", "percentage", "swing"]
        saved_def_target = st.session_state.get('default_target_config') or {}
        saved_def_t_method = saved_def_target.get('method') if saved_def_target else None
        def_target_idx = target_keys_settings.index(saved_def_t_method) if saved_def_t_method in target_keys_settings else 0

        def_target_method_idx = st.selectbox(
            "Default Target",
            range(len(target_methods_settings)),
            index=def_target_idx,
            format_func=lambda i: target_methods_settings[i],
            key="settings_default_target_method",
        )
        def_target_method = target_keys_settings[def_target_method_idx]

        if def_target_method is None:
            def_target_config = None
        else:
            def_target_config = {"method": def_target_method}
            if def_target_method == "risk_reward":
                def_target_config["rr_ratio"] = st.number_input(
                    "R:R Ratio", min_value=0.5, max_value=10.0,
                    value=float(saved_def_target.get('rr_ratio', 2.0)),
                    step=0.5, key="settings_target_rr",
                )
            elif def_target_method == "atr":
                def_target_config["atr_mult"] = st.number_input(
                    "ATR Multiplier", min_value=0.5, max_value=10.0,
                    value=float(saved_def_target.get('atr_mult', 2.0)),
                    step=0.1, key="settings_target_atr",
                )
            elif def_target_method == "fixed_dollar":
                def_target_config["dollar_amount"] = st.number_input(
                    "Dollar Amount", min_value=0.01, max_value=100.0,
                    value=float(saved_def_target.get('dollar_amount', 2.0)),
                    step=0.1, key="settings_target_dollar",
                )
            elif def_target_method == "percentage":
                def_target_config["percentage"] = st.number_input(
                    "Percentage", min_value=0.01, max_value=20.0,
                    value=float(saved_def_target.get('percentage', 1.0)),
                    step=0.05, key="settings_target_pct",
                )
            elif def_target_method == "swing":
                def_target_config["lookback"] = st.number_input(
                    "Lookback Bars", min_value=2, max_value=50,
                    value=int(saved_def_target.get('lookback', 5)),
                    step=1, key="settings_target_lookback",
                )
                def_target_config["padding"] = st.number_input(
                    "Padding $", min_value=0.0, max_value=10.0,
                    value=float(saved_def_target.get('padding', 0.05)),
                    step=0.01, key="settings_target_padding",
                )

        st.session_state['default_target_config'] = def_target_config

    # --- Development (mock data only) ---
    if not is_alpaca_configured():
        st.divider()
        st.subheader("Development")
        data_seed = st.number_input(
            "Default Data Seed",
            value=st.session_state.get('global_data_seed', 42),
            key="settings_data_seed",
            help="Default random seed for mock data generation. Change for different random price patterns.",
        )
        st.session_state['global_data_seed'] = data_seed

    # --- Save Settings ---
    st.divider()
    if st.button("Save Settings", type="primary", use_container_width=True):
        current = {k: st.session_state.get(k, v) for k, v in SETTINGS_DEFAULTS.items()}
        if save_settings(current):
            st.toast("Settings saved")
        else:
            st.error("Failed to save settings.")


# =============================================================================
# CONFLUENCE GROUPS PAGE
# =============================================================================

def render_confluence_groups():
    """Render the Confluence Groups management page."""
    st.header("TF Confluence Packs")
    st.caption("Configure timeframe-specific indicators, interpreters, and triggers for your analysis.")

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
        st.markdown(f"**{len(groups)} packs** ({len(get_enabled_groups(groups))} enabled)")
    with col3:
        if st.button("+ New Pack", use_container_width=True):
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
    st.subheader("Create New Pack")

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
        new_id = st.text_input("Pack ID", value=suggested_id, help="Unique identifier (lowercase, no spaces)")

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
    st.caption("Source code for this confluence pack's indicator, interpreter, and trigger logic.")

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

    if group.base_template in OVERLAY_COMPATIBLE_TEMPLATES:
        st.markdown("**Price Chart with Indicator Overlay**")
        show_indicators = get_overlay_indicators_for_group(group)
        indicator_colors_map = get_overlay_colors_for_group(group)
    elif group.base_template in ("macd_line", "macd_histogram"):
        st.markdown("**Price Chart + MACD**")
    elif group.base_template == "rvol":
        st.markdown("**Price Chart + Relative Volume**")

    secondary_panes = build_secondary_panes(df, [group])

    render_chart_with_candle_selector(
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


def build_secondary_panes(df: pd.DataFrame, groups: list) -> list:
    """Build lightweight-charts secondary panes for oscillator-type confluence groups."""
    panes = []
    has_macd = False
    has_rvol = False
    for group in groups:
        if group.base_template in ("macd_line", "macd_histogram") and not has_macd:
            if "macd_line" in df.columns:
                panes.append(_build_macd_lwc_pane(df, group))
                has_macd = True
        elif group.base_template == "rvol" and not has_rvol:
            if "rvol" in df.columns:
                panes.append(_build_rvol_lwc_pane(df, group))
                has_rvol = True
    return panes


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
        st.error(f"Pack not found: {group_id}")
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


# =============================================================================
# GENERAL PACKS PAGE
# =============================================================================

def render_general_packs():
    """Render the General Packs management page."""
    st.header("General Packs")
    st.caption("Configure strategy-wide conditions and optional triggers (time of day, sessions, calendar).")

    if 'gp_editing' not in st.session_state:
        st.session_state.gp_editing = None
    if 'gp_show_new' not in st.session_state:
        st.session_state.gp_show_new = False

    packs = gp_module.load_general_packs()

    # Top action bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        enabled_count = len(gp_module.get_enabled_general_packs(packs))
        st.markdown(f"**{len(packs)} packs** ({enabled_count} enabled)")
    with col3:
        if st.button("+ New Pack", use_container_width=True, key="gp_new_btn"):
            st.session_state.gp_show_new = True
            st.session_state.gp_editing = None

    st.divider()

    # New pack dialog
    if st.session_state.gp_show_new:
        _render_gp_new_dialog(packs)
        st.divider()

    # Details panel
    if st.session_state.gp_editing:
        _render_gp_details(st.session_state.gp_editing, packs)
        st.divider()

    # Pack list by category
    categories = {}
    for pack in packs:
        template = gp_module.get_template(pack.base_template)
        category = template["category"] if template else "Other"
        categories.setdefault(category, []).append(pack)

    for category, cat_packs in categories.items():
        st.subheader(category)
        for pack in cat_packs:
            _render_gp_card(pack, packs)
        st.markdown("")


def _render_gp_card(pack, all_packs):
    """Render a single general pack card."""
    template = gp_module.get_template(pack.base_template)

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([0.08, 0.52, 0.25, 0.15])

        with col1:
            enabled = st.checkbox("", value=pack.enabled, key=f"gp_enable_{pack.id}",
                                  label_visibility="collapsed")
            if enabled != pack.enabled:
                pack.enabled = enabled
                gp_module.save_general_packs(all_packs)
                st.rerun()

        with col2:
            default_badge = " (default)" if pack.is_default else ""
            st.markdown(f"**{pack.name}**{default_badge}")
            param_str = gp_module.format_parameters(pack.parameters, pack.base_template)
            st.caption(param_str)

        with col3:
            if template:
                outputs = template.get("outputs", [])
                st.caption(f"Outputs: {', '.join(outputs)}")

        with col4:
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Details", key=f"gp_details_{pack.id}", use_container_width=True):
                    st.session_state.gp_editing = pack.id
                    st.session_state.gp_show_new = False
                    st.rerun()
            with action_cols[1]:
                if st.button("Copy", key=f"gp_copy_{pack.id}", use_container_width=True):
                    new_id = gp_module.generate_unique_id(pack.base_template, all_packs)
                    new_pack = gp_module.duplicate_pack(pack, new_id, f"{pack.version} Copy")
                    all_packs.append(new_pack)
                    gp_module.save_general_packs(all_packs)
                    st.session_state.gp_editing = new_id
                    st.rerun()


def _render_gp_new_dialog(all_packs):
    """Render the new general pack creation dialog."""
    st.subheader("Create New Pack")

    col1, col2 = st.columns(2)

    with col1:
        template_options = list(gp_module.TEMPLATES.keys())
        template_labels = [gp_module.TEMPLATES[t]["name"] for t in template_options]
        template_idx = st.selectbox("Base Template", range(len(template_options)),
                                    format_func=lambda i: template_labels[i], key="gp_new_template")
        selected_template = template_options[template_idx]
        template = gp_module.TEMPLATES[selected_template]
        st.caption(template["description"])

    with col2:
        new_version = st.text_input("Version Name", value="Custom", key="gp_new_version",
                                    help="e.g., 'NY Open', 'Power Hour'")
        st.caption(f"Full name will be: **{template['name']} ({new_version})**")
        suggested_id = gp_module.generate_unique_id(selected_template, all_packs)
        new_id = st.text_input("Pack ID", value=suggested_id, key="gp_new_id",
                               help="Unique identifier (lowercase, no spaces)")

    id_valid = gp_module.validate_pack_id(new_id, all_packs)
    version_valid = len(new_version.strip()) > 0
    if not id_valid:
        st.warning("ID must be unique and contain only letters, numbers, and underscores.")
    if not version_valid:
        st.warning("Version name cannot be empty.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Create", disabled=not (id_valid and version_valid),
                     use_container_width=True, key="gp_create_btn"):
            param_schema = gp_module.get_parameter_schema(selected_template)
            default_params = {k: v["default"] for k, v in param_schema.items()}

            new_pack = gp_module.GeneralPack(
                id=new_id, base_template=selected_template, version=new_version,
                description=f"Custom {template['name']} configuration",
                enabled=True, is_default=False, parameters=default_params,
            )
            all_packs.append(new_pack)
            gp_module.save_general_packs(all_packs)
            st.session_state.gp_show_new = False
            st.session_state.gp_editing = new_id
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True, key="gp_cancel_btn"):
            st.session_state.gp_show_new = False
            st.rerun()


def _render_gp_details(pack_id, all_packs):
    """Render the detail panel for a general pack."""
    pack = gp_module.get_pack_by_id(pack_id, all_packs)
    if not pack:
        st.error(f"Pack not found: {pack_id}")
        return

    template = gp_module.get_template(pack.base_template)
    if not template:
        st.error(f"Template not found: {pack.base_template}")
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"Edit: {pack.name}")
        st.caption(f"Based on: {template['name']} | ID: {pack.id}")
    with col2:
        if st.button("Close", use_container_width=True, key="gp_close_details"):
            st.session_state.gp_editing = None
            st.rerun()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Parameters", "Outputs", "Preview", "Code", "Danger Zone"])

    # TAB 1: Parameters
    with tab1:
        st.markdown("**Pack Parameters**")
        param_schema = gp_module.get_parameter_schema(pack.base_template)
        updated_params = {}
        changed = False

        for param_key, schema in param_schema.items():
            current_value = pack.parameters.get(param_key, schema["default"])
            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(schema["label"])
            with col2:
                if schema["type"] == "int":
                    new_value = st.number_input(
                        schema["label"], min_value=schema.get("min", 0),
                        max_value=schema.get("max", 500), value=int(current_value),
                        key=f"gp_param_{pack.id}_{param_key}", label_visibility="collapsed")
                elif schema["type"] == "float":
                    new_value = st.number_input(
                        schema["label"], min_value=float(schema.get("min", 0.0)),
                        max_value=float(schema.get("max", 100.0)), value=float(current_value),
                        step=0.1, key=f"gp_param_{pack.id}_{param_key}", label_visibility="collapsed")
                elif schema["type"] == "bool":
                    new_value = st.checkbox(
                        schema["label"], value=bool(current_value),
                        key=f"gp_param_{pack.id}_{param_key}")
                elif schema["type"] == "select":
                    options = schema.get("options", [])
                    idx = options.index(current_value) if current_value in options else 0
                    new_value = st.selectbox(
                        schema["label"], options, index=idx,
                        key=f"gp_param_{pack.id}_{param_key}", label_visibility="collapsed")
                else:
                    new_value = current_value

                updated_params[param_key] = new_value
                if new_value != current_value:
                    changed = True

        if changed:
            if st.button("Save Parameters", key="gp_save_params"):
                pack.parameters = updated_params
                gp_module.save_general_packs(all_packs)
                st.success("Parameters saved!")
                st.rerun()

    # TAB 2: Outputs
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Conditions**")
            output_descs = gp_module.get_output_descriptions(pack.base_template)
            for output, desc in output_descs.items():
                st.markdown(f"- **{output}**: {desc}")
        with col2:
            st.markdown("**Triggers**")
            triggers = template.get("triggers", [])
            if triggers:
                for t in triggers:
                    st.markdown(f"- **{t['name']}**")
                    st.caption(f"  {t['direction']} {t['type']} | {t.get('execution', 'bar_close')}")
            else:
                st.caption("No triggers (conditions only)")

    # TAB 3: Preview
    with tab3:
        _render_gp_preview(pack)

    # TAB 4: Code
    with tab4:
        _render_gp_code(pack)

    # TAB 5: Danger Zone
    with tab5:
        st.markdown("**Rename Version**")
        st.caption(f"Template: {pack.template_name}")
        new_version = st.text_input("Version Name", value=pack.version, key=f"gp_rename_{pack.id}")
        st.caption(f"Full name will be: **{pack.template_name} ({new_version})**")
        if new_version != pack.version:
            if st.button("Rename", key="gp_rename_btn"):
                pack.version = new_version
                gp_module.save_general_packs(all_packs)
                st.success("Renamed!")
                st.rerun()

        st.markdown("---")

        if pack.is_default:
            st.info("Default packs cannot be deleted. You can disable them instead.")
        else:
            st.markdown("**Delete Pack**")
            st.warning("This action cannot be undone.")
            if st.button("Delete Pack", type="primary", key="gp_delete_btn"):
                all_packs.remove(pack)
                gp_module.save_general_packs(all_packs)
                st.session_state.gp_editing = None
                st.rerun()


def _render_gp_preview(pack):
    """Render Preview tab for a General Pack — condition evaluation on sample data."""
    from mock_data import generate_mock_bars

    st.caption("Live preview using sample data to verify condition behavior.")

    c1, c2 = st.columns(2)
    with c1:
        preview_symbol = st.selectbox(
            "Preview Symbol", AVAILABLE_SYMBOLS, index=0,
            key=f"gp_preview_symbol_{pack.id}"
        )
    with c2:
        extended = st.checkbox(
            "Extended Hours (4 AM - 8 PM)",
            value=pack.base_template in ("trading_session",),
            key=f"gp_preview_ext_{pack.id}",
            help="Include pre-market and after-hours bars. Useful for session-based packs."
        )

    with st.spinner("Generating preview data..."):
        end = datetime.now().replace(hour=20 if extended else 16, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=3)

        bars = generate_mock_bars([preview_symbol], start, end, "1Min", seed=42,
                                  extended_hours=extended)
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

        # Evaluate condition
        condition_col = gp_module.evaluate_condition(df, pack)
        df[pack.get_condition_column()] = condition_col

    # --- Build condition state change markers ---
    col_name = pack.get_condition_column()
    states = df[col_name]
    changes = states[states != states.shift(1)]

    # Color map for outputs
    template = gp_module.get_template(pack.base_template)
    outputs = template.get("outputs", []) if template else []
    # First output gets green (positive), second gets amber/red
    state_colors = {}
    palette = ["#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"]
    for i, out in enumerate(outputs):
        state_colors[out] = palette[i % len(palette)]

    condition_markers = []
    for idx, state in changes.items():
        ts = int(pd.to_datetime(idx).timestamp())
        condition_markers.append({
            'time': ts,
            'position': 'aboveBar',
            'color': state_colors.get(state, '#94a3b8'),
            'shape': 'circle',
            'text': state,
        })

    # --- Price Chart with condition markers ---
    st.markdown("**Price Chart**")
    markers_df = pd.DataFrame()
    render_chart_with_candle_selector(
        df, markers_df, {"direction": "LONG"},
        chart_key=f"gp_preview_chart_{pack.id}",
        extra_markers=condition_markers,
    )

    # --- Condition State Timeline ---
    st.markdown("**Condition States**")

    if len(changes) > 0:
        change_records = []
        for idx, state in changes.tail(30).items():
            change_records.append({
                "Time": str(idx),
                "State": state,
                "Price": f"${df.loc[idx, 'close']:.2f}" if 'close' in df.columns else "N/A",
            })

        st.caption(f"Last {len(change_records)} state changes (of {len(changes)} total):")
        st.dataframe(pd.DataFrame(change_records), use_container_width=True, hide_index=True)

        # Distribution
        st.markdown("**Distribution**")
        dist = states.value_counts()
        dist_pct = (dist / len(states) * 100).round(1)
        dist_cols = st.columns(len(dist))
        for i, (state, count) in enumerate(dist.items()):
            dist_cols[i].metric(state, f"{dist_pct[state]}%", help=f"{count:,} bars")
    else:
        st.caption("No state changes detected in sample data.")


def _render_gp_code(pack):
    """Render Code tab for a General Pack — show source of condition evaluation logic."""
    import inspect

    st.caption("Source code for this pack's condition evaluation logic.")

    # Show effective parameters
    st.markdown("**Active Parameters**")
    template = gp_module.get_template(pack.base_template)
    param_schema = template.get("parameters_schema", {}) if template else {}
    param_items = []
    for key, schema in param_schema.items():
        value = pack.parameters.get(key, schema.get("default", "?"))
        param_items.append(f"`{schema.get('label', key)}` = **{value}**")
    if param_items:
        st.markdown(" | ".join(param_items))

    st.divider()

    # Show the evaluation function source
    logic = template.get("condition_logic", "") if template else ""
    eval_func_map = {
        "time_window": gp_module._eval_time_of_day,
        "session_filter": gp_module._eval_trading_session,
        "day_filter": gp_module._eval_day_of_week,
        "calendar_filter": gp_module._eval_calendar_filter,
    }

    func = eval_func_map.get(logic)
    if func:
        with st.expander("Condition Evaluation Function", expanded=True):
            try:
                source = inspect.getsource(func)
                st.code(source, language="python")
            except (OSError, TypeError):
                st.warning(f"Could not retrieve source for {func.__name__}")
    else:
        st.info("No evaluation function available for this template.")

    # Also show the dispatcher
    with st.expander("Dispatcher"):
        try:
            source = inspect.getsource(gp_module.evaluate_condition)
            st.code(source, language="python")
        except (OSError, TypeError):
            st.warning("Could not retrieve dispatcher source.")


# =============================================================================
# RISK MANAGEMENT PACKS PAGE
# =============================================================================

def render_risk_management_packs():
    """Render the Risk Management Packs management page."""
    st.header("Risk Management Packs")
    st.caption("Configure stop-loss and take-profit configurations from shared parameters.")

    if 'rmp_editing' not in st.session_state:
        st.session_state.rmp_editing = None
    if 'rmp_show_new' not in st.session_state:
        st.session_state.rmp_show_new = False

    packs = rmp_module.load_risk_management_packs()

    # Top action bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        enabled_count = len(rmp_module.get_enabled_risk_management_packs(packs))
        st.markdown(f"**{len(packs)} packs** ({enabled_count} enabled)")
    with col3:
        if st.button("+ New Pack", use_container_width=True, key="rmp_new_btn"):
            st.session_state.rmp_show_new = True
            st.session_state.rmp_editing = None

    st.divider()

    # New pack dialog
    if st.session_state.rmp_show_new:
        _render_rmp_new_dialog(packs)
        st.divider()

    # Details panel
    if st.session_state.rmp_editing:
        _render_rmp_details(st.session_state.rmp_editing, packs)
        st.divider()

    # Pack list by category
    categories = {}
    for pack in packs:
        template = rmp_module.get_template(pack.base_template)
        category = template["category"] if template else "Other"
        categories.setdefault(category, []).append(pack)

    for category, cat_packs in categories.items():
        st.subheader(category)
        for pack in cat_packs:
            _render_rmp_card(pack, packs)
        st.markdown("")


def _render_rmp_card(pack, all_packs):
    """Render a single risk management pack card."""
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([0.08, 0.42, 0.35, 0.15])

        with col1:
            enabled = st.checkbox("", value=pack.enabled, key=f"rmp_enable_{pack.id}",
                                  label_visibility="collapsed")
            if enabled != pack.enabled:
                pack.enabled = enabled
                rmp_module.save_risk_management_packs(all_packs)
                st.rerun()

        with col2:
            default_badge = " (default)" if pack.is_default else ""
            st.markdown(f"**{pack.name}**{default_badge}")
            param_str = rmp_module.format_parameters(pack.parameters, pack.base_template)
            st.caption(param_str)

        with col3:
            stop_str = rmp_module.format_stop_summary(pack)
            target_str = rmp_module.format_target_summary(pack)
            st.caption(f"Stop: {stop_str} | Target: {target_str}")

        with col4:
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Details", key=f"rmp_details_{pack.id}", use_container_width=True):
                    st.session_state.rmp_editing = pack.id
                    st.session_state.rmp_show_new = False
                    st.rerun()
            with action_cols[1]:
                if st.button("Copy", key=f"rmp_copy_{pack.id}", use_container_width=True):
                    new_id = rmp_module.generate_unique_id(pack.base_template, all_packs)
                    new_pack = rmp_module.duplicate_pack(pack, new_id, f"{pack.version} Copy")
                    all_packs.append(new_pack)
                    rmp_module.save_risk_management_packs(all_packs)
                    st.session_state.rmp_editing = new_id
                    st.rerun()


def _render_rmp_new_dialog(all_packs):
    """Render the new risk management pack creation dialog."""
    st.subheader("Create New Pack")

    col1, col2 = st.columns(2)

    with col1:
        template_options = list(rmp_module.TEMPLATES.keys())
        template_labels = [rmp_module.TEMPLATES[t]["name"] for t in template_options]
        template_idx = st.selectbox("Base Template", range(len(template_options)),
                                    format_func=lambda i: template_labels[i], key="rmp_new_template")
        selected_template = template_options[template_idx]
        template = rmp_module.TEMPLATES[selected_template]
        st.caption(template["description"])

    with col2:
        new_version = st.text_input("Version Name", value="Custom", key="rmp_new_version",
                                    help="e.g., 'Tight', 'Aggressive', 'Conservative'")
        st.caption(f"Full name will be: **{template['name']} ({new_version})**")
        suggested_id = rmp_module.generate_unique_id(selected_template, all_packs)
        new_id = st.text_input("Pack ID", value=suggested_id, key="rmp_new_id",
                               help="Unique identifier (lowercase, no spaces)")

    id_valid = rmp_module.validate_pack_id(new_id, all_packs)
    version_valid = len(new_version.strip()) > 0
    if not id_valid:
        st.warning("ID must be unique and contain only letters, numbers, and underscores.")
    if not version_valid:
        st.warning("Version name cannot be empty.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Create", disabled=not (id_valid and version_valid),
                     use_container_width=True, key="rmp_create_btn"):
            param_schema = rmp_module.get_parameter_schema(selected_template)
            default_params = {k: v["default"] for k, v in param_schema.items()}

            new_pack = rmp_module.RiskManagementPack(
                id=new_id, base_template=selected_template, version=new_version,
                description=f"Custom {template['name']} configuration",
                enabled=True, is_default=False, parameters=default_params,
            )
            all_packs.append(new_pack)
            rmp_module.save_risk_management_packs(all_packs)
            st.session_state.rmp_show_new = False
            st.session_state.rmp_editing = new_id
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True, key="rmp_cancel_btn"):
            st.session_state.rmp_show_new = False
            st.rerun()


def _render_rmp_details(pack_id, all_packs):
    """Render the detail panel for a risk management pack."""
    pack = rmp_module.get_pack_by_id(pack_id, all_packs)
    if not pack:
        st.error(f"Pack not found: {pack_id}")
        return

    template = rmp_module.get_template(pack.base_template)
    if not template:
        st.error(f"Template not found: {pack.base_template}")
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"Edit: {pack.name}")
        st.caption(f"Based on: {template['name']} | ID: {pack.id}")
    with col2:
        if st.button("Close", use_container_width=True, key="rmp_close_details"):
            st.session_state.rmp_editing = None
            st.rerun()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Parameters", "Outputs", "Preview", "Code", "Danger Zone"])

    # TAB 1: Parameters
    with tab1:
        st.markdown("**Pack Parameters**")
        param_schema = rmp_module.get_parameter_schema(pack.base_template)
        updated_params = {}
        changed = False

        # For composite (rr_ratio) template, handle conditional visibility
        current_stop_method = pack.parameters.get("stop_method", "atr")

        for param_key, schema in param_schema.items():
            current_value = pack.parameters.get(param_key, schema["default"])

            # Conditional visibility for rr_ratio template
            if pack.base_template == "rr_ratio":
                if param_key == "stop_atr_mult" and current_stop_method != "atr":
                    updated_params[param_key] = current_value
                    continue
                if param_key == "stop_amount" and current_stop_method != "fixed_dollar":
                    updated_params[param_key] = current_value
                    continue
                if param_key == "stop_pct" and current_stop_method != "percentage":
                    updated_params[param_key] = current_value
                    continue
                if param_key in ("lookback", "padding") and current_stop_method != "swing":
                    updated_params[param_key] = current_value
                    continue

            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(schema["label"])
            with col2:
                if schema["type"] == "int":
                    new_value = st.number_input(
                        schema["label"], min_value=schema.get("min", 0),
                        max_value=schema.get("max", 500), value=int(current_value),
                        key=f"rmp_param_{pack.id}_{param_key}", label_visibility="collapsed")
                elif schema["type"] == "float":
                    new_value = st.number_input(
                        schema["label"], min_value=float(schema.get("min", 0.0)),
                        max_value=float(schema.get("max", 100.0)), value=float(current_value),
                        step=0.1, key=f"rmp_param_{pack.id}_{param_key}", label_visibility="collapsed")
                elif schema["type"] == "select":
                    options = schema.get("options", [])
                    idx = options.index(current_value) if current_value in options else 0
                    new_value = st.selectbox(
                        schema["label"], options, index=idx,
                        key=f"rmp_param_{pack.id}_{param_key}", label_visibility="collapsed")
                    # If stop method changes, update the conditional visibility
                    if param_key == "stop_method":
                        current_stop_method = new_value
                else:
                    new_value = current_value

                updated_params[param_key] = new_value
                if new_value != current_value:
                    changed = True

        if changed:
            if st.button("Save Parameters", key="rmp_save_params"):
                pack.parameters = updated_params
                rmp_module.save_risk_management_packs(all_packs)
                st.success("Parameters saved!")
                st.rerun()

    # TAB 2: Outputs
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Stop Loss**")
            stop_config = pack.get_stop_config()
            for k, v in stop_config.items():
                st.markdown(f"- **{k}**: {v}")
            st.caption(f"Summary: {rmp_module.format_stop_summary(pack)}")

        with col2:
            st.markdown("**Take Profit**")
            target_config = pack.get_target_config()
            if target_config:
                for k, v in target_config.items():
                    st.markdown(f"- **{k}**: {v}")
                st.caption(f"Summary: {rmp_module.format_target_summary(pack)}")
            else:
                st.caption("No target configured (stop only)")

    # TAB 3: Preview
    with tab3:
        _render_rmp_preview(pack)

    # TAB 4: Code
    with tab4:
        _render_rmp_code(pack)

    # TAB 5: Danger Zone
    with tab5:
        st.markdown("**Rename Version**")
        st.caption(f"Template: {pack.template_name}")
        new_version = st.text_input("Version Name", value=pack.version, key=f"rmp_rename_{pack.id}")
        st.caption(f"Full name will be: **{pack.template_name} ({new_version})**")
        if new_version != pack.version:
            if st.button("Rename", key="rmp_rename_btn"):
                pack.version = new_version
                rmp_module.save_risk_management_packs(all_packs)
                st.success("Renamed!")
                st.rerun()

        st.markdown("---")

        if pack.is_default:
            st.info("Default packs cannot be deleted. You can disable them instead.")
        else:
            st.markdown("**Delete Pack**")
            st.warning("This action cannot be undone.")
            if st.button("Delete Pack", type="primary", key="rmp_delete_btn"):
                all_packs.remove(pack)
                rmp_module.save_risk_management_packs(all_packs)
                st.session_state.rmp_editing = None
                st.rerun()


def _render_rmp_preview(pack):
    """Render Preview tab for a Risk Management Pack — sample trades with stop/target visualization."""
    from mock_data import generate_mock_bars

    st.caption("Live preview using sample trades to verify stop-loss and take-profit behavior.")

    # Controls: symbol + entry/exit trigger selection
    c1, c2, c3 = st.columns(3)
    with c1:
        preview_symbol = st.selectbox(
            "Symbol", AVAILABLE_SYMBOLS, index=0,
            key=f"rmp_preview_symbol_{pack.id}"
        )

    # Build entry/exit trigger lists from enabled TF Confluence Packs
    groups = load_confluence_groups()
    enabled_groups = get_enabled_groups(groups)

    entry_triggers = {}
    exit_triggers = {}
    for g in enabled_groups:
        template = get_template(g.base_template)
        if not template:
            continue
        for trig_def in template.get("triggers", []):
            cid = g.get_trigger_id(trig_def["base"])
            trig_name = g.get_trigger_name(trig_def["base"], trig_def["name"])
            if trig_def["type"] == "ENTRY":
                entry_triggers[cid] = trig_name
            elif trig_def["type"] == "EXIT":
                exit_triggers[cid] = trig_name

    if not entry_triggers:
        st.warning("No entry triggers available. Enable TF Confluence Packs first.")
        return

    entry_options = list(entry_triggers.keys())
    entry_labels = list(entry_triggers.values())
    exit_options = list(exit_triggers.keys())
    exit_labels = list(exit_triggers.values())

    with c2:
        entry_idx = st.selectbox(
            "Entry Trigger", range(len(entry_options)),
            format_func=lambda i: entry_labels[i],
            key=f"rmp_preview_entry_{pack.id}"
        )
        selected_entry_cid = entry_options[entry_idx]

    with c3:
        if exit_options:
            exit_idx = st.selectbox(
                "Exit Trigger", range(len(exit_options)),
                format_func=lambda i: exit_labels[i],
                key=f"rmp_preview_exit_{pack.id}"
            )
            selected_exit_cid = exit_options[exit_idx]
        else:
            st.caption("No exit triggers — using 4-bar count exit")
            selected_exit_cid = None

    direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True,
                         key=f"rmp_preview_dir_{pack.id}")

    # Generate data and run trades
    with st.spinner("Generating preview trades..."):
        end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=5)

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

        df = run_all_indicators(df)
        for g in enabled_groups:
            df = run_indicators_for_group(df, g)
        df = run_all_interpreters(df)
        df = detect_all_triggers(df)

        # Generate trades with this pack's stop/target config
        base_entry = get_base_trigger_id(selected_entry_cid)
        if selected_exit_cid:
            base_exit = get_base_trigger_id(selected_exit_cid)
            exit_list = [base_exit]
            bar_count = None
        else:
            exit_list = None
            bar_count = 4

        stop_config = pack.get_stop_config()
        target_config = pack.get_target_config()

        rm_general_cols = [c for c in df.columns if c.startswith("GP_")]
        trades = generate_trades(
            df, direction=direction, entry_trigger=base_entry,
            exit_triggers=exit_list, bar_count_exit=bar_count,
            risk_per_trade=100.0, stop_config=stop_config,
            target_config=target_config, general_columns=rm_general_cols,
        )

    # --- Chart with trade markers ---
    st.markdown("**Price Chart with Trades**")
    st.caption(f"Stop: {rmp_module.format_stop_summary(pack)} | "
               f"Target: {rmp_module.format_target_summary(pack)}")

    render_chart_with_candle_selector(
        df, trades, {"direction": direction},
        chart_key=f"rmp_preview_chart_{pack.id}",
    )

    # --- Trade Summary ---
    if len(trades) > 0:
        kpis = calculate_kpis(trades, starting_balance=10000, risk_per_trade=100.0)

        st.markdown("**Trade Summary**")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Trades", kpis["total_trades"])
        k2.metric("Win Rate", f"{kpis['win_rate']:.1f}%")
        pf = kpis['profit_factor']
        k3.metric("PF", "∞" if pf == float('inf') else f"{pf:.2f}")
        k4.metric("Avg R", f"{kpis['avg_r']:+.2f}")
        k5.metric("Total R", f"{kpis['total_r']:+.1f}")
        k6.metric("Max DD", f"{kpis['max_r_drawdown']:+.1f}R")

        # Trade details table
        st.markdown("**Trade Details**")
        trade_records = []
        for _, t in trades.tail(15).iterrows():
            record = {
                "Entry": str(t['entry_time']),
                "Exit": str(t['exit_time']),
                "Entry $": f"${t['entry_price']:.2f}",
                "Exit $": f"${t['exit_price']:.2f}",
                "R": f"{t['r_multiple']:+.2f}",
                "Result": "Win" if t['win'] else "Loss",
            }
            if 'stop_price' in t and pd.notna(t.get('stop_price')):
                record["Stop $"] = f"${t['stop_price']:.2f}"
            if 'target_price' in t and pd.notna(t.get('target_price')):
                record["Target $"] = f"${t['target_price']:.2f}"
            trade_records.append(record)

        st.dataframe(pd.DataFrame(trade_records), use_container_width=True, hide_index=True)
    else:
        st.info("No trades generated with the selected entry/exit triggers and direction.")


def _render_rmp_code(pack):
    """Render Code tab for a Risk Management Pack — show builder function source."""
    import inspect

    st.caption("Source code for this pack's stop-loss and take-profit configuration builders.")

    # Show effective outputs
    st.markdown("**Active Configuration**")
    stop_config = pack.get_stop_config()
    target_config = pack.get_target_config()
    st.code(
        f"stop_config  = {stop_config}\ntarget_config = {target_config}",
        language="python"
    )

    st.divider()

    # Show builder function sources
    template = rmp_module.get_template(pack.base_template)
    if template:
        build_stop = template.get("build_stop")
        build_target = template.get("build_target")

        if build_stop:
            with st.expander("Stop Loss Builder", expanded=True):
                try:
                    source = inspect.getsource(build_stop)
                    st.code(source, language="python")
                except (OSError, TypeError):
                    st.warning(f"Could not retrieve source for {build_stop.__name__}")

        if build_target:
            with st.expander("Take Profit Builder", expanded=True):
                try:
                    source = inspect.getsource(build_target)
                    st.code(source, language="python")
                except (OSError, TypeError):
                    st.warning(f"Could not retrieve source for {build_target.__name__}")

    # Show the dataclass methods
    with st.expander("Pack Methods (get_stop_config / get_target_config)"):
        try:
            source = inspect.getsource(rmp_module.RiskManagementPack.get_stop_config)
            st.code(source, language="python")
        except (OSError, TypeError):
            pass
        try:
            source = inspect.getsource(rmp_module.RiskManagementPack.get_target_config)
            st.code(source, language="python")
        except (OSError, TypeError):
            pass


if __name__ == "__main__":
    main()
