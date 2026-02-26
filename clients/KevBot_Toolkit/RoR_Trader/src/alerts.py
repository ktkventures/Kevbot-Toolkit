"""
Alert Engine for RoR Trader
=============================

Handles alert configuration, signal detection, position tracking,
portfolio enrichment, webhook delivery, webhook templates, and alert history.

The alert monitor (alert_monitor.py) uses this module to detect signals
and deliver notifications. The Streamlit app uses the CRUD functions
to configure alerts and display history.
"""

import json
import os
import math
import secrets
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from data_loader import load_latest_bars
from indicators import run_all_indicators
from interpreters import (
    INTERPRETERS,
    run_all_interpreters,
    detect_all_triggers,
    get_confluence_records,
    get_mtf_confluence_records,
)
from interpreters import INTERPRETERS as _ALL_INTERPRETERS
from triggers import generate_trades, calculate_stop_price
from portfolios import load_portfolios, get_requirement_set_by_id

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALERT_CONFIG_FILE = os.path.join(_SCRIPT_DIR, "alert_config.json")
ALERTS_FILE = os.path.join(_SCRIPT_DIR, "alerts.json")
MONITOR_STATUS_FILE = os.path.join(_SCRIPT_DIR, "monitor_status.json")
WEBHOOK_TEMPLATES_FILE = os.path.join(_SCRIPT_DIR, "webhook_templates.json")

# Minimum floor for signal detection bars
_SIGNAL_DETECTION_BARS_FLOOR = 200
# Longest indicator warmup period (EMA 50)
_INDICATOR_WARMUP = 50


def compute_signal_detection_bars(timeframe: str, session: str = "RTH") -> int:
    """Compute minimum bars needed for accurate signal detection.

    Ensures we load at least:
    - One full trading day of bars (for VWAP accuracy on intraday)
    - Enough bars for the longest indicator warmup (EMA 50)
    - Minimum 200 bars floor
    """
    from data_loader import _bars_per_day
    full_day = int(_bars_per_day(timeframe, session))
    return max(full_day, _INDICATOR_WARMUP, _SIGNAL_DETECTION_BARS_FLOOR)

DEFAULT_GLOBAL_CONFIG = {
    "poll_interval_seconds": 60,
    "market_hours_only": True,
    "enabled": False,
}

DEFAULT_STRATEGY_CONFIG = {
    "alerts_enabled": False,
    "alert_on_entry": True,
    "alert_on_exit": True,
}

DEFAULT_PORTFOLIO_CONFIG = {
    "alerts_enabled": False,
    "alert_on_signal": True,
    "alert_on_compliance_breach": True,
    "webhooks": [],
}


# =============================================================================
# PLACEHOLDER SYSTEM
# =============================================================================

PLACEHOLDER_CATALOG = {
    "symbol": "Ticker symbol (e.g., SPY)",
    "timeframe": "Bar timeframe (e.g., 1Min)",
    "strategy_name": "Strategy display name",
    "strategy_id": "Strategy numeric ID",
    "event_type": "entry_signal, exit_signal, or compliance_breach",
    "order_action": "buy, sell, or close",
    "order_price": "Price at signal time",
    "stop_price": "Calculated stop loss price",
    "market_position": "long, short, or flat",
    "direction": "LONG or SHORT",
    "risk_per_trade": "Dollar risk per trade (portfolio-level)",
    "quantity": "Shares (derived from risk / stop distance)",
    "atr": "ATR value at signal bar",
    "trigger_name": "Trigger ID that fired",
    "confluence_met": "Comma-separated confluence conditions",
    "portfolio_name": "Portfolio display name",
    "portfolio_id": "Portfolio numeric ID",
    "position_risk": "Position risk in this portfolio",
    "timestamp": "ISO timestamp of alert",
    "rule_name": "Breached rule name (compliance only)",
    "rule_limit": "Rule limit display (compliance only)",
    "rule_value": "Actual value display (compliance only)",
}


def build_placeholder_context(alert: dict, portfolio_context: dict = None) -> dict:
    """Build the full placeholder substitution dict from an alert record."""
    alert_type = alert.get("type", "")
    direction = alert.get("direction", "")

    # Derive order_action: buy/sell for entries, close for exits
    if alert_type == "entry_signal":
        order_action = "buy" if direction == "LONG" else "sell"
    elif alert_type == "exit_signal":
        order_action = "close"
    else:
        order_action = ""

    # Derive market_position
    if alert_type == "entry_signal":
        market_position = direction.lower() if direction else ""
    elif alert_type == "exit_signal":
        market_position = "flat"
    else:
        market_position = ""

    # Derive quantity from position_risk / |price - stop_price|
    price = alert.get("price")
    stop_price = alert.get("stop_price")
    position_risk = None
    if portfolio_context:
        position_risk = portfolio_context.get("position_risk")
    if not position_risk:
        position_risk = alert.get("risk_per_trade")

    # For exit signals, use entry parameters so exit quantity matches entry size
    qty_price = price
    qty_stop = stop_price
    if alert_type == "exit_signal":
        entry_price = alert.get("entry_price")
        entry_stop = alert.get("entry_stop_price")
        if entry_price and entry_stop:
            qty_price = entry_price
            qty_stop = entry_stop

    quantity = ""
    if qty_price and qty_stop and position_risk:
        try:
            stop_distance = abs(float(qty_price) - float(qty_stop))
            if stop_distance > 0:
                quantity = str(int(float(position_risk) / stop_distance))
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    ctx = {
        "symbol": str(alert.get("symbol", "")),
        "timeframe": str(alert.get("timeframe", "1Min")),
        "strategy_name": str(alert.get("strategy_name", "")),
        "strategy_id": str(alert.get("strategy_id", "")),
        "event_type": alert_type,
        "order_action": order_action,
        "order_price": str(alert.get("price", "")),
        "stop_price": str(alert.get("stop_price", "")),
        "market_position": market_position,
        "direction": direction,
        "risk_per_trade": str(position_risk or alert.get("risk_per_trade", "")),
        "quantity": quantity,
        "atr": str(alert.get("atr", "")),
        "trigger_name": str(alert.get("trigger", "")),
        "confluence_met": ", ".join(alert.get("confluence_met", [])),
        "timestamp": str(alert.get("timestamp", "")),
        "rule_name": str(alert.get("rule_name", "")),
        "rule_limit": str(alert.get("rule_limit", "")),
        "rule_value": str(alert.get("rule_value", "")),
    }

    # Portfolio-specific overrides
    if portfolio_context:
        ctx["portfolio_name"] = str(portfolio_context.get("portfolio_name", ""))
        ctx["portfolio_id"] = str(portfolio_context.get("portfolio_id", ""))
        ctx["position_risk"] = str(portfolio_context.get("position_risk", ""))
    else:
        ctx["portfolio_name"] = ""
        ctx["portfolio_id"] = ""
        ctx["position_risk"] = ""

    return ctx


def _is_json_numeric(value: str) -> bool:
    """Check if a string value should be treated as a bare JSON number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def render_payload(template: str, context: dict) -> str:
    """Substitute {{placeholder}} tokens in a payload template string.

    Handles both quoted ("{{key}}") and unquoted ({{key}}) placeholders:
    - Already-quoted placeholders are replaced directly (user controls quoting).
    - Bare placeholders auto-quote string values and leave numbers bare,
      producing valid JSON regardless of how the user wrote the template.
    """
    result = template
    for key, value in context.items():
        quoted_token = '"{{' + key + '}}"'
        bare_token = '{{' + key + '}}'
        # Pass 1: replace "{{key}}" with "value" (user already quoted)
        if quoted_token in result:
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            result = result.replace(quoted_token, f'"{escaped}"')
        # Pass 2: replace bare {{key}} — auto-quote strings, leave numbers bare
        if bare_token in result:
            if _is_json_numeric(value):
                result = result.replace(bare_token, value)
            else:
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                result = result.replace(bare_token, f'"{escaped}"')
    return result


# =============================================================================
# ALERT CONFIG CRUD
# =============================================================================

def generate_webhook_id() -> str:
    """Generate a unique webhook ID like 'wh_a1b2c3'."""
    return f"wh_{secrets.token_hex(3)}"


def load_alert_config() -> dict:
    """Load alert configuration. Migrates legacy schema if detected."""
    if os.path.exists(ALERT_CONFIG_FILE):
        try:
            with open(ALERT_CONFIG_FILE, 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, Exception):
            return {
                "global": dict(DEFAULT_GLOBAL_CONFIG),
                "strategies": {},
                "portfolios": {},
            }

        # Detect and migrate legacy schema
        migrated = False

        # Remove global webhook_url
        old_global_url = ""
        if "webhook_url" in config.get("global", {}):
            old_global_url = config["global"].pop("webhook_url", "")
            migrated = True

        # Migrate strategy webhook_urls (remove them)
        for sid, scfg in config.get("strategies", {}).items():
            if "webhook_url" in scfg:
                scfg.pop("webhook_url")
                migrated = True

        # Migrate portfolio webhook_urls to webhooks array
        for pid, pcfg in config.get("portfolios", {}).items():
            if "webhooks" not in pcfg:
                old_url = pcfg.pop("webhook_url", "") or old_global_url
                pcfg["webhooks"] = []
                if old_url:
                    pcfg["webhooks"].append({
                        "id": generate_webhook_id(),
                        "name": "Migrated Webhook",
                        "url": old_url,
                        "enabled": True,
                        "events": {
                            "entry_long": True,
                            "entry_short": True,
                            "exit_long": True,
                            "exit_short": True,
                            "compliance_breach": True,
                        },
                        "payload_template": "",
                    })
                migrated = True

        if migrated:
            save_alert_config(config)

        return config

    return {
        "global": dict(DEFAULT_GLOBAL_CONFIG),
        "strategies": {},
        "portfolios": {},
    }


def save_alert_config(config: dict):
    """Save alert configuration to JSON file."""
    with open(ALERT_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_strategy_alert_config(strategy_id: int) -> dict:
    """Get alert config for a strategy, returning defaults if not configured."""
    config = load_alert_config()
    strat_config = config.get("strategies", {}).get(str(strategy_id))
    if strat_config:
        return strat_config
    return dict(DEFAULT_STRATEGY_CONFIG)


def set_strategy_alert_config(strategy_id: int, strat_config: dict):
    """Set alert config for a strategy."""
    config = load_alert_config()
    if "strategies" not in config:
        config["strategies"] = {}
    config["strategies"][str(strategy_id)] = strat_config
    save_alert_config(config)


def get_portfolio_alert_config(portfolio_id: int) -> dict:
    """Get alert config for a portfolio, returning defaults if not configured."""
    config = load_alert_config()
    port_config = config.get("portfolios", {}).get(str(portfolio_id))
    if port_config:
        return port_config
    return dict(DEFAULT_PORTFOLIO_CONFIG)


def set_portfolio_alert_config(portfolio_id: int, port_config: dict):
    """Set alert config for a portfolio."""
    config = load_alert_config()
    if "portfolios" not in config:
        config["portfolios"] = {}
    config["portfolios"][str(portfolio_id)] = port_config
    save_alert_config(config)


# =============================================================================
# WEBHOOK CRUD (portfolio-level)
# =============================================================================

def get_portfolio_webhooks(portfolio_id: int) -> list:
    """Get all webhooks configured for a portfolio."""
    config = load_alert_config()
    port_config = config.get("portfolios", {}).get(str(portfolio_id), {})
    return port_config.get("webhooks", [])


def add_portfolio_webhook(portfolio_id: int, webhook: dict) -> dict:
    """Add a new webhook to a portfolio. Auto-assigns ID. Returns the saved webhook."""
    config = load_alert_config()
    if "portfolios" not in config:
        config["portfolios"] = {}
    pid = str(portfolio_id)
    if pid not in config["portfolios"]:
        config["portfolios"][pid] = dict(DEFAULT_PORTFOLIO_CONFIG)
    if "webhooks" not in config["portfolios"][pid]:
        config["portfolios"][pid]["webhooks"] = []

    webhook["id"] = generate_webhook_id()
    if "enabled" not in webhook:
        webhook["enabled"] = True
    config["portfolios"][pid]["webhooks"].append(webhook)
    save_alert_config(config)
    return webhook


def update_portfolio_webhook(portfolio_id: int, webhook_id: str, updates: dict) -> bool:
    """Update an existing webhook. Returns True if found and updated."""
    config = load_alert_config()
    webhooks = config.get("portfolios", {}).get(str(portfolio_id), {}).get("webhooks", [])
    for i, wh in enumerate(webhooks):
        if wh.get("id") == webhook_id:
            webhooks[i].update(updates)
            save_alert_config(config)
            return True
    return False


def delete_portfolio_webhook(portfolio_id: int, webhook_id: str) -> bool:
    """Delete a webhook from a portfolio. Returns True if found and removed."""
    config = load_alert_config()
    pid = str(portfolio_id)
    webhooks = config.get("portfolios", {}).get(pid, {}).get("webhooks", [])
    original_len = len(webhooks)
    webhooks = [wh for wh in webhooks if wh.get("id") != webhook_id]
    if len(webhooks) < original_len:
        config["portfolios"][pid]["webhooks"] = webhooks
        save_alert_config(config)
        return True
    return False


def get_all_active_webhooks() -> list:
    """Get all enabled webhooks across all portfolios.
    Returns list of dicts with portfolio context added."""
    config = load_alert_config()
    result = []
    for pid, port_config in config.get("portfolios", {}).items():
        if not port_config.get("alerts_enabled", False):
            continue
        for wh in port_config.get("webhooks", []):
            if wh.get("enabled", True):
                entry = dict(wh)
                entry["portfolio_id"] = int(pid)
                result.append(entry)
    return result


# =============================================================================
# ALERT HISTORY CRUD
# =============================================================================

def load_alerts(limit: int = 100) -> list:
    """Load alert history, most recent first. Returns up to `limit` alerts."""
    if not os.path.exists(ALERTS_FILE):
        return []
    try:
        with open(ALERTS_FILE, 'r') as f:
            alerts = json.load(f)
    except (json.JSONDecodeError, Exception):
        return []
    # Sort most recent first
    alerts.sort(key=lambda a: a.get('timestamp', ''), reverse=True)
    return alerts[:limit]


def _save_all_alerts(alerts: list):
    """Save full alerts list to file."""
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2, default=str)


def save_alert(alert: dict) -> dict:
    """Save a new alert, auto-assigning ID and timestamp."""
    alerts = load_alerts(limit=10000)  # load all
    max_id = max((a.get('id', 0) for a in alerts), default=0)
    alert['id'] = max_id + 1
    if 'timestamp' not in alert:
        alert['timestamp'] = datetime.now(timezone.utc).isoformat()
    if 'acknowledged' not in alert:
        alert['acknowledged'] = False
    alerts.insert(0, alert)  # prepend (most recent first)
    # Keep last 500 alerts
    _save_all_alerts(alerts[:500])
    return alert


def acknowledge_alert(alert_id: int) -> bool:
    """Mark an alert as acknowledged."""
    alerts = load_alerts(limit=10000)
    for alert in alerts:
        if alert.get('id') == alert_id:
            alert['acknowledged'] = True
            _save_all_alerts(alerts)
            return True
    return False


def clear_alerts():
    """Remove all alert history."""
    _save_all_alerts([])


def get_alerts_for_strategy(strategy_id: int, limit: int = 50) -> list:
    """Get alerts for a specific strategy."""
    alerts = load_alerts(limit=10000)
    filtered = [a for a in alerts if a.get('strategy_id') == strategy_id]
    return filtered[:limit]


def delete_alerts_for_strategy(strategy_id: int):
    """Remove all alerts for a strategy from alerts.json."""
    alerts = load_alerts(limit=10000)
    filtered = [a for a in alerts if a.get('strategy_id') != strategy_id]
    _save_all_alerts(filtered)


def get_alerts_for_portfolio(portfolio_id: int, limit: int = 50) -> list:
    """Get alerts related to a specific portfolio."""
    alerts = load_alerts(limit=10000)
    filtered = []
    for a in alerts:
        # Direct portfolio alerts (compliance breach)
        if a.get('portfolio_id') == portfolio_id:
            filtered.append(a)
            continue
        # Strategy alerts that include this portfolio in context
        for ctx in a.get('portfolio_context', []):
            if ctx.get('portfolio_id') == portfolio_id:
                filtered.append(a)
                break
    return filtered[:limit]


# =============================================================================
# QUERY HELPERS FOR UI
# =============================================================================

def get_webhook_delivery_log(limit: int = 100) -> list:
    """Get all webhook deliveries across all alerts, flattened for the Outbound Webhooks tab.
    Returns list sorted by sent_at descending."""
    alerts = load_alerts(limit=500)
    deliveries = []
    for alert in alerts:
        for delivery in alert.get("webhook_deliveries", []):
            entry = dict(delivery)
            entry["alert_id"] = alert.get("id")
            entry["alert_type"] = alert.get("type")
            entry["strategy_name"] = alert.get("strategy_name", "")
            entry["symbol"] = alert.get("symbol", "")
            entry["direction"] = alert.get("direction", "")
            entry["alert_timestamp"] = alert.get("timestamp")
            deliveries.append(entry)
    deliveries.sort(key=lambda d: d.get("sent_at", ""), reverse=True)
    return deliveries[:limit]


def get_active_alert_configs() -> dict:
    """Get only strategies/portfolios with alerts currently enabled.
    Returns {'strategies': [...], 'portfolios': [...]} with enriched info."""
    config = load_alert_config()

    active_strategies = []
    for sid, scfg in config.get("strategies", {}).items():
        if scfg.get("alerts_enabled"):
            active_strategies.append({"strategy_id": int(sid), **scfg})

    active_portfolios = []
    for pid, pcfg in config.get("portfolios", {}).items():
        if pcfg.get("alerts_enabled"):
            active_portfolios.append({"portfolio_id": int(pid), **pcfg})

    return {"strategies": active_strategies, "portfolios": active_portfolios}


# =============================================================================
# MONITOR STATUS
# =============================================================================

def load_monitor_status() -> dict:
    """Load monitor status. Returns default offline status if missing."""
    if os.path.exists(MONITOR_STATUS_FILE):
        try:
            with open(MONITOR_STATUS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "running": False,
        "pid": None,
        "last_poll": None,
        "last_poll_duration_ms": None,
        "strategies_monitored": 0,
        "errors": [],
        "started_at": None,
    }


def save_monitor_status(status: dict):
    """Save monitor status."""
    with open(MONITOR_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2, default=str)


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def _run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full indicator/interpreter/trigger pipeline on a DataFrame."""
    if len(df) == 0:
        return df
    df = run_all_indicators(df)
    df = run_all_interpreters(df)
    df = detect_all_triggers(df)
    return df


def _get_base_trigger_id(confluence_trigger_id: str) -> str:
    """
    Map a confluence group trigger ID to the base trigger ID for DataFrame columns.

    Mirrors get_base_trigger_id() from app.py but importable standalone.
    Falls back to the input ID if mapping fails.
    """
    try:
        from confluence_groups import get_all_triggers, get_enabled_groups, TEMPLATES

        all_triggers = get_all_triggers()
        if confluence_trigger_id in all_triggers:
            trigger_def = all_triggers[confluence_trigger_id]
            base_trigger = trigger_def.base_trigger

            enabled_groups = get_enabled_groups()
            for group in enabled_groups:
                if group.get_trigger_id(base_trigger) == confluence_trigger_id:
                    template = TEMPLATES.get(group.base_template)
                    if template and "trigger_prefix" in template:
                        return f"{template['trigger_prefix']}_{base_trigger}"
                    break
    except Exception:
        pass

    return confluence_trigger_id


def detect_signals(strategy: dict, df: pd.DataFrame = None, feed: str = "sip",
                   secondary_tf_dfs: dict = None,
                   override_in_position: bool = None) -> list:
    """
    Run the full pipeline on recent data and check for entry/exit signals.

    Args:
        strategy: Strategy dict with symbol, triggers, confluence, etc.
        df: Pre-loaded DataFrame of bars. If None, bars are loaded automatically
            using compute_signal_detection_bars() for the strategy's timeframe.
        feed: Data feed — "sip" or "iex"
        secondary_tf_dfs: Optional dict ``{tf_label: pd.DataFrame}`` of secondary
            timeframe bar data for multi-timeframe confluence.  Each DataFrame
            should contain raw OHLCV bars; the pipeline is run on them here.

    Returns list of signal dicts (usually 0 or 1 items).
    """
    symbol = strategy.get('symbol', 'SPY')
    seed = strategy.get('data_seed', 42)
    timeframe = strategy.get('timeframe', '1Min')

    # Load recent bars if not provided
    session = strategy.get('trading_session', 'RTH')
    if df is None:
        bars_needed = compute_signal_detection_bars(timeframe, session)
        df = load_latest_bars(symbol, bars=bars_needed, timeframe=timeframe,
                              seed=seed, feed=feed, session=session)
    if len(df) == 0:
        return []

    # Run pipeline on primary TF
    df = _run_pipeline(df)

    # ── Multi-Timeframe: run pipeline on secondary TFs, forward-fill states ──
    secondary_tf_map = {}
    interp_keys = list(_ALL_INTERPRETERS.keys())
    if secondary_tf_dfs:
        for tf_label, sec_df in secondary_tf_dfs.items():
            if sec_df is None or len(sec_df) == 0:
                continue
            try:
                sec_df = _run_pipeline(sec_df.copy())
                suffixed_cols = []
                for interp_col in interp_keys:
                    if interp_col in sec_df.columns:
                        suffixed = f"{interp_col}__{tf_label}"
                        df[suffixed] = sec_df[interp_col].reindex(df.index, method='ffill')
                        suffixed_cols.append(suffixed)
                if suffixed_cols:
                    secondary_tf_map[tf_label] = suffixed_cols
            except Exception:
                pass

    # Get trigger IDs
    entry_trigger = strategy.get('entry_trigger') or ''
    exit_trigger = strategy.get('exit_trigger') or ''

    # Map confluence trigger IDs to base trigger IDs if available
    if strategy.get('entry_trigger_confluence_id'):
        entry_trigger = _get_base_trigger_id(strategy['entry_trigger_confluence_id'])
    if strategy.get('exit_trigger_confluence_id'):
        exit_trigger = _get_base_trigger_id(strategy['exit_trigger_confluence_id'])

    # Build exit triggers list (new multi-exit format or legacy single)
    exit_triggers_list = None
    if strategy.get('exit_trigger_confluence_ids'):
        exit_triggers_list = [_get_base_trigger_id(tid) for tid in strategy['exit_trigger_confluence_ids'] if tid]
    elif strategy.get('exit_triggers'):
        exit_triggers_list = [et for et in strategy['exit_triggers'] if et]

    # Determine position state
    confluence_set = set(strategy.get('confluence', [])) if strategy.get('confluence') else None

    if override_in_position is not None:
        # Caller (e.g. streaming engine) knows authoritative position state
        in_position = override_in_position
    else:
        trades = generate_trades(
            df,
            direction=strategy.get('direction', 'LONG'),
            entry_trigger=entry_trigger,
            exit_trigger=exit_trigger,
            exit_triggers=exit_triggers_list,
            confluence_required=confluence_set,
            risk_per_trade=strategy.get('risk_per_trade', 100.0),
            stop_atr_mult=strategy.get('stop_atr_mult', 1.5),
            stop_config=strategy.get('stop_config'),
            target_config=strategy.get('target_config'),
            bar_count_exit=strategy.get('bar_count_exit'),
            secondary_tf_map=secondary_tf_map if secondary_tf_map else None,
        )
        in_position = False
        if len(trades) > 0:
            last_trade = trades.iloc[-1]
            if pd.isna(last_trade.get('exit_time')):
                in_position = True

    signals = []
    last_bar = df.iloc[-1]
    interpreter_list = interp_keys
    if secondary_tf_map:
        confluence_records = get_mtf_confluence_records(
            last_bar, interpreter_list, secondary_tf_map)
    else:
        confluence_records = get_confluence_records(last_bar, "1M", interpreter_list)

    # Check the last bar for trigger signals
    entry_col = f"trig_{entry_trigger}"
    # _ib triggers share the boolean column with their bar-close base
    if entry_col not in df.columns and entry_trigger.endswith('_ib'):
        entry_col = f"trig_{entry_trigger[:-3]}"

    # Build list of exit trigger columns to check
    exit_cols = []
    if exit_triggers_list:
        for et in exit_triggers_list:
            ec = f"trig_{et}"
            if ec not in df.columns and et.endswith('_ib'):
                ec = f"trig_{et[:-3]}"
            exit_cols.append(ec)
    elif exit_trigger and exit_trigger not in ("opposite_signal", "fixed_r_2", "fixed_r_3", "trailing_stop", "time_exit_50"):
        ec = f"trig_{exit_trigger}"
        if ec not in df.columns and exit_trigger.endswith('_ib'):
            ec = f"trig_{exit_trigger[:-3]}"
        exit_cols = [ec]

    atr_val = last_bar.get('atr', last_bar.get('close', 100) * 0.01)
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = last_bar.get('close', 100) * 0.01

    direction = strategy.get('direction', 'LONG')
    close_price = float(last_bar['close'])

    if not in_position:
        # Look for entry signal
        if entry_col in df.columns and last_bar.get(entry_col, False):
            # Check confluence
            if confluence_set and not confluence_set.issubset(confluence_records):
                pass  # Confluence not met
            else:
                # Calculate stop price using configured method
                effective_stop = strategy.get('stop_config') or {"method": "atr", "atr_mult": strategy.get("stop_atr_mult", 1.5)}
                stop_price = calculate_stop_price(close_price, direction, last_bar, df, len(df) - 1, effective_stop)

                signals.append({
                    "type": "entry_signal",
                    "trigger": entry_trigger,
                    "confluence_met": list(confluence_records & confluence_set) if confluence_set else [],
                    "bar_time": str(last_bar.name) if hasattr(last_bar, 'name') else datetime.now(timezone.utc).isoformat(),
                    "price": close_price,
                    "stop_price": float(stop_price),
                    "atr": float(atr_val),
                })
    else:
        # Look for exit signal on any exit trigger column
        for ec in exit_cols:
            if ec in df.columns and last_bar.get(ec, False):
                signals.append({
                    "type": "exit_signal",
                    "trigger": ec.replace("trig_", ""),
                    "confluence_met": list(confluence_records),
                    "bar_time": str(last_bar.name) if hasattr(last_bar, 'name') else datetime.now(timezone.utc).isoformat(),
                    "price": close_price,
                    "stop_price": None,
                    "atr": float(atr_val),
                })
                break  # First exit trigger to fire wins

    return signals


# =============================================================================
# PORTFOLIO ENRICHMENT
# =============================================================================

def enrich_signal_with_portfolio_context(signal: dict, strategy_id: int) -> dict:
    """
    Add portfolio context to a strategy-level signal.

    For each portfolio containing this strategy, adds allocation info.
    """
    portfolios = load_portfolios()
    context = []

    for port in portfolios:
        for alloc in port.get('strategies', []):
            if alloc.get('strategy_id') == strategy_id:
                context.append({
                    "portfolio_id": port['id'],
                    "portfolio_name": port.get('name', f"Portfolio {port['id']}"),
                    "position_risk": alloc.get('risk_per_trade', 100.0),
                })
                break

    signal['portfolio_context'] = context
    return signal


# =============================================================================
# WEBHOOK DELIVERY
# =============================================================================

def _format_default_webhook_payload(alert: dict) -> dict:
    """Format alert as a Discord/Slack-compatible webhook payload (fallback)."""
    alert_type = alert.get('type', 'signal')
    symbol = alert.get('symbol', '?')
    direction = alert.get('direction', '?')
    price = alert.get('price', 0)
    stop_price = alert.get('stop_price')
    strategy_name = alert.get('strategy_name', 'Unknown Strategy')

    if alert_type == 'entry_signal':
        emoji = "\U0001f7e2"  # green circle
        content = f"{emoji} ENTRY SIGNAL: {symbol} {direction} @ ${price:.2f}"
        if stop_price:
            content += f" (Stop: ${stop_price:.2f})"
    elif alert_type == 'exit_signal':
        emoji = "\U0001f534"  # red circle
        content = f"{emoji} EXIT SIGNAL: {symbol} {direction} @ ${price:.2f}"
    elif alert_type == 'compliance_breach':
        emoji = "\u26a0\ufe0f"  # warning
        portfolio_name = alert.get('portfolio_name', 'Unknown Portfolio')
        rule_name = alert.get('rule_name', 'Unknown Rule')
        content = f"{emoji} COMPLIANCE BREACH: {portfolio_name} - {rule_name}"
    else:
        content = f"Alert: {alert_type} for {strategy_name}"

    # Build embed fields
    fields = []
    if alert.get('trigger'):
        fields.append({"name": "Trigger", "value": alert['trigger'], "inline": True})
    if price:
        fields.append({"name": "Price", "value": f"${price:.2f}", "inline": True})
    if stop_price:
        fields.append({"name": "Stop", "value": f"${stop_price:.2f}", "inline": True})
    if alert.get('confluence_met'):
        fields.append({"name": "Confluence", "value": ", ".join(alert['confluence_met']), "inline": False})
    if alert.get('portfolio_context'):
        port_text = ", ".join(
            f"{c['portfolio_name']} (${c['position_risk']:.0f})"
            for c in alert['portfolio_context']
        )
        fields.append({"name": "Portfolios", "value": port_text, "inline": False})

    payload = {
        "content": content,
        "embeds": [{
            "title": strategy_name,
            "fields": fields,
            "timestamp": alert.get('timestamp', datetime.now(timezone.utc).isoformat()),
        }]
    }

    return payload


def send_webhook(url: str, alert: dict, custom_payload: str = None) -> dict:
    """
    Send alert to webhook URL. Retries once on failure.
    Returns dict with 'success', 'status_code', 'error', 'payload_sent' keys.
    """
    if not url:
        return {"success": False, "status_code": None, "error": "No URL", "payload_sent": None}

    try:
        import requests
    except ImportError:
        return {"success": False, "status_code": None, "error": "requests not installed", "payload_sent": None}

    if custom_payload:
        try:
            payload = json.loads(custom_payload)
        except json.JSONDecodeError:
            # If template isn't valid JSON, send as raw text
            payload = {"content": custom_payload}
    else:
        payload = _format_default_webhook_payload(alert)

    payload_str = json.dumps(payload)
    last_status = None
    last_error = None

    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=5)
            last_status = resp.status_code
            if resp.status_code < 300:
                return {"success": True, "status_code": resp.status_code, "error": None, "payload_sent": payload_str}
            last_error = f"HTTP {resp.status_code}"
        except Exception as e:
            last_error = str(e)
            if attempt == 0:
                continue

    return {"success": False, "status_code": last_status, "error": last_error, "payload_sent": payload_str}


# =============================================================================
# WEBHOOK TEMPLATES
# =============================================================================

DEFAULT_WEBHOOK_TEMPLATES = [
    {
        "id": "tpl_default_ttp_market_buy",
        "name": "TradeThePool - Market Order - Buy",
        "category": "TradeThePool",
        "is_default": True,
        "payload_template": '{\n  "symbol": "{{symbol}}",\n  "action": "buy",\n  "quantity": {{quantity}}\n}',
    },
    {
        "id": "tpl_default_ttp_market_sell",
        "name": "TradeThePool - Market Order - Sell",
        "category": "TradeThePool",
        "is_default": True,
        "payload_template": '{\n  "symbol": "{{symbol}}",\n  "action": "sell",\n  "quantity": {{quantity}}\n}',
    },
    {
        "id": "tpl_default_ttp_market_close",
        "name": "TradeThePool - Market Order - Close",
        "category": "TradeThePool",
        "is_default": True,
        "payload_template": '{\n  "symbol": "{{symbol}}",\n  "action": "close"\n}',
    },
    {
        "id": "tpl_default_ttp_limit_buy",
        "name": "TradeThePool - Limit Order - Buy",
        "category": "TradeThePool",
        "is_default": True,
        "payload_template": '{\n  "symbol": "{{symbol}}",\n  "action": "buy",\n  "quantity": {{quantity}},\n  "limit_price": {{order_price}}\n}',
    },
    {
        "id": "tpl_default_ttp_limit_sell",
        "name": "TradeThePool - Limit Order - Sell",
        "category": "TradeThePool",
        "is_default": True,
        "payload_template": '{\n  "symbol": "{{symbol}}",\n  "action": "sell",\n  "quantity": {{quantity}},\n  "limit_price": {{order_price}}\n}',
    },
    {
        "id": "tpl_default_ttp_limit_close",
        "name": "TradeThePool - Limit Order - Close",
        "category": "TradeThePool",
        "is_default": True,
        "payload_template": '{\n  "symbol": "{{symbol}}",\n  "action": "close"\n}',
    },
]


def load_webhook_templates() -> list:
    """Load webhook templates. Seeds defaults if file missing."""
    if os.path.exists(WEBHOOK_TEMPLATES_FILE):
        with open(WEBHOOK_TEMPLATES_FILE, 'r') as f:
            templates = json.load(f)
        # Ensure all defaults are present
        existing_ids = {t.get("id") for t in templates}
        added = False
        for default in DEFAULT_WEBHOOK_TEMPLATES:
            if default["id"] not in existing_ids:
                templates.append(dict(default))
                added = True
        if added:
            save_webhook_templates(templates)
        return templates

    # Seed defaults
    templates = [dict(t) for t in DEFAULT_WEBHOOK_TEMPLATES]
    save_webhook_templates(templates)
    return templates


def save_webhook_templates(templates: list):
    """Save webhook templates to file."""
    with open(WEBHOOK_TEMPLATES_FILE, 'w') as f:
        json.dump(templates, f, indent=2)


def add_webhook_template(template: dict) -> dict:
    """Add a user-created webhook template. Auto-assigns ID."""
    templates = load_webhook_templates()
    template["id"] = f"tpl_{secrets.token_hex(3)}"
    template["is_default"] = False
    templates.append(template)
    save_webhook_templates(templates)
    return template


def update_webhook_template(template_id: str, updates: dict) -> bool:
    """Update an existing webhook template. Returns True if found and updated."""
    templates = load_webhook_templates()
    for i, t in enumerate(templates):
        if t.get("id") == template_id:
            templates[i].update(updates)
            save_webhook_templates(templates)
            return True
    return False


def delete_webhook_template(template_id: str) -> bool:
    """Delete a webhook template (only user-created). Returns True if removed."""
    templates = load_webhook_templates()
    original_len = len(templates)
    templates = [t for t in templates if not (t.get("id") == template_id and not t.get("is_default", False))]
    if len(templates) < original_len:
        save_webhook_templates(templates)
        return True
    return False


def get_webhook_template_by_id(template_id: str) -> dict | None:
    """Get a single webhook template by ID."""
    for t in load_webhook_templates():
        if t.get("id") == template_id:
            return t
    return None


# =============================================================================
# LIVE ALERTS VALIDATION (Phase 13)
# =============================================================================

def match_alerts_to_trades(strategy: dict, alerts: list = None) -> dict:
    """Match fired alerts to forward test trades for a strategy.

    Compares alerts from alerts.json against forward test trades to identify:
    - Successfully matched alert→trade pairs (live_executions)
    - Missed alerts (forward test trade exists, no matching alert)
    - Phantom alerts (alert fired but no matching forward test trade)

    Args:
        strategy: Strategy dict with stored_trades and forward_test_start
        alerts: Optional list of alert records. If None, loads from file.

    Returns:
        dict with 'live_executions' and 'discrepancies' lists.
    """
    if alerts is None:
        alerts = load_alerts(limit=10000)

    stored_trades = strategy.get('stored_trades', [])
    ft_start = strategy.get('forward_test_start')
    strategy_id = strategy.get('id')

    if not ft_start or not strategy_id:
        return {'live_executions': [], 'discrepancies': []}

    # Normalize all timestamps to UTC-naive for safe comparison.
    # Alert timestamps are local (no tz), trade times are UTC (+00:00).
    from datetime import timezone as _tz
    def _utc_naive(dt):
        """Convert any datetime to UTC-naive (naive local → UTC, tz-aware → UTC)."""
        return dt.astimezone(_tz.utc).replace(tzinfo=None)

    ft_start_dt = _utc_naive(datetime.fromisoformat(ft_start))

    # Parse reset timestamp (used later to filter discrepancies, NOT trades)
    _reset_dt = None
    _reset_at = strategy.get('alert_tracking_reset_at')
    if _reset_at:
        try:
            _reset_dt = _utc_naive(datetime.fromisoformat(_reset_at))
        except (ValueError, TypeError):
            pass

    # Get ALL forward test trades for matching (don't filter by reset —
    # new alerts may legitimately match older trades)
    ft_trades = []
    for idx, t in enumerate(stored_trades):
        try:
            entry_dt = _utc_naive(datetime.fromisoformat(t['entry_time']))
            if entry_dt >= ft_start_dt:
                ft_trades.append((idx, t, entry_dt))
        except (ValueError, KeyError):
            continue

    # Get alerts for this strategy, pre-parse timestamps once (avoid O(n²) parsing)
    entry_alerts = []  # (parsed_dt, alert)
    exit_alerts = []
    for a in alerts:
        if a.get('strategy_id') != strategy_id:
            continue
        a_type = a.get('type')
        if a_type not in ('entry_signal', 'exit_signal'):
            continue
        try:
            a_dt = _utc_naive(datetime.fromisoformat(a['timestamp']))
        except (ValueError, KeyError):
            continue
        if a_type == 'entry_signal':
            entry_alerts.append((a_dt, a))
        else:
            exit_alerts.append((a_dt, a))

    # Sort by time for faster matching
    entry_alerts.sort(key=lambda x: x[0])
    exit_alerts.sort(key=lambda x: x[0])

    from realtime_engine import TIMEFRAME_SECONDS as _TFS
    _bar_period = _TFS.get(strategy.get('timeframe', '1Min'), 60)
    MATCH_WINDOW_SECONDS = max(300, _bar_period * 5)  # ±5 bar periods, min 5 minutes

    executions = []
    matched_alert_ids = set()
    matched_trade_indices = set()

    for trade_idx, trade, trade_entry_dt in ft_trades:
        trade_exit_dt = None
        try:
            trade_exit_dt = _utc_naive(datetime.fromisoformat(trade['exit_time']))
        except (ValueError, KeyError):
            pass

        # Find closest entry alert match
        entry_match = None
        best_entry_delta = float('inf')
        for a_dt, alert in entry_alerts:
            if alert['id'] in matched_alert_ids:
                continue
            delta = abs((a_dt - trade_entry_dt).total_seconds())
            if delta < MATCH_WINDOW_SECONDS and delta < best_entry_delta:
                entry_match = alert
                best_entry_delta = delta

        # Find closest exit alert match
        exit_match = None
        if trade_exit_dt:
            best_exit_delta = float('inf')
            for a_dt, alert in exit_alerts:
                if alert['id'] in matched_alert_ids:
                    continue
                delta = abs((a_dt - trade_exit_dt).total_seconds())
                if delta < MATCH_WINDOW_SECONDS and delta < best_exit_delta:
                    exit_match = alert
                    best_exit_delta = delta

        if entry_match:
            matched_alert_ids.add(entry_match['id'])
            matched_trade_indices.add(trade_idx)

            alert_price = entry_match.get('price', 0)
            theoretical_price = trade.get('entry_price', alert_price)
            risk = abs(theoretical_price - trade.get('stop_price', theoretical_price)) or 1.0

            # Check webhook delivery status
            deliveries = entry_match.get('webhook_deliveries', [])
            webhook_delivered = any(d.get('success') for d in deliveries)

            execution = {
                'alert_id': entry_match['id'],
                'type': 'entry',
                'alert_timestamp': entry_match.get('timestamp'),
                'bar_time': entry_match.get('bar_time'),
                'source': entry_match.get('source', 'unknown'),
                'alert_price': alert_price,
                'theoretical_price': theoretical_price,
                'slippage_r': round((alert_price - theoretical_price) / risk, 4) if risk > 0 else 0,
                'matched_trade_index': trade_idx,
                'webhook_delivered': webhook_delivered,
                'webhook_delivery_count': len(deliveries),
            }
            executions.append(execution)

            if exit_match:
                matched_alert_ids.add(exit_match['id'])
                exit_alert_price = exit_match.get('price', 0)
                exit_theoretical = trade.get('exit_price', exit_alert_price)

                exit_execution = {
                    'alert_id': exit_match['id'],
                    'type': 'exit',
                    'alert_timestamp': exit_match.get('timestamp'),
                    'bar_time': exit_match.get('bar_time'),
                    'source': exit_match.get('source', 'unknown'),
                    'alert_price': exit_alert_price,
                    'theoretical_price': exit_theoretical,
                    'slippage_r': round((exit_alert_price - exit_theoretical) / risk, 4) if risk > 0 else 0,
                    'matched_trade_index': trade_idx,
                    'webhook_delivered': any(d.get('success') for d in exit_match.get('webhook_deliveries', [])),
                    'webhook_delivery_count': len(exit_match.get('webhook_deliveries', [])),
                }
                executions.append(exit_execution)

    # Build discrepancies
    discrepancies = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # Missed alerts: forward test trades with no matching alert
    # Skip trades before reset — their alerts were intentionally deleted
    for trade_idx, trade, trade_entry_dt in ft_trades:
        if trade_idx not in matched_trade_indices:
            if _reset_dt and trade_entry_dt < _reset_dt:
                continue  # trade predates reset, alerts were cleared
            discrepancies.append({
                'type': 'missed_alert',
                'trade_index': trade_idx,
                'trade_entry_time': trade.get('entry_time'),
                'trade_exit_time': trade.get('exit_time'),
                'detected_at': now_iso,
            })

    # Phantom alerts: alerts that fired but no matching forward test trade
    # Skip alerts before reset (shouldn't exist since we deleted them, but guard)
    all_parsed_alerts = entry_alerts + exit_alerts
    for a_dt, alert in all_parsed_alerts:
        if alert['id'] not in matched_alert_ids:
            if a_dt >= ft_start_dt and (not _reset_dt or a_dt >= _reset_dt):
                discrepancies.append({
                    'type': 'phantom_alert',
                    'alert_id': alert['id'],
                    'alert_type': alert.get('type'),
                    'alert_timestamp': alert.get('timestamp'),
                    'bar_time': alert.get('bar_time'),
                    'source': alert.get('source', 'unknown'),
                    'price': alert.get('price'),
                    'trigger': alert.get('trigger', ''),
                    'detected_at': now_iso,
                })

    return {'live_executions': executions, 'discrepancies': discrepancies}
