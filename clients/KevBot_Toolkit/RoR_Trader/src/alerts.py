"""
Alert Engine for RoR Trader
=============================

Handles alert configuration, signal detection, position tracking,
portfolio enrichment, webhook delivery, and alert history.

The alert monitor (alert_monitor.py) uses this module to detect signals
and deliver notifications. The Streamlit app uses the CRUD functions
to configure alerts and display history.
"""

import json
import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from data_loader import load_latest_bars
from indicators import run_all_indicators
from interpreters import (
    INTERPRETERS,
    run_all_interpreters,
    detect_all_triggers,
    get_confluence_records,
)
from triggers import generate_trades
from portfolios import load_portfolios, get_requirement_set_by_id

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALERT_CONFIG_FILE = os.path.join(_SCRIPT_DIR, "alert_config.json")
ALERTS_FILE = os.path.join(_SCRIPT_DIR, "alerts.json")
MONITOR_STATUS_FILE = os.path.join(_SCRIPT_DIR, "monitor_status.json")

# How many bars to load for signal detection (enough for indicator warmup)
SIGNAL_DETECTION_BARS = 200

DEFAULT_GLOBAL_CONFIG = {
    "webhook_url": "",
    "poll_interval_seconds": 60,
    "market_hours_only": True,
    "enabled": False,
}

DEFAULT_STRATEGY_CONFIG = {
    "alerts_enabled": False,
    "alert_on_entry": True,
    "alert_on_exit": True,
    "webhook_url": "",
}

DEFAULT_PORTFOLIO_CONFIG = {
    "alerts_enabled": False,
    "alert_on_signal": True,
    "alert_on_compliance_breach": True,
    "webhook_url": "",
}


# =============================================================================
# ALERT CONFIG CRUD
# =============================================================================

def load_alert_config() -> dict:
    """Load alert configuration from JSON file. Returns defaults if missing."""
    if os.path.exists(ALERT_CONFIG_FILE):
        with open(ALERT_CONFIG_FILE, 'r') as f:
            return json.load(f)
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
# ALERT HISTORY CRUD
# =============================================================================

def load_alerts(limit: int = 100) -> list:
    """Load alert history, most recent first. Returns up to `limit` alerts."""
    if not os.path.exists(ALERTS_FILE):
        return []
    with open(ALERTS_FILE, 'r') as f:
        alerts = json.load(f)
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
        alert['timestamp'] = datetime.now().isoformat()
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


def detect_signals(strategy: dict) -> list:
    """
    Run the full pipeline on recent data and check for entry/exit signals.

    Returns list of signal dicts (usually 0 or 1 items).
    """
    symbol = strategy.get('symbol', 'SPY')
    seed = strategy.get('data_seed', 42)

    # Load recent bars
    df = load_latest_bars(symbol, bars=SIGNAL_DETECTION_BARS, seed=seed)
    if len(df) == 0:
        return []

    # Run pipeline
    df = _run_pipeline(df)

    # Get trigger IDs
    entry_trigger = strategy.get('entry_trigger', '')
    exit_trigger = strategy.get('exit_trigger', '')

    # Map confluence trigger IDs to base trigger IDs if available
    if strategy.get('entry_trigger_confluence_id'):
        entry_trigger = _get_base_trigger_id(strategy['entry_trigger_confluence_id'])
    if strategy.get('exit_trigger_confluence_id'):
        exit_trigger = _get_base_trigger_id(strategy['exit_trigger_confluence_id'])

    # Determine position state via generate_trades on recent bars
    confluence_set = set(strategy.get('confluence', [])) if strategy.get('confluence') else None
    trades = generate_trades(
        df,
        direction=strategy.get('direction', 'LONG'),
        entry_trigger=entry_trigger,
        exit_trigger=exit_trigger,
        confluence_required=confluence_set,
        risk_per_trade=strategy.get('risk_per_trade', 100.0),
        stop_atr_mult=strategy.get('stop_atr_mult', 1.5),
    )

    signals = []
    last_bar = df.iloc[-1]
    interpreter_list = list(INTERPRETERS.keys())
    confluence_records = get_confluence_records(last_bar, "1M", interpreter_list)

    # Check the last bar for trigger signals
    entry_col = f"trig_{entry_trigger}"
    exit_col = f"trig_{exit_trigger}"

    # Determine if currently in position
    in_position = False
    if len(trades) > 0:
        last_trade = trades.iloc[-1]
        # If last trade has no exit_time or exit_time is NaT, we're in position
        if pd.isna(last_trade.get('exit_time')):
            in_position = True

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
                if direction == "LONG":
                    stop_price = close_price - (atr_val * strategy.get('stop_atr_mult', 1.5))
                else:
                    stop_price = close_price + (atr_val * strategy.get('stop_atr_mult', 1.5))

                signals.append({
                    "type": "entry_signal",
                    "trigger": entry_trigger,
                    "confluence_met": list(confluence_records & confluence_set) if confluence_set else [],
                    "bar_time": str(last_bar.name) if hasattr(last_bar, 'name') else datetime.now().isoformat(),
                    "price": close_price,
                    "stop_price": float(stop_price),
                    "atr": float(atr_val),
                })
    else:
        # Look for exit signal
        if exit_col in df.columns and last_bar.get(exit_col, False):
            signals.append({
                "type": "exit_signal",
                "trigger": exit_trigger,
                "confluence_met": list(confluence_records),
                "bar_time": str(last_bar.name) if hasattr(last_bar, 'name') else datetime.now().isoformat(),
                "price": close_price,
                "stop_price": None,
                "atr": float(atr_val),
            })

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

def _format_webhook_payload(alert: dict) -> dict:
    """Format alert as a Discord/Slack-compatible webhook payload."""
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
            "timestamp": alert.get('timestamp', datetime.now().isoformat()),
        }]
    }

    return payload


def send_webhook(url: str, alert: dict) -> bool:
    """
    Send alert to webhook URL. Retries once on failure.
    Returns True if delivery succeeded.
    """
    if not url:
        return False

    try:
        import requests
    except ImportError:
        print("Warning: requests package not installed, cannot send webhooks")
        return False

    payload = _format_webhook_payload(alert)

    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code < 300:
                return True
        except Exception as e:
            if attempt == 0:
                continue
            print(f"Webhook delivery failed: {e}")

    return False
