"""
Alert Monitor for RoR Trader
==============================

Standalone background script that polls for trading signals on
forward-tested strategies and delivers alerts via webhook + in-app log.

Usage:
    python alert_monitor.py

The Streamlit app starts/stops this process and reads its status via
monitor_status.json.
"""

import os
import sys
import json
import time
import signal
import traceback
from datetime import datetime

# Ensure src/ is on the path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Load .env before any other imports that need it
from dotenv import load_dotenv
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

from alerts import (
    load_alert_config,
    save_alert,
    send_webhook,
    detect_signals,
    enrich_signal_with_portfolio_context,
    load_monitor_status,
    save_monitor_status,
    get_strategy_alert_config,
    get_portfolio_alert_config,
)
from portfolios import (
    load_portfolios,
    get_portfolio_by_id,
    get_requirement_set_by_id,
    get_portfolio_trades,
    calculate_portfolio_kpis,
    evaluate_requirement_set,
)


# =============================================================================
# GLOBALS
# =============================================================================

_running = True


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully."""
    global _running
    _running = False
    print(f"\nReceived signal {signum}, shutting down...")


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# =============================================================================
# HELPERS
# =============================================================================

def load_strategies() -> list:
    """Load all strategies from strategies.json."""
    strat_file = os.path.join(_SCRIPT_DIR, "strategies.json")
    if not os.path.exists(strat_file):
        return []
    with open(strat_file, 'r') as f:
        return json.load(f)


def get_strategy_by_id(strategy_id: int) -> dict | None:
    """Get a single strategy by ID."""
    for s in load_strategies():
        if s.get('id') == strategy_id:
            return s
    return None


def get_monitored_strategies(config: dict) -> list:
    """
    Get all strategies that should be monitored.
    A strategy is monitored if:
    - It has forward_testing enabled
    - It has alerts_enabled in the alert config
    """
    strategies = load_strategies()
    monitored = []

    for strat in strategies:
        if not strat.get('forward_testing'):
            continue
        if 'entry_trigger_confluence_id' not in strat:
            continue  # skip legacy strategies

        strat_config = config.get('strategies', {}).get(str(strat['id']), {})
        if strat_config.get('alerts_enabled', False):
            monitored.append(strat)

    return monitored


def should_poll(config: dict) -> bool:
    """Check if we should poll based on market hours setting."""
    if not config.get('global', {}).get('market_hours_only', True):
        return True

    now = datetime.now()
    # Market hours: 9:30 AM - 4:00 PM ET (simplified, no timezone conversion)
    # For a more robust implementation, use pytz for proper ET handling
    hour = now.hour
    minute = now.minute

    if hour < 9 or (hour == 9 and minute < 30):
        return False
    if hour >= 16:
        return False
    # Skip weekends
    if now.weekday() >= 5:
        return False

    return True


def update_monitor_status(**kwargs):
    """Update specific fields in monitor_status.json."""
    status = load_monitor_status()
    status.update(kwargs)
    save_monitor_status(status)


def log_monitor_error(strategy_id: int, error: str):
    """Append an error to the monitor status error list (keep last 50)."""
    status = load_monitor_status()
    errors = status.get('errors', [])
    errors.append({
        "strategy_id": strategy_id,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    })
    status['errors'] = errors[-50:]  # keep last 50
    save_monitor_status(status)


def deliver_alert(alert: dict, config: dict):
    """
    Deliver an alert via webhook.
    Priority: strategy-specific URL > portfolio-specific URL > global URL.
    """
    strategy_id = alert.get('strategy_id')
    webhook_url = None

    # Strategy-specific webhook
    if strategy_id:
        strat_config = config.get('strategies', {}).get(str(strategy_id), {})
        if strat_config.get('webhook_url'):
            webhook_url = strat_config['webhook_url']

    # Portfolio-specific webhook (use first portfolio with a configured webhook)
    if not webhook_url:
        for ctx in alert.get('portfolio_context', []):
            pid = ctx.get('portfolio_id')
            port_config = config.get('portfolios', {}).get(str(pid), {})
            if port_config.get('webhook_url'):
                webhook_url = port_config['webhook_url']
                break

    # Global webhook
    if not webhook_url:
        webhook_url = config.get('global', {}).get('webhook_url', '')

    if webhook_url:
        success = send_webhook(webhook_url, alert)
        alert['webhook_sent'] = success
    else:
        alert['webhook_sent'] = False


def check_portfolio_compliance(portfolio_id: int, config: dict):
    """
    Check if a portfolio has breached any requirement set rules.
    Only generates alerts for newly breached rules (not repeated).
    """
    portfolio = get_portfolio_by_id(portfolio_id)
    if not portfolio:
        return

    req_set_id = portfolio.get('requirement_set_id')
    if not req_set_id:
        return

    req_set = get_requirement_set_by_id(req_set_id)
    if not req_set:
        return

    kpis = portfolio.get('cached_kpis')
    if not kpis:
        return

    # We need daily_pnl for evaluation — use cached KPIs for a lightweight check
    # For compliance breaches, we check against cached KPIs which are updated
    # when the portfolio is recomputed
    result = evaluate_requirement_set(req_set, portfolio, kpis, pd.DataFrame())

    if not result.get('overall_pass', True):
        for rule in result.get('rules', []):
            if not rule.get('passed', True):
                alert = {
                    "type": "compliance_breach",
                    "level": "portfolio",
                    "portfolio_id": portfolio_id,
                    "portfolio_name": portfolio.get('name', ''),
                    "rule_name": rule.get('name', ''),
                    "rule_limit": rule.get('limit_display', ''),
                    "rule_value": rule.get('value_display', ''),
                    "strategy_id": None,
                    "symbol": None,
                    "direction": None,
                    "price": None,
                    "trigger": None,
                }
                saved = save_alert(alert)
                deliver_alert(saved, config)


# Need pandas for compliance check
import pandas as pd


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    """Main monitor loop."""
    global _running

    print(f"Alert Monitor starting (PID: {os.getpid()})...")

    # Write initial status
    update_monitor_status(
        running=True,
        pid=os.getpid(),
        started_at=datetime.now().isoformat(),
        errors=[],
        strategies_monitored=0,
        last_poll=None,
        last_poll_duration_ms=None,
    )

    config = load_alert_config()
    poll_interval = config.get('global', {}).get('poll_interval_seconds', 60)

    while _running:
        try:
            config = load_alert_config()
            poll_interval = config.get('global', {}).get('poll_interval_seconds', 60)

            if not config.get('global', {}).get('enabled', False):
                time.sleep(poll_interval)
                continue

            if not should_poll(config):
                print(f"Outside market hours, sleeping {poll_interval}s...")
                time.sleep(poll_interval)
                continue

            poll_start = time.time()
            strategies = get_monitored_strategies(config)
            print(f"\nPolling {len(strategies)} strategies at {datetime.now().strftime('%H:%M:%S')}...")

            for strat in strategies:
                strat_config = config.get('strategies', {}).get(str(strat['id']), {})

                try:
                    signals = detect_signals(strat)

                    for sig in signals:
                        # Check if we should alert on this signal type
                        if sig['type'] == 'entry_signal' and not strat_config.get('alert_on_entry', True):
                            continue
                        if sig['type'] == 'exit_signal' and not strat_config.get('alert_on_exit', True):
                            continue

                        # Enrich with strategy info
                        sig['level'] = 'strategy'
                        sig['strategy_id'] = strat['id']
                        sig['strategy_name'] = strat.get('name', f"Strategy {strat['id']}")
                        sig['symbol'] = strat.get('symbol', '?')
                        sig['direction'] = strat.get('direction', '?')
                        sig['risk_per_trade'] = strat.get('risk_per_trade', 100.0)

                        # Add portfolio context
                        sig = enrich_signal_with_portfolio_context(sig, strat['id'])

                        # Save and deliver
                        alert = save_alert(sig)
                        deliver_alert(alert, config)
                        print(f"  Alert: {sig['type']} for {strat['name']} ({strat['symbol']})")

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    print(f"  Error processing {strat.get('name', strat['id'])}: {error_msg}")
                    log_monitor_error(strat['id'], error_msg)

            # Portfolio compliance checks
            for pid_str, pconfig in config.get('portfolios', {}).items():
                if pconfig.get('alerts_enabled') and pconfig.get('alert_on_compliance_breach'):
                    try:
                        check_portfolio_compliance(int(pid_str), config)
                    except Exception as e:
                        print(f"  Compliance check error for portfolio {pid_str}: {e}")

            poll_duration = int((time.time() - poll_start) * 1000)
            update_monitor_status(
                running=True,
                last_poll=datetime.now().isoformat(),
                last_poll_duration_ms=poll_duration,
                strategies_monitored=len(strategies),
            )
            print(f"Poll complete in {poll_duration}ms")

        except Exception as e:
            print(f"Monitor loop error: {e}")
            traceback.print_exc()

        time.sleep(poll_interval)

    # Clean shutdown
    print("Monitor shutting down...")
    update_monitor_status(running=False)
    print("Monitor stopped.")


if __name__ == "__main__":
    main()
