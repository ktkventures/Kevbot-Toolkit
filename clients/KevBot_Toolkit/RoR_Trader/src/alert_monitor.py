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
    build_placeholder_context,
    render_payload,
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
    A strategy is monitored if it has forward_testing enabled AND either:
    - It has alerts_enabled in its own strategy alert config, OR
    - It belongs to any portfolio that has alerts_enabled

    This ensures portfolio webhooks fire even if strategy-level alerts
    are toggled off (those toggles only control Strategy Alerts tab visibility).
    """
    strategies = load_strategies()
    monitored = []

    # Build set of strategy IDs that belong to alert-enabled portfolios
    from portfolios import load_portfolios
    portfolio_strategy_ids = set()
    for pid, pcfg in config.get('portfolios', {}).items():
        if pcfg.get('alerts_enabled', False):
            portfolios = load_portfolios()
            for port in portfolios:
                if str(port['id']) == pid:
                    for alloc in port.get('strategies', []):
                        portfolio_strategy_ids.add(alloc.get('strategy_id'))
                    break

    for strat in strategies:
        if not strat.get('forward_testing'):
            continue
        if 'entry_trigger_confluence_id' not in strat:
            continue  # skip legacy strategies

        strat_config = config.get('strategies', {}).get(str(strat['id']), {})
        strategy_alerts_on = strat_config.get('alerts_enabled', False)
        in_active_portfolio = strat['id'] in portfolio_strategy_ids

        if strategy_alerts_on or in_active_portfolio:
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


def _get_event_key(alert_type: str, direction: str) -> str:
    """Map alert type + direction to webhook event filter key."""
    if alert_type == "entry_signal":
        return "entry_long" if direction == "LONG" else "entry_short"
    elif alert_type == "exit_signal":
        return "exit_long" if direction == "LONG" else "exit_short"
    elif alert_type == "compliance_breach":
        return "compliance_breach"
    return ""


def deliver_alert(alert: dict, config: dict):
    """
    Deliver an alert to all matching portfolio webhooks.
    Populates alert['webhook_deliveries'] with per-webhook results.
    """
    deliveries = []
    alert_type = alert.get("type", "")
    direction = alert.get("direction", "")

    # Map alert type + direction to event key
    event_key = _get_event_key(alert_type, direction)

    # Determine which portfolios to check
    portfolio_ids = set()
    if alert_type == "compliance_breach":
        pid = alert.get("portfolio_id")
        if pid:
            portfolio_ids.add(pid)
    else:
        for ctx in alert.get("portfolio_context", []):
            portfolio_ids.add(ctx.get("portfolio_id"))

    for pid in portfolio_ids:
        port_config = config.get("portfolios", {}).get(str(pid), {})
        if not port_config.get("alerts_enabled", False):
            continue

        # Get portfolio context for this specific portfolio
        port_ctx = None
        for ctx in alert.get("portfolio_context", []):
            if ctx.get("portfolio_id") == pid:
                port_ctx = ctx
                break
        if not port_ctx and alert_type == "compliance_breach":
            port_ctx = {
                "portfolio_id": pid,
                "portfolio_name": alert.get("portfolio_name", ""),
            }

        for wh in port_config.get("webhooks", []):
            if not wh.get("enabled", True):
                continue

            # Check event filter
            events = wh.get("events", {})
            if event_key and not events.get(event_key, False):
                continue

            # Build payload
            placeholder_ctx = build_placeholder_context(alert, port_ctx)
            template = wh.get("payload_template", "")
            custom_payload = render_payload(template, placeholder_ctx) if template else None

            # Send
            result = send_webhook(wh.get("url", ""), alert, custom_payload)

            deliveries.append({
                "webhook_id": wh.get("id", ""),
                "webhook_name": wh.get("name", ""),
                "portfolio_id": pid,
                "sent_at": datetime.now().isoformat(),
                "success": result["success"],
                "status_code": result.get("status_code"),
                "payload_sent": result.get("payload_sent", ""),
                "error": result.get("error", ""),
            })

    alert["webhook_deliveries"] = deliveries
    alert["webhook_sent"] = any(d["success"] for d in deliveries) if deliveries else False

    # Re-save the alert with delivery info
    from alerts import load_alerts, _save_all_alerts
    all_alerts = load_alerts(limit=10000)
    for i, a in enumerate(all_alerts):
        if a.get("id") == alert.get("id"):
            all_alerts[i] = alert
            _save_all_alerts(all_alerts)
            break


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
                strategy_alerts_on = strat_config.get('alerts_enabled', False)

                try:
                    signals = detect_signals(strat)

                    for sig in signals:
                        # Determine strategy-level visibility
                        # (controls whether alert shows in Strategy Alerts tab)
                        strat_visible = strategy_alerts_on
                        if sig['type'] == 'entry_signal' and not strat_config.get('alert_on_entry', True):
                            strat_visible = False
                        if sig['type'] == 'exit_signal' and not strat_config.get('alert_on_exit', True):
                            strat_visible = False

                        # Enrich with strategy info
                        sig['level'] = 'strategy'
                        sig['strategy_id'] = strat['id']
                        sig['strategy_name'] = strat.get('name', f"Strategy {strat['id']}")
                        sig['symbol'] = strat.get('symbol', '?')
                        sig['direction'] = strat.get('direction', '?')
                        sig['risk_per_trade'] = strat.get('risk_per_trade', 100.0)
                        sig['timeframe'] = strat.get('timeframe', '1Min')
                        sig['strategy_alerts_visible'] = strat_visible

                        # Add portfolio context
                        sig = enrich_signal_with_portfolio_context(sig, strat['id'])

                        # Save and deliver (webhooks always fire regardless of strategy toggle)
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
