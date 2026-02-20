"""
Alert Monitor for RoR Trader
==============================

Standalone background script that polls for trading signals on
strategies in webhook-enabled portfolios and delivers alerts via
webhook + in-app log.

Features:
- Smart candle-close-aligned polling (polls each timeframe group
  ~3 seconds after its candle close instead of fixed-interval)
- In-memory data cache with incremental bar fetching
- Symbol deduplication (multiple strategies on same symbol share data)

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
from collections import defaultdict
from datetime import datetime

# Ensure src/ is on the path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Load .env before any other imports that need it
from dotenv import load_dotenv
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

import pandas as pd
from data_loader import load_market_data, load_latest_bars

from alerts import (
    load_alert_config,
    save_alert,
    send_webhook,
    detect_signals,
    compute_signal_detection_bars,
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


# Only register signal handlers when running as standalone script
# (importing this module from another thread would fail otherwise).
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


# =============================================================================
# CANDLE-CLOSE TIMING
# =============================================================================

TIMEFRAME_SECONDS = {
    "5Sec": 5, "10Sec": 10, "15Sec": 15, "30Sec": 30,  # Sub-minute (streaming engine only)
    "1Min": 60, "2Min": 120, "3Min": 180, "5Min": 300,
    "10Min": 600, "15Min": 900, "30Min": 1800,
    "1Hour": 3600, "2Hour": 7200, "4Hour": 14400,
    "1Day": 86400, "1Week": 604800, "1Month": 2592000,
}

# Buffer after candle close before polling (seconds)
_CANDLE_CLOSE_BUFFER = 3


def seconds_until_next_close(timeframe: str) -> float:
    """Seconds until the next candle close + buffer for the given timeframe."""
    now = time.time()
    tf_secs = TIMEFRAME_SECONDS.get(timeframe, 60)
    elapsed = now % tf_secs
    remaining = tf_secs - elapsed + _CANDLE_CLOSE_BUFFER
    # If we're within the buffer window after a close, return 0 (poll now)
    if remaining > tf_secs:
        return 0.0
    return remaining


# =============================================================================
# IN-MEMORY DATA CACHE
# =============================================================================

# {(symbol, timeframe): {"df": pd.DataFrame, "last_bar_time": <index value>}}
_data_cache: dict = {}


def load_cached_bars(
    symbol: str,
    timeframe: str,
    bars_needed: int,
    seed: int = 42,
    feed: str = "sip",
    session: str = "RTH",
) -> pd.DataFrame:
    """Load bars with in-memory caching. Fetches only new bars incrementally."""
    cache_key = (symbol, timeframe, feed, session)

    if cache_key in _data_cache:
        cached = _data_cache[cache_key]
        cached_df = cached["df"]
        last_time = cached_df.index[-1]
        # Fetch only bars since last cached bar
        try:
            new_bars = load_market_data(
                symbol, start_date=last_time, end_date=datetime.now(),
                timeframe=timeframe, seed=seed, feed=feed, session=session,
            )
            if len(new_bars) > 0:
                combined = pd.concat([cached_df, new_bars])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.tail(bars_needed)
                _data_cache[cache_key] = {
                    "df": combined,
                    "last_bar_time": combined.index[-1],
                }
                return combined.copy()
        except Exception:
            pass
        return cached_df.tail(bars_needed).copy()

    # Cold start: full load
    df = load_latest_bars(symbol, bars=bars_needed, timeframe=timeframe,
                          seed=seed, feed=feed, session=session)
    if len(df) > 0:
        _data_cache[cache_key] = {"df": df, "last_bar_time": df.index[-1]}
    return df


# =============================================================================
# HELPERS
# =============================================================================

def load_strategies() -> list:
    """Load all strategies from strategies.json."""
    strat_file = os.path.join(_SCRIPT_DIR, "strategies.json")
    if not os.path.exists(strat_file):
        return []
    try:
        with open(strat_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return []


def get_strategy_by_id(strategy_id: int) -> dict | None:
    """Get a single strategy by ID."""
    for s in load_strategies():
        if s.get('id') == strategy_id:
            return s
    return None


def get_monitored_strategies(config: dict) -> list:
    """
    Get all strategies that should be monitored.

    A strategy is monitored if it belongs to ANY portfolio that has at
    least one enabled webhook.  This replaces the old logic that required
    separate alerts_enabled toggles.
    """
    strategies = load_strategies()
    portfolios = load_portfolios()

    # Build set of strategy IDs in portfolios with active webhooks
    webhook_strategy_ids = set()
    for port in portfolios:
        pid = str(port['id'])
        pcfg = config.get('portfolios', {}).get(pid, {})
        webhooks = pcfg.get('webhooks', [])
        has_active_webhook = any(wh.get('enabled', True) for wh in webhooks)
        if has_active_webhook:
            for alloc in port.get('strategies', []):
                webhook_strategy_ids.add(alloc.get('strategy_id'))

    monitored = []
    for strat in strategies:
        if strat['id'] not in webhook_strategy_ids:
            continue
        if 'entry_trigger_confluence_id' not in strat:
            continue  # skip legacy strategies
        monitored.append(strat)

    return monitored


def should_poll(config: dict, strategies: list = None) -> bool:
    """Check if we should poll based on market hours and strategy sessions."""
    if not config.get('global', {}).get('market_hours_only', True):
        return True

    import pytz
    import datetime as dt_mod
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)

    # Skip weekends
    if now.weekday() >= 5:
        return False

    # Determine the widest session window across monitored strategies
    from data_loader import SESSION_HOURS
    sessions = {s.get('trading_session', 'RTH') for s in (strategies or [])} or {'RTH'}

    earliest_start = dt_mod.time(23, 59)
    latest_end = dt_mod.time(0, 0)
    for sess in sessions:
        sh, sm, eh, em = SESSION_HOURS.get(sess, (9, 30, 16, 0))
        s_time = dt_mod.time(sh, sm)
        e_time = dt_mod.time(eh, em)
        if s_time < earliest_start:
            earliest_start = s_time
        if e_time > latest_end:
            latest_end = e_time

    current_time = now.time()
    return earliest_start <= current_time < latest_end


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
    status['errors'] = errors[-50:]
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


def _portfolio_has_active_webhooks(port_config: dict) -> bool:
    """Check if a portfolio config has at least one enabled webhook."""
    for wh in port_config.get("webhooks", []):
        if wh.get("enabled", True):
            return True
    return False


def deliver_alert(alert: dict, config: dict):
    """
    Deliver an alert to all matching portfolio webhooks.
    Populates alert['webhook_deliveries'] with per-webhook results.

    Delivers to any portfolio that has enabled webhooks (no separate
    alerts_enabled toggle required).
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
        if not _portfolio_has_active_webhooks(port_config):
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


# =============================================================================
# POLLING LOGIC
# =============================================================================

def poll_strategies(strats: list, config: dict, timeframe: str, feed: str = "sip"):
    """
    Poll a group of strategies that share the same timeframe.

    Pre-loads data per unique (symbol, session) pair (deduplication), then runs
    detect_signals with the pre-loaded DataFrame.  For strategies with
    multi-timeframe confluence conditions, secondary TF data is also loaded
    and passed to detect_signals().
    """
    from data_loader import get_required_tfs_from_confluence, get_tf_from_label

    # Deduplicate by (symbol, session) — load data once per unique pair
    symbol_session_seeds = {}
    for strat in strats:
        sym = strat.get('symbol', 'SPY')
        sess = strat.get('trading_session', 'RTH')
        key = (sym, sess)
        if key not in symbol_session_seeds:
            symbol_session_seeds[key] = strat.get('data_seed', 42)

    # Pre-load primary TF data for each unique (symbol, session)
    data_cache = {}
    for (sym, sess), seed in symbol_session_seeds.items():
        bars_needed = compute_signal_detection_bars(timeframe, sess)
        df = load_cached_bars(sym, timeframe, bars_needed, seed=seed,
                              feed=feed, session=sess)
        data_cache[(sym, sess)] = df

    # Cache for secondary TF data: (symbol, session, secondary_tf_str) → df
    sec_data_cache = {}

    # Process each strategy
    for strat in strats:
        sym = strat.get('symbol', 'SPY')
        sess = strat.get('trading_session', 'RTH')
        df = data_cache.get((sym, sess), pd.DataFrame())

        # Load secondary TF data for MTF confluence
        secondary_tf_dfs = None
        req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
        if req_labels:
            secondary_tf_dfs = {}
            seed = strat.get('data_seed', 42)
            for lbl in req_labels:
                sec_tf_str = get_tf_from_label(lbl)
                cache_key = (sym, sess, sec_tf_str)
                if cache_key not in sec_data_cache:
                    sec_bars = compute_signal_detection_bars(sec_tf_str, sess)
                    sec_data_cache[cache_key] = load_cached_bars(
                        sym, sec_tf_str, sec_bars, seed=seed,
                        feed=feed, session=sess)
                sec_df = sec_data_cache[cache_key]
                if len(sec_df) > 0:
                    secondary_tf_dfs[lbl] = sec_df.copy()

        try:
            signals = detect_signals(strat, df=df.copy() if len(df) > 0 else None,
                                     secondary_tf_dfs=secondary_tf_dfs)

            for sig in signals:
                # Enrich with strategy info
                sig['level'] = 'strategy'
                sig['strategy_id'] = strat['id']
                sig['strategy_name'] = strat.get('name', f"Strategy {strat['id']}")
                sig['symbol'] = strat.get('symbol', '?')
                sig['direction'] = strat.get('direction', '?')
                sig['risk_per_trade'] = strat.get('risk_per_trade', 100.0)
                sig['timeframe'] = strat.get('timeframe', '1Min')
                sig['strategy_alerts_visible'] = True

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


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    """Main monitor loop with candle-close-aligned polling."""
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

    # Track which timeframes were polled this cycle to avoid double-polling
    _last_poll_cycle: dict = {}  # {timeframe: last_poll_epoch}

    while _running:
        try:
            config = load_alert_config()
            poll_interval = config.get('global', {}).get('poll_interval_seconds', 60)

            if not config.get('global', {}).get('enabled', False):
                time.sleep(poll_interval)
                continue

            # If the streaming engine is connected, sleep and let it handle alerts
            status = load_monitor_status()
            if status.get('streaming_connected', False):
                time.sleep(5)
                continue

            strategies = get_monitored_strategies(config)

            if not should_poll(config, strategies):
                print(f"Outside market hours, sleeping {poll_interval}s...")
                time.sleep(poll_interval)
                continue

            if not strategies:
                time.sleep(poll_interval)
                continue

            # Group by timeframe
            by_timeframe: dict = defaultdict(list)
            for strat in strategies:
                tf = strat.get('timeframe', '1Min')
                by_timeframe[tf].append(strat)

            # Determine which timeframes need polling now
            poll_start = time.time()
            polled_count = 0
            next_poll_in = float('inf')

            for tf, tf_strats in by_timeframe.items():
                secs = seconds_until_next_close(tf)

                if secs <= 0:
                    # Check if we already polled this candle
                    tf_secs = TIMEFRAME_SECONDS.get(tf, 60)
                    last_poll = _last_poll_cycle.get(tf, 0)
                    candle_epoch = int(time.time() / tf_secs) * tf_secs
                    if last_poll >= candle_epoch:
                        # Already polled for this candle, skip
                        next_close = seconds_until_next_close(tf)
                        next_poll_in = min(next_poll_in, max(next_close, tf_secs - _CANDLE_CLOSE_BUFFER))
                        continue

                    print(f"  [{tf}] Candle closed — polling {len(tf_strats)} strategies...")
                    feed = config.get('global', {}).get('data_feed', 'sip')
                    poll_strategies(tf_strats, config, tf, feed=feed)
                    _last_poll_cycle[tf] = time.time()
                    polled_count += len(tf_strats)

                    # Next poll is one full candle away
                    next_poll_in = min(next_poll_in, tf_secs)
                else:
                    next_poll_in = min(next_poll_in, secs)

            # Portfolio compliance checks (on portfolios with active webhooks)
            for pid_str, pconfig in config.get('portfolios', {}).items():
                if _portfolio_has_active_webhooks(pconfig) and pconfig.get('alert_on_compliance_breach'):
                    try:
                        check_portfolio_compliance(int(pid_str), config)
                    except Exception as e:
                        print(f"  Compliance check error for portfolio {pid_str}: {e}")

            poll_duration = int((time.time() - poll_start) * 1000)
            if polled_count > 0:
                update_monitor_status(
                    running=True,
                    last_poll=datetime.now().isoformat(),
                    last_poll_duration_ms=poll_duration,
                    strategies_monitored=len(strategies),
                )
                print(f"Poll complete: {polled_count} strategies in {poll_duration}ms")
            else:
                update_monitor_status(
                    running=True,
                    strategies_monitored=len(strategies),
                )

        except Exception as e:
            print(f"Monitor loop error: {e}")
            traceback.print_exc()
            next_poll_in = poll_interval

        # Sleep until next candle close (capped, with 1s minimum)
        sleep_time = min(next_poll_in, poll_interval)
        sleep_time = max(1, sleep_time)
        time.sleep(sleep_time)

    # Clean shutdown
    print("Monitor shutting down...")
    update_monitor_status(running=False)
    print("Monitor stopped.")


if __name__ == "__main__":
    main()
