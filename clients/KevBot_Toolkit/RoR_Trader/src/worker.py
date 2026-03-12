#!/usr/bin/env python3
"""
RoR Trader Worker Service — DB-backed RalphEngine orchestration.

Runs as a standalone Railway service. Polls the monitor_status table
to discover which users want their engine running, starts/stops
per-user RalphEngine instances, and routes all I/O through Supabase.

Usage:
    python src/worker.py

Requires environment variables:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    (Per-user Alpaca keys come from user_settings in the database)
"""
import os
import sys
import time
import json
import signal
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# Ensure USE_DB is true and src/ is on path
os.environ["USE_DB"] = "true"
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv(_SCRIPT_DIR / '.env', override=True)

from db import (
    get_admin_client,
    load_all_desired_states,
    load_alert_config_admin,
    load_strategies_admin,
    load_portfolios_admin,
    save_alert_admin,
    save_monitor_status_admin,
    save_engine_state_db,
    load_engine_state_db,
    append_audit_log_db,
    load_general_packs_admin,
    get_monitored_strategies_db,
)

logger = logging.getLogger("worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

# Worker configuration
POLL_INTERVAL = 15          # seconds between desired_state polls
HEARTBEAT_INTERVAL = 30     # seconds between heartbeat writes
CONFIG_CHECK_INTERVAL = 30  # seconds between alert_config change checks
RESTART_BACKOFF_MAX = 300   # max seconds to wait before restarting crashed engine


# ============================================================
# DB-backed AlertDispatcher
# ============================================================

class DBAlertDispatcher:
    """AlertDispatcher that saves alerts to DB and fires webhooks."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._deliver_alert_fn = None
        try:
            from alert_monitor import deliver_alert
            self._deliver_alert_fn = deliver_alert
        except Exception:
            logger.warning("Could not import deliver_alert — webhooks disabled")

    def dispatch(self, signal_data: dict, strategy: dict,
                 config: dict) -> Optional[dict]:
        """Build, enrich, save, and deliver an alert via DB."""
        from alerts import enrich_signal_with_portfolio_context

        direction = strategy.get('direction', 'LONG')
        sig_type = signal_data['type']
        if sig_type == 'exit_signal':
            order_action = 'close'
        elif direction == 'LONG':
            order_action = 'buy'
        else:
            order_action = 'sell'

        alert = {
            'type': sig_type,
            'trigger': signal_data.get('trigger', ''),
            'price': signal_data.get('price', 0),
            'bar_time': signal_data.get('bar_time', ''),
            'stop_price': signal_data.get('stop_price'),
            'target_price': signal_data.get('target_price'),
            'atr': signal_data.get('atr', 0),
            'level': 'strategy',
            'strategy_id': strategy['id'],
            'strategy_name': strategy.get('name', ''),
            'symbol': strategy.get('symbol', '?'),
            'direction': direction,
            'order_action': order_action,
            'risk_per_trade': strategy.get('risk_per_trade', 100.0),
            'timeframe': strategy.get('timeframe', '1Min'),
            'strategy_alerts_visible': True,
            'source': 'ralph',
        }

        if sig_type == 'exit_signal':
            alert['entry_price'] = signal_data.get('entry_price')
            alert['entry_stop_price'] = signal_data.get('entry_stop_price')

        alert = enrich_signal_with_portfolio_context(alert, strategy['id'])
        alert = save_alert_admin(alert, self.user_id)

        logger.info("ALERT [%s]: %s for %s (%s) trigger=%s price=%.2f",
                     self.user_id[:8], sig_type, strategy.get('name'),
                     strategy.get('symbol'), signal_data.get('trigger'),
                     signal_data.get('price', 0))

        # Webhook delivery
        if self._deliver_alert_fn:
            try:
                self._deliver_alert_fn(alert, config)
            except Exception as e:
                logger.error("Webhook delivery failed: %s", e)

        return alert


# ============================================================
# DB-backed FidelityAuditor
# ============================================================

class DBAuditor:
    """Writes audit entries to the engine_audit_log table."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._batch = []
        self._batch_lock = threading.Lock()
        self._last_flush = time.time()

    def log_bar_close(self, symbol, tf_seconds, bar, indicator_values,
                      trigger_booleans, interpreter_states, positions):
        """Buffer an audit entry (flushed periodically to reduce DB calls)."""
        entry = {
            'ts': bar.get('timestamp', ''),
            'symbol': symbol,
            'tf': tf_seconds,
            'bar_close': bar.get('close', 0),
            'indicators': indicator_values,
            'triggers': trigger_booleans,
            'interpreters': interpreter_states,
            'positions': positions,
        }
        with self._batch_lock:
            self._batch.append(entry)

    def flush(self):
        """Flush buffered audit entries to DB."""
        with self._batch_lock:
            batch = self._batch[:]
            self._batch.clear()
        for entry in batch:
            try:
                append_audit_log_db(entry, self.user_id)
            except Exception as e:
                logger.debug("Audit write failed: %s", e)


# ============================================================
# DB-backed RalphEngine subclass
# ============================================================

class DBRalphEngine:
    """Wraps RalphEngine with DB-backed I/O for the worker service.

    Rather than subclassing (which would be fragile given the 1800-line
    base class), this creates a standard RalphEngine and monkey-patches
    its I/O methods to use the database.
    """

    def __init__(self, user_id: str, alpaca_keys: dict):
        self.user_id = user_id
        self.alpaca_keys = alpaca_keys
        self._engine = None
        self._config = None
        self._config_updated_at = ''

    def start(self, strategies: list, config: dict):
        """Start the engine (blocking)."""
        from ralph_engine import (
            RalphEngine, load_engine_state, save_engine_state,
            _load_enabled_general_packs,
        )

        self._config = config
        self._config_updated_at = config.pop('_updated_at', '')

        # Set Alpaca keys in environment for this thread
        os.environ['ALPACA_API_KEY'] = self.alpaca_keys.get('api_key', '')
        os.environ['ALPACA_SECRET_KEY'] = self.alpaca_keys.get('secret_key', '')
        os.environ['ALPACA_DATA_FEED'] = self.alpaca_keys.get('data_feed', 'sip')

        engine = RalphEngine()
        self._engine = engine

        # --- Patch I/O methods ---

        # 1. Position state: read/write from DB
        original_load = load_engine_state

        def db_load_engine_state(path=None):
            state = load_engine_state_db(self.user_id)
            positions = state.get('positions', {})
            from ralph_engine import PositionState
            return {int(k): PositionState.from_dict(v)
                    for k, v in positions.items()}

        def db_save_all_positions(self_engine=None):
            positions = {sid: m.position.state
                         for sid, m in engine.monitors.items()}
            state = {'positions': {str(k): v.to_dict()
                                   for k, v in positions.items()},
                     'saved_at': datetime.now(timezone.utc).isoformat()}
            save_engine_state_db(state, self.user_id)

        engine._save_all_positions = db_save_all_positions

        # Patch the module-level load_engine_state used by start()
        import ralph_engine as re_mod
        re_mod.load_engine_state = db_load_engine_state

        # 2. Status writes: write to DB
        def db_write_status(running: bool, connected: bool = False):
            tick_count = sum(h.tick_count for h in engine.hubs.values())
            status = {
                'running': running,
                'started_at': getattr(engine, '_start_time', ''),
                'connected': connected,
                'symbols': list(engine.hubs.keys()),
                'strategies': len(engine.strategies) if engine.strategies else 0,
                'tick_count': tick_count,
                'streaming_connected': connected,
            }
            try:
                save_monitor_status_admin(self.user_id, status)
            except Exception as e:
                logger.warning("Status write failed: %s", e)

        engine._write_status = db_write_status

        # 3. Alert dispatcher: use DB-backed version
        engine.dispatcher = DBAlertDispatcher(self.user_id)

        # 4. Fidelity auditor: use DB-backed version
        engine.auditor = DBAuditor(self.user_id)

        # 5. Pickle writes: no-op in production
        engine._write_chart_pickles = lambda: None

        # 6. Hot-reload: use DB instead of file flag + file reads
        original_hot_reload = engine._hot_reload_strategies

        def db_hot_reload():
            try:
                config = load_alert_config_admin(self.user_id)
                new_updated_at = config.pop('_updated_at', '')

                strategies = get_monitored_strategies_db(self.user_id)
                engine._config = config

                # Refresh general packs from DB
                packs_raw = load_general_packs_admin(self.user_id)
                if packs_raw:
                    from general_packs import _parse_pack_list
                    all_packs = _parse_pack_list(packs_raw)
                    engine._general_packs = [
                        p for p in all_packs if getattr(p, 'enabled', True)]
                else:
                    engine._general_packs = []

                new_ids = {s['id'] for s in strategies}
                current_ids = set(engine.monitors.keys())

                # Remove strategies no longer monitored
                removed = current_ids - new_ids
                for sid in removed:
                    monitor = engine.monitors.pop(sid, None)
                    if monitor:
                        sym = monitor.symbol
                        hub = engine.hubs.get(sym)
                        if hub and sid in hub.monitors:
                            del hub.monitors[sid]
                        logger.info("Removed strategy %d from monitoring", sid)

                # Add new strategies
                added = new_ids - current_ids
                for strat in strategies:
                    if strat['id'] not in added:
                        continue
                    sym = strat.get('symbol', 'SPY')
                    from ralph_engine import SymbolHub, StrategyMonitor
                    if sym not in engine.hubs:
                        engine.hubs[sym] = SymbolHub(sym)
                    hub = engine.hubs[sym]
                    monitor = StrategyMonitor(
                        strat, None, general_packs=engine._general_packs)
                    hub.add_monitor(monitor)
                    engine.monitors[strat['id']] = monitor
                    logger.info("Added strategy %d to monitoring", strat['id'])

                    # Warmup new strategy
                    try:
                        from data_loader import load_market_data
                        tf_str = strat.get('timeframe', '1Min')
                        df = load_market_data(
                            sym, days=7, timeframe=tf_str,
                            feed=self.alpaca_keys.get('data_feed', 'sip'),
                            session=monitor.session)
                        tf_seconds = monitor.tf_seconds
                        hub.seed_history(tf_seconds, df)
                        monitor.warmup(df)
                    except Exception as e:
                        logger.warning("Warmup failed for strategy %d: %s",
                                       strat['id'], e)

                self._config_updated_at = new_updated_at
                logger.info("Hot-reload complete: %d strategies "
                            "(%d added, %d removed)",
                            len(engine.monitors), len(added), len(removed))

            except Exception as e:
                logger.warning("DB hot-reload failed: %s", e)

        engine._hot_reload_strategies = db_hot_reload

        # 7. Reload flag check: override to check DB timestamp
        # The periodic_tasks_loop checks _ENGINE_RELOAD_FLAG.exists()
        # We patch it to always return False (DB hot-reload is time-based)
        import ralph_engine as re_mod2
        re_mod2._ENGINE_RELOAD_FLAG = type('FakePath', (), {
            'exists': lambda self: False,
            'unlink': lambda self: None,
        })()

        # 8. Flush auditor periodically
        original_periodic = getattr(engine, '_periodic_tasks_loop', None)

        # --- Start the engine ---
        logger.info("[%s] Starting engine: %d strategies",
                    self.user_id[:8], len(strategies))
        engine.start(strategies, config)

    def stop(self):
        """Signal the engine to stop."""
        if self._engine:
            self._engine._running = False

    @property
    def is_running(self):
        return self._engine and self._engine._running


# ============================================================
# Per-user engine instance
# ============================================================

class UserRalphInstance:
    """Manages a single user's RalphEngine in its own thread."""

    def __init__(self, user_id: str, alpaca_keys: dict):
        self.user_id = user_id
        self.alpaca_keys = alpaca_keys
        self._thread: Optional[threading.Thread] = None
        self._db_engine: Optional[DBRalphEngine] = None
        self._started_at: Optional[str] = None
        self._crash_count = 0

    def start(self):
        """Launch the engine in a background thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"ralph-{self.user_id[:8]}")
        self._thread.start()
        self._started_at = datetime.now(timezone.utc).isoformat()

    def _run(self):
        """Thread entry point."""
        try:
            config = load_alert_config_admin(self.user_id)
            strategies = get_monitored_strategies_db(self.user_id)

            if not strategies:
                logger.info("[%s] No strategies to monitor", self.user_id[:8])
                save_monitor_status_admin(self.user_id, {
                    'running': False,
                    'error': 'No strategies configured for monitoring',
                })
                return

            self._db_engine = DBRalphEngine(self.user_id, self.alpaca_keys)
            self._db_engine.start(strategies, config)

        except Exception as e:
            logger.error("[%s] Engine crashed: %s", self.user_id[:8], e,
                         exc_info=True)
            self._crash_count += 1
            try:
                save_monitor_status_admin(self.user_id, {
                    'running': False,
                    'error': str(e),
                    'crash_count': self._crash_count,
                })
            except Exception:
                pass

    def stop(self):
        """Stop the engine and wait for thread to finish."""
        if self._db_engine:
            self._db_engine.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                logger.warning("[%s] Engine thread did not stop cleanly",
                               self.user_id[:8])

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ============================================================
# Worker Manager (main orchestration loop)
# ============================================================

class WorkerManager:
    """Polls the database and manages per-user engine instances."""

    def __init__(self):
        self._running = True
        self._instances: Dict[str, UserRalphInstance] = {}
        self._config_timestamps: Dict[str, str] = {}
        self._last_config_check = 0.0

    def run(self):
        """Main loop — poll DB, start/stop engines, write heartbeats."""
        logger.info("Worker manager starting")

        # Handle SIGTERM gracefully
        def handle_signal(signum, frame):
            logger.info("Received signal %d — shutting down", signum)
            self._running = False
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        while self._running:
            try:
                self._poll_and_reconcile()
                self._check_config_changes()
                # Touch health check file for Docker HEALTHCHECK
                Path('/tmp/worker_alive').touch()
            except Exception as e:
                logger.error("Poll cycle error: %s", e, exc_info=True)

            # Sleep in small increments for responsive shutdown
            for _ in range(POLL_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

        self._shutdown_all()
        logger.info("Worker manager stopped")

    def _poll_and_reconcile(self):
        """Check desired_state for all users, start/stop instances."""
        try:
            rows = load_all_desired_states()
        except Exception as e:
            logger.error("Failed to poll desired states: %s", e)
            return

        desired_running = set()

        for row in rows:
            uid = row['user_id']
            desired = row.get('desired_state', 'stopped')

            if desired == 'running':
                desired_running.add(uid)

                if uid not in self._instances or not self._instances[uid].is_alive:
                    # Need to start (or restart) this user's engine
                    self._start_user_engine(uid)

            elif desired == 'stopped':
                if uid in self._instances and self._instances[uid].is_alive:
                    logger.info("[%s] Stopping engine (desired_state=stopped)",
                                uid[:8])
                    self._instances[uid].stop()
                    del self._instances[uid]

        # Stop any instances whose user no longer has a monitor_status row
        orphan_uids = set(self._instances.keys()) - {r['user_id'] for r in rows}
        for uid in orphan_uids:
            logger.info("[%s] Stopping orphaned engine", uid[:8])
            self._instances[uid].stop()
            del self._instances[uid]

    def _start_user_engine(self, user_id: str):
        """Start or restart an engine for a user."""
        # Alpaca keys are system-level (shared across all users)
        alpaca_keys = {
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'data_feed': os.getenv('ALPACA_DATA_FEED', 'sip'),
        }

        if not alpaca_keys['api_key'] or not alpaca_keys['secret_key']:
            logger.error("[%s] No Alpaca API keys in environment", user_id[:8])
            save_monitor_status_admin(user_id, {
                'running': False,
                'error': 'No Alpaca API keys configured on worker',
            })
            return

        # Stop existing instance if crashed
        if user_id in self._instances:
            old = self._instances[user_id]
            if old.is_alive:
                return  # Already running
            old.stop()

        instance = UserRalphInstance(user_id, alpaca_keys)
        self._instances[user_id] = instance
        instance.start()
        logger.info("[%s] Engine instance started", user_id[:8])

    def _check_config_changes(self):
        """Check for alert_config changes and trigger hot-reloads."""
        now = time.time()
        if now - self._last_config_check < CONFIG_CHECK_INTERVAL:
            return
        self._last_config_check = now

        for uid, instance in list(self._instances.items()):
            if not instance.is_alive:
                continue
            try:
                config = load_alert_config_admin(uid)
                updated_at = config.get('_updated_at', '')
                last_seen = self._config_timestamps.get(uid, '')
                if updated_at and updated_at != last_seen:
                    self._config_timestamps[uid] = updated_at
                    if last_seen:  # Don't trigger on first check
                        logger.info("[%s] Config changed — triggering hot-reload",
                                    uid[:8])
                        if (instance._db_engine and instance._db_engine._engine):
                            instance._db_engine._engine._hot_reload_strategies()
            except Exception as e:
                logger.debug("[%s] Config check failed: %s", uid[:8], e)

    def _shutdown_all(self):
        """Stop all running engine instances."""
        logger.info("Shutting down %d engine instances", len(self._instances))
        for uid, instance in self._instances.items():
            logger.info("[%s] Stopping engine", uid[:8])
            instance.stop()
        self._instances.clear()


# ============================================================
# Entry point
# ============================================================

def main():
    from db import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)

    logger.info("RoR Trader Worker starting")
    logger.info("Supabase URL: %s", SUPABASE_URL)

    manager = WorkerManager()
    manager.run()


if __name__ == "__main__":
    main()
