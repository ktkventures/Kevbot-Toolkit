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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set

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
    get_strategy_by_id_admin,
    update_strategy_admin,
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

    # Portfolio membership cache TTL (seconds). Portfolios don't change
    # per-bar, so caching avoids a Supabase round trip on every alert.
    # Trade_Timestamps_Spec: part of the dispatch-latency optimization
    # push — saves ~200-400ms per alert.
    _PORTFOLIO_CACHE_TTL_S = 60.0

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._deliver_alert_fn = None
        # Portfolio cache: (portfolios_list, loaded_at_epoch_s).
        self._portfolio_cache: Optional[tuple[list, float]] = None
        # Per-portfolio deployed capital tracker (2026-04-22 BP-cap work).
        # Shape: {portfolio_id: {strategy_id: dollars_deployed}}.
        # Lazy-seeded on first dispatch() call from existing open positions.
        # Protected by a lock because RalphEngine fires multiple strategies
        # concurrently on the same DBAlertDispatcher instance; unsynchronized
        # dict mutation across threads surfaces as "dict changed size during
        # iteration" under load.
        self._deployed_capital: Dict[int, Dict[int, float]] = {}
        # Parallel tracker keyed the same way. Holds the executed share
        # quantity at entry time so the paired exit alert can close at the
        # right size. Exit signals don't carry a stop price, which makes
        # per_share_risk == 0 and would yield rpt_qty = 0; that's wrong for
        # exits — we want to unwind the position, not resize it.
        self._position_quantity: Dict[int, Dict[int, int]] = {}
        self._deployed_lock = threading.Lock()
        self._deployed_seeded: bool = False
        try:
            from alert_monitor import deliver_alert
            self._deliver_alert_fn = deliver_alert
        except Exception:
            logger.warning("Could not import deliver_alert — webhooks disabled")

    def _get_portfolios_cached(self) -> list:
        """Load portfolios for this user with a short TTL cache."""
        now = time.time()
        if self._portfolio_cache is not None:
            portfolios, loaded_at = self._portfolio_cache
            if now - loaded_at < self._PORTFOLIO_CACHE_TTL_S:
                return portfolios
        portfolios = load_portfolios_admin(self.user_id)
        self._portfolio_cache = (portfolios, now)
        return portfolios

    def _seed_deployed_capital(self, portfolios: list) -> None:
        """Initialize self._deployed_capital from existing open positions.

        Called lazily on first dispatch. Walks each portfolio's open
        positions (derived from strategy.live_executions / fallback
        alerts via get_portfolio_alert_trades) and records the
        currently-held buying_power_used per (portfolio, strategy).
        Without this, a worker restart mid-session would treat every
        portfolio as fully liquid and over-size the next entry.
        """
        try:
            from portfolios import get_portfolio_alert_trades
        except Exception as e:
            logger.warning("Cannot seed deployed capital (import failed): %s", e)
            return

        # Strategy lookup for get_portfolio_alert_trades. Admin-scoped —
        # we're already running as the service role.
        def _strat_lookup(sid):
            try:
                return get_strategy_by_id_admin(sid, self.user_id)
            except Exception:
                return None

        total_seeded = 0
        for port in portfolios:
            pid = port.get('id')
            if pid is None:
                continue
            try:
                data = get_portfolio_alert_trades(port, _strat_lookup)
            except Exception as e:
                logger.warning(
                    "Deployed-capital seed failed for portfolio %s: %s", pid, e)
                continue
            with self._deployed_lock:
                bucket = self._deployed_capital.setdefault(pid, {})
                qty_bucket = self._position_quantity.setdefault(pid, {})
                for op in data.get('open_positions') or []:
                    sid = op.get('strategy_id')
                    bp_used = op.get('buying_power_used') or 0
                    qty = op.get('quantity') or 0
                    if sid is not None and bp_used > 0:
                        bucket[sid] = float(bp_used)
                        total_seeded += 1
                    if sid is not None and qty > 0:
                        qty_bucket[sid] = int(qty)
        if total_seeded:
            logger.info(
                "Deployed-capital tracker seeded: %d open position(s) "
                "across %d portfolio(s)",
                total_seeded, len(self._deployed_capital))

    def _available_bp_for_portfolio(self, portfolio: dict) -> float:
        """Return balance minus currently-deployed capital for this portfolio."""
        try:
            from portfolios import compute_account_balance
        except Exception:
            return 0.0
        balance = compute_account_balance(portfolio.get('account') or {})
        with self._deployed_lock:
            # Snapshot values inside the lock so concurrent mutation of the
            # inner dict can't blow up mid-sum.
            deployed = sum(
                list(self._deployed_capital.get(portfolio.get('id'), {}).values()))
        return max(balance - deployed, 0.0)

    def dispatch(self, signal_data: dict, strategy: dict,
                 config: dict) -> Optional[dict]:
        """Build, enrich, save, and deliver an alert via DB."""
        direction = strategy.get('direction', 'LONG')
        sig_type = signal_data['type']
        if sig_type == 'exit_signal':
            order_action = 'close'
        elif direction == 'LONG':
            order_action = 'buy'
        else:
            order_action = 'sell'

        # Trade_Timestamps_Spec: each alert row represents ONE event (entry
        # or exit), so trigger_ts / fill_ts are side-agnostic columns on the
        # alerts table. Select the side-appropriate pair from the signal.
        is_entry = sig_type == 'entry_signal'
        side = 'entry' if is_entry else 'exit'
        trigger_ts = signal_data.get(
            'entry_trigger_ts' if is_entry else 'exit_trigger_ts')
        fill_ts = signal_data.get(
            'entry_fill_ts' if is_entry else 'exit_fill_ts')

        alert = {
            'type': sig_type,
            'trigger': signal_data.get('trigger', ''),
            'price': signal_data.get('price', 0),
            # Trade_Timestamps_Spec (2026-04-20): actual_price = near-live
            # market price at alert-fire moment (SymbolHub.latest_close,
            # stamped in Ralph._fire_alert). The gap between actual_price
            # and price (theoretical fill from the engine) = price slippage.
            # Complements the time-slippage delta column.
            'actual_price': signal_data.get('actual_price'),
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
            # Top-level columns (post-migration) for the 4-field model.
            'event_type': 'fill',   # locked default; Behavior A cancels and
                                    # post-check validation_failed override
                                    # this once those paths land.
            'side': side,
            'trigger_ts': trigger_ts,
            'fill_ts': fill_ts,
            'exec_type': signal_data.get('exec_type', ''),
            'trigger_id': signal_data.get('trigger', ''),
            'hold_duration_s': signal_data.get('hold_duration_s', 0),
            'behavior': signal_data.get('behavior', 'B'),
            # Also keep both entry_* and exit_* timestamps in data JSONB for
            # full lifecycle reconstruction (useful for exit rows that want
            # to reference their paired entry_trigger_ts).
            'entry_trigger_ts': signal_data.get('entry_trigger_ts'),
            'entry_fill_ts': signal_data.get('entry_fill_ts'),
            'exit_trigger_ts': signal_data.get('exit_trigger_ts'),
            'exit_fill_ts': signal_data.get('exit_fill_ts'),
        }

        if sig_type == 'exit_signal':
            alert['entry_price'] = signal_data.get('entry_price')
            alert['entry_stop_price'] = signal_data.get('entry_stop_price')

        # Enrich with portfolio context using admin client.
        # Cached with 60s TTL — portfolios don't change per-bar, avoids
        # a Supabase round trip per alert.
        try:
            portfolios = self._get_portfolios_cached()
            # Lazy-seed deployed capital tracker on first dispatch (keeps
            # cold-start cheap — no cost if the engine never fires an alert).
            if not self._deployed_seeded:
                self._seed_deployed_capital(portfolios)
                self._deployed_seeded = True

            # Derive per-share risk from the signal's price + stop. Shared
            # across all portfolios since the strategy's stop is the same
            # regardless of subscriber.
            sig_price = signal_data.get('price') or 0
            sig_stop = signal_data.get('stop_price') or 0
            try:
                per_share_risk = abs(float(sig_price) - float(sig_stop))
            except (TypeError, ValueError):
                per_share_risk = 0.0
            try:
                price_f = float(sig_price) if sig_price else 0.0
            except (TypeError, ValueError):
                price_f = 0.0

            context = []
            for port in portfolios:
                for alloc in port.get('strategies', []):
                    if alloc.get('strategy_id') == strategy['id']:
                        risk_per_trade = alloc.get('risk_per_trade', 100.0)

                        rpt_qty = (int(risk_per_trade / per_share_risk)
                                   if per_share_risk > 0 else 0)
                        available_bp = self._available_bp_for_portfolio(port)
                        bp_qty = (int(available_bp / price_f)
                                  if price_f > 0 else 0)
                        executed_qty = min(rpt_qty, bp_qty)

                        # Exit signals must close the existing position — they
                        # don't get re-sized. Exit alerts usually lack a stop
                        # price, making per_share_risk = 0, which would leave
                        # rpt_qty/executed_qty at 0 and send `"quantity": 0`
                        # to the broker. Override with the tracked position
                        # size for this (portfolio, strategy) pair.
                        if sig_type == 'exit_signal':
                            with self._deployed_lock:
                                held_qty = self._position_quantity.get(
                                    port['id'], {}).get(strategy['id'], 0)
                            if held_qty > 0:
                                executed_qty = int(held_qty)
                                # Surface the held size in both RPT/BP slots
                                # too so the Details drawer shows coherent
                                # numbers instead of "RPT 0 / BP N / Exec N".
                                rpt_qty = int(held_qty)
                                bp_qty = int(held_qty)

                        # Compute balance once; expose for audit trail.
                        try:
                            from portfolios import compute_account_balance
                            account_balance = compute_account_balance(
                                port.get('account') or {})
                        except Exception:
                            account_balance = 0.0

                        context.append({
                            "portfolio_id": port['id'],
                            "portfolio_name": port.get(
                                'name', f"Portfolio {port['id']}"),
                            "position_risk": risk_per_trade,
                            # 2026-04-22 BP-cap work. Webhook payload uses
                            # executed_quantity; Trade History UI shows all
                            # three for visibility.
                            "rpt_quantity": rpt_qty,
                            "bp_quantity": bp_qty,
                            "executed_quantity": executed_qty,
                            "available_bp_at_fire": round(available_bp, 2),
                            "account_balance_at_fire": round(
                                account_balance, 2),
                        })
                        break
            alert['portfolio_context'] = context
        except Exception as e:
            logger.warning("Portfolio enrichment failed: %s", e)
            alert['portfolio_context'] = []

        # Save alert to DB
        try:
            alert = save_alert_admin(alert, self.user_id)
        except Exception as e:
            logger.error("ALERT SAVE FAILED [%s]: %s for %s (%s) — %s",
                         self.user_id[:8], sig_type, strategy.get('name'),
                         strategy.get('symbol'), e)
            return None

        logger.info("ALERT SAVED [%s]: %s for %s (%s) trigger=%s price=%.2f",
                     self.user_id[:8], sig_type, strategy.get('name'),
                     strategy.get('symbol'), signal_data.get('trigger'),
                     signal_data.get('price', 0))

        # Update deployed-capital tracker so the NEXT entry's BP calc
        # reflects this trade. Entry: add executed_qty * price per
        # subscribed portfolio. Exit: release whatever was tracked for
        # that (portfolio, strategy). Ordering matters — must run after
        # save_alert so the alert's portfolio_context exists, but before
        # the webhook dispatch so the deployed state reflects the event
        # that just happened (shouldn't matter for this alert but matters
        # for any concurrent alert firing against the same portfolio).
        try:
            with self._deployed_lock:
                if sig_type == 'entry_signal':
                    for pctx in alert.get('portfolio_context') or []:
                        pid = pctx.get('portfolio_id')
                        exec_qty = pctx.get('executed_quantity') or 0
                        if pid is not None and exec_qty > 0 and price_f > 0:
                            self._deployed_capital.setdefault(pid, {})[
                                strategy['id']] = exec_qty * price_f
                            # Remember the filled size so the paired exit
                            # alert can close the same position.
                            self._position_quantity.setdefault(pid, {})[
                                strategy['id']] = int(exec_qty)
                elif sig_type == 'exit_signal':
                    for pctx in alert.get('portfolio_context') or []:
                        pid = pctx.get('portfolio_id')
                        if pid is not None:
                            self._deployed_capital.get(pid, {}).pop(
                                strategy['id'], None)
                            self._position_quantity.get(pid, {}).pop(
                                strategy['id'], None)
        except Exception as e:
            # Don't fail the dispatch if tracker update has a bug —
            # worst case, next entry's BP qty is slightly wrong.
            logger.warning("Deployed-capital update failed: %s", e)

        # Atomic algo-history append. Exit signals from the unified engine
        # carry a fully-formed trade_record (PositionStateMachine.
        # get_trade_record). Appending here — in the same dispatch that
        # saves the alert — keeps algo history and the alerts table in
        # lockstep. Refresh button still re-runs the full backtest via
        # forward_test_service.recompute_and_persist_stored_trades.
        # Reverts b83629b's write-inversion (see
        # docs/Alert_Recovery_Plan_2026-04-17.md, Phase 1).
        if sig_type == 'exit_signal':
            try:
                self._persist_algo_trade(strategy, signal_data)
            except Exception as e:
                logger.error("ALGO TRADE PERSIST FAILED [%s]: %s — %s",
                             self.user_id[:8], strategy.get('name'), e)
                # Alert is already saved; don't fail the whole dispatch.

        # Webhook delivery — fires immediately on save for all exec types.
        # Trade_Timestamps_Spec (revised 2026-04-20): the webhook fire
        # moment is what "alert history" displays (= alerts.timestamp).
        # C-type already waits for bar-close confirmation at the engine
        # layer before dispatch is called; L-type dispatches at the cross
        # moment. In both cases, the webhook should go out immediately
        # when the engine emits — no scheduling delay. Firing early would
        # defeat the purpose of C-type; firing late would introduce
        # artificial lag.
        if self._deliver_alert_fn:
            # Set admin user context so deliver_alert's internal get_client()
            # calls (get_portfolio_by_id, get_webhook_group_by_id_db,
            # update_alert) route through the admin client and bypass RLS.
            # Without this, anon PATCHes to the alerts table silently write
            # zero rows and webhook_deliveries never persists — which is
            # exactly the state that was masking the Phase 39 schema drift
            # until 2026-04-23. Reset to previous context after the call.
            try:
                from db import set_admin_user_context, clear_current_user
                set_admin_user_context(self.user_id)
            except Exception:
                pass
            try:
                self._deliver_alert_fn(alert, config)
            except Exception as e:
                logger.error("Webhook delivery failed: %s", e)
            finally:
                try:
                    from db import clear_current_user
                    clear_current_user()
                except Exception:
                    pass

        return alert

    def _persist_algo_trade(self, strategy: dict, signal_data: dict) -> None:
        """Append the unified-engine-produced trade record to the strategy's
        stored_trades. Called only on exit signals (which carry a full
        trade record built by PositionStateMachine.get_trade_record).

        Safe to call repeatedly — duplicate detection uses the
        (entry_fill_ts, exit_fill_ts) tuple so replayed exit signals don't
        double-append.

        Trade_Timestamps_Spec Step 6 regression fix: trade_record no
        longer carries legacy entry_time / exit_time aliases. Read the
        canonical entry_fill_ts / exit_fill_ts fields instead.
        """
        trade_record = signal_data.get('trade_record')
        if not trade_record:
            return  # older/other signal shapes that don't carry a record
        entry_fill_ts = trade_record.get('entry_fill_ts')
        exit_fill_ts = trade_record.get('exit_fill_ts')
        if not entry_fill_ts or not exit_fill_ts:
            logger.debug("Skipping algo-trade persist: missing entry/exit fill_ts")
            return

        # Serialize sets (confluence_records) — JSONB doesn't accept sets
        tr_clean = dict(trade_record)
        cr = tr_clean.get('confluence_records')
        if isinstance(cr, set):
            tr_clean['confluence_records'] = list(cr)

        # Flag ON path: single-row INSERT into the trades table. Avoids the
        # full-column JSONB rewrite that was causing statement timeouts.
        try:
            import trades_store
        except Exception:
            trades_store = None
        if trades_store is not None and trades_store._flag_on():
            # Replace any stale open-position placeholder for this entry
            # (forward_test_service may have emitted one while the position
            # was still held; now that we have the real exit, it should
            # go away so the only row left is the closed trade).
            try:
                from db import delete_trade_placeholder_admin
                delete_trade_placeholder_admin(strategy['id'], entry_fill_ts)
            except Exception as e:
                logger.warning("Failed to clear trade placeholder: %s", e)

            inserted = trades_store.insert_trade(
                strategy['id'], self.user_id, tr_clean)
            if inserted:
                logger.info(
                    "ALGO TRADE APPENDED [%s]: strat=%s trades-table r=%.2f",
                    self.user_id[:8], strategy['id'],
                    tr_clean.get('r_multiple', 0) or 0)
            else:
                logger.debug(
                    "Algo trade insert skipped (likely duplicate or error): "
                    "strat=%s", strategy['id'])
            return

        # Flag OFF path (legacy): read strategy, append to stored_trades, write back.
        # Re-fetch the strategy fresh (hot-reload may have mutated it)
        strat_db = get_strategy_by_id_admin(strategy['id'], self.user_id)
        if strat_db is None:
            logger.warning("Strategy %s not found for algo-trade persist",
                           strategy['id'])
            return

        stored_trades = list(strat_db.get('stored_trades') or [])
        # Dedup on (entry_fill_ts, exit_fill_ts) — cheap, O(n) worst case
        for t in stored_trades:
            if (t.get('entry_fill_ts') == entry_fill_ts
                    and t.get('exit_fill_ts') == exit_fill_ts):
                logger.debug(
                    "Algo trade already persisted: (%s, %s)",
                    entry_fill_ts, exit_fill_ts)
                return

        # Remove any prior "open" placeholder with the same entry_fill_ts.
        # forward_test_service recomputes may have emitted an open-position
        # row while the position was still held; now that we have a real
        # exit, replace the placeholder rather than leaving it duplicated.
        stored_trades = [
            t for t in stored_trades
            if not (t.get('entry_fill_ts') == entry_fill_ts
                    and t.get('exit_fill_ts') in (None, 'NaT', ''))
        ]

        stored_trades.append(tr_clean)
        update_strategy_admin(
            strategy['id'], self.user_id,
            {'stored_trades': stored_trades})
        logger.info(
            "ALGO TRADE APPENDED [%s]: strat=%s total=%d r=%.2f",
            self.user_id[:8], strategy['id'],
            len(stored_trades), tr_clean.get('r_multiple', 0) or 0)


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

class _PerStrategyThrottle:
    """Drop bar-close recompute requests that arrive while one is already
    in flight for the same strategy. Prevents overlapping Polygon fetches
    on high-frequency strategies (10-sec bars closing faster than a
    15-30s recompute). Dropped events are captured by the next bar close.
    """

    def __init__(self):
        self._in_flight: Set[int] = set()
        self._lock = threading.Lock()

    def should_skip(self, strategy_id: int) -> bool:
        with self._lock:
            return strategy_id in self._in_flight

    def mark_start(self, strategy_id: int) -> None:
        with self._lock:
            self._in_flight.add(strategy_id)

    def mark_done(self, strategy_id: int) -> None:
        with self._lock:
            self._in_flight.discard(strategy_id)


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
        # Forward-test recompute: per-strategy throttle + bounded thread
        # pool. Recomputes are sync (Polygon fetch + unified engine), so
        # they run on a thread-pool executor off the async tick loop.
        # max_workers=2 lets two distinct strategies recompute concurrently
        # without overwhelming the host.
        self._ft_throttle = _PerStrategyThrottle()
        self._ft_executor: Optional[ThreadPoolExecutor] = None

    def start(self, strategies: list, config: dict):
        """Start the engine (blocking)."""
        from ralph_engine import (
            RalphEngine, load_engine_state, save_engine_state,
            _load_enabled_general_packs,
        )
        from live_bar_publisher import make_publisher_from_env

        self._config = config
        self._config_updated_at = config.pop('_updated_at', '')

        # Set API keys in environment for this thread
        os.environ['ALPACA_API_KEY'] = self.alpaca_keys.get('api_key', '')
        os.environ['ALPACA_SECRET_KEY'] = self.alpaca_keys.get('secret_key', '')
        os.environ['ALPACA_DATA_FEED'] = self.alpaca_keys.get('data_feed', 'sip')
        # Polygon.io key (system-level env var on Railway)
        if os.getenv('POLYGON_API_KEY'):
            os.environ['POLYGON_API_KEY'] = os.getenv('POLYGON_API_KEY')
        os.environ['DATA_PROVIDER'] = os.getenv('DATA_PROVIDER', 'polygon')

        # M8.5: stamp user_id onto every strategy so SymbolHub can route
        # live-bar broadcasts to the right user's Realtime channel.
        for strat in strategies:
            strat.setdefault('user_id', self.user_id)

        engine = RalphEngine()
        # M8.5: live-chart publisher for forming/completed bar broadcasts.
        engine._publisher = make_publisher_from_env()
        self._engine = engine

        # Thread pool: shared by (a) future /refresh-driven forward-test
        # recomputes and (b) alert-dispatch I/O (Phase 3 of Alert_Recovery_
        # Plan_2026-04-17.md). Dedicated pool so Supabase round-trips don't
        # stall the async WS consumer on the event loop.
        # max_workers=8: enough parallelism that burst-firing across many
        # strategies doesn't queue. Supabase connection cap (~60) leaves
        # plenty of headroom. For network-bound work the GIL releases
        # during IO so thread parallelism helps. (Bumped from 2 → 8
        # 2026-04-20 to eliminate alert-queue time as a latency variable.)
        if self._ft_executor is None:
            self._ft_executor = ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix=f'ft-{self.user_id[:8]}-',
            )
        # Wire the executor to the engine so _on_alert can schedule
        # DB writes + webhook delivery off the event loop.
        engine._alert_executor = self._ft_executor

        # Bar-close recompute hook is OFF. stored_trades is now appended
        # atomically by DBAlertDispatcher.dispatch on exit signals (see
        # docs/Alert_Recovery_Plan_2026-04-17.md, Phase 1). The
        # _install_bar_close_hook_on_engine / _on_bar_close / _ft_throttle
        # machinery is retained on this class for:
        #  - manual re-enablement via env flag in the future
        #  - user-initiated refresh via /api/strategies/{id}/refresh, which
        #    calls forward_test_service.recompute_and_persist_stored_trades
        #    directly (not through this hook)
        # self._install_bar_close_hook_on_engine(engine)

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
                    # M8.5: ensure hot-reloaded strategies carry user_id so
                    # SymbolHub can route live-bar broadcasts.
                    strat.setdefault('user_id', self.user_id)
                    sym = strat.get('symbol', 'SPY')
                    from ralph_engine import SymbolHub, StrategyMonitor
                    if sym not in engine.hubs:
                        engine.hubs[sym] = SymbolHub(
                            sym, publisher=engine._publisher)
                    hub = engine.hubs[sym]
                    monitor = StrategyMonitor(
                        strat, None, general_packs=engine._general_packs)
                    hub.add_monitor(monitor)
                    engine.monitors[strat['id']] = monitor
                    logger.info("Added strategy %d to monitoring", strat['id'])

                    # Finalize shadow engines (idempotent — creates only new ones)
                    hub.finalize_shadow_engines()

                    # Warmup new strategy + any new shadow engines
                    try:
                        from data_loader import load_market_data
                        from ralph_engine import SECONDS_TO_TIMEFRAME
                        tf_str = strat.get('timeframe', '1Min')
                        df = load_market_data(
                            sym, days=7, timeframe=tf_str,
                            feed=self.alpaca_keys.get('data_feed', 'sip'),
                            session=monitor.session)
                        tf_seconds = monitor.tf_seconds
                        hub.seed_history(tf_seconds, df)
                        monitor.warmup(df)

                        # Warmup shadow engines for secondary TFs
                        for sec_tf, shadow in hub._shadow_engines.items():
                            if not shadow.indicators._initialized:
                                sec_tf_str = SECONDS_TO_TIMEFRAME.get(sec_tf, '1Min')
                                try:
                                    sec_df = load_market_data(
                                        sym, days=7, timeframe=sec_tf_str,
                                        feed=self.alpaca_keys.get('data_feed', 'sip'),
                                        session=monitor.session)
                                    hub.seed_history(sec_tf, sec_df)
                                    shadow.warmup(sec_df)
                                except Exception as se:
                                    logger.warning("Shadow warmup failed for %s/%s: %s",
                                                   sym, sec_tf_str, se)
                    except Exception as e:
                        logger.warning("Warmup failed for strategy %d: %s",
                                       strat['id'], e)

                self._config_updated_at = new_updated_at
                logger.info("Hot-reload complete: %d strategies "
                            "(%d added, %d removed)",
                            len(engine.monitors), len(added), len(removed))

                # If new symbols were added, force a WebSocket reconnect
                # so the stream subscribes to the updated symbol list.
                if added:
                    new_symbols = {
                        s.get('symbol', 'SPY') for s in strategies
                        if s['id'] in added
                    } - set(engine._subscribed_symbols or [])
                    if new_symbols and (engine._stream_ref or
                                        getattr(engine, '_crypto_stream_ref', None)):
                        logger.info("New symbols detected (%s) — forcing "
                                    "stream reconnect",
                                    ', '.join(new_symbols))
                        for _sr in (engine._stream_ref,
                                    getattr(engine, '_crypto_stream_ref', None)):
                            if _sr:
                                try:
                                    _sr._should_run = False
                                except Exception:
                                    pass
                                try:
                                    loop = engine._event_loop
                                    if loop and loop.is_running():
                                        loop.call_soon_threadsafe(
                                            loop.create_task, _sr.close())
                                except Exception:
                                    pass

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
        """Signal the engine to stop and close the WebSocket."""
        if self._engine:
            self._engine.stop()  # Properly closes WebSocket, not just _running=False
        if self._ft_executor is not None:
            self._ft_executor.shutdown(wait=False, cancel_futures=True)
            self._ft_executor = None

    @property
    def is_running(self):
        return self._engine and self._engine._running

    # ------------------------------------------------------------------
    # Bar-close → forward-test recompute plumbing
    # ------------------------------------------------------------------

    def _install_bar_close_hook_on_engine(self, engine) -> None:
        """Wrap engine.hubs so every SymbolHub created during engine
        startup or hot-reload gets our bar-close handler installed.

        The hook's contract (SymbolHub.on_bar_close_hook):
            async def hook(strategy_id, user_id, tf_seconds) -> None
        """
        # Wrap the dict class so __setitem__ auto-installs on new hubs.
        original_hubs = engine.hubs
        outer_self = self

        class _HubsWithHook(dict):
            def __setitem__(self, sym, hub):
                hub.on_bar_close_hook = outer_self._on_bar_close
                super().__setitem__(sym, hub)

        new_hubs = _HubsWithHook(original_hubs)
        # Install hook on any pre-existing hubs too.
        for hub in new_hubs.values():
            hub.on_bar_close_hook = self._on_bar_close
        engine.hubs = new_hubs

    async def _on_bar_close(self, *, strategy_id: int,
                            user_id: Optional[str], tf_seconds: int) -> None:
        """Fire-and-forget forward-test recompute for the closed bar.

        Throttled: drops if a recompute is already in flight for the
        same strategy. Never blocks the tick loop — actual work runs
        on _ft_executor.
        """
        if not user_id:
            return
        if self._ft_throttle.should_skip(strategy_id):
            logger.debug(
                "[FT-SKIP] strategy=%s tf=%ss — recompute in flight",
                strategy_id, tf_seconds,
            )
            return
        self._ft_throttle.mark_start(strategy_id)
        loop = asyncio.get_event_loop()
        try:
            from api.services.forward_test_service import (
                recompute_and_persist_stored_trades,
            )
            await loop.run_in_executor(
                self._ft_executor,
                recompute_and_persist_stored_trades,
                strategy_id, user_id,
            )
        except Exception as e:
            logger.warning(
                "[FT-FAIL] strategy=%s tf=%ss: %s",
                strategy_id, tf_seconds, e,
            )
        finally:
            self._ft_throttle.mark_done(strategy_id)


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
                # Diagnostic: log why no strategies are monitored
                all_strats = load_strategies_admin(self.user_id)
                all_ports = load_portfolios_admin(self.user_id)
                logger.warning(
                    "[%s] No strategies to monitor. "
                    "Total strategies: %d, total portfolios: %d. "
                    "Check: portfolios need active webhooks in alert_config, "
                    "strategies need entry_trigger_confluence_id.",
                    self.user_id[:8], len(all_strats), len(all_ports))
                # Log which strategies were filtered and why
                for s in all_strats:
                    has_conf = 'entry_trigger_confluence_id' in s
                    logger.info(
                        "[%s]   Strategy '%s' (id=%s): has_confluence_id=%s",
                        self.user_id[:8], s.get('name', '?'),
                        s.get('id', '?'), has_conf)
                save_monitor_status_admin(self.user_id, {
                    'running': False,
                    'error': 'No strategies configured for monitoring',
                })
                return

            for s in strategies:
                logger.info("[%s] Monitoring: '%s' (id=%s, symbol=%s, tf=%s)",
                            self.user_id[:8], s.get('name', '?'),
                            s.get('id'), s.get('symbol'), s.get('timeframe'))

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
    # Emergency kill switch — set WORKER_DISABLED=true on the Railway service
    # to pause the worker without deleting its deployment. Exits cleanly so
    # Railway reports a graceful shutdown rather than a crash loop.
    if os.environ.get("WORKER_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        logger.warning("WORKER_DISABLED is set — exiting without starting engines.")
        sys.exit(0)

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
