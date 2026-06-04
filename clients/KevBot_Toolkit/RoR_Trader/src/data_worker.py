#!/usr/bin/env python3
"""RoR Trader Data-Worker Service — Phases 1 + 2.

Phase 1: the shared per-symbol 1-second bar layer (SymbolBarStore) — a
steady Polygon REST ingest + periodic reconciliation sweep, in memory.

Phase 2: per-strategy streaming backtest-model engines. Each TSLA
strategy ticks at its bar-close cadence, reads bars from the shared
store, runs the unified engine incrementally via snapshot resume, and
writes new closed trades to the `trades` table under `backtest_<model>`.
This replaces the batch "Update New Data" recompute with a continuous
trickle that cannot overwhelm Supabase — a DB circuit breaker enforces
that. See docs/Spec_Data_Worker_Service.md.

Usage:  python src/data_worker.py
Requires: POLYGON_API_KEY and SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY.
"""
import os
import sys
import time
import signal
import logging
import threading
from pathlib import Path

# Match worker.py: force USE_DB, put src/ on path, load .env.
os.environ["USE_DB"] = "true"
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(_SCRIPT_DIR / '.env', override=True)

logger = logging.getLogger("data_worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

# Pack registry — no engines run in Phase 1, but loading it keeps the
# Phase 2 startup a no-diff and is harmless (filesystem scan only).
import pack_registry  # noqa: E402
_loaded_packs = pack_registry.scan_and_load_all()
logger.info("pack_registry loaded %d user pack(s)", len(_loaded_packs))

from datetime import datetime, timezone  # noqa: E402

from bar_store import SymbolBarStore  # noqa: E402
from coarse_bar_store import CoarseBarStore  # noqa: E402
from bar_store_facade import BarStoreFacade  # noqa: E402
from data_worker_ingest import (  # noqa: E402
    IngestMetrics, run_ingest_cycle, run_recon_cycle,
    is_market_window, prime_store,
    prime_coarse_store, run_coarse_ingest_cycle,
)
from data_worker_circuit import DBCircuitBreaker  # noqa: E402
from data_worker_engine import (  # noqa: E402
    StreamingMetrics, StrategyEngineState, classify_strategy,
    tick_strategy, run_startup_catchup, flush_snapshot,
    recompute_kpis_for_strategy,
)

# --- config (env-overridable; sane code defaults) ---
# DATA_WORKER_SYMBOLS is now an OPTIONAL pinned list — symbols here are
# always tracked regardless of strategy state. The default fleet is
# discovered dynamically from streaming-eligible strategies on each
# reload. Unset = pure dynamic.
PINNED_SYMBOLS = [s.strip().upper() for s in
                  os.getenv("DATA_WORKER_SYMBOLS", "").split(",") if s.strip()]
INGEST_INTERVAL = int(os.getenv("DATA_WORKER_INGEST_INTERVAL_SECONDS", "1"))
RECON_INTERVAL = int(os.getenv("DATA_WORKER_RECON_INTERVAL_SECONDS", "60"))
RECON_WINDOW_MIN = int(os.getenv("DATA_WORKER_RECON_WINDOW_MINUTES", "15"))
METRICS_INTERVAL = int(os.getenv("DATA_WORKER_METRICS_INTERVAL_SECONDS", "60"))
BAR_WINDOW_MIN = int(os.getenv("DATA_WORKER_BAR_WINDOW_MINUTES", "90"))
HEALTH_FILE = "/tmp/data_worker_alive"

# --- Phase 2 streaming config (env-overridable) ---
STREAM_INTERVAL = int(os.getenv("DATA_WORKER_STREAM_INTERVAL_SECONDS", "1"))
SNAPSHOT_FLUSH_INTERVAL = int(
    os.getenv("DATA_WORKER_SNAPSHOT_FLUSH_INTERVAL_SECONDS", "300"))
KPI_RECOMPUTE_INTERVAL = int(
    os.getenv("DATA_WORKER_KPI_RECOMPUTE_INTERVAL_SECONDS", "300"))
STRATEGY_RELOAD_INTERVAL = int(
    os.getenv("DATA_WORKER_STRATEGY_RELOAD_SECONDS", "300"))
CATCHUP_STAGGER_SECONDS = int(
    os.getenv("DATA_WORKER_CATCHUP_STAGGER_SECONDS", "5"))
# TTL on the cached DB-backed streaming-on/off flag. 30s by default — the
# admin UI toggle propagates within ~30s without a Railway redeploy.
STREAMING_FLAG_TTL = int(
    os.getenv("DATA_WORKER_STREAMING_FLAG_TTL_SECONDS", "30"))

# --- Phase 2.5 Tier-2 coarse-bar config ---
COARSE_WINDOW_DAYS = int(
    os.getenv("DATA_WORKER_COARSE_WINDOW_DAYS", "150"))
COARSE_INGEST_INTERVAL = int(
    os.getenv("DATA_WORKER_COARSE_INGEST_INTERVAL_SECONDS", "30"))

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # pragma: no cover — degrade gracefully if psutil absent
    _PROC = None
    logger.warning("psutil unavailable — process memory will not be logged")


class DataWorkerManager:
    """Owns the per-symbol bar stores and the ingest / recon / metrics loops."""

    def __init__(self):
        self._running = True
        self._metrics = IngestMetrics()
        # Stores keyed by symbol — dynamically (de)allocated by
        # _ensure_symbol_stores (called from the streaming loop on each
        # strategy reload). Pinned symbols (DATA_WORKER_SYMBOLS env var)
        # are always retained; the rest come and go based on what
        # streaming-eligible strategies are using.
        self._stores: dict = {}
        self._coarse_stores: dict = {}
        self._facades: dict = {}
        self._coarse_primed: dict = {}
        # Guards mutations to _stores / _coarse_stores / _facades /
        # _coarse_primed. Readers (ingest, recon, coarse-ingest, metrics)
        # snapshot via list(...) at iteration time so they don't need
        # to hold the lock.
        self._symbols_lock = threading.Lock()
        self._ingest_stop = threading.Event()
        self._recon_stop = threading.Event()
        self._metrics_stop = threading.Event()
        self._coarse_stop = threading.Event()

        # --- Phase 2 streaming ---
        self._streaming_metrics = StreamingMetrics()
        self._circuit = DBCircuitBreaker()
        self._engines = {}             # strategy_id -> StrategyEngineState
        self._engines_lock = threading.Lock()
        self._stream_stop = threading.Event()
        self._flush_stop = threading.Event()
        self._kpi_stop = threading.Event()
        self._last_catchup_at = 0.0
        self._last_reload_at = 0.0
        # Watchdog phase 2 — bookkeeping for auto-restart safety bounds.
        self._boot_mono = time.monotonic()
        self._stale_count = 0

        # Streaming-toggle cache — DB-backed system_settings flag, refreshed
        # every STREAMING_FLAG_TTL seconds. Each loop pass calls
        # _streaming_enabled_effective() and short-circuits when disabled.
        self._streaming_flag_cache: bool | None = None
        self._streaming_flag_checked_at: float = 0.0
        self._streaming_flag_last_logged: bool | None = None

    def _streaming_enabled_effective(self) -> bool:
        """Whether the streaming pipeline should run on this pass.

        Reads `system_settings.data_worker_streaming_enabled` (DB) with a
        ~30s in-memory cache so flipping the admin UI toggle propagates
        within a minute. Falls back to DATA_WORKER_STREAMING_ENABLED env
        var (default 'true') when the DB row is missing.
        """
        now = time.monotonic()
        if (self._streaming_flag_cache is not None
                and (now - self._streaming_flag_checked_at) < STREAMING_FLAG_TTL):
            return self._streaming_flag_cache

        env_default = os.environ.get(
            "DATA_WORKER_STREAMING_ENABLED", "true"
        ).strip().lower() in ("1", "true", "yes", "on")

        db_value = None
        try:
            from db import get_system_setting
            db_value = get_system_setting("data_worker_streaming_enabled")
        except Exception as e:
            logger.warning("[stream] system_settings read failed: %s", e)

        if isinstance(db_value, bool):
            effective = db_value
        elif isinstance(db_value, str):
            effective = db_value.strip().lower() in ("1", "true", "yes", "on")
        elif db_value is None:
            effective = env_default
        else:
            # Numeric or other truthy/falsy JSONB scalar
            effective = bool(db_value)

        # Log only on state transitions so we don't spam the log every 30s.
        if effective != self._streaming_flag_last_logged:
            logger.info(
                "[stream] effective streaming flag = %s (db=%r, env_default=%s)",
                "ENABLED" if effective else "DISABLED",
                db_value, env_default,
            )
            self._streaming_flag_last_logged = effective

        self._streaming_flag_cache = effective
        self._streaming_flag_checked_at = now
        return effective

    def run(self):
        logger.info("Data-worker starting — pinned_symbols=%s "
                    "(dynamic discovery for the rest)", PINNED_SYMBOLS)

        def handle_signal(signum, frame):
            logger.info("Received signal %d — shutting down", signum)
            self._running = False
            self._ingest_stop.set()
            self._recon_stop.set()
            self._metrics_stop.set()
            self._coarse_stop.set()
            self._stream_stop.set()
            self._flush_stop.set()
            self._kpi_stop.set()
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        self._start_ingest_loop()
        self._start_recon_loop()
        self._start_metrics_loop()
        self._start_coarse_ingest_loop()
        # Streaming backtest loops always start — but each pass checks
        # `_streaming_enabled_effective()` which reads from system_settings
        # (DB) with ~30s cache, falling back to DATA_WORKER_STREAMING_ENABLED
        # env var. Lets operators flip the toggle from the admin UI
        # (/admin/update-jobs) without a Railway redeploy.
        # Bar ingest / recon / metrics keep running regardless.
        self._start_streaming_loop()
        self._start_snapshot_flush_loop()
        self._start_kpi_recompute_loop()
        logger.info(
            "[data_worker] streaming initial state: %s (env=%s) — toggle "
            "via /admin/update-jobs streaming switch",
            "ENABLED" if self._streaming_enabled_effective() else "DISABLED",
            os.environ.get("DATA_WORKER_STREAMING_ENABLED", "true"),
        )

        # Main thread does only the healthcheck — all work is on daemons.
        while self._running:
            try:
                Path(HEALTH_FILE).touch()
            except Exception as e:
                logger.error("healthcheck touch failed: %s", e)
            for _ in range(5):
                if not self._running:
                    break
                time.sleep(1)

        # Best-effort: flush dirty snapshots once before exit so a deploy
        # loses at most the in-flight tick, not the whole flush interval.
        try:
            self._flush_all_snapshots()
        except Exception as e:
            logger.warning("shutdown snapshot flush failed: %s", e)
        logger.info("data-worker stopped")

    def _start_ingest_loop(self):
        """Provisional ingest — ~1s cadence during the market window;
        coarse backoff (up to 60s) when off-hours so Polygon isn't polled
        pointlessly."""
        def loop():
            logger.info("ingest loop started (interval=%ss)", INGEST_INTERVAL)
            backoff = INGEST_INTERVAL
            while not self._ingest_stop.is_set():
                if is_market_window():
                    for store in list(self._stores.values()):
                        try:
                            run_ingest_cycle(store, self._metrics)
                        except Exception as e:
                            logger.error("ingest cycle crashed: %s", e,
                                         exc_info=True)
                    backoff = INGEST_INTERVAL
                else:
                    backoff = min(max(backoff * 2, INGEST_INTERVAL), 60)
                self._ingest_stop.wait(backoff)
        self._ingest_thread = threading.Thread(
            target=loop, daemon=True, name="data-worker-ingest")
        self._ingest_thread.start()

    def _start_recon_loop(self):
        """REST reconciliation — re-pull the trailing window so late-print
        corrections overwrite provisional bars."""
        def loop():
            self._recon_stop.wait(30)  # stagger past the first ingest ticks
            logger.info("recon loop started (interval=%ss window=%smin)",
                        RECON_INTERVAL, RECON_WINDOW_MIN)
            while not self._recon_stop.is_set():
                if is_market_window():
                    for store in list(self._stores.values()):
                        try:
                            run_recon_cycle(store, self._metrics,
                                            RECON_WINDOW_MIN)
                        except Exception as e:
                            logger.error("recon cycle crashed: %s", e,
                                         exc_info=True)
                self._recon_stop.wait(RECON_INTERVAL)
        self._recon_thread = threading.Thread(
            target=loop, daemon=True, name="data-worker-recon")
        self._recon_thread.start()

    def _start_coarse_ingest_loop(self):
        """Phase 2.5 — refresh the Tier-2 1Min coarse store every
        COARSE_INGEST_INTERVAL during the market window. Cheap (1-3 new
        bars per cycle); the heavy work is the one-shot prime in
        _streaming_pass."""
        def loop():
            self._coarse_stop.wait(45)  # stagger past Tier-1 ingest+recon
            logger.info("[data-worker] coarse-ingest loop started "
                        "(interval=%ss)", COARSE_INGEST_INTERVAL)
            while not self._coarse_stop.is_set():
                if is_market_window():
                    for coarse in list(self._coarse_stores.values()):
                        try:
                            run_coarse_ingest_cycle(coarse, self._metrics)
                        except Exception as e:
                            logger.error("coarse ingest cycle crashed: %s",
                                         e, exc_info=True)
                self._coarse_stop.wait(COARSE_INGEST_INTERVAL)
        self._coarse_thread = threading.Thread(
            target=loop, daemon=True, name="data-worker-coarse-ingest")
        self._coarse_thread.start()

    def _start_metrics_loop(self):
        """The Phase 1 measurement gate — structured metrics every ~60s."""
        def loop():
            self._metrics_stop.wait(15)  # stagger
            logger.info("metrics loop started (interval=%ss)", METRICS_INTERVAL)
            while not self._metrics_stop.is_set():
                try:
                    self._log_metrics()
                except Exception as e:
                    logger.error("metrics log crashed: %s", e, exc_info=True)
                self._metrics_stop.wait(METRICS_INTERVAL)
        self._metrics_thread = threading.Thread(
            target=loop, daemon=True, name="data-worker-metrics")
        self._metrics_thread.start()

    # ─── Phase 2 — streaming engines ──────────────────────────────────

    def _ensure_symbol_stores(self, wanted_symbols: set) -> None:
        """Reconcile the per-symbol store fleet against `wanted_symbols`.

        - Adds (and synchronously primes) stores for symbols in
          `wanted_symbols` or PINNED_SYMBOLS that aren't yet allocated.
        - Removes stores for symbols not in either set.

        Synchronous priming hits Polygon REST and can take ~1-4s per
        symbol — acceptable because this runs on the streaming daemon
        (5-min reload cadence), not the request path.

        Pinned symbols (DATA_WORKER_SYMBOLS env var) are always retained
        regardless of strategy state — useful for warm-starting before
        the first reload, or pinning a symbol you want monitored even
        with no strategies on it yet.
        """
        target = set(wanted_symbols) | set(PINNED_SYMBOLS)
        target = {s for s in target if isinstance(s, str) and s.strip()}

        with self._symbols_lock:
            current = set(self._stores.keys())
        to_add = target - current
        to_remove = current - target

        # Adds — do priming OUTSIDE the lock (REST I/O).
        for sym in sorted(to_add):
            logger.info("[stream] provisioning symbol %s "
                        "(Tier-1 + Tier-2 prime)…", sym)
            tier1 = SymbolBarStore(sym, window_minutes=BAR_WINDOW_MIN)
            try:
                prime_store(tier1, BAR_WINDOW_MIN)
            except Exception as e:
                logger.warning("[stream] prime_store %s failed: %s", sym, e)
            coarse = CoarseBarStore(sym, window_days=COARSE_WINDOW_DAYS)
            coarse_primed = False
            try:
                prime_coarse_store(coarse, COARSE_WINDOW_DAYS)
                coarse_primed = True
            except Exception as e:
                logger.warning("[stream] prime_coarse_store %s failed: %s",
                               sym, e)
            facade = BarStoreFacade(tier1, coarse)
            with self._symbols_lock:
                self._stores[sym] = tier1
                self._coarse_stores[sym] = coarse
                self._facades[sym] = facade
                self._coarse_primed[sym] = coarse_primed
            logger.info("[stream] symbol %s provisioned (coarse_primed=%s)",
                        sym, coarse_primed)

        # Removes — under the lock; readers snapshot via list(...) so
        # they won't crash mid-iteration.
        if to_remove:
            with self._symbols_lock:
                for sym in to_remove:
                    logger.info("[stream] deprovisioning symbol %s — "
                                "no eligible strategies + not pinned", sym)
                    self._stores.pop(sym, None)
                    self._coarse_stores.pop(sym, None)
                    self._facades.pop(sym, None)
                    self._coarse_primed.pop(sym, None)

    def _load_streaming_strategies(self):
        """Discover + (re)classify the TSLA strategies to stream.

        Run at streaming-loop start and every STRATEGY_RELOAD_INTERVAL:
        adds new strategies, drops removed ones, and re-classifies the
        rest (a config edit changes the fingerprint → snapshot reset →
        re-catchup).
        """
        from db import load_all_desired_states, load_strategies_monitoring_admin
        try:
            desired = load_all_desired_states() or []
        except Exception as e:
            logger.error("[stream] load_all_desired_states failed: %s", e)
            return
        user_ids = sorted({d.get('user_id') for d in desired
                           if d.get('user_id')})
        seen = set()
        for uid in user_ids:
            try:
                strategies = load_strategies_monitoring_admin(uid) or []
            except Exception as e:
                logger.warning("[stream] load strategies user=%s failed: %s",
                                uid, e)
                continue
            for strat in strategies:
                # Dynamic-symbol mode: accept every streaming-eligible
                # strategy regardless of symbol — _ensure_symbol_stores
                # provisions the bar layer for whatever shows up.
                if not strat.get('symbol'):
                    continue
                if 'entry_trigger_confluence_id' not in strat:
                    continue
                if strat.get('strategy_origin') == 'webhook_inbound':
                    continue
                # 2026-05-27: per-strategy opt-out for the auto-snapshot
                # lane. Set via PATCH /strategies/{id}/snapshot-subscription
                # so the user can park known-broken strategies (no
                # baseline trades, persistent errors) without removing
                # them entirely. Defaults to True so existing strategies
                # behave as before.
                strat_cfg = strat.get('config') or {}
                if not isinstance(strat_cfg, dict):
                    strat_cfg = {}
                if strat_cfg.get('snapshot_subscribe_enabled', True) is False:
                    continue
                sid = strat.get('id')
                if sid is None:
                    continue
                seen.add(sid)
                with self._engines_lock:
                    existing = self._engines.get(sid)
                    if existing is None:
                        state = StrategyEngineState(sid, uid, strat)
                        classify_strategy(state)
                        self._engines[sid] = state
                        if not state.streaming_eligible:
                            logger.info("[stream] sid=%s ineligible: %s",
                                        sid, state.ineligible_reason)
                    else:
                        old_fp = existing.fingerprint
                        existing.strat = strat
                        classify_strategy(existing)
                        if existing.fingerprint != old_fp:
                            logger.info("[stream] sid=%s config changed — "
                                        "re-catchup queued", sid)
                            existing.snapshot_b64 = None
                            existing.catchup_done = False
        with self._engines_lock:
            for sid in list(self._engines.keys()):
                if sid not in seen:
                    del self._engines[sid]
            total = len(self._engines)
            elig = sum(1 for e in self._engines.values()
                       if e.streaming_eligible)
            wanted_symbols = {e.symbol for e in self._engines.values()
                              if e.streaming_eligible and e.symbol}
        logger.info("[stream] tracking %d strategies — %d streaming-eligible, "
                    "%d ineligible · symbols wanted=%s",
                    total, elig, total - elig, sorted(wanted_symbols))
        # Provision / deprovision per-symbol stores to match the active
        # strategy fleet (pinned symbols are always retained).
        self._ensure_symbol_stores(wanted_symbols)

    def _start_streaming_loop(self):
        """Phase 2 — per-strategy streaming engine ticks."""
        def loop():
            self._stream_stop.wait(20)  # stagger past the first ingest fills
            logger.info("[stream] streaming loop started (interval=%ss)",
                        STREAM_INTERVAL)
            self._load_streaming_strategies()
            self._last_reload_at = time.monotonic()
            while not self._stream_stop.is_set():
                try:
                    self._streaming_pass()
                except Exception as e:
                    logger.error("[stream] streaming pass crashed: %s", e,
                                 exc_info=True)
                self._stream_stop.wait(STREAM_INTERVAL)
        self._stream_thread = threading.Thread(
            target=loop, daemon=True, name="data-worker-streaming")
        self._stream_thread.start()

    def _streaming_pass(self):
        """One streaming-loop iteration — prime, reload, catch-up, tick."""
        if not self._streaming_enabled_effective():
            return
        if not is_market_window():
            return
        now_mono = time.monotonic()

        # Per-symbol store provisioning + Tier-1/Tier-2 priming is owned
        # by _ensure_symbol_stores, invoked from _load_streaming_strategies
        # on every reload (initial + every 5 min). Nothing to do here.

        # Periodic strategy reload.
        if now_mono - self._last_reload_at >= STRATEGY_RELOAD_INTERVAL:
            self._load_streaming_strategies()
            self._last_reload_at = now_mono

        with self._engines_lock:
            engines = list(self._engines.values())

        # Catch-up — at most one per pass, staggered, so a fleet of
        # stale strategies doesn't fire a burst of REST appends.
        for eng in engines:
            if not eng.streaming_eligible or eng.catchup_done:
                continue
            if now_mono - self._last_catchup_at < CATCHUP_STAGGER_SECONDS:
                break
            run_startup_catchup(eng, self._streaming_metrics)
            self._last_catchup_at = time.monotonic()
            break

        # Ticks — cadence-gated inside tick_strategy; most return fast.
        # Pass the FACADE so primary→Tier-1 / secondaries→Tier-2 routing
        # is invisible to the engine path.
        for eng in engines:
            if not eng.streaming_eligible or not eng.catchup_done:
                continue
            facade = self._facades.get(eng.symbol)
            if facade is None:
                continue
            try:
                tick_strategy(eng, facade, self._circuit,
                              self._streaming_metrics)
            except Exception as e:
                logger.error("[stream] sid=%s tick crashed: %s",
                              eng.strategy_id, e, exc_info=True)

    def _start_snapshot_flush_loop(self):
        """Periodically persist dirty in-memory snapshots to the DB."""
        def loop():
            self._flush_stop.wait(SNAPSHOT_FLUSH_INTERVAL)
            logger.info("[stream] snapshot-flush loop started (interval=%ss)",
                        SNAPSHOT_FLUSH_INTERVAL)
            while not self._flush_stop.is_set():
                try:
                    self._flush_all_snapshots()
                except Exception as e:
                    logger.error("[stream] snapshot flush crashed: %s", e,
                                 exc_info=True)
                self._flush_stop.wait(SNAPSHOT_FLUSH_INTERVAL)
        self._flush_thread = threading.Thread(
            target=loop, daemon=True, name="data-worker-snapshot-flush")
        self._flush_thread.start()

    def _flush_all_snapshots(self):
        if not self._streaming_enabled_effective():
            return
        with self._engines_lock:
            engines = list(self._engines.values())
        flushed = 0
        for eng in engines:
            if eng.snapshot_dirty:
                if flush_snapshot(eng, self._circuit, self._streaming_metrics):
                    flushed += 1
        if flushed:
            logger.info("[stream] flushed %d snapshot(s) to DB", flushed)

    def _start_kpi_recompute_loop(self):
        """Debounced KPI + equity-curve recompute for strategies that
        traded since the last pass. Kept off the per-tick path."""
        def loop():
            self._kpi_stop.wait(KPI_RECOMPUTE_INTERVAL)
            logger.info("[stream] kpi-recompute loop started (interval=%ss)",
                        KPI_RECOMPUTE_INTERVAL)
            while not self._kpi_stop.is_set():
                try:
                    self._kpi_recompute_pass()
                except Exception as e:
                    logger.error("[stream] kpi recompute crashed: %s", e,
                                 exc_info=True)
                self._kpi_stop.wait(KPI_RECOMPUTE_INTERVAL)
        self._kpi_thread = threading.Thread(
            target=loop, daemon=True, name="data-worker-kpi")
        self._kpi_thread.start()

    def _kpi_recompute_pass(self):
        if not self._streaming_enabled_effective():
            return
        with self._engines_lock:
            engines = [e for e in self._engines.values()
                       if e.has_traded_since_kpi]
        for eng in engines:
            if not self._circuit.allow():
                self._streaming_metrics.record_circuit_skip()
                break
            try:
                recompute_kpis_for_strategy(eng)
                self._circuit.record_success()
                self._streaming_metrics.record_kpi_recompute()
            except Exception as e:
                self._circuit.record_failure()
                logger.warning("[stream] sid=%s kpi recompute failed: %s",
                                eng.strategy_id, e)

    def _log_metrics(self):
        m = self._metrics.snapshot()
        rss_mb = None
        if _PROC is not None:
            try:
                rss_mb = round(_PROC.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                rss_mb = None
        for store in list(self._stores.values()):
            st = store.stats()
            bar_mb = round(st['mem_bytes'] / (1024 * 1024), 3)
            logger.info(
                "[metrics] %s bars=%d span=%.0fs bar_mem=%sMB | rss=%sMB | "
                "ingest_lat_last=%s p95=%s cycle_p95=%s cycles=%d | "
                "recon_cycles=%d revised_total=%d | appended_total=%d "
                "fetch_errors=%d",
                store.symbol, st['bar_count'], st['span_seconds'], bar_mb,
                rss_mb, m['last_ingest_latency_s'], m['ingest_latency_p95_s'],
                m['cycle_dur_p95_s'], m['ingest_cycles'], m['recon_cycles'],
                m['recon_revised_total'], m['provisional_appended_total'],
                m['fetch_errors'])

        # --- Phase 2.5 Tier-2 coarse metrics ---
        for sym, coarse in list(self._coarse_stores.items()):
            cs = coarse.stats()
            cmb = round(cs['mem_bytes'] / (1024 * 1024), 3)
            logger.info(
                "[metrics] coarse %s: bars=%d span=%sd mem=%sMB | "
                "primed=%s | cycles=%d appended=%d revised=%d errors=%d",
                sym, cs['bar_count'], cs.get('span_days'), cmb,
                self._coarse_primed.get(sym, False),
                m['coarse_cycles'], m['coarse_appended_total'],
                m['coarse_revised_total'], m['coarse_fetch_errors'])

        # --- Phase 2 streaming metrics ---
        sm = self._streaming_metrics.snapshot()
        cb = self._circuit.snapshot()
        with self._engines_lock:
            engines = list(self._engines.values())
        elig = [e for e in engines if e.streaming_eligible]
        now = datetime.now(timezone.utc)
        lags = [e.lag_seconds(now) for e in elig if e.last_bar_ts is not None]
        lag_max = round(max(lags)) if lags else None
        caught = sum(1 for e in elig if e.catchup_done and e.snapshot_b64)
        # Watchdog phase 1 (visibility) + phase 2 (auto-heal):
        # If `_last_reload_at` hasn't advanced past 2× STRATEGY_RELOAD_INTERVAL,
        # the streaming thread is likely wedged (e.g., supabase HTTP hang —
        # the 2026-06-02 ghost-roster bug). HTTP timeouts (db.py ClientOptions
        # postgrest_client_timeout=30) close the most common hang path; this
        # is the belt-and-suspenders: after 3 consecutive STALE observations
        # (~3 min) AND uptime > 15 min, force a process restart so Railway
        # respawns the container with a fresh state.
        #
        # 2026-06-04: market-window guard added. The streaming pass returns
        # early when `is_market_window() == False`, so `_last_reload_at`
        # never advances during market-closed hours. That FALSELY triggered
        # STALE → watchdog → os._exit(2) every 15min, killing Data Worker
        # in a restart loop overnight (boot at 23:42 ET, ext-hours starts
        # 4 AM ET → 4+ hours of false STALE before any real bar processing
        # could occur). Skip watchdog evaluation outside market hours so
        # the loop can idle cleanly through pre-market.
        from data_worker_ingest import is_market_window as _imw
        reload_age_s = (round(time.monotonic() - self._last_reload_at)
                        if self._last_reload_at > 0 else None)
        in_market = _imw()
        # Also guard on the streaming-enabled flag — when streaming is
        # toggled OFF via system_settings, the streaming pass returns
        # early before touching `_last_reload_at`, so STALE would
        # falsely trigger the same restart-loop bug that the market-
        # window guard fixed on 2026-06-04.
        streaming_on = self._streaming_enabled_effective()
        reload_stale = (in_market
                        and streaming_on
                        and reload_age_s is not None
                        and reload_age_s > 2 * STRATEGY_RELOAD_INTERVAL)
        if reload_stale:
            self._stale_count = getattr(self, '_stale_count', 0) + 1
        else:
            self._stale_count = 0
        uptime_s = round(time.monotonic() - getattr(self, '_boot_mono', 0))
        # Auto-restart triggers: 3 consecutive stale ticks AND we've been
        # up at least 15 min AND we're inside the market window (so the
        # streaming pass is actually expected to be advancing _last_reload_at).
        if self._stale_count >= 3 and uptime_s > 900 and in_market and streaming_on:
            logger.error(
                "[watchdog] reload stale for %d consecutive ticks "
                "(reload_age=%ss, uptime=%ss) — exiting for Railway respawn",
                self._stale_count, reload_age_s, uptime_s)
            import os as _os
            _os._exit(2)
        log_fn = logger.error if reload_stale else logger.info
        log_fn(
            "[metrics] streaming: engines=%d eligible=%d caught_up=%d | "
            "ticks=%d errored=%d trades_written=%d last_tick=%d | "
            "tick_p95=%s | lag_max=%ss | reload_age=%ss%s | "
            "flushes=%d(err %d) catchups=%d(no_baseline %d) kpi=%d | "
            "circuit=%s(opens %d) skips=%d",
            len(engines), len(elig), caught, sm['ticks_total'],
            sm['ticks_errored'], sm['trades_written_total'],
            sm['trades_written_last_tick'], sm['tick_latency_p95'], lag_max,
            reload_age_s, ' STALE' if reload_stale else '',
            sm['snapshot_flushes_total'], sm['snapshot_flush_errors'],
            sm['catchups_run'], sm['catchups_skipped_no_baseline'],
            sm['kpi_recomputes_total'], cb['state'], cb['open_count'],
            sm['circuit_skips'])


def main():
    # Emergency kill switch — set DATA_WORKER_DISABLED=true on the Railway
    # service to pause it without deleting the deployment (mirrors
    # WORKER_DISABLED on the live worker). Exit 0 = graceful, not a crash.
    if os.environ.get("DATA_WORKER_DISABLED", "").strip().lower() in (
            "1", "true", "yes"):
        logger.warning("DATA_WORKER_DISABLED is set — exiting without starting.")
        sys.exit(0)

    from data_loader import is_polygon_configured
    if not is_polygon_configured():
        logger.error("POLYGON_API_KEY must be set — the data-worker is "
                      "Polygon-driven. Exiting.")
        sys.exit(1)

    # Phase 2 writes trades + snapshots — Supabase creds are required.
    try:
        from db import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
        if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
            logger.error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be "
                         "set — Phase 2 streaming writes trades. Exiting.")
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        logger.error("Supabase config check failed: %s — exiting.", e)
        sys.exit(1)

    logger.info("RoR Trader Data-Worker starting (Phases 1+2 — bar layer + "
                "streaming backtest engines)")
    DataWorkerManager().run()


if __name__ == "__main__":
    main()
