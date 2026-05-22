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
from data_worker_ingest import (  # noqa: E402
    IngestMetrics, run_ingest_cycle, run_recon_cycle,
    is_market_window, prime_store,
)
from data_worker_circuit import DBCircuitBreaker  # noqa: E402
from data_worker_engine import (  # noqa: E402
    StreamingMetrics, StrategyEngineState, classify_strategy,
    tick_strategy, run_startup_catchup, flush_snapshot,
    recompute_kpis_for_strategy,
)

# --- config (env-overridable; sane code defaults) ---
SYMBOLS = [s.strip().upper() for s in
           os.getenv("DATA_WORKER_SYMBOLS", "TSLA").split(",") if s.strip()]
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
        self._stores = {s: SymbolBarStore(s, window_minutes=BAR_WINDOW_MIN)
                        for s in SYMBOLS}
        self._ingest_stop = threading.Event()
        self._recon_stop = threading.Event()
        self._metrics_stop = threading.Event()

        # --- Phase 2 streaming ---
        self._streaming_metrics = StreamingMetrics()
        self._circuit = DBCircuitBreaker()
        self._engines = {}             # strategy_id -> StrategyEngineState
        self._engines_lock = threading.Lock()
        self._stream_stop = threading.Event()
        self._flush_stop = threading.Event()
        self._kpi_stop = threading.Event()
        self._primed = False
        self._last_catchup_at = 0.0
        self._last_reload_at = 0.0

    def run(self):
        logger.info("Data-worker starting — symbols=%s", SYMBOLS)

        def handle_signal(signum, frame):
            logger.info("Received signal %d — shutting down", signum)
            self._running = False
            self._ingest_stop.set()
            self._recon_stop.set()
            self._metrics_stop.set()
            self._stream_stop.set()
            self._flush_stop.set()
            self._kpi_stop.set()
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        self._start_ingest_loop()
        self._start_recon_loop()
        self._start_metrics_loop()
        self._start_streaming_loop()
        self._start_snapshot_flush_loop()
        self._start_kpi_recompute_loop()

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
                    for store in self._stores.values():
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
                    for store in self._stores.values():
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
                if strat.get('symbol') not in SYMBOLS:
                    continue
                if 'entry_trigger_confluence_id' not in strat:
                    continue
                if strat.get('strategy_origin') == 'webhook_inbound':
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
        logger.info("[stream] tracking %d strategies — %d streaming-eligible, "
                    "%d ineligible", total, elig, total - elig)

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
        if not is_market_window():
            return
        now_mono = time.monotonic()

        # Prime the stores once, on first entry into the market window,
        # so the resume windows are covered immediately.
        if not self._primed:
            for store in self._stores.values():
                try:
                    prime_store(store, BAR_WINDOW_MIN)
                except Exception as e:
                    logger.warning("[stream] prime_store failed: %s", e)
            self._primed = True

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
        for eng in engines:
            if not eng.streaming_eligible or not eng.catchup_done:
                continue
            store = self._stores.get(eng.symbol)
            if store is None:
                continue
            try:
                tick_strategy(eng, store, self._circuit,
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
        for store in self._stores.values():
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
        logger.info(
            "[metrics] streaming: engines=%d eligible=%d caught_up=%d | "
            "ticks=%d errored=%d trades_written=%d last_tick=%d | "
            "tick_p95=%s | lag_max=%ss | flushes=%d(err %d) catchups=%d"
            "(no_baseline %d) kpi=%d | circuit=%s(opens %d) skips=%d",
            len(engines), len(elig), caught, sm['ticks_total'],
            sm['ticks_errored'], sm['trades_written_total'],
            sm['trades_written_last_tick'], sm['tick_latency_p95'], lag_max,
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
