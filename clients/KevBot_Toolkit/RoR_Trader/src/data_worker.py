#!/usr/bin/env python3
"""RoR Trader Data-Worker Service — Phase 1: shared per-symbol bar layer.

Stands up the canonical 1-second bar layer of
docs/Spec_Data_Worker_Service.md. Phase 1 is infrastructure +
instrumentation only: it ingests each symbol's 1-second bars steadily
from Polygon REST into an in-memory SymbolBarStore, runs a periodic
REST-reconciliation sweep, and logs memory / throughput / ingest-latency
— the measurement gate before Phase 2's streaming engines.

No streaming engines, no trade writes, no DB writes in Phase 1.

Usage:  python src/data_worker.py
Requires: POLYGON_API_KEY  (SUPABASE_* are used from Phase 2 onward.)
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

from bar_store import SymbolBarStore  # noqa: E402
from data_worker_ingest import (  # noqa: E402
    IngestMetrics, run_ingest_cycle, run_recon_cycle, is_market_window,
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

    def run(self):
        logger.info("Data-worker starting — symbols=%s", SYMBOLS)

        def handle_signal(signum, frame):
            logger.info("Received signal %d — shutting down", signum)
            self._running = False
            self._ingest_stop.set()
            self._recon_stop.set()
            self._metrics_stop.set()
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        self._start_ingest_loop()
        self._start_recon_loop()
        self._start_metrics_loop()

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

    # Supabase isn't written in Phase 1 — warn (not fatal) if absent.
    try:
        from db import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
        if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
            logger.warning("Supabase creds not set — OK for Phase 1 (no DB "
                           "writes), but Phase 2+ will require them.")
    except Exception as e:
        logger.warning("Supabase config check skipped: %s", e)

    logger.info("RoR Trader Data-Worker starting (Phase 1 — bar layer)")
    DataWorkerManager().run()


if __name__ == "__main__":
    main()
