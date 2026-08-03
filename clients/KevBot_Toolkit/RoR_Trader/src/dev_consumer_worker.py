#!/usr/bin/env python3
"""Dev-environment `live_bars` consumer service — board #165 Phase B, step 31.

The Railway service that runs `dev_bars_consumer.DevBarsConsumer` in the **dev**
environment. Shaped like `shadow_worker.py`: SIGTERM/SIGINT graceful stop, a
`/tmp` health file on a separate daemon thread, a `*_DISABLED` kill-switch, and a
boot-line code fingerprint.

IT SHIPS INERT. `RORT_DEV_CONSUMER_ENABLED` defaults to OFF and
`RORT_DEV_CONSUMER_DRY_RUN` defaults to ON, so a deploy of this branch changes
nothing anywhere until two variables are set deliberately.

FOUR THINGS IT REFUSES TO DO
----------------------------
1. **Start in production.** `assert_dev_environment()` — `RORT_ENV_ROLE` unset
   means production, and this is a worker so #286's HTTP middleware never sees
   it. The assertion has to be made here or not at all.
2. **Open a Polygon connection.** It has no market-data client and never
   constructs one; its only input is `live_bars` rows already written by the
   production fleet. `DEV_CONSUMER_FORBIDDEN_IMPORTS` names what a later edit
   must not reintroduce, and the acceptance test parses this source to enforce
   it.
3. **Write anything but `dev_alerts`.** `DevAlertsWriter` is the only write path
   in the module tree.
4. **Start without an explicit strategy list.** `RORT_DEV_CONSUMER_SIDS` is
   required; an empty default would produce a zero-decision session that reads
   as perfect agreement between the fleets.

⛔ `dev_alerts` does not exist yet. `src/migrations/dev_alerts_table.sql` is
AUTHORED, NOT APPLIED — Kevin authorizes it by name, M or Kevin applies it, and
only then may this merge (design §7). Until then this service can run in dry-run
and will still refuse to insert, because there is nowhere to insert to.

Run:  .venv/bin/python src/dev_consumer_worker.py
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dev_bars_consumer as C  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("dev_consumer_worker")

HEALTH_FILE = "/tmp/dev_consumer_alive"
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
DISABLED_VAR = "DEV_CONSUMER_DISABLED"

#: A dev fleet must never touch the provider. Named here so the acceptance test
#: can assert this module tree does not import any of them — a later edit that
#: reintroduces one fails a test instead of opening a second WS connection
#: against an account that already sits near its subscription limit
#: (feedback_polygon_ws_limit).
DEV_CONSUMER_FORBIDDEN_IMPORTS = (
    "polygon", "websocket", "websockets", "alpaca_trade_api", "alpaca",
    "live_bars_writer", "ws_client",
)

_running = True


def _handle_signal(signum, _frame):
    global _running
    logger.info("Received signal %d — stopping", signum)
    _running = False


def _start_health_thread(stop_evt: threading.Event) -> threading.Thread:
    def loop():
        while not stop_evt.is_set():
            try:
                Path(HEALTH_FILE).touch()
            except Exception as e:  # noqa: BLE001
                logger.warning("health touch failed: %s", e)
            stop_evt.wait(5)
    t = threading.Thread(target=loop, name="dev-consumer-health", daemon=True)
    t.start()
    return t


def recover_cursors(client: Any, tag: str, keys) -> Dict[Tuple[str, int],
                                                         Optional[datetime]]:
    """Restart cursor: `max(bar_start)` per `(symbol, tf)` for THIS fleet tag.

    The cursor lives in `dev_alerts` rather than in memory or a cursor table
    (design §3.3), so a restarted consumer cannot re-decide a bar it has already
    recorded. Scoped by `fleet_tag` so two dev fleets never inherit each other's
    position.
    """
    out: Dict[Tuple[str, int], Optional[datetime]] = {}
    for symbol, tf in keys:
        rows = (client.table(C.DEV_ALERTS_TABLE)
                .select("bar_start")
                .eq("fleet_tag", tag)
                .eq("symbol", symbol)
                .eq("timeframe_seconds", int(tf))
                .order("bar_start", desc=True)
                .limit(1)
                .execute()).data or []
        out[(symbol, int(tf))] = (C._as_dt(rows[0]["bar_start"])  # noqa: SLF001
                                  if rows else None)
    return out


def build(now: Optional[datetime] = None):
    """Wire the real consumer. Every refusal happens before anything is polled."""
    # RAIL 1 — before the DB client, before the engine, before anything.
    C.assert_dev_environment()

    sids = C.configured_sids()
    tag = C.fleet_tag()

    from db import get_admin_client, set_admin_user_context
    admin_uid = (os.environ.get("RORT_DEV_CONSUMER_ADMIN_UID")
                 or os.environ.get("ROR_DEV_USER_ID")
                 or os.environ.get("DEV_USER_ID") or "").strip()
    if not admin_uid:
        raise C.DevConsumerConfigError(
            "no admin user id: set RORT_DEV_CONSUMER_ADMIN_UID. The consumer "
            "reads PRODUCTION strategy configs by id — a fleet-vs-fleet "
            "comparison requires the same strategy_ids, which is the reason the "
            "split shares one Supabase at all (design §4.4.3) — and writes "
            "nowhere but dev_alerts.")
    set_admin_user_context(admin_uid)
    client = get_admin_client()

    from dev_consumer_engine import EngineDecider, armed_flag_set
    decider = EngineDecider(sids, admin_uid)
    decider.load(now=now)
    strats = decider.strategies()

    writer = C.DevAlertsWriter(
        client, tag,
        fingerprint=C.code_fingerprint(),
        flag_set=armed_flag_set(),
        dry=C.dry_run())

    poller = C.LiveBarsPoller(
        client,
        symbols=[s["symbol"] for s in strats],
        timeframes=[s["timeframe_seconds"] for s in strats],
        grace=C.grace_s())

    consumer = C.DevBarsConsumer(
        bar_source=poller, decider=decider, writer=writer,
        grace=C.grace_s(), max_lag=C.max_lag_s(),
        max_skip_rows=C.max_skip_rows())

    recovered = recover_cursors(client, tag, sorted(consumer._by_key))  # noqa: SLF001
    consumer.seed_from(recovered, now=now)
    return consumer


def main() -> None:
    if (os.environ.get(DISABLED_VAR) or "").strip().lower() in (
            "1", "true", "yes", "on"):
        logger.warning("%s set — exiting without starting.", DISABLED_VAR)
        return

    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("USE_DB", "true")

    if not C.consumer_enabled():
        logger.warning(
            "[dev-consumer] RORT_DEV_CONSUMER_ENABLED is off — exiting. This "
            "service ships INERT; arm it deliberately, in the dev environment "
            "only, and only after dev_alerts exists (board #165 design §7).")
        return

    logger.warning("[dev-consumer] up id=%s fingerprint=%s env_role=%s "
                   "fleet_tag=%s poll=%ss grace=%ss max_lag=%ss dry_run=%s",
                   WORKER_ID, C.code_fingerprint(),
                   __import__("env_role").env_role(), C.fleet_tag(),
                   C.poll_interval_s(), C.grace_s(), C.max_lag_s(), C.dry_run())

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    stop_evt = threading.Event()
    _start_health_thread(stop_evt)

    consumer = build()
    poll_s = C.poll_interval_s()

    while _running:
        t0 = time.monotonic()
        try:
            delta = consumer.tick(datetime.now(timezone.utc))
            if delta:
                logger.info("[dev-consumer] tick %.2fs %s",
                            time.monotonic() - t0, delta)
        except Exception as e:  # noqa: BLE001
            # One bad tick must not take the session down — but it is never
            # silent, and a persistent failure shows as a run of these lines
            # rather than a fleet that quietly stopped deciding.
            logger.error("[dev-consumer] tick failed: %s", e, exc_info=True)
        for _ in range(poll_s):
            if not _running:
                break
            time.sleep(1)

    stop_evt.set()
    logger.warning("[dev-consumer] stopped — wrote %s", consumer.writer.counts)


if __name__ == "__main__":
    main()
