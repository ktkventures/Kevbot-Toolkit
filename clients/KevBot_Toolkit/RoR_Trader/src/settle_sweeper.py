"""Settle sweeper — deterministic bar_cache correction without full streaming.

The settle pipeline (provisional ingest -> recon -> settle_sweep) lives in the
data-worker's STREAMING loops, which stay MANUAL/off. With streaming off, the
only correction left is bar_cache's revision-horizon re-fetch **on read** — so
bars nobody reads within the horizon fossilize at their pre-revision values
(2026-06-29 pre-market cents; fidelity_parity_suite red on live-ingested days).

This module restores the deterministic guarantee as a small standalone loop:
every RORT_SETTLE_SWEEP_INTERVAL_S (default 600s), for each enabled capture
target symbol, re-fetch the trailing settle window of NATIVE 1Sec and 1Min bars
from Polygon REST and upsert via bar_cache.write_through_bars (last-write-wins,
stamps settled/revised_at), then settle_sweep. Bars therefore converge to
canonical REST within ~1h of close, every day, regardless of read patterns.

Also provides the one-time --retrue CLI to heal the fossilized window (bars
ingested while streaming was off, ~2026-06-05 onward): day-chunked 1Sec + 1Min
re-fetch/upsert, then re-materialize maintained coarse layers.

Gating (all default OFF; the loop is inert unless BOTH are set on the service):
  RORT_SETTLE_SWEEPER=1            enable the loop (data_worker starts the thread)
  RORT_BARCACHE_WRITETHROUGH=1     write_through_bars/settle_sweep are no-ops without it
Tunables: RORT_SETTLE_SWEEP_INTERVAL_S (600), RORT_SETTLE_SWEEP_LOOKBACK_MIN (60),
RORT_SETTLE_SWEEP_LAG_MIN (10).
"""
from __future__ import annotations

import logging
import os
import threading
import time as _time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("settle_sweeper")


def is_enabled() -> bool:
    return os.environ.get("RORT_SETTLE_SWEEPER", "").strip().lower() in (
        "1", "true", "yes", "on")


def _interval_s() -> int:
    try:
        return max(60, int(os.environ.get("RORT_SETTLE_SWEEP_INTERVAL_S", "600")))
    except ValueError:
        return 600


def _lookback_min() -> int:
    try:
        return max(15, int(os.environ.get("RORT_SETTLE_SWEEP_LOOKBACK_MIN", "60")))
    except ValueError:
        return 60


def _lag_min() -> int:
    try:
        return max(1, int(os.environ.get("RORT_SETTLE_SWEEP_LAG_MIN", "10")))
    except ValueError:
        return 10


def _target_symbols() -> list[str]:
    """Distinct symbols of enabled bar-cache capture targets (the per-ticker
    authority from the Bar Cache admin page)."""
    from bar_cache import get_capture_targets
    seen: list[str] = []
    for t in get_capture_targets(enabled_only=True):
        sym = t.get("symbol")
        if sym and sym not in seen:
            seen.append(sym)
    return seen


def sweep_symbol_once(symbol: str, *, lookback_min: int | None = None,
                      lag_min: int | None = None) -> dict:
    """One correction pass for a symbol: re-fetch [now-lookback, now-lag] native
    1Sec + 1Min from Polygon REST, upsert (last-write-wins picks up late-print
    revisions), then flip aged rows settled. Mirrors run_recon_cycle's shape but
    is self-contained (no streaming stores)."""
    import bar_cache
    from data_worker_ingest import _polygon_1s, _polygon_1m

    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=lookback_min or _lookback_min())
    end = now - timedelta(minutes=lag_min or _lag_min())
    out = {"symbol": symbol, "sec_rows": 0, "min_rows": 0, "settled": 0}
    if end <= start:
        return out

    df_s = _polygon_1s(symbol, start, end)
    out["sec_rows"] = bar_cache.write_through_bars(symbol, "1Sec", df_s)
    df_m = _polygon_1m(symbol, start, end)
    out["min_rows"] = bar_cache.write_through_bars(symbol, "1Min", df_m)
    out["settled"] = bar_cache.settle_sweep(symbol)
    return out


_STORE_MAINT_LAST = {"ts": 0.0}


def _maintain_resampled_store() -> None:
    """M-RS2 Phase 2: keep the resampled coarse-bar store's edge fresh by
    re-materializing each target's recent window right AFTER the 1Sec/1Min sweep
    (the store re-derives coarse buckets from the freshly-settled/corrected 1Min
    — same ordering rationale as the sweeper itself). Flag-gated
    RORT_RESAMPLED_STORE_WRITE (default OFF → this is a no-op), own throttle
    RORT_RESAMPLED_STORE_MAINTAIN_S (default 900s) so cadence is independent of
    the sweep interval, chunked + circuit-breakered inside maintain_all_coarse.
    Never raises — the sweeper loop must not be breakable from here.

    Runs AROUND THE CLOCK (unlike the sweep): T+1 revisions land OVERNIGHT —
    the nightly full_recompute re-trues the 1Min cache with Polygon's finalized
    numbers after 00:20Z, and a market-window-only maintain would carry that
    drift until the next session (observed 2026-07-11 05:09Z: a coordinated
    volume-drift wave across the fleet on the prior day's bars). Off-hours
    passes throttle to RORT_RESAMPLED_STORE_MAINTAIN_OFFHOURS_S (default 3600s)
    since only the nightly changes anything overnight."""
    import time as _time
    try:
        import resampled_bar_store as rbs
        if not rbs.writethrough_enabled():
            logger.info("[settle_sweeper] resampled-store maintain: skip "
                        "(RORT_RESAMPLED_STORE_WRITE off)")
            return
        from data_worker_ingest import is_market_window
        if is_market_window():
            every = max(300, int(os.environ.get(
                "RORT_RESAMPLED_STORE_MAINTAIN_S", "900")))
        else:
            every = max(900, int(os.environ.get(
                "RORT_RESAMPLED_STORE_MAINTAIN_OFFHOURS_S", "3600")))
        now = _time.time()
        remaining = every - (now - _STORE_MAINT_LAST["ts"])
        if remaining > 0:
            logger.info("[settle_sweeper] resampled-store maintain: throttled "
                        "(%ds left)", int(remaining))
            return
        _STORE_MAINT_LAST["ts"] = now
        t0 = _time.time()
        r = rbs.maintain_all_coarse()
        # ALWAYS log the outcome — a silent pass is indistinguishable from a
        # never-ran hook (the exact diagnosis gap this line closes, 2026-07-10).
        logger.info("[settle_sweeper] resampled-store maintain RAN in %.0fs: %s",
                    _time.time() - t0, r)
    except Exception as e:  # noqa: BLE001
        logger.warning("[settle_sweeper] resampled-store maintain failed: %s", e)


def run_sweeper_loop(stop_evt: threading.Event) -> None:
    """Daemon loop: sweep every enabled capture symbol each interval. Idles
    outside the weekday extended-hours window (no fresh bars to correct; the
    last pre-close sweep already covered the session tail)."""
    from data_worker_ingest import is_market_window

    logger.warning("[settle_sweeper] loop up — interval=%ss lookback=%sm lag=%sm",
                   _interval_s(), _lookback_min(), _lag_min())
    while not stop_evt.is_set():
        if is_market_window():
            try:
                symbols = _target_symbols()
            except Exception as e:  # noqa: BLE001
                symbols = []
                logger.warning("[settle_sweeper] target discovery failed: %s", e)
            for sym in symbols:
                if stop_evt.is_set():
                    break
                try:
                    r = sweep_symbol_once(sym)
                    if r["sec_rows"] or r["min_rows"] or r["settled"]:
                        logger.info(
                            "[settle_sweeper] %s: upserted 1Sec=%d 1Min=%d settled=%d",
                            sym, r["sec_rows"], r["min_rows"], r["settled"])
                except Exception as e:  # noqa: BLE001
                    logger.warning("[settle_sweeper] sweep %s failed: %s", sym, e)
        # M-RS2-P2: store maintain AFTER the sweep (coarse derives from the
        # just-corrected 1Min) — and ALSO outside the market window, on its own
        # slower off-hours throttle, so overnight T+1 revisions (the nightly
        # full_recompute re-truing 1Min after 00:20Z) propagate into the store
        # instead of drifting until the next session. No-op unless
        # RORT_RESAMPLED_STORE_WRITE=1.
        _maintain_resampled_store()
        stop_evt.wait(_interval_s())


def start_sweeper_thread(stop_evt: threading.Event) -> threading.Thread | None:
    """Start the sweeper daemon if enabled (call from data_worker.main)."""
    if not is_enabled():
        return None
    t = threading.Thread(target=run_sweeper_loop, args=(stop_evt,),
                         daemon=True, name="settle-sweeper")
    t.start()
    return t


# ---------------------------------------------------------------------------
# One-time re-true of a fossilized window (CLI)
# ---------------------------------------------------------------------------

def retrue_range(symbol: str, start_day: datetime, end_day: datetime,
                 *, rematerialize: bool = True) -> dict:
    """Re-fetch [start_day, end_day] native 1Sec + 1Min in one-day chunks and
    upsert, healing rows fossilized pre-revision. Then re-materialize this
    symbol's maintained coarse layers over the range so derived TFs agree with
    the trued sources. Idempotent; safe to re-run."""
    import bar_cache
    from data_worker_ingest import _polygon_1s, _polygon_1m

    total = {"symbol": symbol, "sec_rows": 0, "min_rows": 0, "days": 0}
    day = start_day
    while day <= end_day:
        nxt = day + timedelta(days=1)
        df_s = _polygon_1s(symbol, day, min(nxt, end_day + timedelta(days=1)))
        total["sec_rows"] += bar_cache.write_through_bars(symbol, "1Sec", df_s)
        df_m = _polygon_1m(symbol, day, min(nxt, end_day + timedelta(days=1)))
        total["min_rows"] += bar_cache.write_through_bars(symbol, "1Min", df_m)
        total["days"] += 1
        logger.info("[settle_sweeper] retrue %s %s: 1Sec=%d 1Min=%d",
                    symbol, day.date(), len(df_s), len(df_m))
        day = nxt
    bar_cache.settle_sweep(symbol)
    if rematerialize:
        for t in bar_cache.get_capture_targets(enabled_only=True):
            if t.get("symbol") != symbol:
                continue
            tf = t.get("timeframe")
            if tf in ("1Sec", "1Min") or not tf:
                continue
            try:
                bar_cache.materialize_derived(symbol, tf, start_day,
                                              end_day + timedelta(days=1))
                logger.info("[settle_sweeper] re-materialized %s/%s", symbol, tf)
            except Exception as e:  # noqa: BLE001
                logger.warning("[settle_sweeper] re-materialize %s/%s failed: %s",
                               symbol, tf, e)
    return total


def _main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retrue-from", required=True, help="UTC day, e.g. 2026-06-05")
    p.add_argument("--retrue-to", default=None, help="UTC day inclusive; default yesterday")
    p.add_argument("--symbols", default=None,
                   help="comma list; default = enabled capture targets")
    p.add_argument("--no-rematerialize", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("RORT_BARCACHE_WRITETHROUGH", "1")  # explicit CLI intent
    start = datetime.fromisoformat(args.retrue_from).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.retrue_to).replace(tzinfo=timezone.utc)
           if args.retrue_to else
           datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                              microsecond=0) - timedelta(days=1))
    symbols = ([s.strip() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else _target_symbols())
    for sym in symbols:
        r = retrue_range(sym, start, end, rematerialize=not args.no_rematerialize)
        print(f"{sym}: days={r['days']} 1Sec={r['sec_rows']} 1Min={r['min_rows']}")


if __name__ == "__main__":
    _main()
