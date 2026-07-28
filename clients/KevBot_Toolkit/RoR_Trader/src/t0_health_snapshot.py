"""Daily T+0 health snapshot — record what a NORMAL unsettled session looks like
(board #160).

THE TRAP THIS CLOSES (demonstrated 07-27, cost: most of a trading morning). Every
intraday fidelity check pits TODAY's T+0 unsettled data against fully-settled
prior days — their bars are final and last night's settle rebuilt their lanes,
so they read ~100% while today reads ~87% purely because today isn't settled
yet. That comparison is structurally guaranteed to look bad EVERY day, forever,
so mid-session we can never answer the only question that matters: *is today
abnormal?* We have never recorded what a normal T+0 looks like — and the evidence
is destroyed nightly (the settle pass delete->reinserts each day's trade rows).

THE FIX. Once per session, at a FIXED clock mark (~15:00Z, ~90 min after the open
— what Kevin actually looks at), snapshot per-strategy health and RETAIN it. We
capture the alert-history lane (live alerts vs backtest edges — get_strategy_health's
default) at BOTH 10s and 60s tolerance, plus the coverage/lane-asymmetry context
and alert-lane freshness at capture time, into `t0_health_snapshots`
(migrations/t0_health_snapshots.sql). Written ONLY at the mark, NEVER by the
nightly settle, so the retained value is the genuine unsettled state.

PAYOFF. Tomorrow, "is today weaker than usual?" is a T+0-vs-T+0 lookup answered in
seconds instead of a multi-hour investigation, and the divergence loop finally
gets a trustworthy intraday paired-% signal (memory divergence_loop_cadence).

Gating (default OFF — inert until armed on the data-worker):
  RORT_T0_HEALTH_SNAPSHOT=1        enable the daily capture thread
Tunables:
  RORT_T0_HEALTH_SNAPSHOT_AT_UTC   HH:MM UTC mark to capture at   (default 15:00)
  RORT_T0_HEALTH_SNAPSHOT_TOLS     comma tolerances in seconds    (default 10,60)
  RORT_T0_HEALTH_SNAPSHOT_WINDOW_H divergence window hours        (default 24)
  RORT_T0_HEALTH_SNAPSHOT_LATE_MIN skip if now is >N min past the mark
                                   (keeps the baseline apples-to-apples; default 120)
  RORT_T0_HEALTH_SNAPSHOT_POLL_S   loop poll interval             (default 60)

CLI (manual / one-off, ignores the clock gate):
  ../.venv/bin/python t0_health_snapshot.py --now            # capture for today
  ../.venv/bin/python t0_health_snapshot.py --now --dry-run  # compute, print, no write
  ../.venv/bin/python t0_health_snapshot.py --now --sids 267,341
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime, time as dtime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("t0_health_snapshot")

_TABLE = "t0_health_snapshots"
# The health function never reads `user` beyond its FastAPI dependency; an offline
# caller just needs to pass something so the default Depends(...) isn't invoked.
_SYS_USER = {"id": "t0-health-snapshot", "role": "admin"}
_DEFAULT_TOLS = (10.0, 60.0)


# ── flags / tunables ───────────────────────────────────────────────────────
def is_enabled() -> bool:
    return os.environ.get("RORT_T0_HEALTH_SNAPSHOT", "").strip().lower() in (
        "1", "true", "yes", "on")


def _target_time_utc() -> dtime:
    """Parse RORT_T0_HEALTH_SNAPSHOT_AT_UTC (HH:MM). Bad values → 15:00."""
    raw = os.environ.get("RORT_T0_HEALTH_SNAPSHOT_AT_UTC", "15:00").strip()
    try:
        hh, mm = raw.split(":")
        return dtime(hour=int(hh) % 24, minute=int(mm) % 60, tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return dtime(hour=15, minute=0, tzinfo=timezone.utc)


def _tolerances() -> List[float]:
    raw = os.environ.get("RORT_T0_HEALTH_SNAPSHOT_TOLS", "").strip()
    if not raw:
        return list(_DEFAULT_TOLS)
    out: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out or list(_DEFAULT_TOLS)


def _window_hours() -> int:
    try:
        return max(1, min(168, int(
            os.environ.get("RORT_T0_HEALTH_SNAPSHOT_WINDOW_H", "24"))))
    except ValueError:
        return 24


def _late_cutoff_min() -> int:
    try:
        return max(0, int(os.environ.get(
            "RORT_T0_HEALTH_SNAPSHOT_LATE_MIN", "120")))
    except ValueError:
        return 120


def _poll_s() -> int:
    try:
        return max(15, int(os.environ.get(
            "RORT_T0_HEALTH_SNAPSHOT_POLL_S", "60")))
    except ValueError:
        return 60


# ── scheduling predicate (pure — unit tested) ──────────────────────────────
def _should_capture_now(now: datetime, last_capture_date: Optional[date],
                        target: dtime, late_cutoff_min: int) -> bool:
    """True when `now` is a weekday, at/after the daily mark, within the late
    cutoff, and today hasn't been captured yet in-process.

    - Weekend guard: Sat/Sun have no session — skip. (Holidays fall through to
      an empty/quiet capture; the idempotent table absorbs it and the loop also
      consults the DB guard, so at worst one quiet row lands. Cheap and rare.)
    - `late_cutoff_min` keeps the baseline apples-to-apples: a process that only
      comes up hours after the mark must NOT record a stale-late capture and
      pollute the fixed-mark series. 0 = no cutoff (always allowed once past
      the mark).
    """
    if now.weekday() >= 5:
        return False
    if last_capture_date == now.date():
        return False
    target_dt = datetime.combine(now.date(), target)
    if now < target_dt:
        return False
    if late_cutoff_min > 0:
        late_by_min = (now - target_dt).total_seconds() / 60.0
        if late_by_min > late_cutoff_min:
            return False
    return True


# ── transform (pure — unit tested) ─────────────────────────────────────────
def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_snapshot_rows(
        health_by_tol: Dict[float, List[Dict[str, Any]]],
        *,
        snapshot_day: date,
        captured_at: datetime,
        target_utc: str,
        health_now: Optional[str],
        window_hours: int,
        algo_lane_sids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Merge per-tolerance get_strategy_health outputs into one upsert row per
    strategy. PURE: no DB, no clock — everything is passed in.

    `health_by_tol` maps tolerance → the `rows` list from get_strategy_health at
    that tolerance. 10.0 and 60.0 drive the two stored count sets; the 60.0 rows
    (dashboard default) carry the descriptors + coverage context + full
    health_row. `algo_lane_sids` is the set of strategy_ids that had cache_%
    (algo-lane) trades at capture; None = the presence query didn't run
    (algo_lane_present stays NULL rather than a false "absent").
    """
    def _index(tol: float) -> Dict[int, Dict[str, Any]]:
        return {r.get("strategy_id"): r
                for r in health_by_tol.get(tol, [])
                if r.get("strategy_id") is not None}

    idx10 = _index(10.0)
    idx60 = _index(60.0)
    # Descriptors/coverage come from the 60s pass; fall back to 10s if a
    # tolerance is absent so the transform is robust to a single-tol capture.
    base_idx = idx60 or idx10
    all_sids = sorted(set(idx10) | set(idx60))

    rows: List[Dict[str, Any]] = []
    for sid in all_sids:
        base = base_idx.get(sid) or idx10.get(sid) or {}
        r10 = idx10.get(sid) or {}
        r60 = idx60.get(sid) or {}
        row: Dict[str, Any] = {
            "snapshot_date": snapshot_day.isoformat(),
            "strategy_id": sid,
            "captured_at": captured_at.isoformat(),
            "target_utc": target_utc,
            "health_now": health_now,
            "window_hours": window_hours,
            # descriptors (denormalized)
            "user_id": base.get("user_id"),
            "symbol": base.get("symbol"),
            "timeframe": base.get("timeframe"),
            "direction": base.get("direction"),
            "data_source": base.get("data_source"),
            "backtest_model": base.get("backtest_model"),
            "forward_testing": bool(base.get("forward_testing"))
            if base.get("forward_testing") is not None else None,
            "trade_count_backtest": _as_int(base.get("trade_count_backtest")),
            # alert-history lane @ 10s
            "paired_10": _as_int(r10.get("paired_count_cov")),
            "phantom_10": _as_int(r10.get("phantom_count_cov")),
            "missed_10": _as_int(r10.get("missed_count_cov")),
            "tbd_10": _as_int(r10.get("tbd_count")),
            "combined_pct_10": _as_float(r10.get("combined_pct")),
            # alert-history lane @ 60s
            "paired_60": _as_int(r60.get("paired_count_cov")),
            "phantom_60": _as_int(r60.get("phantom_count_cov")),
            "missed_60": _as_int(r60.get("missed_count_cov")),
            "tbd_60": _as_int(r60.get("tbd_count")),
            "combined_pct_60": _as_float(r60.get("combined_pct")),
            # coverage / lane-asymmetry context
            "last_recompute_until_ts": base.get("last_recompute_until_ts"),
            "bt_current_through": base.get("bt_current_through"),
            "bt_heartbeat_at": base.get("bt_heartbeat_at"),
            "snapshot_age_sec": _as_int(base.get("snapshot_age_sec")),
            "last_alert_at": base.get("last_alert_at"),
            "last_alert_age_sec": _as_int(base.get("last_alert_age_sec")),
            "algo_lane_present": (None if algo_lane_sids is None
                                  else sid in algo_lane_sids),
            # full fidelity
            "health_row": r60 or r10 or None,
        }
        rows.append(row)
    return rows


# ── DB helpers ─────────────────────────────────────────────────────────────
def already_captured_today(snapshot_day: Optional[date] = None) -> bool:
    """True when a row for `snapshot_day` (default: today UTC) already exists —
    the cross-restart guard so a mid-afternoon restart doesn't overwrite the
    15:00 mark with a later capture. Fail-open to False (better to attempt an
    idempotent upsert than to silently skip)."""
    from db import get_admin_client
    day = snapshot_day or datetime.now(timezone.utc).date()
    try:
        resp = (get_admin_client().table(_TABLE)
                .select("id")
                .eq("snapshot_date", day.isoformat())
                .limit(1).execute())
        return bool(resp.data)
    except Exception as e:  # noqa: BLE001
        logger.warning("[t0-health] already-captured check failed: %s", e)
        return False


_ALGO_PAGE = 5000


def _fetch_algo_lane_sids(day: date) -> Optional[set]:
    """SIDs with cache_% (algo-lane) trades created on `day` — the presence
    signal for the exact asymmetry M hit 07-27 (comparing T+0 against the
    nightly-only algo lane). Light projection (strategy_id only), bounded to
    today. PAGINATED: an un-ranged PostgREST select silently caps at ~50k rows
    and would drop whichever sids land past the cap → a false 'algo lane absent'
    for those exact strategies (the bug class strategy_health._paginated_fetch
    documents). None on failure so we record NULL, never a false 'absent'."""
    from db import get_admin_client
    try:
        day_start = datetime.combine(day, dtime(0, 0, tzinfo=timezone.utc))
        c = get_admin_client()
        sids: set = set()
        offset = 0
        while True:
            chunk = (c.table("trades")
                     .select("strategy_id")
                     .like("data_source", "cache_%")
                     .gte("created_at", day_start.isoformat())
                     .order("strategy_id").order("id")
                     .range(offset, offset + _ALGO_PAGE - 1)
                     .execute().data) or []
            for r in chunk:
                sid = r.get("strategy_id")
                if sid is not None:
                    sids.add(sid)
            if len(chunk) < _ALGO_PAGE:
                break
            offset += _ALGO_PAGE
        return sids
    except Exception as e:  # noqa: BLE001
        logger.warning("[t0-health] algo-lane presence query failed: %s", e)
        return None


# ── capture orchestrator ───────────────────────────────────────────────────
def capture_snapshot(
        *,
        now: Optional[datetime] = None,
        tolerances: Optional[Sequence[float]] = None,
        only_sids: Optional[List[int]] = None,
        dry_run: bool = False,
) -> Dict[str, Any]:
    """Compute + persist today's T+0 snapshot. Returns a summary dict.

    Calls get_strategy_health once per tolerance (a light fleet computation),
    merges into one row per strategy, records the algo-lane presence signal, and
    upserts on (snapshot_date, strategy_id) — idempotent, so re-running the same
    day overwrites cleanly rather than duplicating.
    """
    from api.routers import strategy_health as sh

    now = now or datetime.now(timezone.utc)
    tols = list(tolerances) if tolerances is not None else _tolerances()
    win = _window_hours()
    snapshot_day = now.date()

    prev_only = sh._ONLY_SIDS
    if only_sids is not None:
        sh._ONLY_SIDS = list(only_sids)
    health_by_tol: Dict[float, List[Dict[str, Any]]] = {}
    health_now: Optional[str] = None
    try:
        for tol in tols:
            resp = sh.get_strategy_health(
                user=_SYS_USER, window_hours=win, tolerance_seconds=float(tol))
            health_by_tol[float(tol)] = resp.get("rows", []) or []
            # Use the last call's `now` as the capture reference clock.
            health_now = resp.get("now") or health_now
    finally:
        sh._ONLY_SIDS = prev_only

    algo_lane_sids = _fetch_algo_lane_sids(snapshot_day)

    rows = build_snapshot_rows(
        health_by_tol,
        snapshot_day=snapshot_day,
        captured_at=now,
        target_utc=_target_time_utc().strftime("%H:%M"),
        health_now=health_now,
        window_hours=win,
        algo_lane_sids=algo_lane_sids,
    )

    summary = {
        "snapshot_date": snapshot_day.isoformat(),
        "captured_at": now.isoformat(),
        "tolerances": tols,
        "strategies": len(rows),
        "algo_lane_sids": (None if algo_lane_sids is None
                           else len(algo_lane_sids)),
        "written": 0,
        "dry_run": dry_run,
    }

    if dry_run:
        logger.info("[t0-health] DRY-RUN: %d strategies for %s (no write)",
                    len(rows), snapshot_day.isoformat())
        summary["rows"] = rows
        return summary

    if rows:
        from db import get_admin_client
        get_admin_client().table(_TABLE).upsert(
            rows, on_conflict="snapshot_date,strategy_id").execute()
        summary["written"] = len(rows)
    logger.info("[t0-health] captured %d strategies for %s (tols=%s) — written=%d",
                len(rows), snapshot_day.isoformat(), tols, summary["written"])
    return summary


# ── daemon loop ────────────────────────────────────────────────────────────
def run_capture_loop(stop_evt: threading.Event) -> None:
    """Poll on the clock; capture once per weekday at/after the configured mark.

    Guarded three ways so a restart at any time of day is safe: the pure
    predicate (weekday + past-mark + within-late-cutoff), an in-process
    last-capture-date, and a DB existence check (don't overwrite an earlier
    mark). Fail-soft: a capture error is logged and retried next poll, never
    breaks the loop."""
    target = _target_time_utc()
    late = _late_cutoff_min()
    poll = _poll_s()
    logger.warning(
        "[t0-health] loop up — mark=%sZ tols=%s window=%sh late_cutoff=%dm poll=%ds",
        target.strftime("%H:%M"), _tolerances(), _window_hours(), late, poll)
    last_capture_date: Optional[date] = None
    while not stop_evt.is_set():
        try:
            now = datetime.now(timezone.utc)
            if _should_capture_now(now, last_capture_date, target, late):
                if already_captured_today(now.date()):
                    # Another process/an earlier run already marked today.
                    last_capture_date = now.date()
                    logger.info("[t0-health] %s already captured — skipping",
                                now.date().isoformat())
                else:
                    r = capture_snapshot(now=now)
                    last_capture_date = now.date()
                    logger.warning("[t0-health] capture done: %s", r)
        except Exception as e:  # noqa: BLE001
            logger.error("[t0-health] capture cycle crashed: %s", e,
                         exc_info=True)
        stop_evt.wait(poll)


def start_capture_thread(stop_evt: threading.Event) -> Optional[threading.Thread]:
    """Start the daily-capture daemon if enabled (call from data_worker.main)."""
    if not is_enabled():
        return None
    t = threading.Thread(target=run_capture_loop, args=(stop_evt,),
                         daemon=True, name="t0-health-snapshot")
    t.start()
    return t


# ── CLI ────────────────────────────────────────────────────────────────────
def _main() -> None:
    import argparse
    import json
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(override=True)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--now", action="store_true",
                   help="capture immediately, ignoring the clock gate")
    p.add_argument("--dry-run", action="store_true",
                   help="compute + print, do not write")
    p.add_argument("--sids", default=None,
                   help="comma list to scope the capture (default: whole fleet)")
    p.add_argument("--tols", default=None,
                   help="comma tolerances in seconds (default: 10,60)")
    args = p.parse_args()

    if not args.now:
        p.error("pass --now to capture (the scheduled loop runs on the "
                "data-worker via RORT_T0_HEALTH_SNAPSHOT)")

    only_sids = ([int(s) for s in args.sids.split(",") if s.strip()]
                 if args.sids else None)
    tols = ([float(t) for t in args.tols.split(",") if t.strip()]
            if args.tols else None)
    summary = capture_snapshot(only_sids=only_sids, tolerances=tols,
                               dry_run=args.dry_run)
    # Don't dump the (large) per-strategy rows to stdout by default.
    summary.pop("rows", None)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    _main()
