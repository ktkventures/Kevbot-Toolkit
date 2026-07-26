#!/usr/bin/env python3
"""Replay-SIM poller (board #120, V1.16) — runs on Kevin's machine only.

The LOCAL executor for the SIM basis (Railway can't reach this host, so the
"run SIM" button and the nightly sweep are DECLARATIVE — they write
replay_sim_requests rows; THIS poller claims and runs them). Same pattern as the
Run-button dispatcher (board #109), a separate process so the heavy replay never
touches a prod Railway service.

DRY-RUN BY DEFAULT: prints what it WOULD do, makes ZERO writes. --live executes.

  Phase 1 (on-demand button):
    python3 replay_sim_poller.py --live --loop --poll 30
    Claims 'requested' rows FIFO, runs one replay at a time (serialized — the
    replay is heavy), writes the cache row + terminal outcome. Deferred during
    RTH by default (prod-DB load rule) — pass --allow-rth to override.

  Phase 2 (nightly opted-in sweep) — flag-gated, default OFF:
    RORT_SIM_NIGHTLY=1 python3 replay_sim_poller.py --live --nightly
    ONLY when RORT_SIM_NIGHTLY is armed AND --nightly is passed. Enqueues a
    request for each OPTED-IN strategy (replay_sim_optin.enabled) that isn't
    cache-fresh, then drains the queue. Cost = (# opted-in) × ~2min — Kevin holds
    that dial via the per-strategy toggle (Kevin 07-26 scaling refinement).

Guardrails (Kevin-approved, E-specified):
  (a) off-hours only  — in_offhours_window(); post-close → pre-open, weekends.
  (b) serialized      — never runs while a compute_jobs recompute is queued/running.
  (c) 5-day span cap  — SIM_MAX_SPAN_TRADING_DAYS, passed to run_sim_job.
  (d) skip-fresh      — strategies whose cache is < SIM_SKIP_FRESH_DAYS old.

Kill switch: touch tools/team_sim/PAUSE.
"""
import argparse
import logging
import os
import socket
import sys
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
# Resolve RoR_Trader/src relative to this file so the poller works from ANY
# worktree (do not hardcode a single checkout path). Override with RORT_SRC.
SRC = os.environ.get("RORT_SRC") or os.path.normpath(
    os.path.join(HERE, "..", "..", "src"))
PAUSE_FILE = os.path.join(HERE, "PAUSE")

logger = logging.getLogger("replay_sim_poller")

DEFAULT_POLL_S = 30
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"

# Off-hours window: RTH is 09:30–16:00 ET; treat [09:00, 16:00) ET on a weekday
# as "market hours" and everything else (incl. weekends) as off-hours. The
# 30-min pre-open pad keeps a long replay from bleeding into the open.
_MKT_OPEN_HH, _MKT_OPEN_MM = 9, 0
_MKT_CLOSE_HH = 16


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ─────────────────────────── guardrails (light / testable) ───────────────────

def in_offhours_window(now_utc: datetime) -> bool:
    """True when it's safe to run the heavy replay (guardrail a). Post-close →
    pre-open on weekdays, and all weekend. DST-correct via ET conversion."""
    import pytz
    et = now_utc.astimezone(pytz.timezone("America/New_York"))
    if et.weekday() >= 5:                 # Sat/Sun
        return True
    minutes = et.hour * 60 + et.minute
    open_m = _MKT_OPEN_HH * 60 + _MKT_OPEN_MM
    close_m = _MKT_CLOSE_HH * 60
    return minutes < open_m or minutes >= close_m


def no_active_recompute(sb) -> bool:
    """True when no nightly/append recompute is queued or running (guardrail b —
    never contend with the recompute's prod-DB egress)."""
    try:
        rows = (sb.table("compute_jobs").select("id,status")
                .in_("status", ["queued", "running"]).limit(1)
                .execute().data or [])
        return not rows
    except Exception as e:  # noqa: BLE001
        # Fail CLOSED: if we can't tell, assume a recompute may be running and
        # hold off (a missed nightly sweep is cheaper than starving the fleet).
        logger.warning("recompute check failed (%s) — holding off", e)
        return False


def is_cache_fresh(sb, sid: int, skip_fresh_days: int,
                   now_utc: datetime) -> bool:
    """True when sid already has a cache row younger than skip_fresh_days
    (guardrail d — don't re-replay a strategy we just did)."""
    if skip_fresh_days <= 0:
        return False
    rows = (sb.table("replay_sim_basis").select("computed_at")
            .eq("strategy_id", sid).in_("status", ["ok", "partial"])
            .order("computed_at", desc=True).limit(1).execute().data or [])
    if not rows:
        return False
    from replay_sim_job import _parse_iso
    ca = _parse_iso(rows[0]["computed_at"])
    return ca is not None and (now_utc - ca) < timedelta(days=skip_fresh_days)


def opted_in_sids(sb) -> list:
    """Strategy ids with nightly SIM turned ON (Kevin's per-strategy toggle)."""
    rows = (sb.table("replay_sim_optin").select("strategy_id")
            .eq("enabled", True).execute().data or [])
    return sorted(r["strategy_id"] for r in rows)


# ─────────────────────────── DB bootstrap ────────────────────────────────────

def _admin():
    from db import get_admin_client
    return get_admin_client()


def _bootstrap_env():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(SRC, ".env"))
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("USE_DB", "true")


# ─────────────────────────── queue processing ────────────────────────────────

def _claim_one(sb, worker_id: str):
    """Optimistically claim the oldest 'requested' row (replica-safe: the update
    only lands if the row is still 'requested')."""
    rows = (sb.table("replay_sim_requests").select("*")
            .eq("outcome", "requested").order("requested_at").limit(1)
            .execute().data or [])
    if not rows:
        return None
    row = rows[0]
    upd = (sb.table("replay_sim_requests")
           .update({"outcome": "running", "claimed_by": worker_id,
                    "started_at": datetime.now(timezone.utc).isoformat()})
           .eq("id", row["id"]).eq("outcome", "requested").execute())
    return upd.data[0] if upd.data else None      # someone else grabbed it


def _run_request(sb, req, max_span_days, live):
    """Run one claimed request through the job, record the terminal outcome."""
    sid = req["strategy_id"]
    from replay_sim_job import run_sim_job
    try:
        out = run_sim_job(sid, sb=sb, requested_by=req.get("requested_by"),
                          max_span_trading_days=max_span_days, save=live)
    except Exception as e:  # noqa: BLE001
        logger.exception("sid=%s job crashed: %s", sid, e)
        if live:
            sb.table("replay_sim_requests").update({
                "outcome": "error",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "log_tail": f"{type(e).__name__}: {e}"[-2000:]}) \
              .eq("id", req["id"]).execute()
        return "error"
    outcome = "ignored" if out.get("skipped") else out["status"]
    if live:
        sb.table("replay_sim_requests").update({
            "outcome": outcome, "result_id": out.get("result_id"),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "log_tail": (out.get("reason")
                         or f"status={outcome} id={out.get('result_id')}")[-2000:]}) \
          .eq("id", req["id"]).execute()
    logger.info("sid=%s → %s (result_id=%s)", sid, outcome, out.get("result_id"))
    return outcome


def _enqueue_nightly(sb, sids, skip_fresh_days, now_utc, live):
    """Enqueue a nightly request for each opted-in, non-fresh sid without one
    already pending. Returns (enqueued, skipped_fresh, skipped_pending)."""
    enqueued = skipped_fresh = skipped_pending = 0
    for sid in sids:
        if is_cache_fresh(sb, sid, skip_fresh_days, now_utc):
            skipped_fresh += 1
            continue
        pending = (sb.table("replay_sim_requests").select("id")
                   .eq("strategy_id", sid)
                   .in_("outcome", ["requested", "running"]).limit(1)
                   .execute().data or [])
        if pending:
            skipped_pending += 1
            continue
        if live:
            sb.table("replay_sim_requests").insert({
                "strategy_id": sid, "source": "nightly",
                "outcome": "requested"}).execute()
        enqueued += 1
    return enqueued, skipped_fresh, skipped_pending


def one_pass(args):
    if os.path.exists(PAUSE_FILE):
        logger.info("PAUSED (remove %s to resume)", PAUSE_FILE)
        return
    sb = _admin()
    now = datetime.now(timezone.utc)
    offhours = in_offhours_window(now)
    can_run = (offhours or args.allow_rth) and no_active_recompute(sb)
    tag = "LIVE" if args.live else "DRY-RUN"

    # Phase 2 nightly enqueue — flag-gated + guardrailed.
    if args.nightly:
        if not _truthy(os.environ.get("RORT_SIM_NIGHTLY", "")):
            logger.info("[nightly] RORT_SIM_NIGHTLY not armed — sweep disabled")
        elif not offhours and not args.allow_rth:
            logger.info("[nightly] inside market hours — deferring sweep (guardrail a)")
        elif not no_active_recompute(sb):
            logger.info("[nightly] a recompute is active — deferring sweep (guardrail b)")
        else:
            sids = opted_in_sids(sb)
            enq, sf, sp = _enqueue_nightly(sb, sids, args.skip_fresh_days, now,
                                           args.live)
            logger.info("[nightly] opted-in=%d enqueued=%d skip_fresh=%d "
                        "skip_pending=%d (%s)", len(sids), enq, sf, sp, tag)

    # Drain the queue — ONE replay at a time (serialized; the replay is heavy).
    pend = (sb.table("replay_sim_requests").select("id")
            .eq("outcome", "requested").limit(1000).execute().data or [])
    if not pend:
        logger.info("queue empty (%s)", tag)
        return
    if not can_run:
        why = ("market hours (use --allow-rth)" if not (offhours or args.allow_rth)
               else "a recompute is active")
        logger.info("%d request(s) queued but HOLDING — %s (%s)",
                    len(pend), why, tag)
        return

    processed = 0
    while True:
        if os.path.exists(PAUSE_FILE):
            logger.info("PAUSED mid-drain — stopping")
            break
        # Re-check the serialize guard between every heavy run.
        if not (args.allow_rth or in_offhours_window(datetime.now(timezone.utc))):
            logger.info("market opened mid-drain — stopping (processed %d)", processed)
            break
        if not no_active_recompute(sb):
            logger.info("recompute started mid-drain — stopping (processed %d)", processed)
            break
        if not args.live:
            # Dry-run: report what WOULD be claimed without mutating the queue.
            logger.info("WOULD claim + run %d request(s)", len(pend))
            break
        req = _claim_one(sb, WORKER_ID)
        if req is None:
            break
        _run_request(sb, req, args.max_span_days, args.live)
        processed += 1
    if processed:
        logger.info("drained %d request(s)", processed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--nightly", action="store_true",
                    help="enqueue the opted-in nightly sweep this pass "
                         "(requires RORT_SIM_NIGHTLY armed)")
    ap.add_argument("--allow-rth", action="store_true",
                    help="run heavy replays even during market hours (default: "
                         "defer — prod-DB load rule)")
    ap.add_argument("--poll", type=int, default=DEFAULT_POLL_S)
    ap.add_argument("--skip-fresh-days", type=int,
                    default=int(os.environ.get("SIM_SKIP_FRESH_DAYS", "3")))
    ap.add_argument("--max-span-days", type=int,
                    default=int(os.environ.get("SIM_MAX_SPAN_TRADING_DAYS", "5")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _bootstrap_env()
    logger.info("replay_sim_poller up id=%s src=%s poll=%ss %s", WORKER_ID, SRC,
                args.poll, "LIVE" if args.live else "DRY-RUN")
    while True:
        try:
            one_pass(args)
        except Exception as e:  # noqa: BLE001
            logger.exception("pass failed: %s", e)
        if not args.loop:
            break
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
