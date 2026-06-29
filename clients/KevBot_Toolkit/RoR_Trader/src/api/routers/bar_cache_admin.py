"""Bar-cache admin API (M-RS2 Phase 1 control surface).

Admin-gated CRUD over `bar_cache_config` + backfill / maintain triggers +
per-target status (row counts + coverage span). Backfills can be long (1Sec is
~52k rows/day) so they run fire-and-forget in a daemon thread; the UI polls
GET /targets (row counts) + GET /backfill/status. Everything here only touches
the global `bar_cache` / `bar_cache_config` tables — no strategy/config writes.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bar-cache", tags=["bar-cache"])

# Capture resolutions selectable on the admin page. 1Sec + 1Min are the two
# native SOURCE layers; sub-minute is offered for volume-faithful primaries.
# Coarse TFs (1Hour/1Day) are MATERIALIZED from cached 1Min (split-safe) — their
# backfill/maintain delegate to `bar_cache.materialize_derived` (never native
# Polygon daily, which has split-adjustment bugs). They require the symbol's
# 1Min layer to be captured first.
_RESOLUTIONS = ["1Sec", "5Sec", "10Sec", "15Sec", "30Sec", "1Min", "1Hour", "1Day"]

# (symbol, timeframe) -> status string: "running" | "done:+N" | "error:..."
_backfills: dict[tuple, str] = {}


def _admin():
    from db import get_admin_client
    return get_admin_client()


def _row_stats(symbol: str, timeframe: str) -> dict:
    """Row count + coverage span for one (symbol, timeframe) layer."""
    c = _admin()
    try:
        cnt = c.table("bar_cache").select("ts", count="exact") \
            .eq("symbol", symbol).eq("timeframe", timeframe).limit(1) \
            .execute().count
        lo = c.table("bar_cache").select("ts").eq("symbol", symbol) \
            .eq("timeframe", timeframe).order("ts").limit(1).execute().data
        hi = c.table("bar_cache").select("ts").eq("symbol", symbol) \
            .eq("timeframe", timeframe).order("ts", desc=True).limit(1) \
            .execute().data
        return {"rows": cnt,
                "min_ts": lo[0]["ts"] if lo else None,
                "max_ts": hi[0]["ts"] if hi else None}
    except Exception as e:  # noqa: BLE001
        return {"rows": None, "min_ts": None, "max_ts": None,
                "error": str(e)[:120]}


@router.get("/targets")
def list_targets(user=Depends(get_current_user)):
    """Supply coverage + freshness for EVERY tracked symbol (catalog-driven), with
    the configured target depth + live backfill status merged into each row. The
    page is one module now — no separate manual-targets table."""
    import bar_cache
    cfg = {(t["symbol"], t["timeframe"]): t
           for t in bar_cache.get_capture_targets(enabled_only=False)}
    cov = bar_cache.supply_coverage(_tracked_symbols()) or []
    remote_bf = _remote_backfill_status()
    for row in cov:
        k = (row["symbol"], row["timeframe"])
        # Backfills run on the batch-worker now (compute_jobs); prefer that status,
        # fall back to the legacy in-process dict.
        row["backfill_status"] = remote_bf.get(k) or _backfills.get(k)
        row["target_days"] = (cfg.get(k) or {}).get("capture_days")
    return {
        "resolutions": _RESOLUTIONS,
        "read_enabled": _bar_cache_read_enabled(),
        "maintain_cron_enabled": _maintain_cron_enabled(),
        "writethrough_enabled": _writethrough_enabled(),
        "supply_coverage": cov,
        "read_health": _read_health(),
    }


def _tracked_symbols() -> list:
    """Every symbol any strategy uses, across ALL accounts (admin-level) — the
    set the bar cache should be maintaining as the source of truth. Until the
    formal supply registry IS the streaming source (Step 5), this 'demand' set is
    the accurate tracked set; a symbol here with no coverage is a gap to backfill.
    Falls back to whatever's in bar_cache_config if the strategies read fails."""
    try:
        rows = _admin().table("strategies").select("symbol").execute().data or []
        syms = sorted({(r.get("symbol") or "").strip().upper()
                       for r in rows if r.get("symbol")})
        if syms:
            return syms
    except Exception as e:  # noqa: BLE001
        logger.warning("bar-cache _tracked_symbols: strategies read failed: %s", e)
    import bar_cache
    return sorted({t["symbol"] for t in bar_cache.get_capture_targets(enabled_only=False)})


def _remote_backfill_status() -> dict:
    """Map (symbol, timeframe) -> latest backfill job status from compute_jobs
    (the batch-worker queue), over a recent window so the UI shows running +
    just-finished. Values: 'running' | the job's progress_label (e.g. 'done:+N')
    | 'error'."""
    out: dict = {}
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        rows = (_admin().table("compute_jobs")
                .select("payload,status,progress_label,created_at")
                .eq("job_type", "backfill").gte("created_at", since)
                .order("created_at").execute().data) or []
        for r in rows:
            p = r.get("payload") or {}
            sym, tf = p.get("symbol"), p.get("timeframe")
            if not sym or not tf:
                continue
            st = r.get("status")
            if st in ("queued", "running"):
                label = "running"
            elif st == "completed":
                label = r.get("progress_label") or "done"
            elif st == "failed":
                label = "error"
            else:
                label = st
            out[(sym, tf)] = label  # ordered by created_at → latest wins
    except Exception as e:  # noqa: BLE001
        logger.warning("bar-cache remote backfill status failed: %s", e)
    return out


def _writethrough_enabled() -> bool:
    """Whether the data-worker is writing its continuous REST stream through to
    bar_cache (RORT_BARCACHE_WRITETHROUGH). NOTE: this reads the flag on the API
    service's env for display parity; the actual writer is the Data Worker
    service — `live_freshness` (recent revised_at) is the real proof it's writing."""
    import os
    return os.environ.get("RORT_BARCACHE_WRITETHROUGH", "").strip().lower() in (
        "1", "true", "yes", "on")


def _read_health() -> dict:
    """Live proof the REST Bars direct-Postgres read path works on THIS service.

    Confirms the deploy env is wired: is BAR_CACHE_ENABLED on, is
    SUPABASE_CONNECTION_STRING present, and can we ACTUALLY read_bars() a row
    (i.e. the DSN connects + returns data). Surfaced on /admin/bar-cache so the
    api-service env is visually verifiable. NOTE: reflects the API service only;
    the worker (which runs Update-All) has its own env — confirm that from the
    worker logs ('[HIFI] Primed N day(s) ... from REST Bars') on the next run.
    """
    import bar_cache
    out = {
        "read_enabled": _bar_cache_read_enabled(),
        "direct_pg_available": bar_cache.direct_pg_available(),
        "sample_read_ok": None,
        "detail": None,
    }
    if not out["direct_pg_available"]:
        out["detail"] = "SUPABASE_CONNECTION_STRING not set on this service"
        return out
    try:
        targets = (bar_cache.get_capture_targets(enabled_only=True)
                   or bar_cache.get_capture_targets(enabled_only=False))
        if not targets:
            out["detail"] = "DSN present but no capture targets configured"
            return out
        t0 = targets[0]
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=4)
        df = bar_cache.read_bars(t0["symbol"], t0["timeframe"], start, end)
        n = 0 if df is None else len(df)
        out["sample_read_ok"] = n > 0
        out["detail"] = f"{t0['symbol']}/{t0['timeframe']}: {n} rows in last 4d"
    except Exception as e:  # noqa: BLE001
        out["sample_read_ok"] = False
        out["detail"] = str(e)[:160]
    return out


def _bar_cache_read_enabled() -> bool:
    import os
    return os.environ.get("BAR_CACHE_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def _maintain_cron_enabled() -> bool:
    import os
    return os.environ.get("BAR_CACHE_MAINTAIN_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


@router.post("/targets")
def upsert_target(payload: dict = Body(...), user=Depends(get_current_user)):
    """Create/update a capture target. Does NOT backfill — call /backfill."""
    import bar_cache
    sym = (payload.get("symbol") or "").strip().upper()
    tf = (payload.get("timeframe") or "").strip()
    if not sym or not tf:
        raise HTTPException(status_code=400,
                            detail="symbol and timeframe are required")
    if tf not in _RESOLUTIONS:
        raise HTTPException(status_code=400,
                            detail=f"timeframe must be one of {_RESOLUTIONS}")
    cap = payload.get("capture_days")
    ok = bar_cache.set_capture_target(
        sym, tf, enabled=bool(payload.get("enabled", True)),
        capture_days=int(cap) if cap not in (None, "") else None)
    if not ok:
        raise HTTPException(status_code=500, detail="upsert failed")
    return {"ok": True}


@router.delete("/targets/{symbol}/{timeframe}")
def delete_target(symbol: str, timeframe: str, user=Depends(get_current_user)):
    """Remove a capture target row (does NOT delete cached bars)."""
    _admin().table("bar_cache_config").delete() \
        .eq("symbol", symbol).eq("timeframe", timeframe).execute()
    return {"ok": True}


@router.post("/backfill")
def trigger_backfill(payload: dict = Body(...), user=Depends(get_current_user)):
    """Fire-and-forget backfill of [start, now] for one (symbol, timeframe).
    `start_date` (ISO) sets the floor — backfill everything from that date to now;
    falls back to `days` back if no start_date. Persists the chosen depth as the
    capture target (so the layer is tracked at that depth). Returns immediately;
    poll GET /targets (coverage + backfill_status)."""
    import bar_cache
    sym = (payload.get("symbol") or "").strip().upper()
    tf = (payload.get("timeframe") or "").strip()
    if not sym or not tf:
        raise HTTPException(status_code=400,
                            detail="symbol and timeframe are required")
    end = datetime.now(timezone.utc)
    sd = payload.get("start_date")
    if sd:
        try:
            start = datetime.fromisoformat(str(sd).replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400,
                                detail=f"bad start_date: {sd!r} (want ISO 8601)")
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=int(payload.get("days") or 365))
    if start >= end:
        raise HTTPException(status_code=400, detail="start_date must be in the past")
    key = (sym, tf)
    if _backfills.get(key) == "running":
        raise HTTPException(status_code=409,
                            detail="backfill already running for this target")
    # Persist the chosen depth as the tracked capture target (catalog-driven, but
    # the depth is the operator's lever) so re-backfills / future maintenance know
    # the intended history floor.
    try:
        bar_cache.set_capture_target(
            sym, tf, enabled=True, capture_days=max(1, (end - start).days))
    except Exception as e:  # noqa: BLE001
        logger.warning("bar-cache: persist target %s/%s failed (continuing): %s",
                       sym, tf, e)

    # Prefer the batch-worker (off the api service) when the remote queue is on;
    # 1yr of 1Sec is ~5.8M rows, too heavy to run in the API process.
    try:
        import recompute_jobs
        job_id = recompute_jobs.submit_backfill_job(
            sym, tf, start.isoformat(), end.isoformat())
        return {"started": True, "remote": True, "job_id": job_id}
    except Exception as e:  # noqa: BLE001
        logger.warning("bar-cache backfill remote enqueue failed (%s) — running "
                       "in-process fallback", e)

    def run():
        _backfills[key] = "running"
        try:
            n = bar_cache.backfill_symbol(sym, tf, start, end)
            _admin().table("bar_cache_config").update(
                {"last_backfill_at": datetime.now(timezone.utc).isoformat()}
            ).eq("symbol", sym).eq("timeframe", tf).execute()
            _backfills[key] = f"done:+{n}"
        except Exception as e:  # noqa: BLE001
            _backfills[key] = f"error:{str(e)[:120]}"
            logger.error("bar-cache backfill %s/%s failed: %s", sym, tf, e)

    threading.Thread(target=run, daemon=True,
                     name=f"bc-backfill-{sym}-{tf}").start()
    return {"started": True, "remote": False}


@router.get("/backfill/status")
def backfill_status(user=Depends(get_current_user)):
    return {f"{k[0]}/{k[1]}": v for k, v in _backfills.items()}


@router.post("/maintain")
def trigger_maintain(user=Depends(get_current_user)):
    """Manually run the keep-updated loop over all enabled targets (fire-and-
    forget; the worker cron does this automatically when enabled)."""
    import bar_cache

    def run():
        try:
            logger.info("bar-cache manual maintain: %s",
                        bar_cache.maintain_all_enabled())
        except Exception as e:  # noqa: BLE001
            logger.error("bar-cache manual maintain failed: %s", e)

    threading.Thread(target=run, daemon=True, name="bc-maintain-manual").start()
    return {"started": True}
