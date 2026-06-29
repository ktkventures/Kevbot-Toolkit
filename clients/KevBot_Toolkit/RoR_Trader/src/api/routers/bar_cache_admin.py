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
    """All configured capture targets + their live row stats + backfill status."""
    import bar_cache
    targets = bar_cache.get_capture_targets(enabled_only=False)
    for t in targets:
        t.update(_row_stats(t["symbol"], t["timeframe"]))
        t["backfill_status"] = _backfills.get((t["symbol"], t["timeframe"]))
    return {
        "targets": targets,
        "resolutions": _RESOLUTIONS,
        "read_enabled": _bar_cache_read_enabled(),
        "maintain_cron_enabled": _maintain_cron_enabled(),
        "writethrough_enabled": _writethrough_enabled(),
        "live_freshness": bar_cache.live_freshness(),
        "read_health": _read_health(),
    }


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
    """Fire-and-forget backfill of [now - days, now] for one target. Returns
    immediately; poll GET /targets (rows) + GET /backfill/status."""
    import bar_cache
    sym = (payload.get("symbol") or "").strip().upper()
    tf = (payload.get("timeframe") or "").strip()
    days = int(payload.get("days") or 365)
    if not sym or not tf:
        raise HTTPException(status_code=400,
                            detail="symbol and timeframe are required")
    key = (sym, tf)
    if _backfills.get(key) == "running":
        raise HTTPException(status_code=409,
                            detail="backfill already running for this target")

    def run():
        _backfills[key] = "running"
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
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
    return {"started": True}


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
