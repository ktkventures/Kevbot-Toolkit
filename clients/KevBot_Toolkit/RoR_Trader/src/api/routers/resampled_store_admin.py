"""Resampled Store admin API (M-RS2 Phase 2, §8 observability).

Read-only coverage/freshness over `resampled_bar_cache` — the canonical
session-keyed coarse-bar store (2Min..1Day). Mirrors the Bar Cache admin
surface: per (symbol, tf, session) the UI shows coverage span, row count,
freshness (last revised_at) and settled %. ONE aggregate GROUP BY over
direct-PG (the table is ~760k rows; never per-target round trips), same DSN
the store itself uses (`resampled_bar_store._dsn`). No writes, no comparator
runs from this endpoint (the §7 comparator is a batch/shadow concern — too
heavy for a page load).
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/resampled-store",
                   tags=["admin", "resampled-store"])


def _tf_label_map() -> dict:
    """timeframe_seconds -> canonical TF string ('4Hour', '1Day', ...) over the
    store's coarse catalog. Falls back to a generic formatter for any seconds
    value outside COARSE_TFS (future TFs keep rendering instead of 500ing)."""
    import resampled_bar_store as rbs
    out = {}
    for tf in rbs.COARSE_TFS:
        try:
            out[rbs.tf_seconds(tf)] = tf
        except Exception:  # noqa: BLE001
            continue
    return out


def _fmt_tf_seconds(sec: int, labels: dict) -> str:
    if sec in labels:
        return labels[sec]
    if sec < 60:
        return f"{sec}Sec"
    if sec < 3600:
        return f"{sec // 60}Min"
    if sec < 86400:
        return f"{sec // 3600}Hour"
    return f"{sec // 86400}Day"


def _flags() -> dict:
    """The store's feature-flag states (env, this service) for the page pills —
    display parity with the Bar Cache page's read/write/maintain pills."""
    import resampled_bar_store as rbs
    return {
        "write_enabled": rbs.writethrough_enabled(),
        "read_enabled": rbs.read_enabled(),
        "compare_enabled": rbs.compare_enabled(),
        # COMPUTE-SKIP flag lives in strategy_data (serve settled coarse days
        # from the store); read the same env contract for display.
        "serve_enabled": os.getenv("RORT_RESAMPLED_STORE_SERVE", "0") == "1",
    }


@router.get("/coverage")
def coverage(user=Depends(get_current_user)):
    """Per (symbol, tf_label, session): first_ts, last_ts, row_count,
    last_revised_at, settled_pct — ONE aggregate query over the whole store."""
    import resampled_bar_store as rbs
    flags = _flags()
    dsn = rbs._dsn()
    if not dsn:
        return {**flags, "direct_pg_available": False, "coverage": [],
                "detail": "SUPABASE_CONNECTION_STRING not set on this service"}
    labels = _tf_label_map()
    sql = ("select symbol, timeframe_seconds, session, "
           "min(ts) as first_ts, max(ts) as last_ts, count(*) as row_count, "
           "max(revised_at) as last_revised_at, "
           "avg(case when settled then 1.0 else 0.0 end) as settled_pct "
           "from resampled_bar_cache "
           "group by symbol, timeframe_seconds, session "
           "order by symbol, timeframe_seconds, session")
    rows = []
    try:
        import psycopg
        with psycopg.connect(dsn, connect_timeout=15, prepare_threshold=None,
                             autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                for (sym, tfs, sess, first_ts, last_ts, n, rev,
                     settled_pct) in cur.fetchall():
                    rows.append({
                        "symbol": sym,
                        "tf_seconds": int(tfs),
                        "tf_label": _fmt_tf_seconds(int(tfs), labels),
                        "session": sess,
                        "first_ts": first_ts.isoformat() if first_ts else None,
                        "last_ts": last_ts.isoformat() if last_ts else None,
                        "row_count": int(n or 0),
                        "last_revised_at": rev.isoformat() if rev else None,
                        "settled_pct": (float(settled_pct)
                                        if settled_pct is not None else None),
                    })
    except Exception as e:  # noqa: BLE001
        logger.warning("resampled-store coverage query failed: %s", e)
        return {**flags, "direct_pg_available": True, "coverage": [],
                "detail": f"coverage query failed: {str(e)[:160]}"}
    return {**flags, "direct_pg_available": True, "coverage": rows,
            "detail": None}
