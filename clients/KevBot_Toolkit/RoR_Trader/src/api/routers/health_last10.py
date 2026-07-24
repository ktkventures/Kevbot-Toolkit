"""Last-10 health score — board #70 Phase A (Kevin scoping 07-24).

READ-ONLY companion to the Strategy Health page: for each strategy's last 10
COMPLETED backtest-lane trades, score 1 point per entry and per exit that
pairs to a live ALERT within ±tolerance (default 10s) → n/20 (denominator
shrinks to 2k when fewer than 10 completed trades exist).

Pairing semantics are a VERBATIM port of `_pair_phantom_missed`'s greedy
two-pointer walk in routers/strategy_health.py:349 (which is a closure and
not importable) — sorted edges vs sorted alerts, 1:1, |alert-edge| ≤ tol.
The only addition is per-edge bookkeeping (WHICH edges paired) so the modal
can show one row per trade. E reviews these semantics in the PR per the
task spec; this file never writes and touches no engine paths.
"""
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_current_user
from api.routers.strategy_health import _parse_iso  # shared, module-level

router = APIRouter(prefix="/api/strategy-health-last10", tags=["strategy-health"])

_TOL_DEFAULT = 10.0
_ALERT_PAD_SEC = 3600  # bracket the alert query 1h around the edges — light read


def _admin():
    from db import get_admin_client
    return get_admin_client()


def _greedy_pair(edges: List[float], alerts: List[float],
                 tolerance: float) -> set:
    """strategy_health._pair_phantom_missed's walk, returning the set of
    paired edge timestamps instead of counts. Same order, same tie rules —
    counts derived from this set equal the original's paired_edges."""
    edges_sorted = sorted(edges)
    alerts_sorted = sorted(alerts)
    paired: set = set()
    i = j = 0
    while i < len(edges_sorted) and j < len(alerts_sorted):
        diff = alerts_sorted[j] - edges_sorted[i]
        if abs(diff) <= tolerance:
            paired.add(edges_sorted[i])
            i += 1
            j += 1
        elif diff < 0:
            j += 1   # alert too early — skip ahead
        else:
            i += 1   # edge too early — skip ahead
    return paired


def _nearest(ts: float, alerts_sorted: List[float]) -> Optional[float]:
    """Nearest alert to an edge (display only — pairing is the walk above)."""
    if not alerts_sorted:
        return None
    return min(alerts_sorted, key=lambda a: abs(a - ts))


def _score_sid(c, sid: int, tolerance: float,
               want_detail: bool) -> Dict[str, Any]:
    tr = c.table("trades").select(
        "id,entry_fill_ts,exit_fill_ts"
    ).eq("strategy_id", sid).like("data_source", "backtest_%") \
        .not_.is_("exit_fill_ts", "null") \
        .order("exit_fill_ts", desc=True).limit(10).execute().data or []
    trades = list(reversed(tr))  # oldest → newest for display

    edges: List[Tuple[float, int, str]] = []  # (unix, trade_idx, 'entry'|'exit')
    for idx, t in enumerate(trades):
        for kind in ("entry", "exit"):
            dt = _parse_iso(t.get(f"{kind}_fill_ts"))
            if dt is not None:
                edges.append((dt.timestamp(), idx, kind))

    alerts_unix: List[float] = []
    if edges:
        import datetime as _dt
        lo = _dt.datetime.fromtimestamp(min(e[0] for e in edges) - _ALERT_PAD_SEC,
                                        _dt.timezone.utc).isoformat()
        hi = _dt.datetime.fromtimestamp(max(e[0] for e in edges) + _ALERT_PAD_SEC,
                                        _dt.timezone.utc).isoformat()
        al = c.table("alerts").select(
            "strategy_id,fill_ts,trigger_ts,event_type,timestamp"
        ).eq("strategy_id", sid).gte("timestamp", lo).lte("timestamp", hi) \
            .execute().data or []
        for a in al:
            dt = _parse_iso(a.get("fill_ts") or a.get("trigger_ts"))
            if dt is not None:
                alerts_unix.append(dt.timestamp())

    paired = _greedy_pair([e[0] for e in edges], alerts_unix, tolerance)
    alerts_sorted = sorted(alerts_unix)
    points = sum(1 for e in edges if e[0] in paired)
    denom = 2 * len(trades)

    out: Dict[str, Any] = {
        "strategy_id": sid, "points": points, "denom": denom,
        "trade_count": len(trades), "tolerance_seconds": tolerance,
    }
    if want_detail:
        import datetime as _dt

        def _fmt(u: Optional[float]) -> Optional[str]:
            return None if u is None else _dt.datetime.fromtimestamp(
                u, _dt.timezone.utc).isoformat()

        rows = []
        for idx, t in enumerate(trades):
            row: Dict[str, Any] = {"trade_id": t.get("id")}
            for kind in ("entry", "exit"):
                dt = _parse_iso(t.get(f"{kind}_fill_ts"))
                u = dt.timestamp() if dt else None
                near = _nearest(u, alerts_sorted) if u is not None else None
                row[f"{kind}_ts"] = _fmt(u)
                row[f"{kind}_nearest_alert_ts"] = _fmt(near)
                row[f"{kind}_delta_sec"] = (
                    None if (u is None or near is None) else round(near - u, 3))
                row[f"{kind}_paired"] = bool(u is not None and u in paired)
            rows.append(row)
        out["trades"] = rows
    return out


@router.get("")
def last10_scores(sids: Optional[str] = Query(
        None, description="Comma-separated strategy ids; omit for all "
                          "strategies that have backtest trades."),
        tolerance_seconds: float = Query(_TOL_DEFAULT, ge=0.0, le=300.0),
        user=Depends(get_current_user)):
    """Batch n/20 scores keyed by strategy id (the table column)."""
    c = _admin()
    if sids:
        try:
            id_list = [int(x) for x in sids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="sids must be comma-separated integers")
    else:
        rows = c.table("strategies").select("id").order("id").execute().data or []
        id_list = [r["id"] for r in rows]
    return {"tolerance_seconds": tolerance_seconds,
            "scores": {str(s): _score_sid(c, s, tolerance_seconds, False)
                       for s in id_list}}


@router.get("/{sid}")
def last10_detail(sid: int,
                  tolerance_seconds: float = Query(_TOL_DEFAULT, ge=0.0, le=300.0),
                  user=Depends(get_current_user)):
    """Per-trade pairing detail (the modal)."""
    return _score_sid(_admin(), sid, tolerance_seconds, True)
