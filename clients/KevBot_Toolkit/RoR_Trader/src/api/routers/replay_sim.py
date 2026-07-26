"""Replay-simulated basis (SIM lane) — API surface (board #120, V1.16).

Design: docs/_active/Design_Replay_Sim_Basis.md. The producer half (the heavy
replay job + cache table) is replay_sim_job.py; this router is the thin
data-path around it:

  POST /api/strategy-health-sim/{sid}/run     enqueue an on-demand replay
       (declarative — the local poller EXECUTES it; Railway can't reach Kevin's
        machine, same pattern as the Run-button dispatcher, board #109).
  GET  /api/strategy-health-sim/{sid}         READ the cached SIM score — pairs
       the newest cache row's would-have-fired edges vs the CURRENT last-10 bt
       edges at read time (cheap; NEVER runs the replay). Returns the divergence
       ledger + coverage + staleness so F can surface trust next to the score.
  GET  /api/strategy-health-sim/{sid}/requests   recent request rows (button UI).
  GET/PUT /api/strategy-health-sim/{sid}/optin   per-strategy nightly opt-in
       toggle (Kevin 07-26 refinement; default OFF).

The pairing REUSES health_last10._greedy_pair verbatim (no fork — §9 test 1);
the SIM lane differs from the fired/theo/algo lanes only in WHERE the compared
timestamps come from (the cache, not the alerts table). Admin-gated,
service-role client — same posture as health_last10 / run_history / dev_tasks.
This file never runs the engine and issues only light reads/writes.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from api.deps import get_current_user
from api.routers.health_last10 import _greedy_pair
from replay_sim_job import _last10_bt_trades, _parse_iso, apply_staleness

router = APIRouter(prefix="/api/strategy-health-sim", tags=["strategy-health"])

_TOL_DEFAULT = 10.0
_ACTIVE_REQUEST_OUTCOMES = ("requested", "running")


def _admin():
    from db import get_admin_client
    return get_admin_client()


def _edge_ts_list(raw: List[Any]) -> List[float]:
    """Replay edges → unix timestamps. Accepts v1.0 ISO strings and the v1.1
    enriched {'ts':..} dict form (§2) — pairing only needs the timestamp."""
    out: List[float] = []
    for e in raw or []:
        ts = e.get("ts") if isinstance(e, dict) else e
        dt = _parse_iso(ts)
        if dt is not None:
            out.append(dt.timestamp())
    return out


def _live_armed_fp(c) -> Optional[str]:
    """The armed-flag fingerprint as of the last SIM job run (§8 D9), stashed in
    system_settings by replay_sim_job so the API never imports the heavy
    replay_harness. None if never stamped. Best-effort — drift detection is
    provenance, not load-bearing."""
    try:
        rows = (c.table("system_settings").select("value")
                .eq("key", "replay_sim_armed_fp").execute().data or [])
        if rows:
            v = rows[0]["value"]
            return v.get("fp") if isinstance(v, dict) else str(v)
    except Exception:  # noqa: BLE001
        return None
    return None


def score_sim_basis(c, sid: int, tolerance: float) -> Dict[str, Any]:
    """Pair the newest SIM cache row's edges vs the CURRENT last-10 bt edges.

    Reusable by F (fold into health_last10's basis dispatch) and the nightly
    bug-hunt. status='none' ⇒ no cache yet (page shows 'run SIM'); 'unres' ⇒ the
    replay couldn't resolve a ribbon (surface the ledger note, never a 0)."""
    rows = (c.table("replay_sim_basis").select("*")
            .eq("strategy_id", sid).in_("status", ["ok", "partial"])
            .order("computed_at", desc=True).limit(1).execute().data or [])
    if not rows:
        # Distinguish 'never run' from 'last run was unres' so the UI shows the
        # right thing (a run button vs the unresolved-ribbon reason).
        last = (c.table("replay_sim_basis").select("status,divergences,computed_at")
                .eq("strategy_id", sid).order("computed_at", desc=True)
                .limit(1).execute().data or [])
        if last and last[0]["status"] == "unres":
            return {"strategy_id": sid, "status": "unres",
                    "divergences": last[0].get("divergences") or [],
                    "computed_at": last[0].get("computed_at")}
        return {"strategy_id": sid, "status": "none"}

    row = rows[0]
    win_since = _parse_iso(row.get("window_since"))
    win_until = _parse_iso(row.get("window_until"))

    bt_trades = _last10_bt_trades(c, sid)
    stale = apply_staleness(bt_trades, win_since, win_until)
    covered = stale["covered"]

    bt_edges: List[float] = []
    for t in covered:
        for kind in ("entry", "exit"):
            dt = _parse_iso(t.get(f"{kind}_fill_ts"))
            if dt is not None:
                bt_edges.append(dt.timestamp())

    replay_edges = (_edge_ts_list(row.get("replay_entries"))
                    + _edge_ts_list(row.get("replay_exits")))
    # bt edges are the reference set (like fired/theo); pair them vs the SIM lane.
    paired = _greedy_pair(bt_edges, replay_edges, tolerance)
    denom = 2 * stale["n_covered"]

    cached_fp = row.get("flags_fp")
    live_fp = _live_armed_fp(c)
    flags_stale = bool(cached_fp and live_fp and cached_fp != live_fp)

    return {
        "strategy_id": sid,
        "status": row["status"],
        "points": len(paired),
        "denom": denom,
        "trade_count": stale["n_covered"],
        "tolerance_seconds": tolerance,
        "coverage": row.get("coverage"),
        "corrected": row.get("corrected"),
        "divergences": row.get("divergences") or [],
        "computed_at": row.get("computed_at"),
        "compute_secs": row.get("compute_secs"),
        "flags_fp": cached_fp,
        "flags_stale": flags_stale,       # armed flags changed since compute → re-run
        "engine_sha": row.get("engine_sha"),
        "window_since": row.get("window_since"),
        "window_until": row.get("window_until"),
        # staleness (§4.3): current last-10 extends past the cached window.
        "stale": stale["stale"] or (stale["n_covered"] < stale["n_total"]),
        "covered_of_total": [stale["n_covered"], stale["n_total"]],
        "result_id": row.get("id"),
    }


@router.get("/{sid}")
def sim_score(sid: int,
             tolerance_seconds: float = Query(_TOL_DEFAULT, ge=0.0, le=300.0),
             user=Depends(get_current_user)):
    """Read-only cached SIM score for one strategy (the health-page cell)."""
    return score_sim_basis(_admin(), sid, tolerance_seconds)


@router.post("/{sid}/run")
def request_sim_run(sid: int, payload: dict = Body(default={}),
                    user=Depends(get_current_user)):
    """Enqueue an on-demand replay for `sid` (Phase 1 button). Declarative — the
    local poller claims and runs it. 409 if a request is already pending/running
    for this strategy (no point stacking identical heavy replays)."""
    author = (payload.get("author") or "kevin").strip() or "kevin"
    c = _admin()
    srow = (c.table("strategies").select("id").eq("id", sid).execute().data or [])
    if not srow:
        raise HTTPException(status_code=404, detail="strategy not found")
    pending = (c.table("replay_sim_requests").select("id,outcome")
               .eq("strategy_id", sid)
               .in_("outcome", list(_ACTIVE_REQUEST_OUTCOMES))
               .limit(1).execute().data or [])
    if pending:
        raise HTTPException(status_code=409,
                            detail=f"a SIM run is already {pending[0]['outcome']} "
                                   "for this strategy")
    res = (c.table("replay_sim_requests").insert({
        "strategy_id": sid, "requested_by": author, "source": "button",
        "outcome": "requested"}).execute())
    return res.data[0] if res.data else {"status": "requested"}


@router.get("/{sid}/requests")
def sim_requests(sid: int, limit: int = Query(10, ge=1, le=100),
                 user=Depends(get_current_user)):
    """Recent request rows for `sid`, newest-first (button spinner/status)."""
    return (_admin().table("replay_sim_requests").select("*")
            .eq("strategy_id", sid).order("requested_at", desc=True)
            .limit(limit).execute().data or [])


@router.get("/{sid}/optin")
def get_optin(sid: int, user=Depends(get_current_user)):
    """Per-strategy nightly opt-in state. Absent row ⇒ opted OUT (default)."""
    rows = (_admin().table("replay_sim_optin").select("*")
            .eq("strategy_id", sid).execute().data or [])
    if not rows:
        return {"strategy_id": sid, "enabled": False}
    return rows[0]


@router.put("/{sid}/optin")
def set_optin(sid: int, payload: dict = Body(...),
              user=Depends(get_current_user)):
    """Turn nightly SIM on/off for one strategy (Kevin 07-26 — selective, so
    fleet-wide cost doesn't scale with fleet size). Upsert, default OFF."""
    if "enabled" not in payload:
        raise HTTPException(status_code=400, detail="body must include 'enabled'")
    enabled = bool(payload["enabled"])
    author = (payload.get("author") or "kevin").strip() or "kevin"
    c = _admin()
    srow = (c.table("strategies").select("id").eq("id", sid).execute().data or [])
    if not srow:
        raise HTTPException(status_code=404, detail="strategy not found")
    res = (c.table("replay_sim_optin").upsert({
        "strategy_id": sid, "enabled": enabled,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": author}, on_conflict="strategy_id").execute())
    return res.data[0] if res.data else {"strategy_id": sid, "enabled": enabled}
