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
from api.routers.health_last10 import _greedy_pair, _nearest
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


def _row_provenance(row: Dict[str, Any], c) -> Dict[str, Any]:
    """The immutable per-run provenance the UI surfaces next to the score
    (board #177 run-navigation reads a specific row and shows exactly this)."""
    cached_fp = row.get("flags_fp")
    live_fp = _live_armed_fp(c)
    return {
        "coverage": row.get("coverage"),
        "corrected": row.get("corrected"),
        "divergences": row.get("divergences") or [],
        "computed_at": row.get("computed_at"),
        "compute_secs": row.get("compute_secs"),
        "flags_fp": cached_fp,
        # armed flags changed since compute → re-run warranted (§8 D9).
        "flags_stale": bool(cached_fp and live_fp and cached_fp != live_fp),
        "engine_sha": row.get("engine_sha"),
        "window_since": row.get("window_since"),
        "window_until": row.get("window_until"),
        "timeframe_secs": row.get("timeframe_secs"),
        "result_id": row.get("id"),
    }


def _sim_detail_rows(covered: List[Dict[str, Any]], replay_edges: List[float],
                     paired: set, tolerance: float,
                     tf_secs: Optional[int]) -> Dict[str, Any]:
    """Per-trade pairing rows for the SIM tab — a VERBATIM mirror of
    health_last10._score_sid's detail shape (board #177), sourcing the compared
    lane from the replay cache edges instead of the alerts table. No price / no
    direction — the pairing table needs only timestamps, which the cache stores.
    `paired` is the set of bt-edge INDICES (into the flattened entry/exit edge
    list) that greedy-paired to a replay edge."""
    display_window = (max(tolerance, 2 * tf_secs) if tf_secs else tolerance)
    sorted_edges = sorted(replay_edges)

    def _fmt(u: Optional[float]) -> Optional[str]:
        return None if u is None else datetime.fromtimestamp(
            u, timezone.utc).isoformat()

    rows: List[Dict[str, Any]] = []
    pos = 0  # running index into the flattened edge list (entry then exit)
    for t in covered:
        row: Dict[str, Any] = {"trade_id": t.get("id")}
        side: Dict[str, Any] = {}
        for kind in ("entry", "exit"):
            dt = _parse_iso(t.get(f"{kind}_fill_ts"))
            u = dt.timestamp() if dt is not None else None
            row[f"{kind}_ts"] = _fmt(u)
            near = _nearest(u, sorted_edges) if u is not None else None
            # Unpaired edges show BLANK nearest/Δ (not globally-nearest garbage)
            # — same Chart+Trades display rule the alert bases use.
            if near is not None and u is not None and abs(near - u) > display_window:
                near = None
            side[f"{kind}_nearest_alert_ts"] = _fmt(near)
            side[f"{kind}_delta_sec"] = (
                None if (u is None or near is None) else round(near - u, 3))
            side[f"{kind}_paired"] = bool(u is not None and pos in paired)
            if u is not None:
                pos += 1
        row["sim"] = side
        rows.append(row)
    return {"trades": rows, "display_window_sec": display_window}


def score_sim_basis(c, sid: int, tolerance: float,
                    result_id: Optional[int] = None,
                    detail: bool = False) -> Dict[str, Any]:
    """Pair a SIM cache row's edges vs the CURRENT last-10 bt edges.

    Reusable by F (the unified last-10 modal) and the nightly bug-hunt.
    `result_id` selects a SPECIFIC immutable run (board #177 run-navigation —
    "see what I saw then"); default = the newest ok/partial row. `detail=True`
    adds per-trade pairing rows for the modal table (board #177 SIM tab).
    status='none' ⇒ no cache yet (page shows 'run SIM'); 'unres' ⇒ the replay
    couldn't resolve a ribbon (surface the ledger note, never a 0)."""
    if result_id is not None:
        # Run-navigation: load exactly the requested immutable run.
        sel = (c.table("replay_sim_basis").select("*")
               .eq("strategy_id", sid).eq("id", result_id)
               .limit(1).execute().data or [])
        if not sel:
            return {"strategy_id": sid, "status": "none"}
        row = sel[0]
        if row["status"] not in ("ok", "partial"):
            # A navigated unres/error run: surface its provenance + ledger,
            # never fabricate a score.
            out = {"strategy_id": sid, "status": row["status"]}
            out.update(_row_provenance(row, c))
            return out
        rows = [row]
    else:
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

    # Flatten covered bt trades into (entry, exit) edges — the reference set
    # (like fired/theo). `paired` returns the set of edge INDICES that matched.
    bt_edges: List[float] = []
    for t in covered:
        for kind in ("entry", "exit"):
            dt = _parse_iso(t.get(f"{kind}_fill_ts"))
            if dt is not None:
                bt_edges.append(dt.timestamp())

    replay_edges = (_edge_ts_list(row.get("replay_entries"))
                    + _edge_ts_list(row.get("replay_exits")))
    paired = _greedy_pair(bt_edges, replay_edges, tolerance)
    denom = 2 * stale["n_covered"]

    out: Dict[str, Any] = {
        "strategy_id": sid,
        "status": row["status"],
        "points": len(paired),
        # PHANTOM (board #177): replay edges that paired to NO bt edge — SIM's
        # precision miss (the sid-271 "10/20 recall vs 91-vs-51 over-firing"
        # contradiction the modal used to hide). greedy 1:1 ⇒ one replay edge
        # per paired bt edge, so phantom = total replay edges − pairs.
        "phantom": max(0, len(replay_edges) - len(paired)),
        "denom": denom,
        "trade_count": stale["n_covered"],
        "tolerance_seconds": tolerance,
        # staleness (§4.3): current last-10 extends past the cached window.
        "stale": stale["stale"] or (stale["n_covered"] < stale["n_total"]),
        "covered_of_total": [stale["n_covered"], stale["n_total"]],
    }
    out.update(_row_provenance(row, c))
    if detail:
        out.update(_sim_detail_rows(covered, replay_edges, paired, tolerance,
                                    row.get("timeframe_secs")))
    return out


@router.get("/queue")
def sim_queue(user=Depends(get_current_user)):
    """Global active-request queue, oldest-first (board #177). The poller runs
    ONE replay at a time (serialized by design), so a 'requested' row sits
    behind everything ahead of it — this lets the UI show 'queued behind N'
    instead of a spinner indistinguishable from a run in progress. Declared
    BEFORE /{sid} so the literal path wins over the int-typed sid route."""
    rows = (_admin().table("replay_sim_requests")
            .select("id,strategy_id,outcome,requested_at,started_at")
            .in_("outcome", list(_ACTIVE_REQUEST_OUTCOMES))
            .order("requested_at", desc=False).execute().data or [])
    return {"active": rows}


@router.get("/{sid}")
def sim_score(sid: int,
             tolerance_seconds: float = Query(_TOL_DEFAULT, ge=0.0, le=300.0),
             result_id: Optional[int] = Query(
                 None, description="Load a SPECIFIC immutable run (run-nav, "
                                   "board #177); default = newest ok/partial."),
             detail: bool = Query(
                 False, description="Include per-trade pairing rows for the "
                                    "unified last-10 modal's SIM tab."),
             user=Depends(get_current_user)):
    """Read-only cached SIM score for one strategy (the health-page cell +
    the unified modal's SIM tab when detail=true)."""
    return score_sim_basis(_admin(), sid, tolerance_seconds,
                           result_id=result_id, detail=detail)


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
