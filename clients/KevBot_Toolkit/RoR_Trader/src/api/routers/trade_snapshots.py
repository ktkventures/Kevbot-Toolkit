"""Admin API for trade-level fidelity snapshots (M-RS3 follow-up).

Surfaces src/trade_snapshot.py through the admin UI: capture every backtest-lane
trade now, again later, and diff to the EXACT added/removed/changed trades — the
bit-perfect-vs-before check a count/KPI diff can't give. Backs the
`trade_snapshots` table (migrations/trade_snapshots_table.sql); the multi-MB
payload is gzip+base64 in a TEXT column.

Capture runs in a background thread (146k trades is minutes — past the HTTP
edge timeout), inserting a 'running' row that flips to 'ready'. Service-role
client; scoped by user_id.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin/trade-snapshots", tags=["trade-snapshots"])

TABLE = "trade_snapshots"


def _uid(user) -> str:
    uid = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not uid:
        raise HTTPException(status_code=401, detail="missing user_id")
    return uid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _all_sids(client, user_id):
    rows = (client.table('strategies').select('id')
            .eq('user_id', user_id).execute().data)
    return sorted(r['id'] for r in rows)


def _run_capture(snap_id: int, sids, user_id: str) -> None:
    """Background: build the snapshot, gzip it, flip the row to 'ready'."""
    from db import get_admin_client
    import trade_snapshot as ts
    client = get_admin_client()
    try:
        snap = ts.build_snapshot_dict(sids)
        payload = {"strategies": snap}
        raw = json.dumps(payload, default=str).encode("utf-8")
        gz = base64.b64encode(gzip.compress(raw)).decode("ascii")
        n_trades = sum(v.get("n", 0) for v in snap.values())
        client.table(TABLE).update({
            "status": "ready",
            "n_strategies": len(sids),
            "n_trades": n_trades,
            "snapshot_gz": gz,
            "completed_at": _now(),
        }).eq("id", snap_id).execute()
        logger.info("[TRADE-SNAPSHOT] %s ready: %d strategies, %d trades "
                    "(%d KB gz)", snap_id, len(sids), n_trades, len(gz) // 1024)
    except Exception as e:  # noqa: BLE001
        logger.exception("[TRADE-SNAPSHOT] %s failed: %s", snap_id, e)
        try:
            client.table(TABLE).update({
                "status": "failed", "error": f"{type(e).__name__}: {e}",
                "completed_at": _now()}).eq("id", snap_id).execute()
        except Exception:  # noqa: BLE001
            pass


def _load_snapshot(client, snap_id, user_id):
    row = (client.table(TABLE).select('*')
           .eq('id', snap_id).eq('user_id', user_id).limit(1).execute().data)
    if not row:
        raise HTTPException(status_code=404, detail=f"snapshot {snap_id} not found")
    r = row[0]
    if r.get("status") != "ready" or not r.get("snapshot_gz"):
        raise HTTPException(status_code=409,
                            detail=f"snapshot {snap_id} not ready ({r.get('status')})")
    raw = gzip.decompress(base64.b64decode(r["snapshot_gz"]))
    return json.loads(raw).get("strategies", {})


@router.post("")
def create_snapshot(body: dict = Body(...), user=Depends(get_current_user)):
    """Start a snapshot. Body: {label, strategy_ids?: [int] | null(=all)}.
    Returns the row id immediately; poll the list/GET for status=ready."""
    from db import get_admin_client
    user_id = _uid(user)
    label = (body.get("label") or "").strip() or f"snapshot {_now()[:19]}"
    client = get_admin_client()
    sids = body.get("strategy_ids") or _all_sids(client, user_id)
    sids = [int(x) for x in sids]
    row = client.table(TABLE).insert({
        "label": label, "user_id": user_id, "status": "running",
        "strategy_ids": sids, "created_at": _now(),
    }).execute().data[0]
    snap_id = row["id"]
    threading.Thread(target=_run_capture, args=(snap_id, sids, user_id),
                     name=f"trade-snapshot-{snap_id}", daemon=True).start()
    return {"id": snap_id, "status": "running", "n_strategies": len(sids)}


@router.get("")
def list_snapshots(user=Depends(get_current_user)):
    """List snapshots (metadata only — never the payload)."""
    from db import get_admin_client
    user_id = _uid(user)
    rows = (get_admin_client().table(TABLE)
            .select('id,label,status,n_strategies,n_trades,error,'
                    'created_at,completed_at')
            .eq('user_id', user_id).order('created_at', desc=True)
            .limit(100).execute().data)
    return {"snapshots": rows or []}


@router.post("/diff")
def diff_snapshots(body: dict = Body(...), user=Depends(get_current_user)):
    """Body: {before_id, after_id}. Returns the exact per-strategy
    added/removed/changed trade diff."""
    from db import get_admin_client
    import trade_snapshot as ts
    user_id = _uid(user)
    before_id = body.get("before_id")
    after_id = body.get("after_id")
    if not before_id or not after_id:
        raise HTTPException(status_code=400,
                            detail="before_id and after_id required")
    client = get_admin_client()
    before = _load_snapshot(client, before_id, user_id)
    after = _load_snapshot(client, after_id, user_id)
    return ts.diff_dicts(before, after)


@router.delete("/{snap_id}")
def delete_snapshot(snap_id: int, user=Depends(get_current_user)):
    from db import get_admin_client
    user_id = _uid(user)
    (get_admin_client().table(TABLE).delete()
     .eq('id', snap_id).eq('user_id', user_id).execute())
    return {"id": snap_id, "deleted": True}
