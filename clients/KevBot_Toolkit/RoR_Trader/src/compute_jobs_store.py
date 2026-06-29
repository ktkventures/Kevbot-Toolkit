"""Durable cross-service queue for batch compute jobs (M-RS3 Phase 2).

Backs the `compute_jobs` table (migrations/compute_jobs_table.sql). The API
enqueues here when RORT_COMPUTE_REMOTE=1 (instead of spawning an in-process
thread); the dedicated `batch_worker` service claims + runs; both sides read
job state through here so the API response shape is unchanged.

All access is via the service-role admin client (RLS bypassed). Reads/writes
are best-effort with logged failures — a transient DB blip must not crash the
batch-worker loop or the API request.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TABLE = "compute_jobs"


def remote_enabled() -> bool:
    """True when recompute jobs should be enqueued to the DB queue for the
    dedicated batch-worker instead of running in-process. Default OFF —
    today's in-API-thread behavior is unchanged until this is flipped on."""
    return os.getenv("RORT_COMPUTE_REMOTE", "0").strip() == "1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client():
    from db import get_admin_client
    return get_admin_client()


def enqueue(user_id: str, strategy_ids: List[int], job_type: str,
            force: bool = False, payload: Optional[Dict[str, Any]] = None) -> str:
    """INSERT a queued job row; returns the job_id. Mirrors the in-memory job
    dict so a later get_row() is a drop-in for get_job_status(). `payload` carries
    params for non-strategy job types (e.g. 'backfill': {symbol, timeframe, start,
    end}); ignored by the strategy recompute path."""
    job_id = str(uuid.uuid4())
    row = {
        "id": job_id,
        "user_id": user_id,
        "job_type": job_type,
        "strategy_ids": list(strategy_ids),
        "force": bool(force),
        "status": "queued",
        "created_at": _now(),
        "current_strategy_idx": 0,
        "progress_label": "queued",
        "per_strategy_results": [],
        "summary": {
            "total_strategies": len(strategy_ids),
            "total_inserted": 0,
            "total_skipped": 0,
            "total_failed": 0,
            "total_appended": 0,
        },
        "cancelled": False,
        "payload": payload,
    }
    _client().table(TABLE).insert(row).execute()
    logger.info("[COMPUTE-JOBS] enqueued %s type=%s n=%d user=%s",
                job_id[:8], job_type, len(strategy_ids), (user_id or "")[:8])
    return job_id


def claim_one(worker_id: str) -> Optional[Dict[str, Any]]:
    """Optimistic claim of the oldest queued job. Atomic: the UPDATE matches
    only WHERE status='queued', so if another replica grabbed the row first our
    update touches 0 rows and we try the next candidate. Returns the claimed
    row (now status='running') or None when the queue is empty.

    (A SELECT ... FOR UPDATE SKIP LOCKED RPC would be the higher-contention
    upgrade; this optimistic form is correct + dependency-free for a few
    replicas, which is the Phase-2 target.)"""
    cli = _client()
    try:
        cands = (cli.table(TABLE).select("*")
                 .eq("status", "queued").order("created_at")
                 .limit(5).execute().data) or []
    except Exception as e:  # noqa: BLE001
        logger.warning("[COMPUTE-JOBS] claim select failed: %s", e)
        return None
    for row in cands:
        try:
            won = (cli.table(TABLE)
                   .update({"status": "running", "claimed_by": worker_id,
                            "claimed_at": _now(), "started_at": _now(),
                            "progress_label": "claimed"})
                   .eq("id", row["id"]).eq("status", "queued")
                   .execute().data)
        except Exception as e:  # noqa: BLE001
            logger.warning("[COMPUTE-JOBS] claim update %s failed: %s",
                           str(row.get("id"))[:8], e)
            continue
        if won:
            logger.info("[COMPUTE-JOBS] claimed %s type=%s n=%d by=%s",
                        str(row["id"])[:8], row.get("job_type"),
                        len(row.get("strategy_ids") or []), worker_id)
            return won[0]
    return None


def update_row(job_id: str, **fields) -> None:
    """Best-effort partial update (progress / summary / status). Keys must be
    columns of compute_jobs (the recompute worker only ever passes such keys)."""
    if not fields:
        return
    try:
        _client().table(TABLE).update(fields).eq("id", job_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning("[COMPUTE-JOBS] update %s failed: %s",
                       str(job_id)[:8], e)


def get_row(job_id: str) -> Optional[Dict[str, Any]]:
    try:
        data = (_client().table(TABLE).select("*")
                .eq("id", job_id).limit(1).execute().data)
        return data[0] if data else None
    except Exception as e:  # noqa: BLE001
        logger.warning("[COMPUTE-JOBS] get %s failed: %s", str(job_id)[:8], e)
        return None


def list_rows(user_id: str, status: Optional[str] = None,
              limit: int = 50) -> List[Dict[str, Any]]:
    try:
        q = (_client().table(TABLE).select("*").eq("user_id", user_id))
        if status:
            q = q.eq("status", status)
        return (q.order("created_at", desc=True).limit(limit).execute().data) or []
    except Exception as e:  # noqa: BLE001
        logger.warning("[COMPUTE-JOBS] list for %s failed: %s",
                       (user_id or "")[:8], e)
        return []


def is_cancelled(job_id: str) -> bool:
    row = get_row(job_id)
    return bool(row and row.get("cancelled"))


def request_cancel(job_id: str) -> bool:
    """Flag a queued/running job for cooperative cancellation. The batch-worker
    polls cancelled between strategies and bails cleanly."""
    row = get_row(job_id)
    if row is None or row.get("status") not in ("queued", "running"):
        return False
    update_row(job_id, cancelled=True, progress_label="cancelling…")
    return True


def reclaim_stale(max_age_s: int = 1800) -> int:
    """Requeue jobs stuck 'running' past max_age_s (a replica died mid-job).
    Returns count reset. Optional janitor — safe to call from any replica."""
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_s
    cli = _client()
    try:
        running = (cli.table(TABLE).select("id,claimed_at")
                   .eq("status", "running").execute().data) or []
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for r in running:
        ca = r.get("claimed_at")
        try:
            ts = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).timestamp()
        except Exception:  # noqa: BLE001
            continue
        if ts < cutoff:
            update_row(r["id"], status="queued", claimed_by=None,
                       progress_label="requeued (stale claim)")
            n += 1
    if n:
        logger.warning("[COMPUTE-JOBS] reclaimed %d stale running job(s)", n)
    return n
