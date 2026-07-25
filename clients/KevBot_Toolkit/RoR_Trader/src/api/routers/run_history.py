"""Run History API — dispatcher run lifecycle rows (board #109, V4.12).

Read-only over `run_history` (see migrations/run_history.sql). Rows are
WRITTEN by two producers only: the run-request endpoint (dev_tasks router,
outcome='requested') and the local dispatcher via Supabase REST
(claim → 'running', reap → terminal outcome + log tail). One filterable
endpoint serves all three UI consumers: the board page (bulk, latest per
task), the task modal's run-history panel (task_id) and /admin/agents
per-agent lists (agent_letter). Admin-gated, service-role client — same
posture as dev_tasks.
"""
from typing import Optional

from fastapi import APIRouter, Depends

from api.deps import get_current_user

router = APIRouter(prefix="/api/run-history", tags=["run-history"])


def _admin():
    from db import get_admin_client
    return get_admin_client()


@router.get("")
def list_runs(task_id: Optional[int] = None, agent_letter: Optional[str] = None,
              limit: int = 100, user=Depends(get_current_user)):
    """Runs newest-first, optionally filtered by task and/or agent."""
    q = _admin().table("run_history").select("*")
    if task_id is not None:
        q = q.eq("task_id", task_id)
    if agent_letter:
        q = q.eq("agent_letter", agent_letter)
    limit = min(max(limit, 1), 500)
    return q.order("requested_at", desc=True).limit(limit).execute().data or []
