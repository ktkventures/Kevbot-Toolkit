"""Alerts router — alert feed, config, acknowledgement."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Body, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# =============================================================================
# ALERT FEED
# =============================================================================

@router.get("")
def list_alerts(
    limit: int = Query(100, ge=1, le=5000),
    user=Depends(get_current_user),
):
    """Load alert history, most recent first."""
    from alerts import load_alerts
    return load_alerts(limit=limit)


@router.get("/strategy/{strategy_id}")
def get_strategy_alerts(
    strategy_id: int,
    limit: int = Query(50, ge=1, le=1000),
    user=Depends(get_current_user),
):
    """Get alerts for a specific strategy."""
    from alerts import get_alerts_for_strategy
    return get_alerts_for_strategy(strategy_id, limit=limit)


@router.put("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, user=Depends(get_current_user)):
    """Mark an alert as acknowledged."""
    from alerts import acknowledge_alert as _acknowledge
    success = _acknowledge(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged"}


@router.post("/clear")
def clear_alerts(user=Depends(get_current_user)):
    """Clear all alert history."""
    from alerts import clear_alerts as _clear
    _clear()
    return {"status": "cleared"}


@router.post("/clear-by-strategies")
def clear_alerts_by_strategies(
    payload: dict = Body(...),
    user=Depends(get_current_user),
):
    """Delete alerts for the given list of strategy IDs.

    Body: {"strategy_ids": [int, int, ...]}.
    User-scoped via the alerts.user_id RLS predicate; admin client is
    used for the delete itself so RLS doesn't silently drop it. Caller
    is responsible for selecting only their own strategies — backend
    additionally filters strategy_ids belonging to the current user
    before issuing the delete.
    """
    sids = payload.get("strategy_ids") or []
    if not isinstance(sids, list) or not sids:
        raise HTTPException(status_code=400,
                            detail="strategy_ids must be a non-empty list")
    try:
        sids = [int(x) for x in sids]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400,
                            detail="strategy_ids must be integers")

    from db import get_admin_client
    client = get_admin_client()
    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="missing user_id")

    # Restrict to strategies owned by this user — defense in depth
    owned = client.table("strategies").select("id") \
        .in_("id", sids).eq("user_id", user_id).execute()
    owned_ids = [row["id"] for row in (owned.data or [])]
    if not owned_ids:
        return {"status": "noop", "deleted": 0,
                "requested": len(sids), "owned": 0}

    # Delete in chunks for safety on huge selections
    deleted = 0
    BATCH = 500
    for i in range(0, len(owned_ids), BATCH):
        chunk = owned_ids[i:i + BATCH]
        # Page through alert ids for this chunk
        while True:
            r = client.table("alerts").select("id") \
                .in_("strategy_id", chunk).limit(1000).execute()
            ids = [row["id"] for row in (r.data or [])]
            if not ids:
                break
            client.table("alerts").delete().in_("id", ids).execute()
            deleted += len(ids)

    return {
        "status": "cleared",
        "deleted": deleted,
        "requested": len(sids),
        "owned": len(owned_ids),
    }


# =============================================================================
# ALERT CONFIG
# =============================================================================

@router.get("/config")
def get_alert_config(user=Depends(get_current_user)):
    """Load alert configuration (global + per-strategy + per-portfolio)."""
    from alerts import load_alert_config
    return load_alert_config()


@router.put("/config")
def update_alert_config(config: dict = Body(...), user=Depends(get_current_user)):
    """Save alert configuration."""
    from alerts import save_alert_config
    save_alert_config(config)
    return {"status": "saved"}


# =============================================================================
# SINGLE ALERT LOOKUP
# =============================================================================
#
# Declared LAST so the literal-path routes above (/config, /strategy/{...},
# /clear, "") take precedence over this integer-path route. FastAPI matches
# routes in declaration order; if /{alert_id} came before /config, a request
# for /api/alerts/config would fail 422 trying to coerce "config" → int.

@router.get("/{alert_id}")
def get_alert(alert_id: int, user=Depends(get_current_user)):
    """Get a single alert by id, including its webhook_deliveries array.

    Powers the Portfolio Trade History Details drawer (roadmap 9ad):
    the frontend uses this to show the rendered webhook payload that
    was dispatched for a specific trade + the HTTP delivery result.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="DB mode required")
    from db import get_alert_by_id_db
    alert = get_alert_by_id_db(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
