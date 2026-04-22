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
