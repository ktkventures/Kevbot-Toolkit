"""Webhook Groups router — account-based webhook groups (11 events per group)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Body, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhook-groups", tags=["webhook-groups"])


# =============================================================================
# CRUD
# =============================================================================

@router.get("")
def list_groups(user=Depends(get_current_user)):
    """Load all webhook groups for the current user."""
    from alerts import load_webhook_groups
    return load_webhook_groups()


@router.post("")
def create_group(group: dict = Body(...), user=Depends(get_current_user)):
    """Create a new webhook group. Auto-assigns ID."""
    from alerts import add_webhook_group
    return add_webhook_group(group)


@router.get("/{group_id}")
def get_group(group_id: str, user=Depends(get_current_user)):
    """Get a single webhook group by ID."""
    from alerts import get_webhook_group_by_id
    group = get_webhook_group_by_id(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Webhook group not found")
    return group


@router.put("/{group_id}")
def update_group(
    group_id: str,
    updates: dict = Body(...),
    user=Depends(get_current_user),
):
    """Update an existing webhook group."""
    from alerts import update_webhook_group
    success = update_webhook_group(group_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook group not found")
    return {"status": "updated"}


@router.delete("/{group_id}")
def delete_group(group_id: str, user=Depends(get_current_user)):
    """Delete a webhook group."""
    from alerts import delete_webhook_group
    deleted = delete_webhook_group(group_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Webhook group not found")
    return {"status": "deleted"}


@router.post("/{group_id}/duplicate")
def duplicate_group(group_id: str, user=Depends(get_current_user)):
    """Duplicate a webhook group."""
    from alerts import duplicate_webhook_group
    copy = duplicate_webhook_group(group_id)
    if copy is None:
        raise HTTPException(status_code=404, detail="Webhook group not found")
    return copy


# =============================================================================
# TEST DELIVERY
# =============================================================================

@router.post("/{group_id}/test/{event_type}")
def test_event(
    group_id: str,
    event_type: str,
    body: dict = Body(default={}),
    user=Depends(get_current_user),
):
    """Send a test webhook for a specific event in the group.

    Body may include an ``alert`` dict to use for payload formatting; otherwise
    a synthetic test alert is used.
    """
    from alerts import (
        get_webhook_group_by_id,
        send_webhook,
        build_placeholder_context,
        render_payload,
        WEBHOOK_EVENT_TYPES,
    )
    if event_type not in WEBHOOK_EVENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown event type: {event_type}")

    group = get_webhook_group_by_id(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Webhook group not found")

    event_cfg = (group.get("events") or {}).get(event_type) or {}
    # Shared URL mode: all events fire at the same URL
    if group.get("url_mode") == "shared":
        url = group.get("shared_url") or ""
    else:
        url = event_cfg.get("url") or ""
    payload_template = event_cfg.get("payload")
    if not url:
        raise HTTPException(
            status_code=400,
            detail=f"No URL configured for event '{event_type}' in this group",
        )

    # Synthetic alert with realistic defaults so placeholders ({{symbol}},
    # {{quantity}}, {{order_price}}, etc.) can render into valid JSON for
    # webhook service validation.
    alert_type = "entry_signal" if event_type.startswith("entry") else (
        "exit_signal" if event_type.startswith("exit") else event_type
    )
    alert = body.get("alert") or {
        "type": alert_type,
        "event_type": event_type,
        "strategy_name": "Test Strategy",
        "strategy_id": "test_strategy",
        "symbol": "AAPL",
        "direction": "LONG" if "long" in event_type else "SHORT",
        "price": 150.00,
        "stop_price": 148.50,
        "entry_price": 150.00,
        "entry_stop_price": 148.50,
        "risk_per_trade": 150.0,
        "atr": 1.25,
        "trigger": "test_trigger",
        "exec_type": "L" if "limit" in event_type else "C",
        "timeframe": "5Min",
        "message": f"Test delivery for {event_type}.",
    }

    rendered_payload = payload_template
    if payload_template:
        ctx = build_placeholder_context(alert, {
            "portfolio_name": "Test Portfolio",
            "portfolio_id": "test",
            "position_risk": 150.0,
        })
        rendered_payload = render_payload(payload_template, ctx)

    return send_webhook(url, alert, custom_payload=rendered_payload)


# =============================================================================
# METADATA
# =============================================================================

@router.get("/meta/event-types")
def list_event_types(user=Depends(get_current_user)):
    """Return the canonical list of webhook event types."""
    from alerts import WEBHOOK_EVENT_TYPES
    return {"event_types": WEBHOOK_EVENT_TYPES}
