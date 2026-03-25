"""Webhooks router — template CRUD, test delivery, delivery log."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Body, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


# =============================================================================
# TEMPLATES CRUD
# =============================================================================

@router.get("/templates")
def list_templates(user=Depends(get_current_user)):
    """Load all webhook templates (built-in + user-created)."""
    from alerts import load_webhook_templates
    return load_webhook_templates()


@router.post("/templates")
def create_template(template: dict = Body(...), user=Depends(get_current_user)):
    """Create a new user webhook template. Auto-assigns ID."""
    from alerts import add_webhook_template
    return add_webhook_template(template)


@router.get("/templates/{template_id}")
def get_template(template_id: str, user=Depends(get_current_user)):
    """Get a single webhook template by ID."""
    from alerts import get_webhook_template_by_id
    template = get_webhook_template_by_id(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Webhook template not found")
    return template


@router.put("/templates/{template_id}")
def update_template(
    template_id: str,
    updates: dict = Body(...),
    user=Depends(get_current_user),
):
    """Update an existing webhook template."""
    from alerts import update_webhook_template
    success = update_webhook_template(template_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook template not found")
    return {"status": "updated"}


@router.delete("/templates/{template_id}")
def delete_template(template_id: str, user=Depends(get_current_user)):
    """Delete a user-created webhook template. Built-in templates cannot be deleted."""
    from alerts import delete_webhook_template
    deleted = delete_webhook_template(template_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Webhook template not found or is a built-in default",
        )
    return {"status": "deleted"}


# =============================================================================
# TEST DELIVERY
# =============================================================================

@router.post("/test")
def test_webhook(body: dict = Body(...), user=Depends(get_current_user)):
    """Send a test webhook delivery.

    Expects JSON body with:
        - ``url``: Webhook URL to test.
        - ``payload``: Optional custom payload string.
        - ``alert``: Optional alert dict to use for payload formatting.
    """
    url = body.get('url')
    if not url:
        raise HTTPException(status_code=400, detail="'url' is required")

    custom_payload = body.get('payload')
    alert = body.get('alert', {
        'type': 'test',
        'strategy_name': 'Test Strategy',
        'symbol': 'TEST',
        'direction': 'LONG',
        'message': 'This is a test webhook delivery.',
    })

    from alerts import send_webhook
    result = send_webhook(url, alert, custom_payload=custom_payload)
    return result


# =============================================================================
# DELIVERY LOG
# =============================================================================

@router.get("/delivery-log")
def get_delivery_log(
    limit: int = Query(100, ge=1, le=1000),
    user=Depends(get_current_user),
):
    """Get webhook delivery history across all alerts."""
    from alerts import get_webhook_delivery_log
    return get_webhook_delivery_log(limit=limit)
