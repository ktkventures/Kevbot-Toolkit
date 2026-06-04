"""System settings router — read/write platform-wide runtime flags.

First key: `data_worker_streaming_enabled` (bool). Data Worker polls the
DB with a ~30s cache, so flipping this in the UI propagates without a
redeploy.

The "admin" gate is currently any authenticated user (matches the rest
of the admin endpoints — there's no role check elsewhere). When proper
admin-roles ship this will tighten naturally.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/system-settings", tags=["system-settings"])


# Known keys + their defaults (env-var fallback + description).
# Keep the source of truth here so the UI can render the toggle even
# when the DB row hasn't been written yet.
KNOWN_SETTINGS: dict[str, dict] = {
    "data_worker_streaming_enabled": {
        "env_var": "DATA_WORKER_STREAMING_ENABLED",
        "env_default": "true",
        "description": (
            "Whether the Data Worker runs the always-on streaming "
            "backtest catchup + snapshot flush + KPI recompute loops. "
            "When OFF, use on-demand /admin/update-jobs to refresh "
            "trades + KPIs explicitly. Bar ingest / recon / metrics "
            "loops keep running regardless. Cached ~30s in Data Worker "
            "so changes take effect within a minute."
        ),
        "type": "bool",
    },
}


def _env_bool(env_val: str | None, default: str) -> bool:
    """Parse an env var like the streaming-loop check does."""
    return (env_val if env_val is not None else default).strip().lower() in (
        "1", "true", "yes", "on",
    )


def _effective_value(key: str, db_value):
    """If the DB row is missing, fall back to the env var default."""
    if db_value is not None:
        return db_value
    spec = KNOWN_SETTINGS.get(key)
    if not spec:
        return None
    env_val = os.environ.get(spec["env_var"])
    if spec["type"] == "bool":
        return _env_bool(env_val, spec["env_default"])
    return env_val if env_val is not None else spec.get("env_default")


@router.get("")
def list_settings(user=Depends(get_current_user)):
    """List all known settings + their current effective values."""
    from db import USE_DB, list_system_settings
    if not USE_DB:
        return {"settings": []}
    rows = {r["key"]: r for r in list_system_settings()}
    out = []
    for key, spec in KNOWN_SETTINGS.items():
        row = rows.get(key)
        db_value = row.get("value") if row else None
        effective = _effective_value(key, db_value)
        out.append({
            "key": key,
            "value": effective,
            "db_value": db_value,
            "source": "db" if db_value is not None else "env",
            "description": (row.get("description") if row else None) or spec["description"],
            "type": spec["type"],
            "env_var": spec["env_var"],
            "env_default": spec["env_default"],
            "updated_at": row.get("updated_at") if row else None,
            "updated_by": row.get("updated_by") if row else None,
        })
    # Append any extra DB rows whose keys aren't in KNOWN_SETTINGS (forward-compat).
    for key, row in rows.items():
        if key in KNOWN_SETTINGS:
            continue
        out.append({
            "key": key,
            "value": row.get("value"),
            "db_value": row.get("value"),
            "source": "db",
            "description": row.get("description"),
            "type": "unknown",
            "updated_at": row.get("updated_at"),
            "updated_by": row.get("updated_by"),
        })
    return {"settings": out}


@router.get("/{key}")
def get_setting(key: str, user=Depends(get_current_user)):
    """Read a single setting's current effective value."""
    from db import USE_DB, get_system_setting
    if not USE_DB:
        raise HTTPException(status_code=501, detail="DB mode required")
    db_value = get_system_setting(key)
    effective = _effective_value(key, db_value)
    return {
        "key": key,
        "value": effective,
        "db_value": db_value,
        "source": "db" if db_value is not None else "env",
    }


@router.patch("/{key}")
def update_setting(
    key: str,
    payload: dict = Body(...),
    user=Depends(get_current_user),
):
    """Upsert a setting. Body: `{value: <bool|number|string|object>}`.

    Returns the new effective value. Data Worker picks it up within ~30s
    via its cached read.
    """
    from db import USE_DB, set_system_setting, get_system_setting
    if not USE_DB:
        raise HTTPException(status_code=501, detail="DB mode required")
    if "value" not in payload:
        raise HTTPException(status_code=400, detail="payload must include 'value'")
    value = payload["value"]
    spec = KNOWN_SETTINGS.get(key, {})
    description = payload.get("description") or spec.get("description")

    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    ok = set_system_setting(key, value,
                            description=description,
                            updated_by=user_id)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to persist setting")

    db_value = get_system_setting(key)
    effective = _effective_value(key, db_value)
    logger.info(
        "[system_settings] key=%s set by user=%s value=%r (effective=%r)",
        key, user_id, value, effective)
    return {
        "key": key,
        "value": effective,
        "db_value": db_value,
        "source": "db",
    }
