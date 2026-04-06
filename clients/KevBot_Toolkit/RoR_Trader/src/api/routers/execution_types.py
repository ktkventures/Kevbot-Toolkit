"""Execution Types Router — list, inspect, and configure execution type modules."""

from fastapi import APIRouter, Depends, HTTPException, Body
from api.deps import get_current_user

router = APIRouter(prefix="/api/execution-types", tags=["execution-types"])


@router.get("")
def list_execution_types(user=Depends(get_current_user)):
    """List all registered execution type modules with user's enable/disable state."""
    from execution_types import list_modules

    # Load user's config (enabled state + parameter overrides)
    config = _load_config()

    result = []
    for m in list_modules():
        d = m.to_dict()
        user_cfg = config.get(m.slug, {})
        d['enabled'] = user_cfg.get('enabled', m.slug in ('bar_close', 'level'))  # C and L enabled by default
        d['user_params'] = user_cfg.get('params', {})
        result.append(d)
    return result


@router.get("/config")
def get_config(user=Depends(get_current_user)):
    """Load user's execution type configuration."""
    return _load_config()


@router.put("/config")
def save_config(config: dict = Body(...), user=Depends(get_current_user)):
    """Save user's execution type configuration (enabled state + params)."""
    _save_config(config)
    return {"status": "saved"}


@router.get("/{slug}")
def get_execution_type(slug: str, user=Depends(get_current_user)):
    """Get a specific execution type module by slug."""
    from execution_types import list_modules
    config = _load_config()
    for m in list_modules():
        if m.slug == slug:
            d = m.to_dict()
            user_cfg = config.get(m.slug, {})
            d['enabled'] = user_cfg.get('enabled', m.slug in ('bar_close', 'level'))
            d['user_params'] = user_cfg.get('params', {})
            return d
    raise HTTPException(status_code=404, detail=f"Execution type '{slug}' not found")


@router.put("/{slug}/toggle")
def toggle_execution_type(slug: str, user=Depends(get_current_user)):
    """Toggle an execution type's enabled state."""
    config = _load_config()
    current = config.get(slug, {})
    current_enabled = current.get('enabled', slug in ('bar_close', 'level'))
    current['enabled'] = not current_enabled
    config[slug] = current
    _save_config(config)
    return {"slug": slug, "enabled": current['enabled']}


@router.put("/{slug}/params")
def update_params(slug: str, params: dict = Body(...), user=Depends(get_current_user)):
    """Update an execution type's parameter values."""
    config = _load_config()
    current = config.get(slug, {})
    current['params'] = params
    config[slug] = current
    _save_config(config)
    return {"slug": slug, "params": params}


# ---- Storage helpers ----
# Store execution type config in user_settings under the key 'execution_types'

def _load_config() -> dict:
    from db import USE_DB
    if USE_DB:
        from db import load_settings_db
        settings = load_settings_db()
        return settings.get('execution_types', {})
    return {}


def _save_config(config: dict):
    from db import USE_DB
    if USE_DB:
        from db import load_settings_db, save_settings_db
        settings = load_settings_db()
        settings['execution_types'] = config
        save_settings_db(settings)
