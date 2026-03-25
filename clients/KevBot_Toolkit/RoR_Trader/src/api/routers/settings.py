"""Settings router — GET/PUT user settings."""

import json
import os
import logging

from fastapi import APIRouter, Depends, Body

from api.deps import get_current_user
from db import USE_DB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Path for JSON fallback (single-user local dev)
_SETTINGS_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
_SETTINGS_FILE = os.path.join(_SETTINGS_DIR, 'settings.json')


@router.get("")
def get_settings(user=Depends(get_current_user)):
    """Load user settings."""
    if USE_DB:
        from db import load_settings_db
        return load_settings_db()

    # Fallback: JSON file
    if os.path.exists(_SETTINGS_FILE):
        with open(_SETTINGS_FILE) as f:
            return json.load(f)
    return {}


@router.put("")
def update_settings(settings: dict = Body(...), user=Depends(get_current_user)):
    """Save user settings."""
    if USE_DB:
        from db import save_settings_db
        save_settings_db(settings)
        return {"status": "saved"}

    # Fallback: JSON file
    with open(_SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    return {"status": "saved"}
