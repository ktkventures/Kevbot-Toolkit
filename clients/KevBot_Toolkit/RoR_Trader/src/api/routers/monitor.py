"""Monitor router — engine status, start/stop control, engine state."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


# =============================================================================
# STATUS
# =============================================================================

@router.get("/status")
def get_status(user=Depends(get_current_user)):
    """Load current monitor status (running, pid, last_poll, etc.)."""
    from alerts import load_monitor_status
    return load_monitor_status()


# =============================================================================
# START / STOP
# =============================================================================

@router.post("/start")
def start_monitor(user=Depends(get_current_user)):
    """Request the monitor engine to start.

    Sets desired_state to 'running' in the database. The worker process
    polls this value and starts the engine accordingly.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(
            status_code=501,
            detail="Monitor control requires DB mode (cloud deployment)",
        )

    from db import set_desired_state_db
    set_desired_state_db('running')
    logger.info("Monitor start requested by user %s", user.get('id'))
    return {"status": "start_requested", "desired_state": "running"}


@router.post("/stop")
def stop_monitor(user=Depends(get_current_user)):
    """Request the monitor engine to stop.

    Sets desired_state to 'stopped' in the database.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(
            status_code=501,
            detail="Monitor control requires DB mode (cloud deployment)",
        )

    from db import set_desired_state_db
    set_desired_state_db('stopped')
    logger.info("Monitor stop requested by user %s", user.get('id'))
    return {"status": "stop_requested", "desired_state": "stopped"}


# =============================================================================
# ENGINE STATE
# =============================================================================

@router.get("/engine-state")
def get_engine_state(user=Depends(get_current_user)):
    """Load engine state (positions, symbols, etc.) from the database."""
    from db import USE_DB
    if USE_DB:
        from db import load_engine_state_db
        state = load_engine_state_db()
        return state if state else {}

    # JSON fallback: read local engine_state.json if it exists
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'engine_state.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}
