"""
FastAPI dependencies — JWT auth, user context.

Validates Supabase JWTs and sets the thread-local user context
that db.py expects for RLS-scoped queries.
"""

import os
import logging

from fastapi import Depends, HTTPException, Header
from jose import jwt, JWTError

from db import set_current_user

logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")


def get_current_user(authorization: str = Header(default="")):
    """Extract and validate Supabase JWT. Sets thread-local user context for db.py.

    All router functions using this dependency MUST be sync (def, not async def)
    so FastAPI runs them in a threadpool — each request gets its own thread,
    matching db.py's threading.local() pattern.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization[7:]

    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_JWT_SECRET not configured on server",
        )

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except JWTError as e:
        logger.warning("JWT validation failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No subject in token")

    # Set thread-local context — db.py uses this for RLS
    set_current_user(user_id, token)

    return {"id": user_id, "email": payload.get("email", "")}
