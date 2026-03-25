"""Auth router — login, signup, refresh, /me."""

import os
import logging

from fastapi import APIRouter, HTTPException, Depends

from api.schemas.auth import LoginRequest, LoginResponse, RefreshRequest
from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Authenticate via Supabase Auth and return JWT pair."""
    from supabase import create_client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    client = create_client(url, key)
    try:
        result = client.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password,
        })
    except Exception as e:
        logger.warning("Login failed for %s: %s", req.email, e)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session = result.session
    user = result.user
    if not session or not user:
        raise HTTPException(status_code=401, detail="Authentication failed")

    return LoginResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_at=session.expires_at,
        user={"id": str(user.id), "email": user.email or ""},
    )


@router.post("/refresh", response_model=LoginResponse)
def refresh(req: RefreshRequest):
    """Refresh an expired JWT using a refresh token."""
    from supabase import create_client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    client = create_client(url, key)
    try:
        result = client.auth.refresh_session(req.refresh_token)
    except Exception as e:
        logger.warning("Token refresh failed: %s", e)
        raise HTTPException(status_code=401, detail="Refresh failed")

    session = result.session
    user = result.user
    if not session or not user:
        raise HTTPException(status_code=401, detail="Refresh failed")

    return LoginResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_at=session.expires_at,
        user={"id": str(user.id), "email": user.email or ""},
    )


@router.get("/me")
def me(user=Depends(get_current_user)):
    """Return the currently authenticated user."""
    return user
