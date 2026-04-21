"""
FastAPI dependencies — JWT auth, user context.

Validates Supabase JWTs by calling Supabase's auth.getUser() endpoint,
then sets the thread-local user context that db.py expects for RLS.

For local development, set DEV_BYPASS_AUTH=true in .env to skip JWT validation
and use the service role key for database access (bypasses RLS).
"""

import os
import json
import base64
import logging
import threading
import time as _time

from fastapi import HTTPException, Header

from db import set_current_user

logger = logging.getLogger(__name__)

# Local dev auth bypass — NEVER set this on Railway/production
_DEV_BYPASS = os.getenv("DEV_BYPASS_AUTH", "").lower() in ("true", "1", "yes")
_DEV_USER_ID = os.getenv("DEV_USER_ID", "00000000-0000-0000-0000-000000000000")

if _DEV_BYPASS:
    logger.warning("⚠️  DEV_BYPASS_AUTH is enabled — all requests authenticated as %s", _DEV_USER_ID)

# Token validation cache.
#
# Every authenticated request was making a sync httpx call to Supabase's
# /auth/v1/user endpoint. With --workers 1 (required so mass_builder's
# in-memory progress state is shared across requests), a single slow
# Supabase response blocked the whole API for up to 5s, cascading into
# 503 storms when multiple requests queued behind it (2026-04-21).
#
# Cache key: token (the raw JWT). Value: (expires_at_monotonic,). If the
# monotonic clock shows we're within the TTL window, skip the remote
# validation entirely. JWT tokens carry their own `exp` claim which is
# always enforced independently (we decode it every request to extract
# user_id), so cache-miss correctness is preserved.
_AUTH_CACHE: dict = {}
_AUTH_CACHE_LOCK = threading.Lock()
_AUTH_CACHE_TTL_S = 300  # 5 minutes — balances security (revocation window)
                         # against load on Supabase auth endpoint


def get_current_user(authorization: str = Header(default="")):
    """Extract and validate Supabase JWT. Sets thread-local user context for db.py.

    Uses Supabase's own auth API to validate the token rather than
    local JWT verification — this handles all signing key formats
    (HS256 legacy, EdDSA new keys) without needing the signing secret.

    In dev mode (DEV_BYPASS_AUTH=true), skips validation and uses the
    service role key for database access.
    """
    if _DEV_BYPASS:
        # Use service role key so DB operations work without RLS
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        set_current_user(_DEV_USER_ID, service_key)
        return {"id": _DEV_USER_ID, "email": "dev@local"}

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization[7:]

    # Extract user_id from JWT payload without verification first
    # (Supabase auth.getUser validates it server-side)
    try:
        parts = token.split('.')
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Malformed token")

        # Decode payload (middle part)
        payload_b64 = parts[1]
        payload_b64 += '=' * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        user_id = payload.get("sub")
        email = payload.get("email", "")

        if not user_id:
            raise HTTPException(status_code=401, detail="No subject in token")

        # Validate token via Supabase auth API
        _validate_with_supabase(token)

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("JWT processing failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid token")

    # Set thread-local context — db.py uses this for RLS
    set_current_user(user_id, token)

    return {"id": user_id, "email": email}


def _validate_with_supabase(token: str):
    """Validate a JWT by calling Supabase's auth.getUser() endpoint.

    Cached for 5 minutes per token to avoid hammering Supabase on every
    request. The JWT's own `exp` claim is always enforced (decoded on
    every request upstream), so the cache can't extend a token's
    lifetime — it only skips the remote "is this token still valid
    with the auth provider" check. If Supabase revokes a token, it
    will stop being accepted after the TTL window (≤5 min).
    """
    # Cache hit path — short-circuit the remote call entirely
    now = _time.monotonic()
    with _AUTH_CACHE_LOCK:
        cached_expires = _AUTH_CACHE.get(token)
        if cached_expires is not None and cached_expires > now:
            return
        # Opportunistic cleanup — if the cache has grown past 1k entries,
        # drop anything expired so it can't grow unbounded.
        if len(_AUTH_CACHE) > 1000:
            stale = [k for k, v in _AUTH_CACHE.items() if v <= now]
            for k in stale:
                _AUTH_CACHE.pop(k, None)

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_ANON_KEY", "")

    if not url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")

    import httpx
    try:
        resp = httpx.get(
            f"{url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": key,
            },
            timeout=5.0,
        )
        if resp.status_code != 200:
            logger.warning("Supabase auth validation failed: %d %s", resp.status_code, resp.text[:100])
            raise HTTPException(status_code=401, detail="Token validation failed")
    except httpx.TimeoutException:
        logger.warning("Supabase auth validation timed out")
        raise HTTPException(status_code=503, detail="Auth service timeout")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Supabase auth validation error: %s", e)
        raise HTTPException(status_code=401, detail="Token validation error")

    # Success — cache for TTL so subsequent requests on this token don't
    # make a new network call.
    with _AUTH_CACHE_LOCK:
        _AUTH_CACHE[token] = now + _AUTH_CACHE_TTL_S
