"""Environment-role write guard — board #286.

WHY THIS EXISTS
---------------
The dev environment (#165) shares ONE Supabase database with production. The
#285 audit (`docs/_active/Env_Write_Surface_Audit.md`) classified all 249 API
routes and found **29 fleet-wide WRITES** — routes that use the admin client and
touch every user's rows, so per-user scoping does not contain them at all.

The sharpest of those needs no user action:

    `cleanup_orphaned_mass_searches()` / `cleanup_orphaned_update_jobs()` are
    documented as "Called once on api boot". They flip EVERY user's
    `status='running'` row to `orphaned`. **Starting the dev API is itself a
    fleet-wide write** — the first dev deploy would mark production's genuinely
    running mass searches and UAD jobs as dead, and it would repeat on every
    dev deploy.

So this module is a single environment-role marker plus the two mechanisms that
consume it: boot sweeps skip, fleet-wide write routes refuse.

THE THREE DESIGN CONSTRAINTS (from the task, and they are load-bearing)
-----------------------------------------------------------------------
1. **Runtime-read, never import-time.** `env_role()` calls `os.getenv` on every
   read. An import-time read cannot be flipped without a redeploy and — worse —
   cannot be tested, because the module is already imported by the time a test
   sets the variable.

2. **Unset == today's behaviour, exactly.** Production does not set
   `RORT_ENV_ROLE`. With it unset `is_write_guarded()` is False and every entry
   point here returns before doing anything at all: the middleware calls
   `call_next` on its first line, the sweeps run their original code, no route
   is matched, no regex is evaluated. Production is byte-identical.

3. **Refuse, do not silently no-op.** A route that quietly does nothing teaches
   the dev UI to lie about what happened: "Add ticker" would appear to succeed
   and the operator would believe production had been reconfigured. Blocked
   routes return **403** with a body naming the env role and the route.
   (The BOOT path is the one place where "skip" is the right verb — nobody is
   waiting on a reply — and it logs a WARNING so the skip is never invisible.)

FAIL-SAFE ON AN UNKNOWN VALUE
-----------------------------
Unset / blank / `production` / `prod` → production. **Anything else guards**,
including a typo. The two failure modes are not symmetric: a typo'd role on a
dev box that silently writes production is the entire bug this guards; a typo'd
role on production refuses 29 admin routes loudly, with a message that names the
offending value, and is undone by unsetting one variable.

THE BOARD CARVE-OUT
-------------------
§5.5 of the audit deliberately dissents on 18 of the 29: `dev_tasks`,
`mentions`, `m_session`, `agents`, `strategy_notes`. The board is *meant* to be
one shared surface across all sessions, and a dev environment that cannot reach
it cannot be driven by the dispatcher. They are still blocked BY DEFAULT here —
safe is the right default — but `RORT_ENV_ROLE_ALLOW_BOARD=1` re-permits exactly
that group. Without this knob the guard is all-or-nothing, and an all-or-nothing
guard that breaks the dispatcher invites someone to unset `RORT_ENV_ROLE`
entirely, which would take the urgent boot-sweep half down with it.

NOT FIXED HERE, ON PURPOSE
--------------------------
* `replay_sim`'s missing ownership check on `/run` and `/optin` is a real bug.
  This guards the boundary; it does not refactor behind it. (#285 follow-up.)
* `cleanup_orphaned_mass_searches` has a known `queued` gap (board #32).
  Noted, untouched.
* `DEV_BYPASS_AUTH` must never be set in the shared-DB dev environment (§6 of
  the audit). That is env config, not code, and is out of scope here.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_VAR = "RORT_ENV_ROLE"
ALLOW_BOARD_VAR = "RORT_ENV_ROLE_ALLOW_BOARD"

PRODUCTION = "production"
DEV = "dev"

# Values that mean "behave exactly as production always has".
_PRODUCTION_ALIASES = ("", PRODUCTION, "prod")


def env_role() -> str:
    """The current environment role. **Read at call time, every time.**

    Unset / blank / `production` / `prod` → `"production"`.
    Anything else is returned lowercased and is treated as guarded.
    """
    raw = (os.getenv(ENV_VAR) or "").strip().lower()
    if raw in _PRODUCTION_ALIASES:
        return PRODUCTION
    return raw


def is_write_guarded() -> bool:
    """True when fleet-wide writes must be refused (i.e. role is not production)."""
    return env_role() != PRODUCTION


def board_writes_allowed() -> bool:
    """True when the §5.5 shared-coordination routes are re-permitted.

    Only consulted when the role is guarded; under production everything is
    allowed and this is never called.
    """
    return (os.getenv(ALLOW_BOARD_VAR) or "").strip().lower() in ("1", "true", "yes")


class EnvRoleWriteBlocked(RuntimeError):
    """A fleet-wide write was attempted from a non-production environment."""


# ── The 29 fleet-wide writes (§5.6 of Env_Write_Surface_Audit.md) ────────────
#
# Order and spelling mirror §5.6 exactly so the two can be diffed by eye.
# `test_env_role_write_guard_286.py` parses the routers' source and asserts
# every pair below resolves to a real decorated route — so a rename that
# silently un-guards a route fails a test instead of shipping.

_ENGINE_AND_DATA_ROUTES: Tuple[Tuple[str, str], ...] = (
    ("PATCH", "/api/admin/system-settings/{key}"),
    ("POST", "/api/r-session/heartbeat"),
    ("POST", "/api/bar-cache/add-ticker"),
    ("POST", "/api/bar-cache/targets"),
    ("DELETE", "/api/bar-cache/targets/{symbol}/{timeframe}"),
    ("POST", "/api/bar-cache/backfill"),
    ("POST", "/api/bar-cache/maintain"),
    ("POST", "/api/mass-builder/cleanup-orphans"),
    ("POST", "/api/update-jobs/cleanup-orphans"),
    ("POST", "/api/strategy-health-sim/{sid}/run"),
    ("PUT", "/api/strategy-health-sim/{sid}/optin"),
)

# §5.5 — the shared team board and session state (18 routes: dev_tasks 9,
# m_session 4, agents 2, mentions 2, strategy_notes 1 — §5.5's prose header says
# "17", but its own enumeration and the §5.6 list both come to 18; §5.6 is the
# authoritative list and is what this file mirrors). Blocked by default; see the
# module docstring for why these are separable.
BOARD_COORDINATION_ROUTES: Tuple[Tuple[str, str], ...] = (
    ("POST", "/api/agents"),
    ("PATCH", "/api/agents/{agent_id}"),
    ("POST", "/api/strategy-notes/{sid}"),
    ("POST", "/api/mentions/seen-all"),
    ("POST", "/api/mentions/{mention_id}/seen"),
    ("PUT", "/api/m-session/canvas"),
    ("PUT", "/api/m-session/order"),
    ("POST", "/api/m-session/marks"),
    ("DELETE", "/api/m-session/marks/{event_key:path}"),
    ("POST", "/api/dev-tasks"),
    ("PATCH", "/api/dev-tasks/{task_id}"),
    ("DELETE", "/api/dev-tasks/{task_id}"),
    ("POST", "/api/dev-tasks/{task_id}/run-request"),
    ("POST", "/api/dev-tasks/{task_id}/stamp"),
    ("POST", "/api/dev-tasks/{task_id}/steps/complete"),
    ("POST", "/api/dev-tasks/{task_id}/steps/raise-issue"),
    ("POST", "/api/dev-tasks/{task_id}/steps/stamp"),
    ("POST", "/api/dev-tasks/{task_id}/comments"),
)

FLEET_WRITE_ROUTES: Tuple[Tuple[str, str], ...] = (
    _ENGINE_AND_DATA_ROUTES + BOARD_COORDINATION_ROUTES
)

_BOARD_ROUTE_SET = frozenset(BOARD_COORDINATION_ROUTES)


# ── Path-template matching ──────────────────────────────────────────────────
#
# Self-contained rather than reaching into Starlette's compiled `path_regex`:
# the matcher is then exercisable in a test without constructing the app (and
# constructing the app is exactly what runs the boot sweeps this guards).

def _template_to_regex(template: str) -> "re.Pattern[str]":
    out = ["^"]
    for literal, param in _split_template(template):
        out.append(re.escape(literal))
        if param is not None:
            # `{name:path}` may span slashes; every other param may not.
            out.append(".+" if param.endswith(":path") else "[^/]+")
    out.append("/?$")
    return re.compile("".join(out))


def _split_template(template: str) -> Iterable[Tuple[str, Optional[str]]]:
    pos = 0
    for m in re.finditer(r"\{([^}]*)\}", template):
        yield template[pos:m.start()], m.group(1)
        pos = m.end()
    yield template[pos:], None


_COMPILED: Tuple[Tuple[str, str, "re.Pattern[str]"], ...] = tuple(
    (method, path, _template_to_regex(path)) for method, path in FLEET_WRITE_ROUTES
)


def blocked_route(method: str, path: str) -> Optional[Tuple[str, str]]:
    """Return the `(METHOD, template)` this request is blocked by, else None.

    Assumes the caller has already established that the role is guarded — it
    does NOT re-check, so production never reaches a single regex here.
    """
    method = (method or "").upper()
    board_ok = board_writes_allowed()
    for m, template, rx in _COMPILED:
        if m != method:
            continue
        if not rx.match(path or ""):
            continue
        if board_ok and (m, template) in _BOARD_ROUTE_SET:
            return None
        return (m, template)
    return None


def refusal_detail(method: str, template: str) -> str:
    """The 403 body. Names the role AND the route — a dev operator must be able
    to tell this apart from an ordinary auth failure without reading logs."""
    return (
        f"Refused by the environment-role write guard: {ENV_VAR}={env_role()!r}. "
        f"{method} {template} is a fleet-wide write against the SHARED production "
        f"database (it uses the admin client and ignores user scoping), so it is "
        f"blocked outside the production environment. "
        f"Nothing was written. See board #286 / docs/_active/Env_Write_Surface_Audit.md."
    )


# ── Boot sweeps ─────────────────────────────────────────────────────────────

def boot_sweep_blocked(name: str) -> bool:
    """True when a cross-user boot sweep must NOT run in this environment.

    `name` is the sweep's function name; it goes into the log line so the skip
    is attributable. Logged at WARNING, once per call, because a silently
    skipped sweep is its own confusing failure mode ("why is this row still
    running?") and the operator should see which environment made that choice.
    """
    if not is_write_guarded():
        return False
    logger.warning(
        "[env-role] SKIPPED cross-user boot sweep %s() — %s=%r. This sweep writes "
        "EVERY user's rows via the admin client; running it from a non-production "
        "environment against the shared database would mark production's live jobs "
        "as dead (board #286).",
        name, ENV_VAR, env_role(),
    )
    return True


def assert_write_allowed(label: str) -> None:
    """Raise `EnvRoleWriteBlocked` if fleet-wide writes are guarded here.

    For non-HTTP callers that would rather fail than skip.
    """
    if is_write_guarded():
        raise EnvRoleWriteBlocked(
            f"{label} is a fleet-wide write and {ENV_VAR}={env_role()!r} "
            f"(board #286)."
        )


# ── FastAPI wiring ──────────────────────────────────────────────────────────

def install_env_role_guard(app) -> None:
    """Attach the refusal middleware.

    Must be installed BEFORE the CORS middleware. Starlette builds the stack so
    that the LAST-added middleware is outermost; if this were added after CORS
    it would wrap CORS, and its 403 would come back without CORS headers — the
    browser would then show the dev UI an opaque network error instead of the
    message that explains what happened, which is the same lie the refusal
    exists to prevent.
    """
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def _env_role_write_guard(request, call_next):
        # Production fast path — one dict lookup, then out. No matching, no
        # regex, no behaviour change of any kind.
        if not is_write_guarded():
            return await call_next(request)

        hit = blocked_route(request.method, request.url.path)
        if hit is None:
            return await call_next(request)

        method, template = hit
        detail = refusal_detail(method, template)
        logger.warning("[env-role] REFUSED %s %s — %s", request.method,
                       request.url.path, detail)
        return JSONResponse(status_code=403, content={"detail": detail})

    return None
