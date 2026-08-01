"""Acceptance test — board #286: a dev API boot cannot mutate production.

The #285 audit found that `cleanup_orphaned_mass_searches()` and
`cleanup_orphaned_update_jobs()` are documented as "Called once on api boot",
use the ADMIN client, and flip EVERY user's `status='running'` rows to
`orphaned`. Under the shared-database dev environment (#165) that means
**starting the dev API is itself a fleet-wide production write** — nobody has to
click anything, and it repeats on every dev deploy.

`src/env_role.py` is the guard. Every rail below FAILS without its line of it —
several are mutation-checked rather than merely asserted, because the whole
value of this guard is the OFF path being untouched, and "I added a flag" is not
evidence of that.

  R1  the mass-search boot sweep DOES NOT reach the DB with role=dev
  R2  … and DOES with the role unset            ← production is unchanged
  R3  the update-jobs boot sweep DOES NOT reach the DB with role=dev
  R4  … and DOES with the role unset            ← production is unchanged
  R5  the api-boot call sites are structurally guarded (AST over main.py),
      including the THIRD sweep the audit missed (`_stale_parity_cleanup`)
  R6  a representative fleet-wide WRITE route REFUSES 403 with role=dev,
      naming the role and the route — over real HTTP, through the real
      middleware
  R7  … and the SAME route is untouched with the role unset
  R8  a write route that is NOT on the list still works under role=dev —
      the guard is a list, not a blanket write-block
  R9  all 29 §5.6 routes refuse under role=dev, and none of them refuses
      with the role unset
  R10 path-parameterised templates match real URLs, `{…:path}` included
  R11 role parsing: unset/blank/production/prod = production; anything else,
      typos included, guards (fail-safe)
  R12 the board carve-out (§5.5): RORT_ENV_ROLE_ALLOW_BOARD=1 re-permits the
      18 board routes and NOT the other 11
  R13 the 403 carries CORS headers — install order is guard-then-CORS, so the
      browser sees the message instead of an opaque network error
  R14 registry drift: every one of the 29 (METHOD, path) pairs resolves to a
      real decorated route in `src/api/routers/` source

Offline + hermetic: no DB, no network, no `create_app()` (constructing the real
app is precisely what runs the sweeps under test). HTTP rails go through a real
FastAPI app + TestClient with stub handlers mounted at the real paths, so
`install_env_role_guard` is genuinely exercised.

Run:  .venv/bin/python src/test_env_role_write_guard_286.py   (from RoR_Trader root)
"""
import ast
import importlib
import os
import re
import sys
import types

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


def set_role(value):
    """Set/unset RORT_ENV_ROLE. `None` = unset = production."""
    if value is None:
        os.environ.pop("RORT_ENV_ROLE", None)
    else:
        os.environ["RORT_ENV_ROLE"] = value


def set_allow_board(value):
    if value is None:
        os.environ.pop("RORT_ENV_ROLE_ALLOW_BOARD", None)
    else:
        os.environ["RORT_ENV_ROLE_ALLOW_BOARD"] = value


set_role(None)
set_allow_board(None)

import env_role  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# R11 — role parsing, and the fail-safe on an unknown value
# ══════════════════════════════════════════════════════════════════════════

for raw in (None, "", "   ", "production", "PRODUCTION", "prod", " Prod "):
    set_role(raw)
    ok(f"R11 role={raw!r} → production, unguarded",
       env_role.env_role() == "production" and not env_role.is_write_guarded(),
       f"got {env_role.env_role()!r}")

for raw in ("dev", "DEV", " dev ", "development", "staging", "prodcution"):
    set_role(raw)
    ok(f"R11 role={raw!r} → guarded (fail-safe)",
       env_role.is_write_guarded() and env_role.env_role() == raw.strip().lower())

# Runtime-read, not import-time: the module was imported with the role unset
# and still sees every change above without a reload.
set_role("dev")
ok("R11 runtime-read (no reload needed)", env_role.is_write_guarded())
set_role(None)
ok("R11 runtime-read flips back", not env_role.is_write_guarded())


# ══════════════════════════════════════════════════════════════════════════
# Fake `db` — records whether a sweep reached the database at all.
# ══════════════════════════════════════════════════════════════════════════

DB_CALLS = []


class _Chain:
    """Permissive PostgREST-ish stub: any method returns self, execute() → []."""

    def __getattr__(self, _name):
        def _call(*_a, **_kw):
            return self
        return _call

    def execute(self):
        return types.SimpleNamespace(data=[])


def _install_fake_db():
    fake = types.ModuleType("db")
    fake.USE_DB = True

    def get_admin_client():
        DB_CALLS.append("get_admin_client")
        return _Chain()

    def load_orphaned_update_jobs():
        DB_CALLS.append("load_orphaned_update_jobs")
        return []

    def mark_update_job_orphaned(*_a, **_kw):
        DB_CALLS.append("mark_update_job_orphaned")

    def update_mass_search(*_a, **_kw):
        DB_CALLS.append("update_mass_search")

    fake.get_admin_client = get_admin_client
    fake.load_orphaned_update_jobs = load_orphaned_update_jobs
    fake.mark_update_job_orphaned = mark_update_job_orphaned
    fake.update_mass_search = update_mass_search
    sys.modules["db"] = fake
    return fake


_install_fake_db()

import mass_builder  # noqa: E402
import update_jobs  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# R1–R4 — the boot sweeps. The proof of "did not run" is that the sweep
# never touched the DB, not that it returned a particular dict.
# ══════════════════════════════════════════════════════════════════════════

set_role("dev")
DB_CALLS.clear()
summary = mass_builder.cleanup_orphaned_mass_searches()
ok("R1 mass-search sweep touched NO DB with role=dev", DB_CALLS == [], f"calls={DB_CALLS}")
ok("R1 … and says so in its summary", summary.get("skipped_env_role") is True, f"{summary}")
ok("R1 … and orphaned nothing", summary.get("orphaned") == 0 and summary.get("resumed") == 0)

set_role(None)
DB_CALLS.clear()
mass_builder.cleanup_orphaned_mass_searches()
ok("R2 mass-search sweep DOES reach the DB with the role unset",
   "get_admin_client" in DB_CALLS, f"calls={DB_CALLS}")
ok("R2 … and reports no env-role skip", "skipped_env_role" not in
   mass_builder.cleanup_orphaned_mass_searches())

set_role("dev")
DB_CALLS.clear()
summary = update_jobs.cleanup_orphaned_update_jobs()
ok("R3 update-jobs sweep touched NO DB with role=dev", DB_CALLS == [], f"calls={DB_CALLS}")
ok("R3 … and says so in its summary", summary.get("skipped_env_role") is True, f"{summary}")

set_role(None)
DB_CALLS.clear()
update_jobs.cleanup_orphaned_update_jobs()
ok("R4 update-jobs sweep DOES reach the DB with the role unset",
   "load_orphaned_update_jobs" in DB_CALLS, f"calls={DB_CALLS}")


# ══════════════════════════════════════════════════════════════════════════
# R5 — the api-boot call sites, checked structurally (AST). Importing
# api.main and calling create_app() would run the very sweeps under test.
# ══════════════════════════════════════════════════════════════════════════

MAIN_SRC = open(os.path.join(SRC, "api", "main.py"), encoding="utf-8").read()
MAIN_TREE = ast.parse(MAIN_SRC)


def _find_funcdef(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _calls_named(node, name):
    """Every ast.Call to a bare function `name` inside `node`."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                and sub.func.id == name:
            out.append(sub)
    return out


startup = _find_funcdef(MAIN_TREE, "_startup_cleanup")
ok("R5 _startup_cleanup exists", startup is not None)

for sweep in ("cleanup_orphaned_mass_searches", "cleanup_orphaned_update_jobs"):
    # Every call to the sweep must sit inside an `if not boot_sweep_blocked(...)`.
    guarded = False
    for stmt in startup.body:
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
            continue
        if not _calls_named(test, "boot_sweep_blocked"):
            continue
        if any(_calls_named(s, sweep) for s in stmt.body):
            guarded = True
    ok(f"R5 boot call site for {sweep} is inside `if not boot_sweep_blocked(...)`",
       guarded)
    # …and nowhere else in the boot function.
    outside = [s for s in startup.body
               if not isinstance(s, ast.If) and _calls_named(s, sweep)]
    ok(f"R5 … and {sweep} is not ALSO called unguarded", outside == [])

parity = _find_funcdef(MAIN_TREE, "_stale_parity_cleanup")
ok("R5 _stale_parity_cleanup exists (the third sweep, missed by #285)",
   parity is not None)
first = parity.body[0]
ok("R5 _stale_parity_cleanup returns early on boot_sweep_blocked",
   isinstance(first, ast.If) and _calls_named(first.test, "boot_sweep_blocked")
   and any(isinstance(s, ast.Return) for s in first.body))

# Install order: the guard must be added BEFORE CORS so CORS ends up outermost.
guard_at = MAIN_SRC.find("install_env_role_guard(app)")
cors_at = MAIN_SRC.find("CORSMiddleware,")
ok("R5 guard is installed before CORS in create_app()",
   0 < guard_at < cors_at, f"guard@{guard_at} cors@{cors_at}")


# ══════════════════════════════════════════════════════════════════════════
# HTTP rails — a real FastAPI app with stub handlers at the real paths.
# ══════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# One control route that WRITES but is not on the fleet-wide list.
CONTROL_ROUTE = ("POST", "/api/mass-builder/run")


def build_app():
    app = FastAPI(redirect_slashes=False)
    env_role.install_env_role_guard(app)          # same order as api/main.py
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    routes = list(env_role.FLEET_WRITE_ROUTES) + [CONTROL_ROUTE]
    for method, path in routes:
        # `{k:path}` is a Starlette converter and is valid in the decorator.
        app.add_api_route(
            path,
            lambda: {"ran": True},
            methods=[method],
            name=f"stub_{method}_{path}",
        )
    return app


CLIENT = TestClient(build_app())


def call(method, path, **kw):
    return CLIENT.request(method, path, **kw)


def concrete(path):
    """Fill a path template with plausible values so it matches a real URL."""
    out = re.sub(r"\{[^}]*:path\}", "some/nested/key", path)
    return re.sub(r"\{[^}]*\}", "42", out)


REPRESENTATIVE = ("POST", "/api/bar-cache/add-ticker")

set_role("dev")
r = call(*REPRESENTATIVE)
ok("R6 representative fleet-write route refuses with role=dev", r.status_code == 403,
   f"got {r.status_code}")
body = r.json().get("detail", "")
ok("R6 … the 403 names the env role", "RORT_ENV_ROLE='dev'" in body, body)
ok("R6 … names the route", "/api/bar-cache/add-ticker" in body, body)
ok("R6 … says nothing was written", "Nothing was written" in body, body)
ok("R6 … it REFUSED rather than no-op'd (handler never ran)",
   "ran" not in r.text)

set_role(None)
r = call(*REPRESENTATIVE)
ok("R7 the same route is untouched with the role unset",
   r.status_code == 200 and r.json() == {"ran": True}, f"{r.status_code} {r.text}")

set_role("dev")
r = call(*CONTROL_ROUTE)
ok("R8 a NON-listed write route still works under role=dev",
   r.status_code == 200 and r.json() == {"ran": True}, f"{r.status_code} {r.text}")

# R9 — all 29, both directions.
set_role("dev")
set_allow_board(None)
refused = []
for method, path in env_role.FLEET_WRITE_ROUTES:
    resp = call(method, concrete(path))
    if resp.status_code != 403:
        refused.append((method, path, resp.status_code))
ok(f"R9 all {len(env_role.FLEET_WRITE_ROUTES)} §5.6 routes refuse under role=dev",
   refused == [], f"not refused: {refused}")

set_role(None)
leaked = []
for method, path in env_role.FLEET_WRITE_ROUTES:
    resp = call(method, concrete(path))
    if resp.status_code != 200:
        leaked.append((method, path, resp.status_code))
ok("R9 …and NONE of them refuses with the role unset (production unchanged)",
   leaked == [], f"unexpectedly blocked: {leaked}")

ok("R9 the list is exactly 29 routes", len(env_role.FLEET_WRITE_ROUTES) == 29,
   str(len(env_role.FLEET_WRITE_ROUTES)))

# R10 — parameterised templates, including the `:path` converter.
set_role("dev")
for method, url in (
    ("POST", "/api/dev-tasks/286/steps/complete"),
    ("DELETE", "/api/bar-cache/targets/SPY/1Min"),
    ("PATCH", "/api/admin/system-settings/RORT_MASS_RECOVER_QUEUED"),
    ("DELETE", "/api/m-session/marks/deploy/2026-07-31/wave38"),
    ("PUT", "/api/strategy-health-sim/339/optin"),
):
    ok(f"R10 {method} {url} matches its template", call(method, url).status_code == 403)

ok("R10 a sibling path does NOT match by accident",
   env_role.blocked_route("POST", "/api/bar-cache/add-ticker-please") is None)
ok("R10 a single-segment param does not swallow a slash",
   env_role.blocked_route("PATCH", "/api/admin/system-settings/a/b") is None)

# R12 — the board carve-out.
set_role("dev")
set_allow_board("1")
board_leaks = []
for method, path in env_role.BOARD_COORDINATION_ROUTES:
    resp = call(method, concrete(path))
    if resp.status_code != 200:
        board_leaks.append((method, path, resp.status_code))
ok("R12 ALLOW_BOARD=1 re-permits all 18 board routes", board_leaks == [],
   f"still blocked: {board_leaks}")

non_board = [r for r in env_role.FLEET_WRITE_ROUTES
             if r not in set(env_role.BOARD_COORDINATION_ROUTES)]
ok("R12 the non-board remainder is 11", len(non_board) == 11, str(len(non_board)))
still_blocked = all(call(m, concrete(p)).status_code == 403 for m, p in non_board)
ok("R12 …and ALLOW_BOARD does NOT re-permit the other 11", still_blocked)

set_allow_board(None)
ok("R12 board routes block again once ALLOW_BOARD is unset",
   call("POST", "/api/dev-tasks/286/comments").status_code == 403)

# R13 — CORS headers survive the refusal.
set_role("dev")
r = call("POST", "/api/bar-cache/add-ticker",
         headers={"Origin": "http://localhost:3000"})
ok("R13 the 403 still carries CORS headers (guard installed inside CORS)",
   r.status_code == 403
   and r.headers.get("access-control-allow-origin") == "http://localhost:3000",
   dict(r.headers))


# ══════════════════════════════════════════════════════════════════════════
# R14 — registry drift: every listed route must exist in the real routers.
# ══════════════════════════════════════════════════════════════════════════

ROUTERS_DIR = os.path.join(SRC, "api", "routers")
_PREFIX_RX = re.compile(r"APIRouter\(\s*prefix=[\"']([^\"']*)[\"']")
_DECOR_RX = re.compile(r"@router\.(get|post|put|patch|delete)\(\s*[\"']([^\"']*)[\"']")

real_routes = set()
for fname in sorted(os.listdir(ROUTERS_DIR)):
    if not fname.endswith(".py"):
        continue
    text = open(os.path.join(ROUTERS_DIR, fname), encoding="utf-8").read()
    m = _PREFIX_RX.search(text)
    prefix = m.group(1) if m else ""
    for meth, path in _DECOR_RX.findall(text):
        real_routes.add((meth.upper(), (prefix + path) or "/"))

missing = [r for r in env_role.FLEET_WRITE_ROUTES if r not in real_routes]
ok("R14 every guarded route exists in api/routers/ source", missing == [],
   f"missing: {missing}")

# The two cleanup-orphans routes are the reason this task exists — name them.
for r in (("POST", "/api/mass-builder/cleanup-orphans"),
          ("POST", "/api/update-jobs/cleanup-orphans")):
    ok(f"R14 {r[0]} {r[1]} is both real and guarded",
       r in real_routes and r in env_role.FLEET_WRITE_ROUTES)

set_role(None)
set_allow_board(None)

print(f"\nPASS — {PASS} checks")
sys.exit(0)
