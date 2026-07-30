"""Acceptance test — board #217: a headless run can tick its step, and nothing
else got easier to reach.

The #217 diagnosis established that EVERY HTTP `POST /steps/complete` a
dispatched run has ever made was refused `401 Missing Bearer token` — the route
gated on a Supabase *user* JWT, which a headless agent has no way to obtain.
The fix opts the actor-attributed step routes into `get_service_or_user`, which
also accepts the service-role key.

The whole risk of that fix is that it accidentally opens more than those
routes, so this file is deliberately half proof-it-works, half proof-it-still-
refuses. Every rail here fails without its line of the fix:

  R1  service-role bearer TICKS       — /steps/complete over real HTTP
  R2  … with full handler fidelity    — completed_at/by + hand-off + comment
  R3  no bearer is still 401          — the gate did not become optional
  R4  a WRONG bearer is still 401     — we compare the key, not merely its shape
  R5  ordinary routes still 401       — get_current_user was NOT widened
  R6  /steps/stamp still 401          — Kevin's approval is not agent-reachable
  R7  unset service key FAILS CLOSED  — no key ⇒ no bearer can ever match
  R8  /steps/raise-issue accepts it   — the "I could not finish" path works too
  R9  dispatcher renders an ABSOLUTE, credentialed call — no host to guess
  R10 spawn() exports host + key      — the run does not hunt for src/.env

Requests go through a real FastAPI app with TestClient, so `Depends(...)` is
actually exercised — an in-process function call would prove nothing here,
since bypassing ASGI is precisely how the two 07-29 "successes" happened.
Offline + hermetic: fake Supabase, no DB, no network.

Run:  .venv/bin/python src/test_steps_actor_auth_217.py   (from RoR_Trader root)
"""
import base64
import copy
import importlib
import json
import os
import sys
import time
import types

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


# ── In-memory fake Supabase (same subset as test_process_chain_182.py) ──────

class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, tname):
        self.store, self.tname = store, tname
        self.op, self.payload, self.filters = "select", None, []

    def select(self, *_):
        self.op = "select"
        return self

    def insert(self, row):
        self.op = "insert"
        self.payload = [dict(r) for r in row] if isinstance(row, list) \
            else [dict(row)]
        return self

    def update(self, row):
        self.op, self.payload = "update", dict(row)
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def order(self, *_, **__):
        return self

    def limit(self, *_):
        return self

    def _match(self, row):
        return all(row.get(k) == v for k, v in self.filters)

    def execute(self):
        rows = self.store.setdefault(self.tname, [])
        if self.op == "insert":
            out = []
            for payload in self.payload:
                row = dict(payload)
                row["id"] = self.store["_seq"] = self.store.get("_seq", 0) + 1
                rows.append(row)
                out.append(copy.deepcopy(row))
            return _Res(out)
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(copy.deepcopy(self.payload))
            return _Res([copy.deepcopy(r) for r in hit])
        return _Res([copy.deepcopy(r) for r in rows if self._match(r)])


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return _Query(self.store, name)


FAKE = FakeSupabase()
FAKE.store["agents"] = [{"letter": ltr} for ltr in ("M", "E", "F", "P", "R")]
db_stub = types.ModuleType("db")
db_stub.set_current_user = lambda *a, **k: None
db_stub.get_admin_client = lambda: FAKE
sys.modules["db"] = db_stub

# The service key is JWT-SHAPED but carries no `sub` — exactly like the real
# Supabase service_role key. That shape matters: a garbage string would make
# R5 pass for the wrong reason (rejected as malformed rather than as
# subject-less), so the "ordinary routes still refuse it" rail is tested with
# the same token the step routes accept.
SERVICE_KEY = "eyJhbGciOiJIUzI1NiJ9." + base64.urlsafe_b64encode(
    json.dumps({"role": "service_role",
                "exp": int(time.time()) + 10 * 365 * 86400}).encode()
).decode().rstrip("=") + ".sig"

os.environ["SUPABASE_SERVICE_ROLE_KEY"] = SERVICE_KEY
os.environ["DEV_BYPASS_AUTH"] = "false"   # must be OFF or nothing is proven

from api import deps                                    # noqa: E402
importlib.reload(deps)                                  # pick up the env above
ok("DEV_BYPASS_AUTH is OFF for this run (else every rail is vacuous)",
   deps._DEV_BYPASS is False)

from api.routers import dev_tasks as dt                 # noqa: E402
importlib.reload(dt)                                    # rebind to reloaded deps

from fastapi import FastAPI                             # noqa: E402
from fastapi.testclient import TestClient               # noqa: E402

app = FastAPI()
app.include_router(dt.router)
client = TestClient(app)

SVC = {"Authorization": f"Bearer {SERVICE_KEY}"}


def make_task(assignee="F"):
    """A two-step chain: F builds, M closes."""
    row = dt.create_task({
        "title": "chain under test", "assignee": assignee,
        "checklist": [
            {"owner": "F", "title": "Build the thing", "mode": "execute",
             "done": False},
            {"owner": "M", "title": "Close", "mode": "execute",
             "done": False},
        ]}, user=None)
    return row["id"]


def comments_for(tid):
    return [c["body"] for c in FAKE.store.get("dev_task_comments", [])
            if c["task_id"] == tid]


# ── R1/R2: the headless-shaped call now succeeds, with full fidelity ────────
print("― R1/R2: service-role bearer ticks the step, server-owned logic runs ―")
tid = make_task()
r = client.post(f"/api/dev-tasks/{tid}/steps/complete",
                json={"actor": "M·auto"}, headers=SVC)
ok("HTTP POST /steps/complete with the service key → 200 (was 401)",
   r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
body = r.json()
step0 = body["checklist"][0]
ok("step 1 is ticked", step0["done"] is True)
ok("completed_by records the ACTOR, not the credential",
   step0["completed_by"] == "M·auto", f"got {step0.get('completed_by')}")
ok("completed_at is stamped (a PostgREST PATCH would have skipped it)",
   bool(step0.get("completed_at")))
ok("hand-off ran: assignee F → M (the next step's owner)",
   body["assignee"] == "M", f"got {body['assignee']}")
ok("system comment posted with the hand-off",
   any("complete (by M·auto)" in b and "F → M" in b for b in comments_for(tid)),
   f"got {comments_for(tid)}")

# ── R3: an unauthenticated caller is still refused ─────────────────────────
print("― R3: no bearer is still refused ―")
tid2 = make_task()
r = client.post(f"/api/dev-tasks/{tid2}/steps/complete", json={"actor": "M·auto"})
ok("no Authorization header → 401", r.status_code == 401,
   f"got {r.status_code}: {r.text[:200]}")
ok("… with the original message", r.json().get("detail") == "Missing Bearer token")
ok("… and the step is UNTOUCHED",
   dt._task_row(FAKE, tid2)["checklist"][0]["done"] is False)

# ── R4: a wrong bearer is still refused (we compare the key, not the shape) ─
print("― R4: a wrong bearer is still refused ―")
wrong = "eyJhbGciOiJIUzI1NiJ9." + base64.urlsafe_b64encode(
    json.dumps({"role": "service_role",
                "exp": int(time.time()) + 3600}).encode()
).decode().rstrip("=") + ".sig"
ok("the decoy is JWT-shaped and NOT the key", wrong != SERVICE_KEY)
r = client.post(f"/api/dev-tasks/{tid2}/steps/complete",
                json={"actor": "M·auto"},
                headers={"Authorization": f"Bearer {wrong}"})
ok("a look-alike service token → 401", r.status_code == 401,
   f"got {r.status_code}: {r.text[:200]}")
r = client.post(f"/api/dev-tasks/{tid2}/steps/complete",
                json={"actor": "M·auto"},
                headers={"Authorization": "Bearer "})
ok("an EMPTY bearer → 401", r.status_code == 401, f"got {r.status_code}")
ok("… and the step is STILL untouched",
   dt._task_row(FAKE, tid2)["checklist"][0]["done"] is False)

# ── R5: the shared user gate was NOT widened ───────────────────────────────
print("― R5: the service key opens ONLY the opted-in routes ―")
r = client.get("/api/dev-tasks", headers=SVC)
ok("GET /api/dev-tasks with the service key → 401 (list is user-gated)",
   r.status_code == 401, f"got {r.status_code}: {r.text[:200]}")
r = client.patch(f"/api/dev-tasks/{tid2}", json={"assignee": "M"}, headers=SVC)
ok("PATCH /api/dev-tasks/<id> with the service key → 401",
   r.status_code == 401, f"got {r.status_code}: {r.text[:200]}")
r = client.get(f"/api/dev-tasks/{tid2}/comments", headers=SVC)
ok("GET …/comments with the service key → 401",
   r.status_code == 401, f"got {r.status_code}: {r.text[:200]}")

# ── R6: Kevin's stamp is NOT agent-reachable ───────────────────────────────
print("― R6: /steps/stamp stays on the user gate (approval ≠ deliverable) ―")
r = client.post(f"/api/dev-tasks/{tid2}/steps/stamp",
                json={"actor": "kevin", "decision": "approve"}, headers=SVC)
ok("a service-key caller cannot manufacture an approval → 401",
   r.status_code == 401, f"got {r.status_code}: {r.text[:200]}")

# ── R7: an unconfigured service key fails CLOSED ───────────────────────────
print("― R7: no configured key ⇒ no bearer can match ―")
_saved = os.environ.pop("SUPABASE_SERVICE_ROLE_KEY")
try:
    ok("_service_role_key() is empty when unset", deps._service_role_key() == "")
    for hdr, label in ((SVC, "the real key"),
                       ({"Authorization": "Bearer "}, "an empty bearer"),
                       ({"Authorization": "Bearer x"}, "any bearer")):
        r = client.post(f"/api/dev-tasks/{tid2}/steps/complete",
                        json={"actor": "M·auto"}, headers=hdr)
        ok(f"key unset + {label} → 401", r.status_code == 401,
           f"got {r.status_code}: {r.text[:200]}")
finally:
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = _saved
ok("the key is restored and accepted again",
   client.post(f"/api/dev-tasks/{tid2}/steps/complete",
               json={"actor": "M·auto"}, headers=SVC).status_code == 200)

# ── R8: raise-issue — the "I could not finish" path — works headless too ───
print("― R8: /steps/raise-issue accepts the service key, refuses without ―")
tid3 = make_task()
r = client.post(f"/api/dev-tasks/{tid3}/steps/raise-issue",
                json={"actor": "M·auto", "body": "⚠️ Raise-an-issue: bad hand-off"})
ok("raise-issue with no bearer → 401", r.status_code == 401,
   f"got {r.status_code}")
r = client.post(f"/api/dev-tasks/{tid3}/steps/raise-issue",
                json={"actor": "M·auto", "body": "⚠️ Raise-an-issue: bad hand-off"},
                headers=SVC)
ok("raise-issue with the service key → 200", r.status_code == 200,
   f"got {r.status_code}: {r.text[:200]}")
ok("… routed to M without ticking anything",
   r.json()["assignee"] == "M" and r.json()["checklist"][0]["done"] is False)

# ── R9/R10: the dispatcher no longer makes the run guess ───────────────────
print("― R9/R10: absolute, pre-credentialed call in the prompt + run env ―")
sys.path.insert(0, os.path.join(os.path.dirname(SRC), "tools", "team_dispatcher"))
import dispatcher as D                                  # noqa: E402

ok("BOARD_API_URL is an absolute https host",
   D.BOARD_API_URL.startswith("https://") and "://" in D.BOARD_API_URL,
   D.BOARD_API_URL)

chain_task = {
    "id": 217, "title": "t", "description": "d", "assignee": "M",
    "status": "In Progress", "impact": "contained",
    "checklist": [
        {"owner": "M", "title": "Do the thing", "body": "SOP", "done": False},
        {"owner": "R", "title": "Ship it", "body": "SOP", "done": False},
    ],
}
agent = {"letter": "M", "scope": "docs", "boundaries": "none", "worktree": D.REPO}
p1 = D.build_prompt(agent, chain_task, [])
p2 = D.build_prompt(agent, chain_task, [])
ok("the rendered tick call is IDENTICAL across builds (deterministic)", p1 == p2)
ok("the tick call is absolute — it carries the host",
   f"{D.BOARD_API_URL}/api/dev-tasks/217/steps/complete" in p1
   or "$RORT_BOARD_API_URL/api/dev-tasks/217/steps/complete" in p1, p1[-1500:])
ok("the host is named in the prompt so it is never guessed",
   D.BOARD_API_URL in p1)
ok("the call carries the credential the route now accepts",
   "Authorization: Bearer $RORT_BOARD_SERVICE_KEY" in p1)
# Scoped to the GENERATED TICK BLOCK, not the whole prompt. `build_prompt` embeds
# the task's LIVE comment thread (dispatcher.py:778), so asserting across all of p1
# means any human or agent who quotes the old bare-path string in a #217 comment
# fails this check. That is exactly what happened on 07-30: M's review comment
# explained the fix by quoting `POST /api/dev-tasks/217/steps/complete`, and this
# assertion went red on a REBASE with no code change behind it.
# The rail is about what the dispatcher RENDERS, so it is asserted there.
_tick = p1[p1.index("STEP-TICK CONTRACT"):] if "STEP-TICK CONTRACT" in p1 else ""
ok("the tick block exists to assert against", bool(_tick))
ok("no bare hostless POST line survives IN THE TICK BLOCK",
   "POST /api/dev-tasks/217/steps/complete" not in _tick, _tick[:400])

env = D.run_env()
ok("run_env() exports the API host", env.get("RORT_BOARD_API_URL") == D.BOARD_API_URL)
ok("run_env() exports a service key", bool(env.get("RORT_BOARD_SERVICE_KEY")))
ok("run_env() INHERITS the dispatcher environment (not a bare env)",
   all(k in env for k in os.environ if k not in
       ("RORT_BOARD_API_URL", "RORT_BOARD_SERVICE_KEY")))

# … and spawn() must actually HAND it to the child. run_env() being correct
# proves nothing if Popen never receives it, which is how the gap existed.
captured = {}


class _FakePopen:
    def __init__(self, argv, **kw):
        captured["argv"], captured["kw"] = argv, kw
        self.pid = 1


_real_popen = D.subprocess.Popen
D.subprocess.Popen = _FakePopen
_log = os.path.join(os.environ.get("TMPDIR", "/tmp"), "_217_spawn_probe.log")
try:
    D.spawn(agent, chain_task, "prompt", _log)
finally:
    D.subprocess.Popen = _real_popen
    if os.path.exists(_log):
        os.remove(_log)
child_env = captured["kw"].get("env") or {}
ok("spawn() passes env= to the child at all", bool(child_env))
ok("the child inherits the host", child_env.get("RORT_BOARD_API_URL") == D.BOARD_API_URL)
ok("the child inherits the credential", bool(child_env.get("RORT_BOARD_SERVICE_KEY")))
ok("the child still inherits PATH (env= did not blank the environment)",
   child_env.get("PATH") == os.environ.get("PATH"))

print(f"\nALL {PASS} CHECKS PASSED — #217 step 3")
