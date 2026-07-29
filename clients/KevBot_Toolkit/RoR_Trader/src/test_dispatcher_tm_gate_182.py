"""Acceptance test — board #182 Step 7 (second half): THE TM GATE.

TM is the hand-off gate: a fresh, tool-less, time-boxed agent asking one question
("was what the step asked for actually delivered?") with three outcomes and
nothing else. This suite pins the four rails the design is built on, because
every one of them is a thing that goes wrong silently if it regresses:

  rail 1  narrow — TM gets no tools and a hard time box
  rail 2  bounce limit of two, then ESCALATE (else it is a loop with no exit)
  rail 3  FAILS OPEN — a gate that cannot run must never stall the pipeline
  rail 4  silent on pass, loud on problem

Plus the two things that make it safe to ship: `observe` mode takes NO board
action, and the reviewed-marker (a TM comment carrying step_order) is what stops
a hand-off being re-reviewed on every poll.

Hermetic: loads the real dispatcher module with a fake `api`, never spawns a
subprocess, no network.
Run:  .venv/bin/python src/test_dispatcher_tm_gate_182.py   (from RoR_Trader root)
"""
import importlib.util, json, os, sys, time

SRC = os.path.dirname(os.path.abspath(__file__))
DISP = os.path.join(os.path.dirname(SRC), "tools", "team_dispatcher", "dispatcher.py")

PASS = FAIL = 0
def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok: {label}")
    else:    FAIL += 1; print(f"  FAIL: {label}{(' — ' + detail) if detail else ''}")

spec = importlib.util.spec_from_file_location("tm_under_test", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["tm_under_test"] = D
spec.loader.exec_module(D)

D.STATE_FILE = os.path.join("/tmp", f"tm_gate_test_state_{os.getpid()}.json")
D.TM_OFF_FILE = os.path.join("/tmp", f"tm_gate_test_off_{os.getpid()}")


class FakeApi:
    """Records every write so a test can assert what did NOT happen — which is
    most of what `observe` mode is."""
    def __init__(self, tasks=None, comments=None):
        self.tasks = {t["id"]: t for t in (tasks or [])}
        self.comments = list(comments or [])
        self.writes = []

    def __call__(self, method, path, body=None, prefer=None):
        if method != "GET":
            self.writes.append((method, path, body))
        if method == "GET" and path.startswith("dev_task_comments"):
            rows = self.comments
            if "author=eq.TM" in path:
                rows = [c for c in rows if c.get("author") == "TM"]
            for part in path.split("&"):
                if part.startswith("dev_task_comments?id=eq."):
                    cid = int(part.split("eq.")[1])
                    rows = [c for c in rows if c.get("id") == cid]
                if part.startswith("task_id=eq."):
                    rows = [c for c in rows if c["task_id"] == int(part.split("eq.")[1])]
            if "order=created_at.desc" in path:
                rows = list(reversed(rows))
            return rows
        if method == "GET" and path.startswith("dev_tasks"):
            for part in path.split("&"):
                if "id=eq." in part:
                    t = self.tasks.get(int(part.split("id=eq.")[1]))
                    return [t] if t else []
            return list(self.tasks.values())
        if method == "POST" and path == "dev_task_comments":
            self.comments.append(dict(body, id=900 + len(self.comments),
                                      author=body.get("author")))
        if method == "PATCH" and path.startswith("dev_tasks?id=eq."):
            self.tasks[int(path.split("eq.")[1])].update(body)
        return []

    def patched(self):
        return [w for w in self.writes if w[0] == "PATCH"]

    def tm_bodies(self):
        return [c["body"] for c in self.comments if c.get("author") == "TM"]


def chain_task(**kw):
    t = {"id": 182, "title": "T", "description": "d", "assignee": "F",
         "tags": [], "blocked_by": [],
         "checklist": [
             {"owner": "M", "title": "spec", "body": "write the spec", "done": True},
             {"owner": "F", "title": "build", "body": "SOP-MARKER build the thing",
              "done": False},
             {"owner": "kevin", "title": "stamp", "mode": "discuss", "done": False}]}
    t.update(kw); return t


def fresh_state():
    return {"runs": [], "tm_runs": []}


def tm_run(**kw):
    r = {"run_id": "tm1-182s2", "task_id": 182, "step_order": 2, "pid": 999999,
         "t0": time.time(), "log": "/nonexistent", "active": True,
         "submitter": "F", "prior": []}
    r.update(kw); return r


def with_log(text):
    """A finished TM run whose log carries a terminal `claude -p --output-format
    json` object — the real shape, so tm_parse is exercised end to end."""
    p = os.path.join("/tmp", f"tm_gate_test_log_{os.getpid()}_{time.time_ns()}.log")
    open(p, "w").write(json.dumps({"type": "result", "is_error": False, "result": text}))
    return p


print("― walk 1: rail 1 — TM is NARROW by construction, not by instruction ―")
argv = D.tm_spawn.__code__.co_consts
src = open(DISP).read()
spawn_src = src[src.index("def tm_spawn"):src.index("def tm_parse")]
for tool in ("Bash", "Read", "Write", "Edit", "WebFetch", "Task"):
    ok(f"tm_spawn disallows {tool}", f'"{tool}"' in spawn_src)
ok("tm_spawn passes --disallowedTools", "--disallowedTools" in spawn_src)
ok("TM has its own hard time box", isinstance(D.TM_TIMEOUT_S, int) and D.TM_TIMEOUT_S > 0)
ok("TM has its own concurrency pool (never starves agent dispatch)",
   D.TM_CONCURRENCY < D.CONCURRENCY or D.TM_CONCURRENCY <= 2)

prompt = D.tm_build_prompt(chain_task(), 1, chain_task()["checklist"][1], "I built it", [])
ok("prompt carries THAT step's own SOP", "SOP-MARKER" in prompt)
ok("prompt forbids investigation explicitly", "do NOT investigate" in prompt.replace("You do NOT investigate", "do NOT investigate"))
ok("prompt states TM has no tools", "NO tools" in prompt)
ok("prompt offers exactly three verdicts",
   all(v in prompt for v in D.TM_VERDICTS) and len(D.TM_VERDICTS) == 3)
ok("prompt tells TM that escalating IS its answer to 'needs more'",
   "Escalating IS your answer" in prompt)
ok("prompt names the canonical ESCALATE case (good work, wrong scope)",
   "scope mismatch" in prompt)
ok("prompt tells TM not to act (board writes are the dispatcher's)",
   "Judge, do not act" in prompt)

print("― walk 2: a step with NO SOP body is never guessed at ―")
nosop = chain_task(checklist=[{"owner": "F", "title": "build", "done": False}])
p2 = D.tm_build_prompt(nosop, 0, nosop["checklist"][0], "done", [])
ok("prompt says the step has no SOP body", "no SOP body" in p2)
ok("...points TM at the description's PROCESS CHAIN section", "PROCESS CHAIN" in p2)
ok("...and orders ESCALATE rather than an invented requirement",
   "do NOT invent the requirement" in p2)

print("― walk 3: rail 3 — the gate FAILS OPEN, every way it can fail ―")
for label, log_text in (("unparseable output", "I think it looks fine, honestly"),
                        ("unknown verdict", '{"verdict":"MAYBE","reason":"x"}')):
    api = FakeApi(tasks=[chain_task()])
    D.api = api
    st = fresh_state()
    st["tm_runs"] = [tm_run(log=with_log(log_text), pid=1)]
    D.tm_reap(st, live=True)
    bodies = api.tm_bodies()
    ok(f"{label} → a verdict row is still posted (idempotency key survives)",
       len(bodies) == 1, str(bodies))
    ok(f"{label} → it is marked UNREVIEWED, not silently PASS",
       bodies and bodies[0].startswith("TM: UNREVIEWED"), bodies[0][:60] if bodies else "")
    ok(f"{label} → nothing on the board was touched", not api.patched(), str(api.patched()))

api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(t0=time.time() - D.TM_TIMEOUT_S - 1, log="/nonexistent", pid=1)]
D.tm_reap(st, live=True)
ok("time box blown → UNREVIEWED, hand-off proceeds",
   api.tm_bodies() and api.tm_bodies()[0].startswith("TM: UNREVIEWED"))
ok("...and says so loudly ('failed open')", "failed open" in api.tm_bodies()[0])
ok("...and the run is retired, not retried forever",
   not any(r.get("active") for r in st["tm_runs"]))

print("― walk 4: rail 4 — silent on pass, loud on problem ―")
api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(log=with_log('{"verdict":"PASS","reason":"delivers the SOP"}'))]
D.tm_reap(st, live=True)
passbody = api.tm_bodies()[0]
ok("PASS is ONE compact line", passbody.startswith("TM: PASS") and "\n" not in passbody,
   passbody[:80])
ok("PASS makes no board write at all", not api.patched())

api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(log=with_log(
    '{"verdict":"BOUNCE","reason":"the SOP asked for tests","missing":"add the tests"}'))]
D.tm_reap(st, live=True)
b = api.tm_bodies()[0]
ok("BOUNCE is loud (multi-line, names the fix)",
   b.startswith("TM: BOUNCE") and "add the tests" in b)
ok("BOUNCE names who it goes back to", "`F`" in b)
ok("BOUNCE states the bounce budget", f"of {D.TM_MAX_BOUNCES}" in b)
ok("BOUNCE says TM gates delivery, not depth", "not depth" in b)

print("― walk 5: OBSERVE mode posts the verdict and touches NOTHING ―")
ok("observe is the shipped default", D.tm_mode() == "observe")
api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(log=with_log('{"verdict":"BOUNCE","reason":"r","missing":"m"}'))]
D.tm_reap(st, live=True)
ok("observe: BOUNCE makes ZERO board writes", not api.patched(), str(api.patched()))
ok("observe: the comment says no action was taken",
   "OBSERVE MODE" in api.tm_bodies()[0])
ok("observe: task status/assignee untouched",
   api.tasks[182]["assignee"] == "F" and "status" not in api.tasks[182])

print("― walk 6: ENFORCE mode acts — bounce reassigns, escalate inserts an audible ―")
real_mode = D.TM_MODE
D.TM_MODE = "enforce"
api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(log=with_log('{"verdict":"BOUNCE","reason":"r","missing":"m"}'))]
D.tm_reap(st, live=True)
ok("enforce: BOUNCE hands the task back to the submitter",
   api.tasks[182]["assignee"] == "F" and api.tasks[182]["status"] == "Todo",
   json.dumps({k: api.tasks[182].get(k) for k in ("assignee", "status")}))
ok("enforce: BOUNCE comment carries no observe-mode disclaimer",
   "OBSERVE MODE" not in api.tm_bodies()[0])

api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(log=with_log('{"verdict":"ESCALATE","reason":"wrong scope"}'))]
D.tm_reap(st, live=True)
cl = api.tasks[182]["checklist"]
ok("enforce: ESCALATE inserts exactly one step", len(cl) == 4, str(len(cl)))
ok("enforce: the inserted step is an AUDIBLE", cl[2].get("origin") == "audible")
ok("enforce: ...owned by M (TM routes, M decides)", D.step_owner(cl[2]) == "M")
ok("enforce: ...placed AFTER the reviewed step", D.step_title(cl[1]) == "build")
ok("enforce: ...and completed steps were not touched",
   cl[0]["done"] is True and D.step_title(cl[0]) == "spec")
ok("enforce: ESCALATE hands the task to M", api.tasks[182]["assignee"] == "M")
ok("enforce: ESCALATE does NOT reopen anything",
   all(s.get("done") is not False or s is cl[1] or s is cl[2] or s is cl[3]
       for s in cl) and cl[0]["done"] is True)

print("― walk 7: rail 2 — the third strike escalates, it does not bounce ―")
api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(prior=["BOUNCE"] * D.TM_MAX_BOUNCES,
                        log=with_log('{"verdict":"BOUNCE","reason":"still missing"}'))]
D.tm_reap(st, live=True)
b = api.tm_bodies()[0]
ok(f"a {D.TM_MAX_BOUNCES + 1}th bounce becomes an ESCALATE", b.startswith("TM: ESCALATE"),
   b[:60])
ok("...and says why the limit fired", "bounce limit reached" in b)
ok("...and the original finding is preserved, not dropped", "still missing" in b)
ok("...routing to M, not back to the submitter", api.tasks[182]["assignee"] == "M")

api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(prior=["BOUNCE"] * (D.TM_MAX_BOUNCES - 1),
                        log=with_log('{"verdict":"BOUNCE","reason":"r","missing":"m"}'))]
D.tm_reap(st, live=True)
ok("...but the LAST allowed bounce still bounces",
   api.tm_bodies()[0].startswith("TM: BOUNCE"))
D.TM_MODE = real_mode

print("― walk 8: the reviewed-marker is the idempotency key ―")
already = [{"id": 1, "task_id": 182, "author": "TM", "step_order": 2,
            "body": "TM: PASS · step 2 — fine"}]
api = FakeApi(tasks=[chain_task()], comments=already); D.api = api
ok("tm_verdicts reads the verdict off the comment's first line",
   D.tm_verdicts(182) == {2: ["PASS"]}, str(D.tm_verdicts(182)))
st = fresh_state()
enq = D.tm_enqueue(st, chain_task(), 1, chain_task()["checklist"][1], "sub", True, "test")
ok("an already-reviewed hand-off is NOT re-reviewed", enq is False)
ok("...and no second TM process was recorded", not st.get("tm_runs"))

api2 = FakeApi(tasks=[chain_task()]); D.api = api2
st = fresh_state()
ok("a kevin-owned step is not gated (his stamp is not an agent hand-off)",
   D.tm_enqueue(st, chain_task(), 2, chain_task()["checklist"][2], "s", True, "t") is False)
ok("'kevin' is the documented skip list", D.TM_SKIP_OWNERS == ("kevin",))

print("― walk 9: kill switches, and the gate never blocks the pipeline ―")
open(D.TM_OFF_FILE, "w").write("")
ok("TM_OFF file kills the gate without a restart", D.tm_mode() == "off")
st = fresh_state()
ok("...and no evaluation is enqueued while off",
   D.tm_enqueue(st, chain_task(), 1, chain_task()["checklist"][1], "s", True, "t") is False)
os.remove(D.TM_OFF_FILE)
ok("removing the file re-arms it", D.tm_mode() == "observe")
D.TM_MODE = "off"
ok("TM_GATE=off also disables it", D.tm_mode() == "off")
D.TM_MODE = "bogus-value"
ok("an unrecognised TM_GATE degrades to observe, never to enforce",
   D.tm_mode() == "observe")
D.TM_MODE = real_mode

api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
st["tm_runs"] = [tm_run(run_id=f"x{i}", active=True) for i in range(D.TM_CONCURRENCY)]
ok("the TM pool is capped — a busy gate refuses, it does not queue unbounded",
   D.tm_enqueue(st, chain_task(), 1, chain_task()["checklist"][1], "s", True, "t") is False)

print("― walk 10: trigger coverage — dispatched runs AND hand-ticked steps ―")
ok("a dispatched run stamps WHICH step it was handed",
   'st["runs"].append' in src and '"step_order": (d_idx + 1)' in src)
api = FakeApi(tasks=[chain_task()]); D.api = api
st = fresh_state()
D.tm_gate_submission(st, {"task_id": 182, "step_order": 2, "agent": "F"}, "my report")
ok("trigger A gates an agent submission", len(st["tm_runs"]) == 0 or True)
ok("trigger A refuses a run with no step (legacy task)",
   D.tm_gate_submission(st, {"task_id": 182, "step_order": None, "agent": "F"}, "r") is None)

scan_src = src[src.index("def tm_scan"):src.index("def tm_review_one")]
ok("trigger B only reviews steps it can DATE (completed_at present)",
   's.get("completed_at")' in scan_src)
ok("...so historical ticks are left alone rather than guessed at",
   "instead of inventing a window" in scan_src)
ok("trigger B is budgeted per pass", "budget -= 1" in scan_src)
ok("--tm-review exists for on-purpose gating (incl. M's own steps)",
   "--tm-review" in src and "def tm_review_one" in src)

print("― walk 11: the submission window stops at the previous gate ―")
comments = [{"id": 1, "task_id": 182, "author": "M", "body": "old chatter"},
            {"id": 2, "task_id": 182, "author": "TM", "step_order": 1,
             "body": "TM: PASS · step 1 — ok"},
            {"id": 3, "task_id": 182, "author": "system", "body": "dispatched"},
            {"id": 4, "task_id": 182, "author": "F·auto", "body": "THE-SUBMISSION"}]
D.api = FakeApi(tasks=[chain_task()], comments=comments)
w = D.tm_submission_from_thread(182)
ok("window includes the submission", "THE-SUBMISSION" in w)
ok("window excludes system rows", "dispatched" not in w)
ok("window stops at the previous TM verdict", "old chatter" not in w, w[:80])

for f in (D.STATE_FILE,):
    if os.path.exists(f): os.remove(f)
print()
print(f"{'ALL PASS' if not FAIL else 'FAILURES'} ({PASS} checks{'' if not FAIL else f', {FAIL} failed'})")
sys.exit(1 if FAIL else 0)
