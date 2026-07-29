"""Acceptance test — board #182 Step 7 (second half): THE INBOUND CHECK.

Before starting its own step, a dispatched agent verifies that the PREVIOUS step
delivered what its SOP promised. This replaces the separate "TM gate" agent (see
docs/_active/TM_Gate_Verification_2026-07-29.md).

What this suite pins, and why each one is a thing that breaks silently:

  1 the block renders for a mid-chain step, naming the RIGHT step number
  2 it is ABSENT on the first step of a chain (no predecessor to check)
  3 it is ABSENT when the previous step is kevin-owned (approvals aren't deliverables)
  4 it carries the previous step's OWN SOP — without it there is nothing to check against
  5 a missing SOP body degrades to "proceed", never to a guessed requirement
  6 it renders BEFORE the agent's own step block (order is the whole point — a check
    that arrives after the work has started is not a gate)
  7 the escape hatch is present and specific: raise-an-issue + reassign to M + stop
  8 the anti-obstruction framing survives ("a SUCCESS, not a FAILURE" / proceed if
    ambiguous). This is prompt behaviour, and prompt behaviour regresses by deletion.

NOTE ON SCOPE: this is a hermetic PROMPT-CONSTRUCTION test. It proves the instruction
is delivered to the agent; it cannot prove the agent obeys it. Behavioural calibration
is a live exercise — see the "calibration" note on board #182.

Hermetic: loads the real dispatcher with a stubbed `api`, no network, no subprocess.
Run:  python3 src/test_dispatcher_inbound_check_182.py   (from RoR_Trader root)
"""
import importlib.util, os, sys

SRC = os.path.dirname(os.path.abspath(__file__))
DISP = os.path.join(os.path.dirname(SRC), "tools", "team_dispatcher", "dispatcher.py")

PASS = FAIL = 0


def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}{(' — ' + detail) if detail else ''}")


spec = importlib.util.spec_from_file_location("inbound_under_test", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["inbound_under_test"] = D
spec.loader.exec_module(D)

D.api = lambda *a, **k: []          # no thread fetch, no network

AGENT = {"letter": "F", "scope": "frontend", "boundaries": "no engine files"}
MARK = "=== FIRST — INBOUND CHECK ON STEP"


def task(cl, tid=900):
    return {"id": tid, "title": "demo task", "description": "desc", "checklist": cl}


def step(title, owner, done, body=None):
    s = {"title": title, "text": title, "owner": owner, "role": owner, "done": done}
    if body is not None:
        s["body"] = body
    if done:
        s["completed_at"] = "2026-07-29T00:00:00Z"
    return s


print("\n#182 Step 7 — inbound check")

# 1 — renders mid-chain, names the right predecessor
cl = [step("Design the thing", "M", True, "Produce a spec with three options."),
      step("Build it", "F", False, "Implement the chosen option.")]
p = D.build_prompt(AGENT, task(cl))
ok("1 renders for a mid-chain step", MARK in p)
ok("1 names the PREVIOUS step number (1), not its own (2)",
   "INBOUND CHECK ON STEP 1" in p and "INBOUND CHECK ON STEP 2" not in p)
ok("1 names the previous step's title", 'Design the thing' in p.split("=== END INBOUND")[0])

# 2 — absent on the first step
p1 = D.build_prompt(AGENT, task([step("Design", "M", False, "spec")]))
ok("2 absent on the first step of a chain", MARK not in p1)

# 3 — absent when the predecessor is kevin's
clk = [step("Approve the plan", "kevin", True, "stamp it"),
       step("Build it", "F", False, "implement")]
ok("3 absent when the previous step is kevin-owned", MARK not in D.build_prompt(AGENT, task(clk)))
ok("3 kevin is the configured skip-owner", "kevin" in D.INBOUND_SKIP_OWNERS)

# 4 — carries the predecessor's own SOP
ok("4 carries the previous step's SOP text",
   "Produce a spec with three options." in p.split("=== END INBOUND")[0])

# 5 — a missing SOP degrades to proceed, not to a guess
cl_nb = [step("Design", "M", True), step("Build", "F", False, "implement")]
pnb = D.build_prompt(AGENT, task(cl_nb))
head = pnb.split("=== END INBOUND")[0]
ok("5 renders with no predecessor SOP body", MARK in pnb)
ok("5 degrades to 'proceed', does not invent a requirement",
   "proceed" in head.lower() and "no SOP body" in head)

# 6 — ORDER: the check precedes the agent's own step
ok("6 inbound block comes BEFORE the agent's own step block",
   p.index(MARK) < p.index("=== YOUR STEP"))

# 7 — the escape hatch is specific and actionable
ok("7 tells the agent to raise an issue", "Raise-an-issue" in head)
ok("7 tells it to reassign to M", "`M`" in head or "assignee` to `M`" in head)
ok("7 tells it to STOP rather than build on a bad hand-off", "stop" in head.lower())
ok("7 demands the SPECIFIC missing thing, not a vague complaint", "SPECIFIC" in head)

# 8 — anti-obstruction framing (the thing that keeps this from becoming a stall)
ok("8 frames raising as a success, not a failure",
   "SUCCESS, NOT A FAILURE" in head)
ok("8 says proceed when genuinely ambiguous", "PROCEED" in head and "ambiguous" in head)
ok("8 scopes the check to DELIVERY, not correctness",
   "DELIVERY, not correctness" in head)

# 9 — the check must not fire on a non-chain task (no regression to plain dispatch)
plain = D.build_prompt(AGENT, {"id": 901, "title": "t", "description": "d", "checklist": []})
ok("9 absent for a task with no process chain", MARK not in plain)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
