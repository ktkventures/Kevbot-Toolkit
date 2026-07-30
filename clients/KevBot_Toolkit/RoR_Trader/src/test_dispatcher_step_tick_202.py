"""Acceptance test — board #202: THE STEP-TICK CONTRACT.

A dispatched agent obeys the #171 ASSIGNEE CONTRACT reliably and left its finished
step UNTICKED four times out of four on 07-29 (#195/#197/#198/#200) — because no
contract asked it to tick. build_prompt now carries a STEP-TICK CONTRACT alongside
the assignee one.

Each check below is a RAIL — something that breaks silently if the block is edited
carelessly, and that this file fails on if the rail is removed:

  1  the block renders for a dispatched chain step
  2  it is ABSENT for a legacy {role,text,done} checklist — no dispatched step, so
     nothing to tick (and the legacy free-toggle path must stay untouched)
  3  it is ABSENT when there is no checklist at all, and when the chain is finished
  4  it names the step the agent was actually dispatched for, not a guess
  5  it names the endpoint that OWNS completion, with this task's real id. A generic
     checklist PATCH is refused for a chain (409 in _prepare_checklist_patch), so an
     instruction to "tick in the same PATCH" would be impossible to obey
  6  the actor payload carries the dispatched lane, so completed_by is not 'system'
  7  the exception is plain: DID NOT FINISH → do NOT tick, report, reassign. Without
     it the contract becomes pressure to tick whatever happened
  8  a 'STEP DONE:' report line is called out as PROSE, not a tick — that is exactly
     what the four measured runs did instead of ticking
  9  it forbids the direct-PATCH workaround and names the 409, so nobody "fixes" a
     refusal by writing `done` straight into the JSONB
  10 a /steps/complete refusal (e.g. a stamp-gated step) is REPORTED, not bypassed
  11 ORDER: it sits after the ASSIGNEE CONTRACT and before the final-report
     paragraph, and defers to that contract instead of contradicting it
  12 it never tells the agent to change `status` — the dispatcher owns status
  13 it renders on the LAST step of a chain (chain-complete is a real hand-off to M)
     and with no SOP body on the step
  14 regressions: assignee / git / one-shot contracts, the step block and the inbound
     check all still render

NOTE ON SCOPE: hermetic PROMPT-CONSTRUCTION test. It proves the instruction is
delivered; it cannot prove the agent obeys it. Behavioural verification against a
real dispatch is board #202's last step.

Hermetic: loads the real dispatcher with a stubbed `api`, no network, no subprocess.
Run:  python3 src/test_dispatcher_step_tick_202.py   (from RoR_Trader root)
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


spec = importlib.util.spec_from_file_location("step_tick_under_test", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["step_tick_under_test"] = D
spec.loader.exec_module(D)

D.api = lambda *a, **k: []          # no thread fetch, no network

AGENT = {"letter": "F", "scope": "frontend", "boundaries": "no engine files"}
MARK = "STEP-TICK CONTRACT (board #202"
TID = 902


def task(cl, tid=TID):
    return {"id": tid, "title": "demo task", "description": "desc", "checklist": cl}


def step(title, owner, done, body=None):
    """A #182 chain step (carries new-shape keys → is_process_chain True)."""
    s = {"title": title, "text": title, "owner": owner, "role": owner, "done": done}
    if body is not None:
        s["body"] = body
    if done:
        s["completed_at"] = "2026-07-29T00:00:00Z"
    return s


def legacy(text, role, done):
    """A pre-#182 checklist item — ONLY {role,text,done}, no new-shape key."""
    return {"text": text, "role": role, "done": done}


def region(prompt):
    """The contract block alone — so a rail can't pass on text borrowed from
    somewhere else in the prompt."""
    assert MARK in prompt, "no step-tick block to slice"
    return prompt[prompt.index(MARK):].split("Do the task.")[0]


print("\n#202 — step-tick contract")

# ---- 1: renders for a dispatched chain step --------------------------------
CHAIN = [step("Approve it", "kevin", True, "stamp"),
         step("Build it", "F", False, "implement the thing"),
         step("M review", "M", False, "review it")]
p = D.build_prompt(AGENT, task(CHAIN))
ok("1 renders for a dispatched chain step", MARK in p)
r = region(p)

# ---- 2: absent for a legacy checklist (nothing was dispatched to tick) -----
leg = [legacy("do the thing", "F", False), legacy("review", "M", False)]
p_leg = D.build_prompt(AGENT, task(leg))
ok("2 absent for a legacy {role,text,done} checklist", MARK not in p_leg)
ok("2 legacy prompt still carries the ASSIGNEE CONTRACT (not a blanket regression)",
   "ASSIGNEE CONTRACT" in p_leg)
ok("2 legacy prompt has no step block either (unchanged pre-#182 behaviour)",
   "=== YOUR STEP" not in p_leg)

# ---- 3: absent with no checklist, and once the chain is finished -----------
ok("3 absent for a task with no checklist",
   MARK not in D.build_prompt(AGENT, task([])))
ok("3 absent when every chain step is already done",
   MARK not in D.build_prompt(AGENT, task([step("a", "M", True, "x"),
                                           step("b", "F", True, "y")])))

# ---- 4: names the step actually dispatched ---------------------------------
ok("4 names the dispatched step (2 of 3)", "STEP 2 of 3" in r)
ok("4 does not name a different step number", "STEP 1 of 3" not in r and "STEP 3 of 3" not in r)
ok("4 the step number matches the step block's own", "STEP 2 of 3" in p.split(MARK)[0])

# ---- 5: names the endpoint that owns completion, with the real task id -----
# Asserted SUBSTANTIVELY, not as one contiguous string (board #217). The tick block
# now renders an absolute, pre-credentialed call —
#   curl -sS -X POST "$RORT_BOARD_API_URL/api/dev-tasks/<id>/steps/complete"
# — so the host variable sits between the verb and the path and a single-substring
# match fails while the rail itself is MORE satisfied than before (the run no longer
# has to guess the host). The rail's purpose is that the tick instruction names its
# target and instructs a POST; both are checked, so it still fails if either is lost.
ok("5 names the /steps/complete path with this task's id",
   f"/api/dev-tasks/{TID}/steps/complete" in r)
ok("5 instructs a POST to it (not a GET, not a bare mention)", "POST" in r)
ok("5 does NOT instruct a checklist PATCH (a chain PATCH is refused, 409)",
   "checklist[" not in r and "`done` = true" not in r and "done = true" not in r)
ok("5 the API really does refuse a PATCH tick (the reason this rail exists)",
   "completion is managed by /steps/complete" in open(
       os.path.join(SRC, "api", "routers", "dev_tasks.py"), encoding="utf-8").read())

# ---- 6: actor payload carries the dispatched lane --------------------------
ok("6 actor payload names this lane, so completed_by is not 'system'",
   '"actor": "F·auto"' in r)
ok("6 the actor key is the one /steps/complete reads",
   'payload.get("actor")' in open(
       os.path.join(SRC, "api", "routers", "dev_tasks.py"), encoding="utf-8").read())

# ---- 7: the exception, stated plainly --------------------------------------
ok("7 says DO NOT TICK when the step was not finished", "DO NOT TICK" in r)
ok("7 tells it to say so in the report", "report" in r.lower())
ok("7 tells it to reassign instead of ticking", "reassign" in r.lower())
ok("7 states WHY (a false tick is the worse error)",
   "worse than" in r and "not done" in r)

# ---- 8: prose is not a tick (the measured failure mode) -------------------
ok("8 calls a 'STEP DONE:' report line prose, not a tick",
   "STEP DONE" in r and "PROSE, not a tick" in r)
ok("8 the legacy 'STEP DONE:' report instruction is still in the prompt "
   "(so the disambiguation has something to disambiguate)",
   "list them as 'STEP DONE: <text>' lines" in p)

# ---- 9: no direct-PATCH workaround ---------------------------------------
ok("9 forbids flipping `done` with a generic PATCH", "generic PATCH" in r)
ok("9 names the 409 so the refusal is expected, not a surprise to route around",
   "409" in r)

# ---- 10: a refusal is reported, not bypassed ------------------------------
ok("10 stamp-gated refusal → do not work around it", "do not work around it" in r)
ok("10 routes the refusal to M rather than to a hack", "let M route it" in r)

# ---- 11: order + deference to the assignee contract ----------------------
ok("11 sits AFTER the ASSIGNEE CONTRACT", p.index("ASSIGNEE CONTRACT") < p.index(MARK))
ok("11 sits BEFORE the final-report paragraph", p.index(MARK) < p.index("Do the task."))
ok("11 defers to the assignee contract instead of restating it",
   "ASSIGNEE CONTRACT above" in r)
ok("11 keeps the manual reassign for the case the chain cannot know",
   "PATCH `assignee` only if reality" in r)

# ---- 12: never touches status -------------------------------------------
ok("12 does not instruct any status change (dispatcher owns status)",
   "status" not in r.lower())

# ---- 13: last step, and a step with no SOP body -------------------------
last = [step("Build", "M", True, "built"), step("Ship it", "F", False, "ship")]
r_last = region(D.build_prompt(AGENT, task(last)))
ok("13 renders on the LAST step of a chain", "STEP 2 of 2" in r_last)
nobody = [step("Approve", "kevin", True, "stamp"), step("Build", "F", False)]
ok("13 renders when the dispatched step has no SOP body",
   MARK in D.build_prompt(AGENT, task(nobody)))

# ---- 14: regressions ----------------------------------------------------
ok("14 ASSIGNEE CONTRACT still renders", "ASSIGNEE CONTRACT (board #171" in p)
ok("14 GIT CONTRACT still renders", "GIT CONTRACT (board #153" in p)
ok("14 ONE-SHOT CONTRACT still renders", "ONE-SHOT CONTRACT" in p)
ok("14 step block still renders", "=== YOUR STEP — STEP 2 of 3" in p)
ok("14 inbound check unaffected — absent when the predecessor is kevin's",
   "=== FIRST — INBOUND CHECK ON STEP" not in p)
mid = [step("Design", "M", True, "spec it"), step("Build", "F", False, "build it")]
ok("14 inbound check still renders for a non-kevin predecessor",
   "=== FIRST — INBOUND CHECK ON STEP 1" in D.build_prompt(AGENT, task(mid)))
ok("14 dispatcher version banner records V4.25",
   "V4.25 (board #202)" in open(DISP, encoding="utf-8").read())

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
