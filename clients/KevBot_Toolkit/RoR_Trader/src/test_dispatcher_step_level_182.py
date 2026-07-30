"""Acceptance test — board #182 Step 7: STEP-LEVEL DISPATCH in the dispatcher.

The dispatcher must send the CURRENT STEP (title + its own SOP body), refuse to
hand a mode='discuss' step to a headless agent, and refuse when the chain's owner
and the task's assignee disagree — while leaving legacy {role,text,done}
checklists behaving exactly as they did in V4.16 (assignee alone gates).

Hermetic: loads the real dispatcher module with a fake `api`, no network.
Run:  .venv/bin/python src/test_dispatcher_step_level_182.py   (from RoR_Trader root)
"""
import importlib.util, os, sys

SRC = os.path.dirname(os.path.abspath(__file__))
DISP = os.path.join(os.path.dirname(SRC), "tools", "team_dispatcher", "dispatcher.py")

PASS = FAIL = 0
def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok: {label}")
    else:    FAIL += 1; print(f"  FAIL: {label}{(' — ' + detail) if detail else ''}")

spec = importlib.util.spec_from_file_location("dispatcher_under_test", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["dispatcher_under_test"] = D
spec.loader.exec_module(D)

AGENTS = {"M": {"letter": "M", "scope": "s", "boundaries": "b"},
          "F": {"letter": "F", "scope": "s", "boundaries": "b"}}

# status="Todo" is explicit since board #198: the filter that used to SELECT these
# rows (`status=eq.Todo`) is now an OR with `ai_eligible`, and these walks describe
# the LEGACY Todo arm — including walk 4, where a fully-ticked chain sitting in
# Todo still dispatches on assignee alone.
def task(**kw):
    t = {"id": 1, "title": "T", "description": "d", "assignee": "F",
         "status": "Todo", "ai_eligible": False,
         "tags": [], "blocked_by": [], "checklist": []}
    t.update(kw); return t

def run_triage(tasks):
    D.api = lambda m, p, body=None, prefer=None: tasks if p.startswith(f"dev_tasks?{D.GATE_FILTER}") else []
    return D.triage_todo(AGENTS, set())

print("― walk 1: legacy checklists are UNCHANGED (V4.16 behaviour) ―")
legacy = task(checklist=[{"role": "kevin", "text": "Kevin stamps", "done": False},
                         {"role": "F", "text": "build", "done": False}])
out, skipped = run_triage([legacy])
ok("legacy checklist with an un-ticked kevin step STILL dispatches to F",
   len(out) == 1 and not skipped, f"out={len(out)} skipped={[(s[1]) for s in skipped]}")
ok("is_process_chain False for a legacy checklist", not D.is_process_chain(legacy["checklist"]))

print("― walk 2: chain gates on the CURRENT step's owner ―")
chain_ok = task(checklist=[{"owner": "M", "title": "spec", "done": True},
                           {"owner": "F", "title": "build", "body": "SOP here", "done": False}])
out, skipped = run_triage([chain_ok])
ok("chain whose current step owner == assignee dispatches", len(out) == 1 and not skipped)

mismatch = task(assignee="F",
                checklist=[{"owner": "M", "title": "spec", "done": False},
                           {"owner": "F", "title": "build", "done": False}])
out, skipped = run_triage([mismatch])
ok("chain/assignee disagreement is REFUSED", len(out) == 0 and len(skipped) == 1)
ok("...and it is reported LOUD (is_stuck=True)", skipped and skipped[0][2] is True)
ok("...with both sides named in the reason",
   skipped and "owned by M" in skipped[0][1] and "assignee is F" in skipped[0][1],
   skipped[0][1] if skipped else "")

print("― walk 3: mode='discuss' is never dispatched to a headless agent ―")
disc = task(checklist=[{"owner": "F", "title": "talk it through", "mode": "discuss", "done": False}])
out, skipped = run_triage([disc])
ok("discuss step is REFUSED", len(out) == 0 and len(skipped) == 1)
ok("...and is reported as WAITING, not stuck (no 4h tripwire)",
   skipped and skipped[0][2] is False, str(skipped[0][2]) if skipped else "")
ok("...reason says it needs a reply, not a build",
   skipped and "REPLY" in skipped[0][1], skipped[0][1] if skipped else "")

exe = task(checklist=[{"owner": "F", "title": "build", "mode": "execute", "done": False}])
ok("mode='execute' still dispatches", len(run_triage([exe])[0]) == 1)
nomode = task(checklist=[{"owner": "F", "title": "build", "done": False}])
ok("missing mode defaults to execute (dispatches)", len(run_triage([nomode])[0]) == 1)

print("― walk 4: finished chain falls back to assignee-only ―")
fin = task(checklist=[{"owner": "M", "title": "a", "done": True},
                      {"owner": "M", "title": "b", "done": True}])
out, skipped = run_triage([fin])
ok("all-steps-done chain dispatches on assignee alone", len(out) == 1 and not skipped)
i, s = D.current_step(fin)
ok("current_step() is (None, None) on a finished chain", i is None and s is None)

print("― walk 5: the prompt carries THAT STEP's SOP ―")
D.api = lambda m, p, body=None, prefer=None: []
p3 = task(checklist=[{"owner": "M", "title": "spec it", "done": True},
                     {"owner": "F", "title": "build the accordion",
                      "body": "UNIQUE-SOP-MARKER: render steps as accordions",
                      "origin": "audible", "done": False},
                     {"owner": "kevin", "title": "review", "done": False}])
prompt = D.build_prompt(AGENTS["F"], p3)
ok("prompt names the current step number and total", "STEP 2 of 3" in prompt)
ok("prompt embeds the step's own SOP body", "UNIQUE-SOP-MARKER" in prompt)
ok("prompt marks an audible step as such", "AUDIBLE" in prompt)
ok("prompt lists what earlier steps concluded", "spec it" in prompt.split("Already completed")[1][:200])
ok("prompt says do this step and STOP", "DO THIS ONE AND STOP" in prompt)
ok("prompt tells the agent to RAISE scope disagreements, not expand",
   "RAISE IT" in prompt and "expand scope unilaterally" in prompt)
ok("prompt refuses to let a missing SOP be guessed at",
   "DO NOT GUESS" in D.build_prompt(AGENTS["F"],
       task(checklist=[{"owner": "F", "title": "x", "done": False}])))

legacy_prompt = D.build_prompt(AGENTS["F"], legacy)
ok("legacy task prompt has NO step block", "YOUR STEP — STEP" not in legacy_prompt)

print()
print(f"{'ALL PASS' if not FAIL else 'FAILURES'} ({PASS} checks{'' if not FAIL else f', {FAIL} failed'})")
sys.exit(1 if FAIL else 0)
