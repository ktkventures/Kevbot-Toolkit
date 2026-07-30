"""Acceptance test — board #218 step 2: the GIT CONTRACT stops telling agents they
cannot push.

Every dispatched agent was handed the line "You cannot push (headless) and must NOT
try to background one". That is FALSE, and was falsified four times for four on
07-29/30 — #198's Staged patch, #193's dashboard, #197's reconcile and #211's
release-brief generator each pushed their own branch, one of them opening a PR
unaided. The lie was not cosmetic: it made V4.23's run-end push the ONLY publication
path, so every weakness in that machinery (board #218's ownership mis-attribution)
became a single point of failure for work the agent could have pushed itself.

Each check below is a RAIL — this file FAILS if the rail is removed. Rails 1 and 2
are the ones that fail against the PRE-#218 prompt; the rest exist because the fix
is a text edit inside a load-bearing block that four other prompt tests depend on,
and the easy way to get rails 1-2 green is to damage something else.

  1  THE BUG — the prompt nowhere claims the agent cannot push. Asserted as a
     pattern ("cannot/can't/unable to ... push"), not as one literal string, so
     rewording the same falsehood does not sneak past
  2  THE FIX — it affirmatively tells the agent to push its OWN branch, and names
     the actual command rather than gesturing at it
  3  the push is SYNCHRONOUS/foreground — the ONE-SHOT CONTRACT's whole point is
     that a backgrounded step dies with the process, and a push is a step
  4  the dangerous pushes are still forbidden: --force, and dev/main
  5  the loop's run-end push is demoted to a BACKSTOP, explicitly not the agent's
     publication path — the framing #218 step 3 depends on being true
  6  it still says an UNCOMMITTED change is what makes work unpushable — the one
     failure mode neither the agent's push nor the backstop can rescue
  7  it still asks for the branch NAME in the final report (the #193 dashboard and
     the release-brief flow both read that)
  8  REGRESSION, #153 — cut from LATEST origin/dev, the explicit command form, and
     the `git log origin/dev..HEAD` self-check all survive intact. These are the
     load-bearing parts of the block and step 2 was told to keep them
  9  it renders in EVERY prompt shape — chain step, legacy checklist, no checklist.
     The git contract is unconditional; a fix that only landed on chains would be
     invisible on exactly the ad-hoc runs that push most
  10 ORDER — GIT CONTRACT still sits after the ONE-SHOT CONTRACT and before the
     ASSIGNEE CONTRACT, and the other blocks still render
  11 the module docstring no longer asserts the agent is told it cannot push, and
     the version banner records V4.26

NOTE ON SCOPE: hermetic PROMPT-CONSTRUCTION test. It proves the corrected
instruction is delivered; it cannot prove an agent obeys it. Behavioural
verification on a real concurrent run is board #218's last step. This file touches
NO ownership logic — that is #218 step 3.

Hermetic: loads the real dispatcher with a stubbed `api`, no network, no subprocess.
Run:  python3 src/test_dispatcher_git_contract_push_218.py   (from RoR_Trader root)
"""
import importlib.util, os, re, sys

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


spec = importlib.util.spec_from_file_location("git_contract_under_test", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["git_contract_under_test"] = D
spec.loader.exec_module(D)

D.api = lambda *a, **k: []          # no thread fetch, no network

AGENT = {"letter": "F", "scope": "frontend", "boundaries": "no engine files"}
MARK = "GIT CONTRACT (board #153"

# "you cannot push" in any of its plausible rewordings. NOT a blanket ban on the
# word — the block legitimately says "never push dev/main" and "unpushable".
CANNOT_PUSH = re.compile(
    r"(cannot|can'?t|can\s+not|unable\s+to|could\s*n'?o?t)\s+(\w+\s+){0,2}push",
    re.IGNORECASE)


def task(cl, tid=918):
    return {"id": tid, "title": "demo task", "description": "desc", "checklist": cl}


def step(title, owner, done, body=None):
    """A #182 chain step (carries new-shape keys → is_process_chain True)."""
    s = {"title": title, "text": title, "owner": owner, "role": owner, "done": done}
    if body is not None:
        s["body"] = body
    if done:
        s["completed_at"] = "2026-07-30T00:00:00Z"
    return s


def legacy(text, role, done):
    """A pre-#182 checklist item — ONLY {role,text,done}, no new-shape key."""
    return {"text": text, "role": role, "done": done}


def region(prompt):
    """The git-contract block alone — so a rail cannot pass on text borrowed from
    elsewhere in the prompt."""
    assert MARK in prompt, "no GIT CONTRACT block to slice"
    return prompt[prompt.index(MARK):].split("ASSIGNEE CONTRACT")[0]


print("\n#218 step 2 — GIT CONTRACT: agents CAN push")

CHAIN = [step("Approve it", "kevin", True, "stamp"),
         step("Fix the contract line", "M", False, "rewrite it"),
         step("M review", "M", False, "review it")]
p = D.build_prompt(AGENT, task(CHAIN))
r = region(p)

# ---- 1: THE BUG — the prompt never claims the agent cannot push -------------
m = CANNOT_PUSH.search(r)
ok("1 git contract does not claim the agent cannot push",
   m is None, f"found {m.group(0)!r}" if m else "")
m_all = CANNOT_PUSH.search(p)
ok("1 NO part of the prompt claims the agent cannot push",
   m_all is None, f"found {m_all.group(0)!r}" if m_all else "")
ok("1 the exact retired sentence is gone", "You cannot push (headless)" not in p)

# ---- 2: THE FIX — push your own branch, with the command --------------------
ok("2 tells the agent to push it itself", "PUSH IT YOURSELF" in r)
ok("2 states the capability positively", "you CAN push" in r)
ok("2 names the actual command", "git push -u origin" in r)
ok("2 cites the board task that authorised the change", "board #218" in r)
ok("2 cites the evidence rather than asserting it",
   "four runs did" in r and "07-29/30" in r)

# ---- 3: the push is foreground, per the ONE-SHOT CONTRACT -------------------
ok("3 requires a synchronous/foreground push", "IN THE FOREGROUND" in r)
ok("3 forbids backgrounding the push", "never background a push" in r)
ok("3 ONE-SHOT CONTRACT still renders (the reason 3 exists)", "ONE-SHOT CONTRACT" in p)

# ---- 4: the dangerous pushes are still forbidden ---------------------------
ok("4 forbids --force", "--force" in r)
ok("4 forbids pushing protected branches", "never push dev/main" in r)

# ---- 5: the loop's push is a BACKSTOP, not the publication path ------------
ok("5 names the run-end push a backstop", "BACKSTOP" in r)
ok("5 still credits the mechanism that provides it", "board #195" in r)
ok("5 tells the agent not to rely on it", "do not rely on it" in r)
ok("5 scopes it to a run that dies first", "dies before it gets there" in r)

# ---- 6: uncommitted work is still the unrescuable case --------------------
ok("6 COMMIT is still demanded first", "COMMIT everything" in r)
ok("6 still names uncommitted work as what makes work unpushable",
   "unpushable" in r and "uncommitted change" in r)
ok("6 says the backstop cannot save uncommitted work",
   "cannot save UNCOMMITTED work" in r)

# ---- 7: the branch name still reaches the report -------------------------
ok("7 still asks for the branch name in the final report",
   "name any branch you created in your final report" in r.lower())

# ---- 8: REGRESSION — the #153 half of the block is intact ----------------
ok("8 still requires cutting from LATEST origin/dev", "from LATEST origin/dev" in r)
ok("8 still gives the cut command", "git branch <name> origin/dev" in r)
ok("8 still gives the checkout -b form", "git checkout -b <name> origin/dev" in r)
ok("8 still demands the origin/dev..HEAD self-check",
   "git log origin/dev..HEAD --oneline" in r)
ok("8 still explains WHY a HEAD-cut branch is wrong",
   "inherits whatever unmerged commits" in r)
ok("8 still says only your own commits may appear",
   "ONLY your own commits" in r)

# ---- 9: renders in EVERY prompt shape ------------------------------------
p_leg = D.build_prompt(AGENT, task([legacy("do the thing", "F", False)]))
p_none = D.build_prompt(AGENT, task([]))
for label, pr in (("legacy checklist", p_leg), ("no checklist", p_none)):
    ok(f"9 git contract renders — {label}", MARK in pr)
    ok(f"9 corrected line present — {label}", "PUSH IT YOURSELF" in region(pr))
    mm = CANNOT_PUSH.search(pr)
    ok(f"9 no cannot-push claim — {label}", mm is None,
       f"found {mm.group(0)!r}" if mm else "")

# ---- 10: ORDER + the neighbouring blocks -------------------------------
ok("10 GIT CONTRACT sits after the ONE-SHOT CONTRACT",
   p.index("ONE-SHOT CONTRACT") < p.index(MARK))
ok("10 GIT CONTRACT sits before the ASSIGNEE CONTRACT",
   p.index(MARK) < p.index("ASSIGNEE CONTRACT (board #171"))
ok("10 ASSIGNEE CONTRACT still renders", "ASSIGNEE CONTRACT (board #171" in p)
ok("10 STEP-TICK CONTRACT still renders", "STEP-TICK CONTRACT (board #202" in p)
ok("10 step block still renders", "=== YOUR STEP — STEP 2 of 3" in p)

# ---- 11: the source's own account of itself ----------------------------
SRC_TEXT = open(DISP, encoding="utf-8").read()
head = SRC_TEXT[:SRC_TEXT.index("def ")]
ok("11 version banner records V4.26", "V4.26 (board #218" in head)
# The MODULE version is asserted ">= V4.26", not "== V4.26". This step shipped as
# V4.26 and its own sibling (#218 step 3, worktree ownership) landed STACKED on top
# as V4.27, rewriting the header line — so an equality check goes red on a change
# that is not a regression. R hit exactly this in Wave 23 and correctly refused to
# improvise past it. What this rail protects is that the module still DECLARES a
# version and has not gone BACKWARDS; the changelog assertion above already proves
# V4.26's own contribution survived the stack.
_mv = re.search(r"Team dispatcher \(V(\d+)\.(\d+)\)", head)
ok("11 module declares a version at all", _mv is not None, head[:120])
ok("11 module version is >= V4.26 (a later step may legitimately bump it)",
   _mv is not None and (int(_mv.group(1)), int(_mv.group(2))) >= (4, 26),
   _mv.group(0) if _mv else "no version string")
ok("11 docstring no longer says agents are TOLD they cannot push",
   "is TOLD it cannot push" not in head)
ok("11 V4.23's entry is preserved, not deleted", "V4.23 (board #195)" in head)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
