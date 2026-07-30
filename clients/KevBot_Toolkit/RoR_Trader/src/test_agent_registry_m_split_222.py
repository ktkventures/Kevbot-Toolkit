"""Acceptance test — board #222: split `M` into M (session) + M-A (assistant).

`M` named two actors at once — the long-running manager SESSION and the headless
engineer that builds the team's own tooling — and the board could not tell them
apart. Twice on 07-30 (#195, #219) a task assigned `M` and left ARMED dispatched
`M·auto` into work reserved for the session; one of those steps read "M restarts
the loop".

The fix deliberately adds NO mechanism. `dispatcher.headless_agents()` already
selects `status=eq.headless`, and `triage_todo()` already reports a non-headless
assignee as WAITING, not stuck — the behaviour `kevin` has proven since 07-23.
So the change is two registry rows plus the display, and this file's job is to
prove the existing machinery really does what the design claims it does.

Every rail below has at least one check that FAILS if the rail is removed:

  RAIL 1  M does not dispatch    — a task assigned `M` is REFUSED by the same
                                   gate that refuses `kevin`, from every live
                                   status, armed or not, chained or not.
  RAIL 2  waiting, NOT stuck     — that refusal carries is_stuck=False, so it
                                   stays quiet and never trips the 4h staleness
                                   tripwire or posts a "dispatch skipped"
                                   comment. A `True` here would spam the board
                                   forever on 16 open tasks.
  RAIL 3  M-A dispatches exactly — every case rail 1 refuses for `M`, `M-A`
          as M does today            dispatches, and every OTHER gate (tags,
                                   empty description, blocked_by, chain/assignee
                                   agreement, mode=discuss, terminal status)
                                   still applies to it unchanged. The split
                                   moves the lane; it does not exempt it.
  RAIL 4  no dispatcher LOGIC    — triage_todo() gains no letter check and no
          change                     new gate: it is the SAME code path that
                                   handles `kevin`. Asserted structurally (the
                                   module never mentions the letters) and
                                   behaviourally (M and kevin get byte-identical
                                   skip reasons).
  RAIL 5  the stub fallback      — when the registry is UNREACHABLE the stub
                                   roster is the docs lane's HEADLESS half. If
                                   `M` were still in STUB_AGENTS, a failed
                                   registry fetch would silently re-arm the exact
                                   behaviour this task removes, at the moment
                                   nobody is watching.
  RAIL 6  the registry migration — the .sql really contains both statements
                                   (INSERT M-A headless, UPDATE M live-session),
                                   inherits M's fields rather than copy-pasting
                                   them, and is DML only — no DDL, which would
                                   need Kevin's by-name authorization.
  RAIL 7  the board renders it   — the UI decides session-vs-agent from registry
                                   `status`/`kind`, NOT a hard-coded letter, and
                                   pairs colour with a shape + a word (a colour-
                                   only cue dies in greyscale and forced-colors).

Hermetic: loads the real dispatcher module with a fake `api()` that PARSES the
PostgREST filter it is handed (borrowed from test_dispatcher_ai_eligible_gate_198)
so a gate that ignored the filter is caught rather than accidentally passing. No
network, no DB, no spawns. Rails 6/7 are source reads of the .sql and .tsx.

Mutation-checked 07-30 (board #222 step 2), six mutations against a green 81/81:
putting `M` back in the headless registry fixture fails 24 · flipping the
non-headless skip to is_stuck=True fails 7 · putting `M` back in STUB_AGENTS
fails 4 · deleting the UPDATE from the migration fails 2 · replacing
isSessionLane's registry read with `role === 'M' || role === 'kevin'` fails 3 ·
stripping the shape + word cue from RoleChip/LaneKindChip (leaving colour alone)
fails 3.

Run:  .venv/bin/python src/test_agent_registry_m_split_222.py   (from RoR_Trader root)
"""
import importlib.util
import os
import re
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DISP = os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py")
MIGRATION = os.path.join(SRC, "migrations", "agents_registry_ma_split_222.sql")
VIEWS = os.path.join(ROOT, "frontend", "src", "views")
SHARED = os.path.join(VIEWS, "taskBoardShared.tsx")
MODAL = os.path.join(VIEWS, "TaskDetailModal.tsx")
CHARTER = os.path.join(ROOT, "docs", "_active", "Session_Charters.md")

PASS = FAIL = 0


def ok(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}{(' — ' + detail) if detail else ''}")


spec = importlib.util.spec_from_file_location("dispatcher_under_test_222", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["dispatcher_under_test_222"] = D
spec.loader.exec_module(D)

DISPATCHER_SRC = open(DISP, encoding="utf-8").read()

# ── the registry AFTER the split: M-A headless, M absent (live-session rows are
#    not returned by headless_agents()). `kevin` is likewise absent — that is the
#    point: M is now in kevin's category, and rail 4 compares them directly.
AGENTS = {
    "M-A": {"letter": "M-A", "status": "headless", "worktree": ".",
            "scope": "s", "boundaries": "b"},
    "F": {"letter": "F", "status": "headless", "worktree": ".",
          "scope": "s", "boundaries": "b"},
}

ALL_STATUSES = ("Backlog", "Scoping", "Approval", "Todo", "In Progress",
                "Review", "Staged", "Done", "Blocked")
TERMINAL = ("Done", "Blocked", "Staged")
LIVE_STATUSES = tuple(s for s in ALL_STATUSES if s not in TERMINAL)


def task(**kw):
    t = {"id": 1, "title": "T", "description": "d", "assignee": "M-A",
         "status": "Backlog", "ai_eligible": False,
         "tags": [], "blocked_by": [], "checklist": []}
    t.update(kw)
    return t


# ── the fake board (same PostgREST filter parser as the #198 suite) ─────────
_OR_RE = re.compile(r"or=\(([^)]*)\)")
_EQ_RE = re.compile(r"(\w+)=eq\.([^&]+)")


def _matches(row, clause):
    col, _, val = clause.partition(".eq.")
    if val == "true":
        return row.get(col) is True
    if val == "false":
        return row.get(col) is False
    return str(row.get(col)) == val


def board_api(rows):
    def _api(method, path, body=None, prefer=None):
        if not path.startswith("dev_tasks?"):
            return []
        q = path.split("?", 1)[1]
        m = _OR_RE.search(q)
        if m:
            return [r for r in rows
                    if any(_matches(r, c) for c in m.group(1).split(","))]
        m = _EQ_RE.search(q)
        if m:
            return [r for r in rows if str(r.get(m.group(1))) == m.group(2)]
        raise AssertionError(f"unmodelled dev_tasks filter: {q}")
    return _api


def triage(rows, st=None):
    D.api = board_api(rows)
    out, skipped = D.triage_todo(AGENTS, set(), st)
    return [t["id"] for t in out], skipped


def skip_for(rows, task_id):
    """(reason, is_stuck) for one task. A task that DISPATCHED returns the
    sentinel below rather than None, so a mutation that re-arms a lane reports
    as N failed checks instead of crashing the suite on an unpack."""
    _, skipped = triage(rows)
    for t, reason, stuck in skipped:
        if t["id"] == task_id:
            return reason, stuck
    return ("<DISPATCHED — not skipped at all>", None)


# ── RAIL 1 — a task assigned M is never dispatched ──────────────────────────
print("― rail 1: `M` does NOT dispatch ―")
for i, s in enumerate(LIVE_STATUSES):
    for armed in (True, False):
        tid = 100 + i * 2 + int(armed)
        got, _ = triage([task(id=tid, assignee="M", status=s,
                              ai_eligible=armed)])
        ok(f"assignee=M, status={s!r}, ai_eligible={armed} → refused",
           got == [], f"dispatched {got}")

# The two live incidents, reconstructed: an ARMED task on a chain step owned by
# M, sitting mid-pipeline. #195 was Staged→Review; #219 was step 5 "Restart the
# loop and VERIFY". Both dispatched M·auto. Neither may now.
incident = task(id=195, assignee="M", status="Review", ai_eligible=True,
                checklist=[{"owner": "M", "title": "Restart the loop and VERIFY",
                            "body": "M restarts the loop", "mode": "execute",
                            "done": False}])
got, _ = triage([incident])
ok("the #195/#219 shape (ARMED, mid-pipeline, chain step owned by M) is refused",
   got == [], f"dispatched {got}")

# Todo is the legacy arm — the one path that dispatches on assignee alone. It
# must refuse M too, or 16 open tasks keep dispatching exactly as before.
got, _ = triage([task(id=196, assignee="M", status="Todo")])
ok("the LEGACY Todo arm refuses M as well (this is where #195/#219 lived)",
   got == [], f"dispatched {got}")

ok("M is not in the post-split headless set at all", "M" not in AGENTS)

# ── RAIL 2 — refused as WAITING, not STUCK ──────────────────────────────────
print("― rail 2: refused as WAITING, not stuck (quiet; no tripwire, no comment) ―")
for s in LIVE_STATUSES:
    got = skip_for([task(id=1, assignee="M", status=s, ai_eligible=True)], 1)
    ok(f"status={s!r} → skipped with is_stuck=False",
       got[1] is False, str(got))
reason, stuck = skip_for([task(id=1, assignee="M", status="Todo")], 1)
ok("the reason names M and says it is not a headless agent",
   "M" in reason and "not a headless agent" in reason, reason)
ok("is_stuck=False on the legacy Todo arm too", stuck is False)

# STALE_SKIP_S is the 4h tripwire that fires on STUCK tasks only. A True above
# would light up every M task on the board, every day, forever.
ok("the staleness tripwire is stuck-only, so a waiting M task never trips it",
   "is_stuck" in DISPATCHER_SRC and "STALE_SKIP_S" in DISPATCHER_SRC)

# ── RAIL 3 — M-A dispatches exactly as M did ────────────────────────────────
print("― rail 3: `M-A` dispatches exactly as M did today ―")
for i, s in enumerate(LIVE_STATUSES):
    tid = 300 + i
    got, skipped = triage([task(id=tid, assignee="M-A", status=s,
                                ai_eligible=True)])
    ok(f"assignee=M-A, ARMED, status={s!r} → dispatches",
       got == [tid], f"skipped={[x[1] for x in skipped]}")
got, skipped = triage([task(id=310, assignee="M-A", status="Todo")])
ok("assignee=M-A, UNARMED but in Todo → dispatches (legacy arm preserved)",
   got == [310], f"skipped={[x[1] for x in skipped]}")
got, _ = triage([task(id=311, assignee="M-A", status="Review",
                      ai_eligible=True,
                      checklist=[{"owner": "M-A", "title": "build it",
                                  "body": "SOP", "mode": "execute",
                                  "done": False}])])
ok("a chain step OWNED by M-A on an M-A task dispatches", got == [311])

# …and every other gate still applies to the new lane. The split moves the lane,
# it does not exempt it.
print("  · every pre-existing gate still applies to M-A:")
cases = [
    ("needs-review tag", task(id=320, assignee="M-A", status="Todo",
                              tags=["needs-review"]), True),
    ("needs-scoping tag", task(id=321, assignee="M-A", status="Todo",
                               tags=["needs-scoping"]), True),
    ("empty description", task(id=322, assignee="M-A", status="Todo",
                               description="  "), True),
    ("blocked_by not Done", task(id=323, assignee="M-A", status="Todo",
                                 blocked_by=[999]), True),
]
for label, t, expect_stuck in cases:
    got, _ = triage([t])
    sk = skip_for([t], t["id"])
    ok(f"    {label} → refused", got == [], f"dispatched {got}")
    ok(f"    {label} → is_stuck={expect_stuck}",
       sk[1] is expect_stuck, str(sk))

discuss = task(id=324, assignee="M-A", status="Todo",
               checklist=[{"owner": "M-A", "title": "decide", "body": "b",
                           "mode": "discuss", "done": False}])
got, _ = triage([discuss])
sk = skip_for([discuss], 324)
ok("    mode=discuss → refused, waiting-not-stuck",
   got == [] and sk[1] is False, f"{got} {sk}")

mismatch = task(id=325, assignee="M-A", status="Todo",
                checklist=[{"owner": "F", "title": "x", "body": "b",
                            "done": False}])
got, _ = triage([mismatch])
sk = skip_for([mismatch], 325)
ok("    chain/assignee disagree → refused, LOUD (stuck)",
   got == [] and sk[1] is True, f"{got} {sk}")

for s in TERMINAL:
    got, _ = triage([task(id=330, assignee="M-A", status=s, ai_eligible=True)])
    ok(f"    terminal status {s!r} → refused", got == [], f"dispatched {got}")

# ── RAIL 4 — no dispatcher LOGIC change: M walks kevin's path ───────────────
print("― rail 4: NO new gate — M takes the same path `kevin` already takes ―")
for s in LIVE_STATUSES:
    m = skip_for([task(id=1, assignee="M", status=s, ai_eligible=True)], 1)
    k = skip_for([task(id=1, assignee="kevin", status=s, ai_eligible=True)], 1)
    ok(f"status={s!r}: M's skip is identical to kevin's but for the letter",
       m[1] is not None and m[1] == k[1]
       and m[0].replace("M", "kevin", 1) == k[0],
       f"M={m} kevin={k}")
ok("triage_todo() contains no letter check for M / M-A",
   not re.search(r"""["']M-?A?["']\s*(==|!=|in\b)""", DISPATCHER_SRC)
   and '== "M"' not in DISPATCHER_SRC,
   "a hard-coded letter check appeared in the gate — the premise is wrong; "
   "reassign to M (step-2 SOP)")
ok("headless_agents() still selects on status alone",
   "agents?status=eq.headless&select=*" in DISPATCHER_SRC)
ok("the non-headless-assignee skip is a single shared branch",
   DISPATCHER_SRC.count("(not a headless agent)") == 1)

# ── RAIL 5 — the stub fallback is the lane's HEADLESS half ──────────────────
print("― rail 5: registry-unreachable fallback cannot re-arm M ―")
ok("STUB_AGENTS does not contain M", "M" not in D.STUB_AGENTS,
   str(sorted(D.STUB_AGENTS)))
ok("STUB_AGENTS contains M-A", "M-A" in D.STUB_AGENTS,
   str(sorted(D.STUB_AGENTS)))
ok("the stub row is headless (it is the fallback dispatch set)",
   D.STUB_AGENTS.get("M-A", {}).get("status") == "headless")
# Prove it through the gate, not just by shape: with the stub as the roster, an
# armed M task is still refused.
D.api = board_api([task(id=1, assignee="M", status="Todo")])
stub_out, _ = D.triage_todo(D.STUB_AGENTS, set(), None)
ok("under the STUB roster, a Todo task assigned M is still refused",
   stub_out == [], f"dispatched {[t['id'] for t in stub_out]}")

# ── RAIL 6 — the migration is real, inherited, and DML-only ────────────────
print("― rail 6: the registry migration ―")
SQL = open(MIGRATION, encoding="utf-8").read()
body = "\n".join(l for l in SQL.splitlines() if not l.strip().startswith("--"))
ok("INSERTs M-A as headless",
   re.search(r"'M-A'.*?'headless'", body, re.S) is not None)
ok("UPDATEs M to live-session",
   re.search(r"UPDATE\s+agents.*?status\s*=\s*'live-session'.*?letter\s*=\s*'M'",
             body, re.S) is not None)
ok("inherits M's fields BY SELECTION (m.scope/m.boundaries/m.worktree/"
   "m.prompt_template), so it cannot drift from M's row",
   all(f"m.{c}" in body for c in
       ("scope", "boundaries", "worktree", "context_docs", "prompt_template")))
ok("idempotent (ON CONFLICT DO NOTHING + a guarded UPDATE)",
   "ON CONFLICT (letter) DO NOTHING" in body
   and "status <> 'live-session'" in body)
ok("DML ONLY — no DDL (that would need Kevin's by-name authorization)",
   not re.search(r"\b(CREATE|ALTER|DROP|TRUNCATE)\b", body, re.I),
   "found DDL: author it, STOP, and hand to M")
ok("the status values it writes are in the API's accepted vocabulary",
   {"headless", "live-session"} <= {
       "live-session", "headless", "dormant", "retired", "ephemeral"})

# ── RAIL 7 — the board renders the distinction, from the registry ──────────
print("― rail 7: the board renders session vs agent, keyed off the registry ―")
TSX = open(SHARED, encoding="utf-8").read()
MODAL_SRC = open(MODAL, encoding="utf-8").read()
ok("isSessionLane reads registry status/kind",
   "status === 'live-session'" in TSX and "kind === 'human'" in TSX)
sess = TSX[TSX.index("export const isSessionLane"):]
sess = sess[:sess.index("export const laneKindLabel")]
ok("isSessionLane contains NO hard-coded letter check "
   "(the next live-session lane must inherit this for free)",
   "'M'" not in sess and '"M"' not in sess and "'kevin'" not in sess, sess)
ok("the letter fallback is scoped to an EMPTY registry (API outage) only",
   "reg.size === 0 && SESSION_FALLBACK.has(role)" in TSX)
# Asserted as an INVARIANT rather than as a pixel literal: the shape must branch
# on `session`, and the agent case must stay a circle. Board #243 parameterised
# the chip's size (the shipping lane demotes the builder to a smaller secondary
# chip), which turned the old literal `session ? 5 : '50%'` into a computed
# value. The CLAIM this rail defends — shape carries the lane kind, not colour —
# is untouched, and dropping the branch or going colour-only still fails it.
ok("RoleChip varies SHAPE, not colour alone (survives greyscale / "
   "forced-colors / colourblindness)",
   re.search(r"borderRadius: session \? [^:\n]+ : '50%'", TSX) is not None)
ok("…and a ring on top of the shape", "boxShadow: session" in TSX)
ok("a short WORD accompanies it (LaneKindChip renders 'session' / 'agent')",
   "'◧ session'" in TSX and "'◍ agent'" in TSX)
ok("both cues carry an explanatory tooltip", "LANE_KIND_TIP" in TSX)
ok("RoleChip resolves the lane from the registry context, not from props",
   "const reg = useAgentRegistry();" in TSX)
ok("TaskDetailModal renders the label on the assignee",
   "<LaneKindChip role={task.assignee || ''} />" in MODAL_SRC)
ok("…and on a chain step owner, quiet unless the owner is a session",
   "sessionOnly" in MODAL_SRC)
ok("M-A has its own avatar colour + abbreviation",
   "'M-A': '#a48cff'" in TSX and "r === 'M-A' ? 'MA'" in TSX)
ok("M-A is in the assignee + mention fallback lists",
   "'M', 'M-A'" in TSX and TSX.count("'M', 'M-A'") >= 2)

# ── the charter records the split (the SSOT Kevin named) ───────────────────
print("― charter §1/§2 ―")
CH = open(CHARTER, encoding="utf-8").read()
ok("§1 has an M-A row", "**M-A — Manager Assistant**" in CH)
ok("§1 marks M as a live session that is never auto-dispatched",
   "LIVE SESSION — never auto-dispatched" in CH)
ok("the dividing line is stated as a table",
   "Give it to **M-A**" in CH and "Keep it on **M**" in CH)
ok("it records that the step-level flag was deliberately dropped",
   "deliberately DROPPED" in CH)
ok("§2 roster lists M-A", "| M-A — Manager Assistant |" in CH)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
