"""Acceptance test — board #229: split `R` into R (conductor) + R-A (executor).

The sibling of test_agent_registry_m_split_222.py, and deliberately the same
shape: if the second split needed different machinery from the first, the first
one's central claim ("no new mechanism; the next live-session lane inherits it
for free") was false. It did not, and rail 7 measures exactly that.

The REASON differs from #222's, and that matters for what this file must prove.
`M` was split because `M·auto` dispatched into work reserved for the session.
`R` is split for the opposite reason: across 27 runs / 24 `ok` / 17 trains,
`R·auto`'s judgement is good — it aborts on red gates rather than improvising,
self-reports its own misses, watches deploys to settlement. The cost is that
every one of those CORRECT hand-backs lands on M, who then reconciles the gate,
rebases the conflict, or rewrites the brief. So the split does not restrict the
headless half at all: R-A must keep dispatching EXACTLY as R does today (rail 3),
and only the conductor's seat moves off the dispatch queue (rails 1/2).

  RAIL 1  R does not dispatch    — after the flip, a task assigned `R` is REFUSED
                                   by the same gate that refuses `kevin` and `M`,
                                   from every live status, armed or not, chained
                                   or not, including the legacy Todo arm.
  RAIL 2  waiting, NOT stuck     — that refusal carries is_stuck=False, so a
                                   Staged-queue task parked on R stays quiet:
                                   no 4h staleness tripwire, no repeated
                                   "dispatch skipped" comment on the board.
  RAIL 3  R-A dispatches exactly — every case rail 1 refuses for `R`, `R-A`
          as R does today            dispatches, and every OTHER gate (tags,
                                   empty description, blocked_by, chain/assignee
                                   agreement, mode=discuss, terminal status)
                                   still applies to it unchanged. Given WHY this
                                   lane is being split, a rail-3 failure would be
                                   the whole task backfiring — it would take the
                                   good executor off the queue too.
  RAIL 4  no dispatcher LOGIC    — triage_todo() gains no letter check and no new
          change                     gate: R walks the identical code path as
                                   `kevin`. The step-2 SOP is explicit that
                                   editing that gate means the premise is wrong,
                                   so this rail is also the abort condition.
  RAIL 5  the stub fallback      — when the registry is UNREACHABLE the stub
                                   roster stays the DOCS lane only. R was never
                                   in it and R-A must not be added: a release
                                   train dispatched off a fallback roster, at
                                   the moment the registry is down, is the last
                                   thing anyone wants running unwatched.
  RAIL 6  the registry migration — the .sql contains both statements (INSERT R-A
                                   headless, UPDATE R live-session), inherits R's
                                   fields BY SELECTION rather than copy-paste,
                                   is DML only (no DDL, which would need Kevin's
                                   by-name authorization), and is idempotent.
  RAIL 7  the board renders it   — R-A renders as an agent and R as a session
          FOR FREE                   through the SAME registry-driven path #222
                                   built: no letter check was added, and the
                                   generic renderers are untouched. Asserted as
                                   MERGE-STABLE properties (no `R-A` letter test
                                   outside the abbreviation lookup; `R-A` data
                                   confined to the shared token module) rather
                                   than as a diff against origin/dev — a diff
                                   assertion is true on the branch and false the
                                   moment another car edits the same file, which
                                   is how it went red in the Wave 24 simulation.

Hermetic: loads the real dispatcher module with a fake `api()` that PARSES the
PostgREST filter it is handed (borrowed from test_dispatcher_ai_eligible_gate_198)
so a gate that ignored the filter is caught rather than accidentally passing. No
network, no DB, no spawns. Rails 6/7 are source reads of the .sql and .tsx.

Mutation-checked 07-30 (board #229 step 2) against a green run — see the run log
in the task report for the per-mutation failure counts.

Run:  .venv/bin/python src/test_agent_registry_r_split_229.py   (from RoR_Trader root)
"""
import glob
import importlib.util
import os
import re
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DISP = os.path.join(ROOT, "tools", "team_dispatcher", "dispatcher.py")
MIGRATION = os.path.join(SRC, "migrations", "agents_registry_ra_split_229.sql")
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


spec = importlib.util.spec_from_file_location("dispatcher_under_test_229", DISP)
D = importlib.util.module_from_spec(spec)
sys.modules["dispatcher_under_test_229"] = D
spec.loader.exec_module(D)

DISPATCHER_SRC = open(DISP, encoding="utf-8").read()

# ── the registry AFTER the split: R-A headless, R absent (live-session rows are
#    not returned by headless_agents()). `kevin` and `M` are likewise absent —
#    that is the point: R joins their category, and rail 4 compares them.
AGENTS = {
    "R-A": {"letter": "R-A", "status": "headless", "worktree": ".",
            "scope": "s", "boundaries": "b"},
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
    t = {"id": 1, "title": "T", "description": "d", "assignee": "R-A",
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


# ── RAIL 1 — a task assigned R is never dispatched ─────────────────────────
print("― rail 1: `R` does NOT dispatch ―")
for i, s in enumerate(LIVE_STATUSES):
    for armed in (True, False):
        tid = 100 + i * 2 + int(armed)
        got, _ = triage([task(id=tid, assignee="R", status=s,
                              ai_eligible=armed)])
        ok(f"assignee=R, status={s!r}, ai_eligible={armed} → refused",
           got == [], f"dispatched {got}")

# The shape a release task actually has: an armed brief in Todo whose chain step
# is owned by R. This is what dispatches R·auto today, seventeen trains' worth.
brief = task(id=229, assignee="R", status="Todo", ai_eligible=True,
             description="Release brief: merge PR #x, watch deploy, log.",
             checklist=[{"owner": "R", "title": "Ship via a release train",
                         "body": "merge -> gate -> deploy-watch -> log",
                         "mode": "execute", "done": False}])
got, _ = triage([brief])
ok("the RELEASE BRIEF shape (armed, Todo, chain step owned by R) is refused",
   got == [], f"dispatched {got}")

# Todo is the legacy arm — the one path that dispatches on assignee alone, and
# the one every release task has travelled. It must refuse R too.
got, _ = triage([task(id=196, assignee="R", status="Todo")])
ok("the LEGACY Todo arm refuses R as well (this is the release path today)",
   got == [], f"dispatched {got}")

ok("R is not in the post-split headless set at all", "R" not in AGENTS)

# ── RAIL 2 — refused as WAITING, not STUCK ─────────────────────────────────
print("― rail 2: refused as WAITING, not stuck (quiet; no tripwire, no comment) ―")
for s in LIVE_STATUSES:
    got = skip_for([task(id=1, assignee="R", status=s, ai_eligible=True)], 1)
    ok(f"status={s!r} → skipped with is_stuck=False",
       got[1] is False, str(got))
reason, stuck = skip_for([task(id=1, assignee="R", status="Todo")], 1)
ok("the reason names R and says it is not a headless agent",
   "R" in reason and "not a headless agent" in reason, reason)
ok("is_stuck=False on the legacy Todo arm too", stuck is False)

# STALE_SKIP_S is the 4h tripwire that fires on STUCK tasks only. A True above
# would light up the whole Staged queue, every day, forever — and the Staged
# queue is precisely what the R session is supposed to sit on and work through.
ok("the staleness tripwire is stuck-only, so a waiting R task never trips it",
   "is_stuck" in DISPATCHER_SRC and "STALE_SKIP_S" in DISPATCHER_SRC)

# ── RAIL 3 — R-A dispatches exactly as R did ───────────────────────────────
print("― rail 3: `R-A` dispatches exactly as R does today ―")
for i, s in enumerate(LIVE_STATUSES):
    tid = 300 + i
    got, skipped = triage([task(id=tid, assignee="R-A", status=s,
                                ai_eligible=True)])
    ok(f"assignee=R-A, ARMED, status={s!r} → dispatches",
       got == [tid], f"skipped={[x[1] for x in skipped]}")
got, skipped = triage([task(id=310, assignee="R-A", status="Todo")])
ok("assignee=R-A, UNARMED but in Todo → dispatches (legacy arm preserved)",
   got == [310], f"skipped={[x[1] for x in skipped]}")

# The same release brief rail 1 refuses for R, now assigned R-A: it must run.
ra_brief = dict(brief, id=311, assignee="R-A",
                checklist=[{"owner": "R-A", "title": "Ship via a release train",
                            "body": "merge -> gate -> deploy-watch -> log",
                            "mode": "execute", "done": False}])
got, _ = triage([ra_brief])
ok("the SAME release brief, assigned R-A, DOES dispatch", got == [311],
   "the split took the executor off the queue too — that is the task backfiring")

# …and every other gate still applies to the new lane. The split moves the lane,
# it does not exempt it.
print("  · every pre-existing gate still applies to R-A:")
cases = [
    ("needs-review tag", task(id=320, assignee="R-A", status="Todo",
                              tags=["needs-review"]), True),
    ("needs-scoping tag", task(id=321, assignee="R-A", status="Todo",
                               tags=["needs-scoping"]), True),
    ("empty description", task(id=322, assignee="R-A", status="Todo",
                               description="  "), True),
    ("blocked_by not Done", task(id=323, assignee="R-A", status="Todo",
                                 blocked_by=[999]), True),
]
for label, t, expect_stuck in cases:
    got, _ = triage([t])
    sk = skip_for([t], t["id"])
    ok(f"    {label} → refused", got == [], f"dispatched {got}")
    ok(f"    {label} → is_stuck={expect_stuck}",
       sk[1] is expect_stuck, str(sk))

discuss = task(id=324, assignee="R-A", status="Todo",
               checklist=[{"owner": "R-A", "title": "decide", "body": "b",
                           "mode": "discuss", "done": False}])
got, _ = triage([discuss])
sk = skip_for([discuss], 324)
ok("    mode=discuss → refused, waiting-not-stuck",
   got == [] and sk[1] is False, f"{got} {sk}")

mismatch = task(id=325, assignee="R-A", status="Todo",
                checklist=[{"owner": "M", "title": "x", "body": "b",
                            "done": False}])
got, _ = triage([mismatch])
sk = skip_for([mismatch], 325)
ok("    chain/assignee disagree → refused, LOUD (stuck)",
   got == [] and sk[1] is True, f"{got} {sk}")

# `Staged` is terminal to the dispatcher (board #198). It stays that way: the R
# SESSION reads that queue by eye — the flip must not turn Staged into a
# dispatch trigger, or a train would launch itself off an unwritten brief.
for s in TERMINAL:
    got, _ = triage([task(id=330, assignee="R-A", status=s, ai_eligible=True)])
    ok(f"    terminal status {s!r} → refused", got == [], f"dispatched {got}")
ok("    `Staged` is still terminal — R's queue is READ, never auto-dispatched",
   "Staged" in D.TERMINAL_STATUSES)

# ── RAIL 4 — no dispatcher LOGIC change: R walks kevin's path ──────────────
print("― rail 4: NO new gate — R takes the same path `kevin` already takes ―")
for s in LIVE_STATUSES:
    r = skip_for([task(id=1, assignee="R", status=s, ai_eligible=True)], 1)
    k = skip_for([task(id=1, assignee="kevin", status=s, ai_eligible=True)], 1)
    ok(f"status={s!r}: R's skip is identical to kevin's but for the letter",
       r[1] is not None and r[1] == k[1]
       and r[0].replace("R", "kevin", 1) == k[0],
       f"R={r} kevin={k}")
ok("triage_todo() contains no letter check for R / R-A",
   not re.search(r"""["']R-?A?["']\s*(==|!=|in\b)""", DISPATCHER_SRC)
   and '== "R"' not in DISPATCHER_SRC,
   "a hard-coded letter check appeared in the gate — the premise is wrong; "
   "reassign to M (step-2 SOP)")
ok("headless_agents() still selects on status alone",
   "agents?status=eq.headless&select=*" in DISPATCHER_SRC)
ok("the non-headless-assignee skip is a single shared branch",
   DISPATCHER_SRC.count("(not a headless agent)") == 1)
# Rail 4, stated so it SURVIVES A MERGE. The original form asserted
# `git diff origin/dev -- dispatcher.py` was empty — true on this branch, and
# false the moment any OTHER car legitimately edits the dispatcher. It went red
# on the Wave 24 combined-tree simulation next to #232 (which adds
# FINISHED_STATUSES) and #228, neither of which is a regression.
#
# The claim worth protecting is not "nobody touched this file", it is "R/R-A
# needed NO dispatcher logic": if the split had required any, the dispatcher
# would have to NAME the lane. So assert the lane is absent from it. That holds
# on the branch, holds after the merge, and still fails the day someone adds a
# hard-coded R-A branch to the gate — which is the thing this rail exists for.
ok("dispatcher.py names no R-A logic — the split needed zero dispatcher code",
   "R-A" not in DISPATCHER_SRC,
   "\n".join(l for l in DISPATCHER_SRC.splitlines() if "R-A" in l)[:400])

# ── RAIL 5 — the registry-unreachable fallback stays the DOCS lane ─────────
print("― rail 5: the stub fallback cannot dispatch a release ―")
ok("STUB_AGENTS does not contain R", "R" not in D.STUB_AGENTS,
   str(sorted(D.STUB_AGENTS)))
ok("STUB_AGENTS does not contain R-A either — a release train must never run "
   "off a fallback roster with the registry down",
   "R-A" not in D.STUB_AGENTS, str(sorted(D.STUB_AGENTS)))
ok("the stub roster is the DOCS lane only", set(D.STUB_AGENTS) == {"M-A"},
   str(sorted(D.STUB_AGENTS)))
# Prove it through the gate, not just by shape: with the stub as the roster,
# both halves of the release lane are refused.
for letter in ("R", "R-A"):
    D.api = board_api([task(id=1, assignee=letter, status="Todo")])
    stub_out, _ = D.triage_todo(D.STUB_AGENTS, set(), None)
    ok(f"under the STUB roster, a Todo task assigned {letter} is refused",
       stub_out == [], f"dispatched {[t['id'] for t in stub_out]}")

# ── RAIL 6 — the migration is real, inherited, and DML-only ───────────────
print("― rail 6: the registry migration ―")
SQL = open(MIGRATION, encoding="utf-8").read()
body = "\n".join(l for l in SQL.splitlines() if not l.strip().startswith("--"))
ok("INSERTs R-A as headless",
   re.search(r"'R-A'.*?'headless'", body, re.S) is not None)
ok("UPDATEs R to live-session",
   re.search(r"UPDATE\s+agents.*?status\s*=\s*'live-session'.*?letter\s*=\s*'R'",
             body, re.S) is not None)
ok("inherits R's fields BY SELECTION (r.scope/r.boundaries/r.worktree/"
   "r.context_docs/r.prompt_template), so it cannot drift from R's row",
   all(f"r.{c}" in body for c in
       ("scope", "boundaries", "worktree", "context_docs", "prompt_template")))
# R's prompt_template — unlike M's, which was NULL — opens "You are R·auto",
# and the dispatcher composes its own "You are {letter}·auto" header. An
# inherited copy would hand R-A two contradicting identities.
ok("the inherited prompt_template has the LETTER substituted, so R-A is not "
   "told it is R·auto",
   "replace(r.prompt_template" in body and "'R-A·auto'" in body)
ok("idempotent (ON CONFLICT DO NOTHING + a guarded UPDATE)",
   "ON CONFLICT (letter) DO NOTHING" in body
   and "status <> 'live-session'" in body)
ok("DML ONLY — no DDL (that would need Kevin's by-name authorization)",
   not re.search(r"\b(CREATE|ALTER|DROP|TRUNCATE)\b", body, re.I),
   "found DDL: author it, STOP, and hand to M")
ok("the UPDATE is documented as HELD until after the step-3 classification",
   re.search(r"HOLD THIS UNTIL AFTER #229 step 3", SQL) is not None)
ok("the status values it writes are in the API's accepted vocabulary",
   {"headless", "live-session"} <= {
       "live-session", "headless", "dormant", "retired", "ephemeral"})
# It must not quietly rewrite R's scope prose in the same statement: the flip's
# only job is `status`, and R-A's INSERT selects FROM that same row. Falls back
# to a sentinel rather than raising, so DELETING the UPDATE reports as failed
# checks instead of crashing the suite mid-run (same posture as skip_for()).
_u = body.upper().find("UPDATE AGENTS")
stmt2 = body[_u:] if _u >= 0 else "<NO UPDATE STATEMENT AT ALL — scope SET name>"
ok("the flip sets `status` and nothing else",
   stmt2.count("SET") == 1 and "scope" not in stmt2 and "name" not in stmt2,
   stmt2)

# ── RAIL 7 — the board renders it, and it cost NOTHING ────────────────────
print("― rail 7: the board renders R-A as an agent / R as a session, for free ―")
TSX = open(SHARED, encoding="utf-8").read()
MODAL_SRC = open(MODAL, encoding="utf-8").read()
ok("isSessionLane still reads registry status/kind",
   "status === 'live-session'" in TSX and "kind === 'human'" in TSX)
sess = TSX[TSX.index("export const isSessionLane"):]
sess = sess[:sess.index("export const laneKindLabel")]
ok("isSessionLane contains NO hard-coded letter check — not for R either",
   not re.search(r"""['"](?:M|R|R-A|M-A|kevin)['"]""", sess), sess)
ok("the letter fallback is still scoped to an EMPTY registry (API outage) only",
   "reg.size === 0 && SESSION_FALLBACK.has(role)" in TSX)
ok("R-A has its own avatar colour + abbreviation",
   "'R-A': '#5fc9c2'" in TSX and "r === 'R-A' ? 'RA'" in TSX)
ok("R-A is in the assignee + mention fallback lists",
   TSX.count("'R', 'R-A'") >= 2)
ok("the outage fallback treats R as a session (under-promises dispatch)",
   "SESSION_FALLBACK = new Set(['kevin', 'M', 'R'])" in TSX)
# The generic renderers #222 built must be untouched — that is the claim.
# Shape is checked as an INVARIANT, not as a pixel literal: board #243
# parameterised RoleChip's size (the shipping lane demotes the builder to a
# smaller secondary chip), so `session ? 5 : '50%'` is now computed. What #222
# claimed — shape branches on the lane kind and an agent stays a circle — still
# has to hold, and a colour-only regression still fails this.
ok("RoleChip still varies SHAPE, not colour alone",
   re.search(r"borderRadius: session \? [^:\n]+ : '50%'", TSX) is not None)
for label, needle in (
        ("…with a ring on top of the shape", "boxShadow: session"),
        ("LaneKindChip still renders the WORD", "'◧ session'"),
        ("…and its agent counterpart", "'◍ agent'"),
        ("both cues still carry a tooltip", "LANE_KIND_TIP"),
        ("RoleChip still resolves the lane from context",
         "const reg = useAgentRegistry();")):
    ok(label, needle in TSX)
ok("TaskDetailModal still labels the assignee",
   "<LaneKindChip role={task.assignee || ''} />" in MODAL_SRC)
ok("…and the chain-step owner, quiet unless that owner is a session",
   "sessionOnly" in MODAL_SRC)

# Rail 7, stated so it SURVIVES A MERGE. The original diffed `frontend/src/views`
# against origin/dev and required every added non-comment line to be registry
# data. True on this branch; false on any tree that also carries another car's
# frontend work — it went red beside #228's cap control in the Wave 24 simulation,
# which is not a regression.
#
# The claim worth protecting: R-A is rendered from REGISTRY DATA, never from a
# hard-coded letter test. So assert the negative directly — no comparison against
# the literal 'R-A' anywhere in the views. Data lookups (`ASSIGNEES`, colour and
# abbrev maps) are unaffected; a `role === 'R-A'` branch fails it, on the branch
# and forever after. That is #222's "the next live-session lane inherits the
# rendering for free" kept as a test rather than a memory.
# One letter test IS legitimate and pre-dates this task: the abbreviation chain
# `r === 'M-A' ? 'MA' : r === 'R-A' ? 'RA' : r`, which is a lookup table written
# as a ternary (#222 established the shape for M-A). That is data. Anything else
# comparing against 'R-A' would be render logic, and is what this rail forbids.
_offending = [l.strip() for l in (TSX + "\n" + MODAL_SRC).splitlines()
              if re.search(r"[=!]==\s*'R-A'|'R-A'\s*[=!]==", l)
              and "? 'RA'" not in l]
ok("no 'R-A' letter test outside the abbreviation lookup — rendering stays registry-driven",
   not _offending, " | ".join(_offending[:3]))
ok("R-A IS present as registry data (colour/abbrev/list), so the negative above is meaningful",
   "R-A" in TSX)
# The "exactly one frontend file" rail was diff-based and died on a merged tree.
# Its first restatement — "R-A appears in NO other view" — was ALSO wrong, and
# board #245 proved it within hours: the shipping lane legitimately needs to know
# which lanes own a ship step (`SHIP_STEP_OWNERS = ['R', 'R-A']`), because R-A is
# the lane that ships. Naming a lane as DATA is not the leak this rail exists to
# catch.
#
# What it actually protects: R-A must never be RENDER LOGIC — no view may branch
# on the literal. So the rail is now a letter-test ban across every view, with the
# abbreviation lookup exempted as before. Data lists pass; `role === 'R-A'` does not.
_offenders = []
for _p in glob.glob(os.path.join(VIEWS, "*.tsx")):
    for _line in open(_p, encoding="utf-8").read().splitlines():
        if re.search(r"[=!]==\s*'R-A'|'R-A'\s*[=!]==", _line) and "? 'RA'" not in _line:
            _offenders.append(f"{os.path.basename(_p)}: {_line.strip()[:60]}")
ok("no view BRANCHES on the literal 'R-A' — it may be data, never render logic",
   not _offenders, " | ".join(_offenders[:3]))

# ── the charter records the split (the SSOT Kevin named) ──────────────────
print("― charter §1/§2/§3/§8 ―")
CH = open(CHARTER, encoding="utf-8").read()
ok("§1 has an R-A row", "**R-A — Release Executor**" in CH)
ok("§1 marks R as a live session that is never auto-dispatched",
   re.search(r"\*\*R — Release\*\*.*LIVE SESSION — never auto-dispatched", CH)
   is not None)
ok("the dividing line is stated as a table",
   "`R` — session, the CONDUCTOR" in CH and "`R-A` — headless, the EXECUTOR" in CH)
ok("it records that R·auto is NOT the problem (27 runs / 24 ok / 17 trains)",
   "is not the problem" in CH and "27 runs, 24 `ok`, 17 trains shipped" in CH)
ok("it records what M actually absorbed on 07-30 — the load being removed",
   "rewrote Wave 21's brief" in CH and "three times" in CH)
ok("M's remaining role in releases is stated as marking a task Staged",
   "M's remaining role in releases: mark a task `Staged`" in CH)
ok("the session is recorded as the FALLBACK when automation fails",
   "fallback" in CH.lower() and "automation" in CH.lower())
ok("the two lessons from M's split are carried over (heartbeat, disarm-first)",
   "goes quiet" in CH and "disarm before restatusing" in CH.lower())
ok("§2 roster lists R-A", "| R-A — Release Executor |" in CH)
ok("§2 roster records R as the persistent conductor",
   "| R — Release (conductor) |" in CH)
ok("§3 names the -A convention and retires E2 rather than repurposing it",
   "`<LETTER>-A`" in CH and "retire" in CH and "E2" in CH)
ok("§8 dispatches the train to R-A, and hands a failure back to R (not M)",
   "`assignee=R-A`" in CH and "hands back to R, not to M" in CH)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
