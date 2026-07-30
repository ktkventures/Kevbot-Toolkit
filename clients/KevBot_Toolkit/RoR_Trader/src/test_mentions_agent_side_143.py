"""Acceptance test — board #143 step 3: the AGENT side of @-mentions.

F's half (PR #102) writes task_mentions and gives Kevin a Messages tab. This
covers the other direction: a lane's UNSEEN mentions are injected into a
dispatched agent's prompt and into a human role session's SessionStart context,
and are marked seen ONLY once they have actually been delivered.

Under test:
  tools/team_dispatcher/mentions.py                    (shared implementation)
  tools/team_dispatcher/dispatcher.py                  (prompt inject + mark timing)
  tools/team_dispatcher/hooks/team_board_context.py    (SessionStart inject)

THE RAIL, and why most of these checks exist: FAIL OPEN. The dispatcher went
silently dead for 41h once (board #171) and must not gain a second way to do it,
so every mentions leg — fetch, enrich, render, mark, even the module import —
is exercised in its broken state to prove dispatch still happens. And marking
seen must never RACE the fail-open path: walk 4 pins the ordering (spawn first,
mark after), because a mention marked seen but never delivered is a lost message.

Hermetic: real modules, fake api(), fake spawn(), temp state — no network, no DB.
Run:  .venv/bin/python src/test_mentions_agent_side_143.py   (from RoR_Trader root)
"""
import importlib.util
import io
import os
import sys
import types
from contextlib import redirect_stdout

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
TD = os.path.join(ROOT, "tools", "team_dispatcher")

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# mentions FIRST and under its real module name, so the dispatcher's own
# `import mentions` and the hook's lazy import both resolve to THIS object.
sys.path.insert(0, TD)
M = load("mentions", os.path.join(TD, "mentions.py"))
disp = load("team_dispatcher", os.path.join(TD, "dispatcher.py"))
hook = load("team_board_context", os.path.join(TD, "hooks", "team_board_context.py"))

ok("dispatcher imported the mentions module (not the disabled fallback)",
   disp.mentions is M)


# ── fake transport for mentions.py ──────────────────────────────────────────
MCALLS = []
UNSEEN = []          # rows the fake DB serves as unseen
MFAIL = {"get": False, "patch": False, "enrich": False}


def fake_mapi(method, path, body=None):
    MCALLS.append((method, path, body))
    if method == "GET" and path.startswith("task_mentions?mentioned="):
        if MFAIL["get"]:
            raise RuntimeError("supabase down")
        return list(UNSEEN)
    if method == "GET" and path.startswith("task_mentions?seen_at=is.null"):
        return [{"mentioned": "M"}, {"mentioned": "M"}, {"mentioned": "F"}]
    if method == "GET" and path.startswith("dev_tasks?id=in."):
        if MFAIL["enrich"]:
            raise RuntimeError("join failed")
        return [{"id": 143, "title": "V2.12 · Mentions + unread Messages tab",
                 "status": "Todo"}]
    if method == "GET" and path.startswith("dev_task_comments?id=in."):
        if MFAIL["enrich"]:
            raise RuntimeError("join failed")
        return [{"id": 900, "body": "@M Design is good enough for me for now so if "
                                    "you want to finish it up please proceed."},
                {"id": 901, "body": "@M noting my own follow-up"}]
    if method == "PATCH" and path.startswith("task_mentions?id=in."):
        if MFAIL["patch"]:
            raise RuntimeError("write refused")
        inner = path.split("in.(", 1)[1].split(")", 1)[0]
        return [{"id": int(i)} for i in inner.split(",") if i]
    return []


M._api = fake_mapi


def mention(mid, author="kevin", cid=900, tid=143):
    return {"id": mid, "task_id": tid, "comment_id": cid, "mentioned": "M",
            "author": author, "created_at": "2026-07-28T18:00:00+00:00"}


# ── walk 1: reading the inbox — the query, the enrichment, the fail-open ────
print("― walk 1: fetch + enrich against the table F built ―")
UNSEEN = [mention(1)]
MCALLS.clear()
rows = M.unseen_for("M")
q = [p for m, p, b in MCALLS if m == "GET" and p.startswith("task_mentions?mentioned=")][0]
ok("queries the SAME semantics F's API uses: mentioned=eq.<role> + seen_at IS NULL",
   "mentioned=eq.M" in q and "seen_at=is.null" in q)
ok("bounded by MENTION_LIMIT — a flood can't blow up a prompt",
   f"limit={M.MENTION_LIMIT}" in q)
ok("row is enriched with task title + status + comment excerpt",
   rows[0]["task_title"].startswith("V2.12") and rows[0]["task_status"] == "Todo"
   and "Design is good enough" in rows[0]["excerpt"])

MCALLS.clear()
ok("an empty role never even calls the API", M.fetch_unseen("") == [] and not MCALLS)

MFAIL["get"] = True
ok("FAIL-OPEN: a dead API yields [] rather than raising", M.unseen_for("M") == [])
MFAIL["get"] = False

MFAIL["enrich"] = True
part = M.unseen_for("M")
ok("FAIL-OPEN: a broken enrichment join still returns the mention, un-enriched",
   len(part) == 1 and part[0]["task_title"] == "")
MFAIL["enrich"] = False


# ── walk 2: rendering — what the agent actually reads ───────────────────────
print("― walk 2: the rendered block ―")
ok("nothing to say → empty string (no empty header in the prompt)",
   M.render_block([], "M") == "")
block = M.render_block(M.unseen_for("M"), "M")
ok("names the role and the count", "UNSEEN @MENTIONS FOR M (1)" in block)
ok("carries author, task id and the excerpt",
   "kevin" in block and "#143" in block and "Design is good enough" in block)
ok("carries a clickable link to the task modal", "/admin/tasks?task=143" in block)
ok("warns against scope-creeping off a mention",
   "do NOT silently do it" in block)

UNSEEN = [mention(1, author="kevin"), mention(2, author="M·auto", cid=901)]
block2, ids2 = M.for_dispatch("M")
ok("an agent's OWN byline is not served back to it (no self-mention loop)",
   "noting my own follow-up" not in block2 and "UNSEEN @MENTIONS FOR M (1)" in block2)
ok("...but the suppressed row is still in the mark-seen set, so it can't "
   "accumulate as phantom unread", sorted(ids2) == [1, 2])

MFAIL["get"] = True
ok("FAIL-OPEN: for_dispatch on a dead API is ('', [])", M.for_dispatch("M") == ("", []))
MFAIL["get"] = False


# ── walk 3: marking seen ────────────────────────────────────────────────────
print("― walk 3: mark_seen ―")
MCALLS.clear()
n = M.mark_seen([1, 2])
patch = [(m, p, b) for m, p, b in MCALLS if m == "PATCH"][0]
ok("PATCHes exactly the delivered ids", "id=in.(1,2)" in patch[1])
ok("guards on seen_at IS NULL so a re-mark can't rewrite an older stamp",
   "seen_at=is.null" in patch[1])
ok("stamps seen_at", "seen_at" in (patch[2] or {}))
ok("returns how many rows it stamped", n == 2)
MCALLS.clear()
ok("empty id list is a no-op, no API call", M.mark_seen([]) == 0 and not MCALLS)
MFAIL["patch"] = True
ok("FAIL-OPEN: a failed mark returns 0 and raises nothing (mention re-delivers)",
   M.mark_seen([1]) == 0)
MFAIL["patch"] = False


# ── walk 4: the dispatcher — inject, and mark ONLY after spawn ──────────────
print("― walk 4: dispatcher injection + mark-seen ordering ―")
scratch = os.environ.get("TMPDIR", "/tmp")
disp.STATE_FILE = os.path.join(scratch, "test143_state.json")
disp.LOG_DIR = os.path.join(scratch, "test143_logs")
disp.PAUSE_FILE = os.path.join(scratch, "test143_no_such_pause")
if os.path.exists(disp.STATE_FILE):
    os.remove(disp.STATE_FILE)

AGENT = {"letter": "M", "status": "headless", "worktree": ROOT,
         "scope": "s", "boundaries": "b"}
TASK = {"id": 143, "title": "V2.12 · Mentions", "description": "d", "assignee": "M",
        "tags": [], "blocked_by": [], "checklist": [], "status": "Todo"}

prompt = disp.build_prompt(AGENT, TASK, mentions_block=block)
ok("build_prompt injects the block", "UNSEEN @MENTIONS FOR M" in prompt)
ok("...without disturbing the existing prompt contract",
   "ONE-SHOT CONTRACT" in prompt and "ASSIGNEE CONTRACT" in prompt
   and "Recent thread:" in prompt)
ok("no block → no stray marker in the prompt",
   "@MENTIONS" not in disp.build_prompt(AGENT, TASK))

EVENTS = []


def fake_api(method, path, body=None, prefer=None):
    # Board #198: the gate query is GATE_FILTER (`or=(ai_eligible.eq.true,
    # status.eq.Todo)`), no longer `status=eq.Todo`. Follow the dispatcher's own
    # constant instead of re-pinning a literal, so a future gate change cannot
    # silently make this fake serve an EMPTY board — which is what the stale
    # literal did: no task dispatched, and the mark-seen ordering check below went
    # red for a reason that had nothing to do with mentions. TASK is status=Todo,
    # so it matches either spelling.
    if method == "GET" and path.startswith(f"dev_tasks?{disp.GATE_FILTER}"):
        return [dict(TASK)]
    # Board #232: the blocked_by lookup is the shared FINISHED set now
    # (`status=in.(Done,Closed)`), not the literal `status=eq.Done`.
    if method == "GET" and path.startswith(f"dev_tasks?{disp.FINISHED_FILTER}"):
        return []
    if method == "GET" and path.startswith("dev_task_comments?task_id="):
        return []
    return []


class FakeProc:
    pid = 4242


def fake_spawn(agent, task, prompt, log_path):
    EVENTS.append(("spawn", prompt))
    return FakeProc()


disp.api = fake_api
disp.spawn = fake_spawn
disp.headless_agents = lambda: ({"M": AGENT}, "test")
real_mark = M.mark_seen


def spy_mark(ids, api=None, now=None):
    EVENTS.append(("mark", list(ids)))
    return real_mark(ids, api=api, now=now)


M.mark_seen = spy_mark
UNSEEN = [mention(1)]

with redirect_stdout(io.StringIO()):
    disp.one_pass(live=True)
kinds = [e[0] for e in EVENTS]
ok("a live dispatch spawns and then marks", kinds == ["spawn", "mark"])
ok("the mentions reached the SPAWNED PROMPT — not just the log",
   "UNSEEN @MENTIONS FOR M" in EVENTS[0][1])
ok("marking happens AFTER the spawn (delivery precedes the write)",
   kinds.index("spawn") < kinds.index("mark") and EVENTS[1][1] == [1])

EVENTS.clear()
os.remove(disp.STATE_FILE)
with redirect_stdout(io.StringIO()):
    disp.one_pass(live=False)
ok("DRY-RUN marks NOTHING seen (a rehearsal must not consume the inbox)",
   [e[0] for e in EVENTS] == [])


# ── walk 5: fail-open, end to end — a mentions bug can't stall a dispatch ───
print("― walk 5: fail-open under a totally broken mentions module ―")
EVENTS.clear()
os.remove(disp.STATE_FILE)
broken = types.SimpleNamespace(
    for_dispatch=lambda role: (_ for _ in ()).throw(RuntimeError("boom")),
    mark_seen=lambda ids: (_ for _ in ()).throw(RuntimeError("boom")))
disp.mentions = broken
try:
    with redirect_stdout(io.StringIO()):
        disp.one_pass(live=True)
    dispatched = [e for e in EVENTS if e[0] == "spawn"]
except Exception as e:  # noqa: BLE001 — this failing IS the regression
    dispatched, boom = [], e
    ok("dispatch survived a raising mentions module", False, str(boom))
ok("a mentions module that RAISES still lets the dispatch spawn",
   len(dispatched) == 1)
ok("...and the prompt simply carries no mentions block",
   "@MENTIONS" not in dispatched[0][1])

EVENTS.clear()
os.remove(disp.STATE_FILE)
disp.mentions = None          # the import-failed path
with redirect_stdout(io.StringIO()):
    disp.one_pass(live=True)
ok("mentions=None (import failed) dispatches exactly as V4.18 did",
   [e[0] for e in EVENTS] == ["spawn"])
disp.mentions = M


# ── walk 6: the SessionStart hook ──────────────────────────────────────────
print("― walk 6: SessionStart hook injection ―")
hook.print_api_queue = lambda roles: print("[Team Board] fake board")
# resolve_roles() reads the checkout path / .claude/role marker; pin it so the
# test doesn't depend on where it happens to be running from.
hook.resolve_roles = lambda: (["M"], ROOT)
EVENTS.clear()
UNSEEN = [mention(7)]
buf = io.StringIO()
with redirect_stdout(buf):
    hook.main()
out = buf.getvalue()
ok("a role session sees BOTH the board and its unseen mentions at session start",
   "[Team Board] fake board" in out and "UNSEEN @MENTIONS FOR M" in out)
ok("the board block comes first, mentions last (closest to the prompt)",
   out.index("[Team Board]") < out.index("UNSEEN @MENTIONS"))

EVENTS.clear()
buf = io.StringIO()
with redirect_stdout(buf):
    hook.print_mentions(["M"])
out = buf.getvalue()
ok("hook prints the role's mention block", "UNSEEN @MENTIONS FOR M (1)" in out)
ok("hook marks them seen, and only after printing",
   [e[0] for e in EVENTS] == ["mark"] and EVENTS[0][1] == [7])

EVENTS.clear()
buf = io.StringIO()
with redirect_stdout(buf):
    hook.print_mentions([])
out = buf.getvalue()
ok("NO role resolved → a counts-only digest, not somebody else's inbox",
   "[@Mentions] unread by role" in out and "M:2" in out)
ok("...and a role-less session marks NOTHING seen (a digest can't lose a message)",
   EVENTS == [])

MFAIL["get"] = True
buf = io.StringIO()
with redirect_stdout(buf):
    hook.main()
ok("FAIL-OPEN: a dead mentions API still lets the board block print",
   "[Team Board] fake board" in buf.getvalue())
MFAIL["get"] = False


def explode(_roles):
    raise RuntimeError("hook boom")


hook.print_mentions = explode
buf = io.StringIO()
with redirect_stdout(buf):
    hook.main()          # must not raise — a hook crash costs every session
ok("a raising print_mentions never blocks session start",
   "[Team Board] fake board" in buf.getvalue())

M.mark_seen = real_mark
if os.path.exists(disp.STATE_FILE):
    os.remove(disp.STATE_FILE)

print(f"\nALL PASS ({PASS} checks)")
