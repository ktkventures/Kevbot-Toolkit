#!/usr/bin/env python3
"""Team dispatcher (V4.23) — dispatches board tasks to headless Claude agents.

V4.23 (board #195): PUSH AT RUN END. A headless agent can build, test and COMMIT
but is TOLD it cannot push (see the GIT CONTRACT below), and in practice most
runs left their work on local disk only — visible solely to whoever read
preflight invariant (2). reap() now pushes the run's own branch. Rails, each with
a test that fails without it: never a protected branch (dev/main/master), never
--force, never a dirty worktree, never a branch this run did not create (HEAD must
have moved OFF the branch the run started on AND the branch must not already exist
on origin), and FAIL OPEN + LOUD — a failed push logs and leaves preflight (2) to
catch the remainder, it never fails the run or the reap. Kill switch: touch
tools/team_dispatcher/NO_PUSH. The pushed branch is recorded on run_history
(pushed_branch/pushed_at) so the #193 dashboard can show built → pushed → PR'd →
merged without re-deriving it from git.

WHERE the branch is is DISCOVERED, not assumed — the fix for the first cut, which
passed every test and never once fired. It anchored the push to the agent's
REGISTERED worktree, but charter §4 has each lane work in a FRESH worktree
(`git worktree add ../Kevbot-<lane> -b <branch> origin/dev`), so at reap the
registered tree was still parked on dev and every push declined as `protected`.
So: `git worktree list` is snapshotted at DISPATCH, and reap pushes from the
worktrees that were not in that snapshot (plus the registered one, if the run did
move it). Ownership is inferred conservatively — a worktree another in-flight run
could equally have created is left alone, because publishing a peer's unfinished
branch would also poison that peer's own push (`exists-on-origin`).
V4.22 (board #198, step 8 audible): `Staged` JOINS THE TERMINAL SET. Kevin's
ruling landed after V4.21 shipped: "if something is staged, maybe you consider
that something similar to done or blocked". `Staged` means REVIEWED, brief held,
waiting on a release train, and the actor is R executing a brief M wrote
(charter §8) — an agent self-dispatching from `Staged` bypasses the brief, the
artifact that makes a release auditable. This deliberately REVERSES the "#193
bonus" V4.21 noted (an armed `Staged` task self-dispatching to R·auto): worth
having as a considered train-builder design, not as a side effect of an
eligibility flag. See `TERMINAL_STATUSES`.

V4.21 (board #198): ELIGIBILITY, NOT STATUS, GATES DISPATCH. The gate was
`status=eq.Todo`, which made the kanban column do two unrelated jobs at once —
WHERE a task sits in its lifecycle, and MAY an AI touch it — and the second kept
corrupting the first (tasks shoved back to Todo purely to re-arm them between
chain steps). The gate is now `ai_eligible = true OR status = 'Todo'`: ADDITIVE,
so every task on the board today dispatches exactly as it did and the new arm is
opt-in per task (same shape as the process chain's new-shape keys, same shape as
a default-OFF engine flag). `status` is now WRITE-ONLY for the dispatcher — it
sets In Progress/Review, and no task is ever refused for not being in Todo.
The Todo query was also the accidental re-dispatch guard (a finished run moved
the task to Review, which dropped it out of the query), so this version replaces
that side-effect with two STATUS-BLIND rails: a step-guard (no repeat of a
chain step a finished run already covered) and a completed-chain refusal. See
`triage_todo`, `step_sig`, `step_already_ran`. The one thing status still says is
FINALITY: a terminal task is never dispatched, armed or not
(`TERMINAL_STATUSES`) — that is not the permission coupling #198 removed.

V4.19 (board #143, V2.12): @-MENTION DELIVERY. A dispatched agent's prompt now
carries that lane's UNSEEN mentions — the same `task_mentions` rows F's Messages
tab reads — and they are marked seen only AFTER spawn() has the prompt. Every leg
fails open: mentions are optional context and must never be able to stall a
dispatch. See tools/team_dispatcher/mentions.py.

V4.18 (board #182 Step 7, second half): THE INBOUND CHECK. Before starting its own
step, a dispatched agent first verifies that the PREVIOUS step delivered what its
SOP promised; if something explicitly promised is missing it raises an issue to M
and stops instead of building on a bad hand-off. This replaces the separate "TM
gate" agent design (see docs/_active/TM_Gate_Verification_2026-07-29.md): the goal
never required a new agent class, and giving the job to the next agent in line adds
no spawn, no concurrency pool, no daily-cap bypass and no unenforceable tool-less
guarantee — while giving the checker BETTER context, since it is the consumer of
the hand-off. Skipped for the first step of a chain and for kevin-owned steps.

V4.17 (board #182 Step 7): STEP-LEVEL DISPATCH. For a #182 process chain the
dispatcher sends the CURRENT STEP — its title + its own SOP body — instead of "the
whole task", and refuses to hand a `mode='discuss'` step to a headless agent (it
would build something nobody asked for). Assignee stays authoritative; the chain
adds a second, narrower gate ON TOP of it, never instead of it.

V4.16 (board #171): `assignee` is authoritative — the checklist no longer gates
dispatch (the `next_actor()` override silently killed the Todo queue for ~41h on
07-26/27), and organic-queue skips are now LOUD + deduped (log-on-change, comment-
once, staleness tripwire) instead of a silent `continue`.

DRY-RUN BY DEFAULT: prints/logs what it WOULD dispatch, makes ZERO writes.
Live mode requires --live. Runs on Kevin's machine only (never Railway).

Usage:
  python3 dispatcher.py                    # one dry-run pass
  python3 dispatcher.py --loop             # dry-run every POLL_S seconds
  python3 dispatcher.py --live             # real dispatching (claims, spawns, reports)
  python3 dispatcher.py --live --loop --poll 20   # serve Run buttons (board #109)

Kill switches: touch tools/team_dispatcher/PAUSE (idles the loop);
touch tools/team_dispatcher/NO_PUSH (disables push-at-run-end, #195);
per-agent: registry status must be 'headless' to be dispatchable at all.
Design: docs/_active/Design_Agent_Dispatcher.md (decisions approved 2026-07-23).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader"
ENV = f"{REPO}/src/.env"
CHARTER = f"{REPO}/docs/_active/Session_Charters.md"
PAUSE_FILE = f"{HERE}/PAUSE"
NO_PUSH_FILE = f"{HERE}/NO_PUSH"   # board #195 kill switch, PAUSE family
STATE_FILE = f"{HERE}/state.json"
LOG_DIR = f"{HERE}/logs"

# Board #143 — @-mention delivery. Imported DEFENSIVELY: mentions are optional
# context, so even an import failure (file removed, syntax error, half-applied
# deploy) has to leave dispatch behaving exactly as it did in V4.18. Every call
# site is guarded by `if mentions is not None`.
try:
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import mentions
except Exception as _mentions_import_err:  # pragma: no cover — belt and braces
    mentions = None
    print(f"WARN: @-mention injection disabled ({_mentions_import_err}); "
          f"dispatch continues", flush=True)

CONCURRENCY = 3          # Kevin 07-23; blocked_by is the long-term governor
DAILY_CAP = 24           # circuit breaker only
POLL_S = 900
RUN_TIMEOUT_S = 45 * 60  # lease
CLAUDE_BIN = "claude"

# Misc caps (board #135) — every timeout/truncation knob in one place.
API_TIMEOUT_S = 15         # Supabase REST call timeout
THREAD_COMMENTS = 5        # recent thread comments quoted in the prompt
COMMENT_SNIP = 400         # chars kept of each quoted comment
REAP_TAIL = 3000           # chars of log tail reap works from
LEASE_COMMENT_TAIL = 800   # chars of tail quoted in the lease-expired comment
RESULT_COMMENT_MAX = 6000  # chars of agent result posted as a task comment
LOG_TAIL_DB_MAX = 4000     # chars of log tail stored in run_history
STALE_SKIP_S = 4 * 3600    # board #171 tripwire: a STUCK Todo (headless-assigned,
                           # hard-gated) non-dispatchable this long → one-time flag
GIT_TIMEOUT_S = 120        # board #195: any single git call in the push leg
PUSH_BASE = "origin/dev"   # "has this branch got work on it?" is measured vs dev
PROTECTED_BRANCHES = ("dev", "main", "master")  # never pushed by the loop, ever

# Board #109 (Registry Phase 2): the Run button is DECLARATIVE — the API adds
# this tag + a run_history row (outcome='requested'); this --loop is what
# EXECUTES button presses. Requested tasks jump the queue (still eligibility-
# gated); the tag is cleared on claim. Ineligible requests are cleared LOUDLY
# (comment + run_history 'ignored') instead of rotting on the task.
RUN_REQUESTED_TAG = "run-requested"

# Board #198 — THE DISPATCH GATE, as a PostgREST filter. `ai_eligible` is the
# ARMING switch ("may an AI run this"); `status` is the kanban POSITION ("where
# is it in its lifecycle"). The OR keeps the legacy Todo queue working unchanged
# while the new arm is opt-in per task, so shipping this changes nothing about
# what runs today: every existing row is ai_eligible=false
# (migrations/dev_tasks_ai_eligible.sql) and every Todo task still matches.
# `eq.true` rather than `is.true` because Step 2 round-tripped that exact
# spelling against the real PostgREST endpoint.
GATE_FILTER = "or=(ai_eligible.eq.true,status.eq.Todo)"

# Board #198 AUDIBLE (inserted by M review of step 4; `Staged` added by the
# step-8 audible on Kevin's 07-29 ruling) — STATUS AS FINALITY.
# #198 removed status-as-PERMISSION; it did not make a FINISHED task
# dispatchable. Each entry earns its place:
#   `Done`   = the work is over.
#   `Blocked`= parked, and reap() itself puts a failed dispatch there.
#   `Staged` = REVIEWED and waiting on a RELEASE, which is R's job via the brief
#              M writes (charter §8) — not a dispatch. An agent self-dispatching
#              from `Staged` would ship without the brief, the artifact that
#              makes a release auditable. It is not an odd third entry: DO NOT
#              "clean it up" as inconsistent with Done/Blocked.
# None of the three is work to hand an agent, armed or not.
#
# `Staged` here deliberately REVERSES the "#193 bonus" V4.21 noted — an armed
# `Staged` task self-dispatching to R·auto and building half the release train
# for free. Kevin's call: that is a train-builder design to make on purpose, not
# a side effect of an eligibility flag.
#
# Reading status to refuse a TERMINAL state is NOT the coupling #198 removed —
# it is status as finality, and it is the ONE status read in the gate that
# withholds anything. DO NOT "clean this up" as leftover Todo coupling.
#
# Why the other rails do not already cover it: step_already_ran() is bounded by
# the dispatcher's rolling run state, and step_sig() of a CHAINLESS task is the
# constant "task" — measured on the live board 07-29, 129 of 132 Done tasks are
# chainless and 89 are assigned to a headless lane, so the step-guard has no
# current step to protect them with. The hole is reachable by the ORDINARY flow:
# Kevin stamps a contained task (the stamp arms it) → it runs → M closes it →
# `Done` + armed + chainless + headless assignee, with nothing disarming it. It
# would re-run finished work and eat the 24-run DAILY_CAP.
# The API also disarms on close (dev_tasks.update_task) — hygiene, and it makes
# the data honest — but that depends on every future close path remembering,
# whereas THIS rail cannot be forgotten. Both, deliberately.
TERMINAL_STATUSES = ("Done", "Blocked", "Staged")

# #182 Step 7 — inbound check. Owners whose completed steps are NOT re-checked by
# the next agent. Kevin's stamps are approvals, not deliverables; gating them would
# put an agent in the position of auditing his sign-off.
INBOUND_SKIP_OWNERS = ("kevin",)

# Fallback roster until the agents registry (V2.6) is live. Only lanes listed
# here can dispatch in fallback mode — Phase B = docs lane only.
STUB_AGENTS = {
    "M": {"letter": "M", "status": "headless", "worktree": REPO,
          "scope": "docs/organization lane", "boundaries": "Edit only under docs/; board API; no git push, no deploys, no engine files, no prod-DB writes, no flags"},
}


def _creds():
    url = key = None
    for line in open(ENV, encoding="utf-8"):
        line = line.strip()
        if line.startswith("SUPABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
            key = line.split("=", 1)[1].strip().strip('"')
    return url.rstrip("/"), key


def api(method, path, body=None, prefer=None):
    url, key = _creds()
    headers = {"apikey": key, "Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{url}/rest/v1/{path}", data=data,
                                 headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as r:
        txt = r.read().decode()
        return json.loads(txt) if txt.strip() else None


def headless_agents():
    try:
        rows = api("GET", "agents?status=eq.headless&select=*")
        if rows:
            return {r["letter"]: r for r in rows}, "registry"
    except Exception:
        pass
    return STUB_AGENTS, "stub (registry not live yet — docs lane only)"


def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"runs": []}


def save_state(st):
    json.dump(st, open(STATE_FILE, "w"), indent=1)


def today_run_count(st):
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(1 for r in st["runs"] if r["ts"][:10] == today)


def active_runs(st):
    now = time.time()
    return [r for r in st["runs"] if r.get("active") and now - r["t0"] < RUN_TIMEOUT_S]



# ── #182 process-chain helpers ───────────────────────────────────────────────
# Mirrors api/routers/dev_tasks.py (_NEW_SHAPE_KEYS / _is_process_chain /
# _step_owner / _current_step_index) so the dispatcher and the API can never
# disagree about which step is current. Keep the two in sync.
_NEW_SHAPE_KEYS = ("id", "owner", "title", "body", "mode", "origin", "stamp")


def is_process_chain(checklist):
    """True once ANY step carries a new-shape key — the opt-in signal that a
    checklist is a #182 process chain and not a legacy {role,text,done} list."""
    return isinstance(checklist, list) and any(
        isinstance(s, dict) and any(k in s for k in _NEW_SHAPE_KEYS)
        for s in checklist)


def step_owner(step):
    o = step.get("owner")
    return o if o is not None else step.get("role")


def step_title(step):
    t = step.get("title")
    return t if t is not None else (step.get("text") or "")


def current_step(task):
    """(index, step) of the first incomplete step, or (None, None) when the
    chain is finished or the task has no chain."""
    cl = task.get("checklist") or []
    if not is_process_chain(cl):
        return None, None
    for i, s in enumerate(cl):
        if isinstance(s, dict) and not s.get("done"):
            return i, s
    return None, None


def step_sig(task):
    """Identity of the WORK a dispatch would cover — the current chain step, or
    the whole task when there is no chain (board #198).

    Deliberately status-blind. It answers the only question the re-dispatch guard
    needs: "has the chain moved on since the last run, or is the SAME step about
    to run a second time?". Includes the title as well as the index so an audible
    step inserted ahead of the current one reads as new work rather than a
    repeat."""
    idx, step = current_step(task)
    if step is None:
        return "task"          # legacy checklist, no checklist, or chain finished
    return f"{idx}:{step_title(step)[:80]}"


def step_already_ran(st, task):
    """True when a FINISHED dispatched run already covered this task's current
    step (board #198).

    Why this exists: the pre-#198 gate was `status=eq.Todo`, so reap()'s move to
    `Review` doubled as the re-dispatch guard — a completed task simply fell out
    of the query. `ai_eligible` does not fall out of anything, so without this
    rail an armed task whose step was left un-ticked (measured 4-for-4 on 07-29,
    board #202) would re-dispatch the SAME step every poll and eat the daily cap.
    The chain advancing is what re-arms it: a new current step is a new
    signature, which is exactly Kevin's "the chain re-arms itself".

    Status-blind by construction — it reads the dispatcher's OWN run log, not the
    board. `st=None` (unit tests, the `dispatchable()` shim) makes it inert.

    #197 COUPLING (unshipped at the time of writing): that branch returns a
    FAILED run's task to its pre-dispatch status so the next poll retries it.
    Once both land, `report_failed_run()` must drop this task's finished runs
    from state (`forget_task_runs(st, task_id)`) or an armed task's auto-retry
    will be refused here. Refusing is the SAFE side of that race — it matches
    today's behaviour, where a failed run parks the task in Review — so the two
    can land in either order without a runaway."""
    if not st:
        return False
    sig = step_sig(task)
    return any(r.get("task_id") == task["id"] and r.get("step_sig") == sig
               and not r.get("active") for r in st.get("runs", []))


def forget_task_runs(st, task_id):
    """Drop a task's finished runs from the step-guard's view, re-arming it for
    another pass at its CURRENT step (board #198). Nothing calls this yet: it is
    the hook #197's retry leg needs — see step_already_ran()."""
    st["runs"] = [r for r in st.get("runs", [])
                  if r.get("task_id") != task_id or r.get("active")]
    return st["runs"]


def triage_todo(agents, done_ids, st=None):
    """Classify the dispatch queue → (dispatchable, skipped).

    Board #198 — ELIGIBILITY GATES, STATUS DESCRIBES. The query is the OR of
    `ai_eligible` and the legacy `status=Todo` queue (GATE_FILTER), and no task
    is ever refused here for sitting in the wrong lifecycle column. `status` is
    read in exactly TWO places below, neither of them a permission read:
      • TERMINAL_STATUSES — finality. `Done`/`Blocked`/`Staged` is not "not
        allowed yet", it is "there is nothing HERE to hand an agent"; see the
        constant for why the step-guard cannot cover a chainless closed task
        (audible, step 5) and why `Staged` belongs (audible, step 8).
      • the arm SELECTOR — it picks which arm's BEHAVIOUR applies, so the
        pre-#198 Todo queue keeps behaving byte-identically (a finished chain
        sitting in Todo still dispatches on assignee alone — board #182) while
        the new arm gets the tighter rails.
    Nothing here refuses a task for NOT being in Todo, which was the coupling.

    Board #171 — ASSIGNEE IS AUTHORITATIVE, the checklist is advisory. The old
    `next_actor()` gate let an un-ticked checklist step OVERRIDE the assignee and
    silently made the WHOLE Todo queue undispatchable for ~41h on 07-26/27 (no
    error, log or comment). That gate is gone: a Todo task dispatches iff its
    `assignee` is a headless agent and no HARD gate blocks it. Nothing reads the
    checklist for control flow any more, which also makes the #151 stamp-tick bug
    non-blocking retroactively.

    `skipped` is [(task, reason, is_stuck)] for LOUD reporting (silence was the
    deeper defect — see process_skips):
      is_stuck=False → assignee is not a headless agent (kevin/human). The task
                       NATURALLY waits; this is the designed behaviour, and it is
                       exactly what Kevin sorts on. Quiet.
      is_stuck=True  → assignee IS a headless lane but a hard gate blocks it
                       (blocked_by / needs-review|scoping / empty description). A
                       real blocker or an unforeseen gate bug hides here → loud."""
    tasks = api("GET", f"dev_tasks?{GATE_FILTER}&select=*"
                       "&order=is_urgent.desc,priority_phase,priority_seq")
    out, skipped = [], []
    for t in tasks:
        if t.get("status") in TERMINAL_STATUSES:
            # HARD RAIL, first in the gate: a finished, parked or release-staged
            # task is never dispatched, however it got armed (see
            # TERMINAL_STATUSES — finality, not permission). Quiet, not stuck: a
            # closed task is not a stuck QUEUE, and flagging it would comment on
            # a finished thread and then trip the 4h staleness tripwire on it
            # every day forever. `Staged` is quiet for the same reason — it waits
            # on a release train, which is R's job, not a stalled dispatch. The
            # SKIP line below still names it on the pass its reason first appears.
            skipped.append((t, f"status is {t['status']} — terminal; a finished, "
                               f"parked or release-staged task is never "
                               f"dispatched, armed or not (board #198 audible)",
                            False))
            continue
        a = t.get("assignee")
        if a not in agents:
            skipped.append((t, f"waiting on {a or '—'} (not a headless agent)", False))
            continue
        tags = t.get("tags") or []
        held = [x for x in ("needs-review", "needs-scoping") if x in tags]
        if held:
            skipped.append((t, f"tagged {'+'.join(held)}", True))
            continue
        if not (t.get("description") or "").strip():
            skipped.append((t, "description is empty — unscoped, never dispatched", True))
            continue
        unmet = [b for b in (t.get("blocked_by") or []) if b not in done_ids]
        if unmet:
            skipped.append((t, f"blocked by {unmet} (not Done)", True))
            continue
        # #182 Step 7 — chain gates. Only for process chains; legacy checklists
        # are untouched (they gate on assignee alone, exactly as in V4.16).
        idx, step = current_step(t)
        if step is not None:
            owner = step_owner(step)
            if owner != a:
                # Assignee and the chain disagree. NOT a silent skip — that is
                # the #171 bug. Loud, because one of the two is wrong.
                skipped.append((t, f"step {idx + 1} is owned by {owner or '—'} "
                                   f"but assignee is {a} — chain/assignee disagree", True))
                continue
            if (step.get("mode") or "execute") == "discuss":
                # SAFETY-CRITICAL: a headless agent handed a discussion step
                # builds something nobody asked for. Waiting, not stuck — so it
                # never trips the 4h staleness tripwire.
                skipped.append((t, f"step {idx + 1} is mode=discuss — needs a REPLY, "
                                   f"not a build; never dispatched to a headless agent",
                                False))
                continue
        # ── Board #198 rails. The `status=eq.Todo` query used to be the accidental
        # re-dispatch guard: reap() moved a finished task to Review, which dropped
        # it out of the query. An armed task never drops out of anything, so the
        # guard has to be stated instead of inherited — status-blind, from the
        # dispatcher's own run log and the chain itself.
        # THE ONE STATUS READ (see the docstring): purely an arm selector. A task
        # in Todo keeps its pre-#198 behaviour to the letter, including the legacy
        # fallback where a fully-ticked chain still dispatches on assignee alone
        # (board #182 walk 4). Nothing here refuses a task for NOT being Todo.
        legacy_todo_arm = t.get("status") == "Todo"
        if not legacy_todo_arm:
            if step is None and is_process_chain(t.get("checklist") or []):
                # Every step ticked. The chain IS the scope (#182), so there is
                # nothing left to hand an agent — dispatching here would re-run
                # the whole task from the top. Waiting on a human to sign off and
                # close, which is the designed state: quiet, no tripwire.
                skipped.append((t, "chain complete — every step is ticked; nothing "
                                   "left to dispatch (sign off and close it, or add "
                                   "a step)", False))
                continue
            if step_already_ran(st, t):
                # A finished run already covered this exact step. Classified as
                # WAITING, not stuck: the task is waiting on a human to review the
                # output / tick the step, and marking it stuck would post a
                # "dispatch skipped" comment on EVERY dispatched task one poll
                # after its result. The log line below still names it every pass.
                where = (f"step {idx + 1}" if step is not None else "this task")
                skipped.append((t, f"{where} already had a finished dispatched run — "
                                   f"waiting on review / the step tick; the chain "
                                   f"advancing is what re-arms it (board #198)", False))
                continue
        out.append(t)
    return out, skipped


def dispatchable(agents, done_ids, st=None):
    """Back-compat shim: the dispatchable subset only. one_pass() calls
    triage_todo() directly because it also needs the skip diagnostics."""
    return triage_todo(agents, done_ids, st)[0]


def process_skips(st, skipped, live, exclude_ids):
    """LOUD, deduped skip reporting for the ORGANIC queue (board #171).

    The pre-#171 queue skipped tasks with a bare `continue` — a task could fail a
    gate every poll for days emitting NOTHING; there was no signal separating "the
    queue is empty" from "the queue is full of permanently-stuck work". This gives
    the organic queue the same loud treatment run_requested() already gives button
    presses. Per pass:
      • log a skip whose reason CHANGED since last pass (dedupe = not spam);
      • comment ONCE per STUCK task (headless-assigned but hard-gated), mirroring
        clear_run_request()'s loud cleanup so the thread records it;
      • staleness tripwire: a stuck task non-dispatchable > STALE_SKIP_S gets a
        one-time flag comment — the backstop for gate bugs nobody predicted.
    `exclude_ids` = tasks the run-request path already reported loudly, so we do
    not double-comment. Dedupe state lives in st['skips'] (survives restarts)."""
    skips = st.setdefault("skips", {})
    seen, now = set(), time.time()
    stuck_ids, waiting_ids = [], []
    for t, reason, is_stuck in skipped:
        tid = str(t["id"])
        seen.add(tid)
        if t["id"] in exclude_ids:
            continue  # already reported by the run-request path this pass
        (stuck_ids if is_stuck else waiting_ids).append(t["id"])
        prev = skips.get(tid)
        if prev is None or prev.get("reason") != reason:
            skips[tid] = {"reason": reason, "since": now,
                          "commented": False, "flagged": False}
            print(f"  SKIP #{t['id']} ({t.get('assignee') or '—'}): {reason}"
                  f"{' [STUCK]' if is_stuck else ''}", flush=True)
        rec = skips[tid]
        if not is_stuck:
            continue  # human-waiting is the designed state: no comment, no tripwire
        if live and not rec["commented"]:
            api("POST", "dev_task_comments", body={
                "task_id": t["id"], "author": "system",
                "body": f"⛔ dispatch skipped: {reason}. assignee "
                        f"'{t.get('assignee') or '—'}' is a headless lane, so this is a real "
                        f"blocker — clear it or reassign. (Won't re-comment until the reason "
                        f"changes; board #171.)"})
            rec["commented"] = True
        age_h = (now - rec["since"]) / 3600.0
        if live and not rec["flagged"] and (now - rec["since"]) >= STALE_SKIP_S:
            api("POST", "dev_task_comments", body={
                "task_id": t["id"], "author": "system",
                "body": f"⚠️ STALENESS TRIPWIRE (board #171): this Todo has been "
                        f"non-dispatchable for ~{age_h:.1f}h — reason: {reason}. Flagging as a "
                        f"possible stuck-queue / gate bug; needs a human look."})
            print(f"  TRIPWIRE #{t['id']} stuck ~{age_h:.1f}h: {reason}", flush=True)
            rec["flagged"] = True
    # Forget entries no longer skipped (dispatched or gate cleared) so a future
    # re-stick re-comments and the dedupe map can't grow unbounded.
    for tid in [k for k in skips if k not in seen]:
        print(f"  UNSTUCK #{tid} (dispatched or gate cleared)", flush=True)
        del skips[tid]
    if stuck_ids or waiting_ids:
        print(f"  skips: {len(stuck_ids) + len(waiting_ids)} "
              f"(stuck={stuck_ids or '—'} · waiting={waiting_ids or '—'})", flush=True)
    save_state(st)


def run_requested(agents, done_ids):
    """Button-requested tasks (board #109) — served ahead of the organic
    queue, FIFO by request time (tag-add bumps updated_at). Spec gates only:
    description non-empty, not blocked, agent headless-enrolled OR stub —
    deliberately NO next-actor / needs-review / status=Todo gates (the button
    is an explicit human override of queue ORDER). Hard refusals: In Progress /
    Done / Blocked (can't be started) and Scoping / needs-scoping (not
    workable yet — the button never overrides that). Returns
    (eligible, [(task, reason)])."""
    rows = api("GET", f"dev_tasks?tags=cs.{{{RUN_REQUESTED_TAG}}}&select=*"
                      "&order=updated_at")
    ok, bad = [], []
    for t in rows or []:
        # Board #198 leaves this list ALONE, deliberately. These are the BUTTON's
        # refusals — a human just pressed Run on a specific task — and they are
        # about "this cannot be started" / "this is not workable yet", not about
        # the eligibility question #198 moved onto `ai_eligible`. The organic gate
        # in triage_todo() no longer refuses anything for its status; loosening
        # the button to honour a press on a stage it currently refuses is a design
        # call for Kevin/M, not a mechanical consequence of the gate change.
        # Step-8 audible note: this list and TERMINAL_STATUSES now AGREE on all
        # three terminal stages — Done, Blocked and Staged are refused on both
        # paths — so the V4.21 divergence on `Staged` closed itself. Nothing here
        # was changed to achieve that.
        # The button overrides queue ORDER, never "not workable yet" (M, #109
        # review): Scoping status / needs-scoping tag are hard refusals. The
        # board-#136 pipeline stages joined the list — Approval isn't approved
        # yet, Review/Staged already ran (claim-time net for tasks moved after
        # the button press; the request endpoint refuses them up front).
        if t.get("status") in ("In Progress", "Done", "Blocked", "Scoping",
                               "Approval", "Review", "Staged"):
            bad.append((t, f"status is {t['status']}"))
        elif "needs-scoping" in (t.get("tags") or []):
            bad.append((t, "tagged needs-scoping — not workable yet"))
        elif not (t.get("description") or "").strip():
            bad.append((t, "description is empty — never dispatch unscoped work"))
        elif t.get("assignee") not in agents:
            bad.append((t, f"assignee '{t.get('assignee') or '—'}' is not headless-enrolled"))
        elif any(b not in done_ids for b in (t.get("blocked_by") or [])):
            bad.append((t, f"blocked by {t['blocked_by']}"))
        else:
            ok.append(t)
    return ok, bad


def clear_run_request(task, reason):
    """LOUD ineligible-request cleanup (live mode only): drop the tag, mark
    the pending run_history row 'ignored', explain on the thread."""
    tags = [x for x in (task.get("tags") or []) if x != RUN_REQUESTED_TAG]
    api("PATCH", f"dev_tasks?id=eq.{task['id']}", body={"tags": tags})
    api("PATCH", f"run_history?task_id=eq.{task['id']}&outcome=eq.requested",
        body={"finished_at": datetime.now(timezone.utc).isoformat(),
              "outcome": "ignored", "log_tail": reason})
    api("POST", "dev_task_comments", body={
        "task_id": task["id"], "author": "system",
        "body": f"run-request ignored: {reason} — tag cleared, re-request once fixed"})


def rh_claim(task, run_id):
    """run_history lifecycle, claim leg: requested → running. Button runs
    already have a row (outcome='requested', from the API endpoint) — take
    the oldest pending one; organic queue dispatches insert theirs here."""
    now = datetime.now(timezone.utc).isoformat()
    rows = api("GET", f"run_history?task_id=eq.{task['id']}&outcome=eq.requested"
                      "&order=requested_at&limit=1&select=id")
    if rows:
        api("PATCH", f"run_history?id=eq.{rows[0]['id']}",
            body={"run_id": run_id, "started_at": now, "outcome": "running"})
        return
    api("POST", "run_history", body={
        "task_id": task["id"], "agent_letter": task.get("assignee") or "?",
        "run_id": run_id, "started_at": now, "outcome": "running"})


def rh_finish(run_id, outcome, log_tail):
    """run_history lifecycle, finish leg (reap): running → terminal."""
    api("PATCH", f"run_history?run_id=eq.{run_id}", body={
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome, "log_tail": (log_tail or "")[-LOG_TAIL_DB_MAX:]})


def mentions_for(role):
    """This lane's unseen @-mentions as (block_text, ids_to_mark). Board #143.

    Belt AND braces: mentions.for_dispatch() already fails open internally, but
    the CALL SITE swallows too — the rail is that no state of that module (broken,
    stubbed, half-deployed, refactored) can stall a dispatch, and internal
    fail-open only protects against the failures its author foresaw."""
    if mentions is None:
        return "", []
    try:
        return mentions.for_dispatch(role)
    except Exception as e:  # noqa: BLE001 — optional context, never fatal
        print(f"  mentions: lookup failed for {role} ({e}) — dispatching without",
              flush=True)
        return "", []


def mentions_delivered(role, ids):
    """Mark delivered mentions seen. Call ONLY after the prompt has reached
    spawn(). Fail-open: an unmarked mention is shown twice; a mention marked
    before delivery is lost."""
    if mentions is None or not ids:
        return 0
    try:
        marked = mentions.mark_seen(ids)
        print(f"  mentions: {marked}/{len(ids)} marked seen for {role}", flush=True)
        return marked
    except Exception as e:  # noqa: BLE001 — re-delivers next pass, never fatal
        print(f"  mentions: mark-seen failed for {role} ({e}) — will re-deliver",
              flush=True)
        return 0


def build_prompt(agent, task, mentions_block=""):
    """Board #135: the ONE-SHOT CONTRACT block exists because two real runs
    (#121/#68) armed background suite-watchers and ended their turn — a
    headless `claude -p` process exits with its final message and reap kills
    the whole session group, so backgrounded work silently never happens."""
    comments = api("GET", f"dev_task_comments?task_id=eq.{task['id']}"
                          f"&order=created_at.desc&limit={THREAD_COMMENTS}&select=author,body")
    thread = "\n".join(f"- {c['author']}: {c['body'][:COMMENT_SNIP]}" for c in reversed(comments or []))
    cl = task.get("checklist") or []
    idx, cur = current_step(task)
    steps = "\n".join(
        f"- [{'x' if s.get('done') else ' '}] ({step_owner(s) or ''}) {step_title(s)}"
        for s in cl)
    # #182 Step 7 — STEP-LEVEL DISPATCH. When the task is a process chain the
    # agent is told to do ONE step and stop, and is handed that step's own SOP
    # plus what earlier steps concluded. This is the agent-quality payoff: M was
    # hand-writing per-dispatch context that the task already contained.
    if cur is not None:
        prior = [f"  {i + 1}. ({step_owner(s) or '—'}) {step_title(s)}"
                 for i, s in enumerate(cl) if s.get("done")]
        # #182 Step 7 (second half) — THE INBOUND CHECK. Kevin's goal was "a quick
        # check that what was promised was delivered, before it goes to the next
        # agent". The first design gave that job to a separate TM agent; this gives
        # it to the agent that was already being dispatched. Two reasons it is
        # better, not just cheaper: (1) it adds NO new agent — no extra spawn, no
        # second concurrency pool, no daily-cap bypass, no run_history gap, and no
        # tool-posture guarantee that the CLI cannot actually enforce; (2) the next
        # agent is the CONSUMER of the hand-off, so it can ask the question that
        # matters — "did step N give me what I need?" — where a blinded gate could
        # only ask "does this text look like it addresses the SOP?".
        # Deliberately NOT gated: the last step of a chain (no successor — Review
        # covers it) and steps owned by kevin (his stamps are not operationally
        # gated, carried over from the TM design).
        inbound = ""
        if idx > 0:
            p = cl[idx - 1]
            p_owner = step_owner(p) or "—"
            if p_owner not in INBOUND_SKIP_OWNERS:
                p_sop = (p.get("body") or "").strip()
                inbound = f"""
=== FIRST — INBOUND CHECK ON STEP {idx} (do this before anything else) ===
Step {idx} ({p_owner}) — "{step_title(p)}" — was handed to you as complete.
What that step promised:
{p_sop or '(no SOP body — judge against what the thread says it set out to do; '
          'if that is not determinable, proceed)'}

Ask ONE question: did step {idx} deliver what it promised — enough for you to do YOUR step?
  • YES — or it is merely imperfect, debatable or unpolished → say nothing and proceed.
    You are checking DELIVERY, not correctness. Depth is not your job here.
  • NO — something it EXPLICITLY promised is missing → do NOT start your step. Post a
    comment beginning "⚠️ Raise-an-issue: step {idx} hand-off", name the SPECIFIC missing
    thing, set `assignee` to `M`, and stop.

RAISING AN ISSUE IS A SUCCESS, NOT A FAILURE. You are not being obstructive, and you are
not graded on reaching your own step. A bad hand-off caught here costs minutes; the same
one caught three steps later costs the chain. But if it is genuinely ambiguous, PROCEED —
do not block on a maybe.
=== END INBOUND CHECK ===
"""
        step_block = f"""{inbound}
=== YOUR STEP — STEP {idx + 1} of {len(cl)} — DO THIS ONE AND STOP ===
Owner: {step_owner(cur) or '—'}   Mode: {cur.get('mode') or 'execute'}\
{'   [AUDIBLE — inserted mid-flight, not in the original plan]' if cur.get('origin') == 'audible' else ''}

{step_title(cur)}

SOP for this step:
{(cur.get('body') or '(no SOP body written for this step — if that leaves the '
  'requirement ambiguous, DO NOT GUESS: reassign to M and say what is missing.)')}

Already completed on this task (their conclusions are in the thread below):
{chr(10).join(prior) or '  (none — this is the first step)'}

SCOPE: do step {idx + 1} only. Do NOT start later steps, even if they look
adjacent or trivially related. If you believe the chain is wrong — steps should
be merged, split or reordered — RAISE IT: reassign to M with the reason. Do not
expand scope unilaterally. M owns the chain (board #182 authoring standard).
=== END OF YOUR STEP ==="""
    else:
        step_block = ""
    return f"""You are {agent['letter']}·auto, a headless dispatched agent on the RoR Trader team.
Identity/scope: {agent.get('scope','')}
HARD BOUNDARIES (violating any = abort and report): {agent.get('boundaries','')}
Charter (read first): {CHARTER}

YOUR TASK — board #{task['id']}: {task['title']}
Description:
{task.get('description') or '(none)'}
Process chain (assignee drives dispatch — board #171; the chain scopes the work — board #182):
{steps or '(none)'}
{step_block}
Recent thread:
{thread or '(none)'}
{(chr(10) + mentions_block + chr(10)) if mentions_block else ''}
ONE-SHOT CONTRACT: you are a single headless process on a {RUN_TIMEOUT_S // 60}-minute
lease. The moment your final message ends, the process exits and its whole process
group is KILLED — background shells, watchers, monitors, scheduled wakeups and
"notify me when done" hooks die with it and NEVER fire. Run every step synchronously
and wait for it in-process: a long step (test suite, backtest) is run in the
foreground to completion, budgeted against the lease — never backgrounded to "check
on later". Anything you cannot finish in-process goes in your final report as
remaining work, not into the background.

GIT CONTRACT (board #153 — cut from a clean base): if you create a branch, cut it
from LATEST origin/dev, never from your worktree's current HEAD:
`git fetch origin && git branch <name> origin/dev` (or `git checkout -b <name> origin/dev`).
A branch cut from worktree HEAD inherits whatever unmerged commits sit there and
carries them into your diff (tonight one branch dragged in 10 unrelated commits).
`git log origin/dev..HEAD --oneline` must show ONLY your own commits before you report
ready. You cannot push (headless) and must NOT try to background one — COMMIT your work
and name any branch you created in your final report. When your run ends the dispatcher
pushes that branch for you (board #195) IF the worktree is clean and the branch is your
own; an uncommitted change is the one thing that makes your work unpushable.

ASSIGNEE CONTRACT (board #171 — assignee = whoever the task is WAITING ON): the dispatcher
now routes on `assignee` alone; the checklist is advisory display only. Before your final
message, set this task's `assignee` to whoever it now waits on:
  • blocked on Kevin's input/decision before it can progress → set assignee to `kevin` and
    state exactly what you need (a comment ALONE is not a handoff — the reassignment is);
  • handing to another lane → set assignee to that letter (M/E/F/P/R);
  • done and awaiting M sign-off → leave assignee as-is; the dispatcher moves you to Review.
Leaving the task assigned to YOU while you are blocked is the failure mode this rule kills.
Reassign via `PATCH /api/dev-tasks/<id>` setting `assignee` ONLY — do NOT change status.

Do the task. Your FINAL message becomes a comment on task #{task['id']} — make it a
self-contained result report (what you did, files touched, what needs review). If you
completed checklist steps, list them as 'STEP DONE: <text>' lines. Do not change task
status; do not touch anything outside your scope."""


def spawn(agent, task, prompt, log_path):
    with open(log_path, "w") as lf:
        return subprocess.Popen(
            [CLAUDE_BIN, "-p", prompt, "--output-format", "json"],
            cwd=agent_worktree(agent), stdout=lf,
            stderr=subprocess.STDOUT, start_new_session=True)


def kill_run_group(pid):
    """SIGKILL a run's whole session — spawn() uses start_new_session=True, so
    the child leads its own process group and killpg reaches lingering
    grandchildren too (board #131). Then reap the zombie (bounded wait) so
    /proc/{pid} clears; a fresh dispatcher process isn't the parent, in which
    case init reaps it."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    for _ in range(10):
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            return
        if done:
            return
        time.sleep(0.05)


def parse_terminal(full):
    """Terminal result object of a `claude -p --output-format json` run, or
    None while the run is still mid-stream. Reverse scan: a lingering child
    that inherited stdout can append stray lines AFTER the terminal JSON
    (board #131, run r1785000766-122), so the last line alone isn't
    trustworthy. Parses whole lines of the FULL log — the terminal JSON is ONE
    long line, and a pre-parse tail truncates it into a false outcome=error
    (M, #109 review)."""
    for line in reversed(full.strip().splitlines()):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and (obj.get("type") == "result" or "result" in obj):
            return obj
    return None


# ── board #195: push at run end ──────────────────────────────────────────────
# A headless agent commits but cannot push. Without this leg every successful
# run leaves its branch on local disk until a human reads preflight (2) — which
# is a lucky catch, not a process. Everything here is advisory: it may decline,
# and it may fail, but it must NEVER fail a run or a reap.

def _git(worktree, *args, timeout=GIT_TIMEOUT_S):
    """(rc, stdout, stderr) of one git call in `worktree`. Raises only on
    timeout/OS failure — every caller sits under try_push_run_branch()."""
    p = subprocess.run(["git", "-C", worktree, *args],
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def agent_worktree(agent):
    return agent.get("worktree") or REPO


def current_branch(worktree):
    """Checked-out branch name, or None if detached / not a git worktree /
    anything at all went wrong. Used at DISPATCH time too, where it must not be
    able to stop a spawn."""
    try:
        rc, out, _ = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")
    except Exception:  # noqa: BLE001 — never blocks a dispatch
        return None
    if rc != 0 or out in ("", "HEAD"):
        return None
    return out


def list_worktrees(path):
    """Every worktree root of the repo containing `path`, realpath'd.

    [] on ANY failure — like current_branch() this runs at DISPATCH time, where
    it must never be able to stop a spawn. An empty snapshot degrades the
    fresh-worktree leg to "off" for that run, never to "push anything".
    """
    try:
        rc, out, _ = _git(path, "worktree", "list", "--porcelain")
    except Exception:  # noqa: BLE001 — never blocks a dispatch
        return []
    if rc != 0:
        return []
    return [os.path.realpath(ln[len("worktree "):].strip())
            for ln in out.splitlines() if ln.startswith("worktree ")]


def _decline(tag, reason, msg, worktree=None):
    print(f"  {tag}: {msg}", flush=True)
    return {"pushed": False, "branch": None, "reason": reason,
            "worktree": worktree, "results": []}


def push_worktree_branch(wt, tag, branch0=None):
    """Every rail, applied to ONE worktree. Returns the same dict shape as
    push_run_branch(): {"pushed", "branch", "reason", "worktree"}.

    `branch0` is the branch this worktree was on at dispatch, and is only
    meaningful for the run's REGISTERED worktree — half 1 of "a branch this run
    did not create" is "HEAD moved off it". None means the rail does not apply,
    which is the case for a worktree that did not exist at dispatch (its very
    existence is a stronger half 1) and for a run that started detached.
    """
    branch = current_branch(wt)
    if branch is None:
        return _decline(tag, "detached", f"SKIP — {wt} is on a detached HEAD", wt)
    if branch in PROTECTED_BRANCHES:
        return _decline(tag, "protected",
                        f"REFUSED — {branch} ({wt}) is a protected branch; the "
                        "loop pushes agent branches only", wt)
    if branch0 is not None and branch == branch0:
        return _decline(tag, "not-this-runs-branch",
                        f"REFUSED — {wt} never left {branch}, the branch the run "
                        "started on; this run did not create it", wt)
    # A dirty tree means the agent did not finish cleanly. Pushing the committed
    # half of an unfinished state is worse than pushing nothing.
    rc, out, err = _git(wt, "status", "--porcelain")
    if rc != 0:
        return _decline(tag, "status-failed",
                        f"SKIP — `git status` failed in {wt}: {err[:200]}", wt)
    if out:
        return _decline(tag, "dirty",
                        f"REFUSED — {branch} has {len(out.splitlines())} uncommitted "
                        "path(s); a partial state is worse than no push", wt)
    rc, out, err = _git(wt, "rev-list", "--count", f"{PUSH_BASE}..{branch}")
    if rc != 0 or not out.isdigit():
        return _decline(tag, "rev-list-failed",
                        f"SKIP — cannot count {PUSH_BASE}..{branch}: {err[:200]}", wt)
    if int(out) == 0:
        return _decline(tag, "nothing-ahead",
                        f"SKIP — {branch} has no commits ahead of {PUSH_BASE}", wt)
    ahead = int(out)
    # "A branch this run did not create", half 2: it must be new to origin.
    # Anything already published is someone else's history to advance.
    rc, out, err = _git(wt, "ls-remote", "--heads", "origin", branch)
    if rc != 0:
        return _decline(tag, "ls-remote-failed",
                        f"SKIP — cannot prove {branch} is new on origin: {err[:200]}",
                        wt)
    if out:
        return _decline(tag, "exists-on-origin",
                        f"REFUSED — {branch} already exists on origin; advancing a "
                        "published branch is not this loop's job", wt)
    # Plain push. No --force, ever: the branch is provably new on origin, so
    # there is nothing to force past.
    rc, out, err = _git(wt, "push", "-u", "origin", branch)
    if rc != 0:
        detail = (err or out)[-400:]
        print(f"  {tag}: PUSH FAILED for {branch} — {detail}", flush=True)
        return {"pushed": False, "branch": branch, "worktree": wt,
                "reason": f"push-failed: {detail}"}
    print(f"  {tag}: PUSHED {branch} from {wt} "
          f"({ahead} commit(s) ahead of {PUSH_BASE})", flush=True)
    return {"pushed": True, "branch": branch, "worktree": wt, "reason": "ok"}


def fresh_worktrees(r, anchor, tag, other_active=()):
    """Worktree roots that did not exist when this run was dispatched.

    THIS is the shape that matters, and the one the first cut of #195 missed:
    charter §4 has every lane work in a FRESH worktree
    (`git worktree add ../Kevbot-<lane> -b <branch> origin/dev`), so at reap the
    agent's REGISTERED worktree is still parked on dev — the push anchored to it
    declined as `protected` and the feature never fired. The work is in a
    directory that did not exist at dispatch, so that is what we look for.

    Ownership is inferred, not known, so it is inferred CONSERVATIVELY: a
    worktree is only this run's if no other run still in flight could also have
    created it. Another active run whose own dispatch snapshot lacks the
    worktree predates it too — ambiguous, decline and let preflight (2) have it.
    Pushing a peer's branch mid-run is the one genuinely harmful outcome here:
    it would publish unfinished work AND poison that peer's own push at reap
    (`exists-on-origin`).
    """
    if "worktrees0" not in r:
        return []                                    # pre-#195 run record
    before = set(r["worktrees0"] or ())
    now = list_worktrees(anchor)
    if not now:
        print(f"  {tag}: SKIP fresh-worktree leg — cannot list worktrees "
              f"under {anchor}", flush=True)
        return []
    out = []
    for w in now:
        if w in before:
            continue
        rival = next((o for o in other_active
                      if w not in set(o.get("worktrees0") or ())), None)
        if rival is not None:
            print(f"  {tag}: SKIP {w} — run {rival.get('run_id')} is still in "
                  "flight and predates it too; owner ambiguous", flush=True)
            continue
        out.append(w)
    return out


def _aggregate(tag, results):
    """Collapse per-worktree results into the run-level answer reap() acts on.

    Normally there is exactly one target and this is a pass-through. A run CAN
    produce two (it moved its registered tree AND cut a fresh worktree), so
    `branch` is comma-joined; `results` always carries the full detail.
    """
    pushed = [x for x in results if x["pushed"]]
    if pushed:
        return {"pushed": True, "reason": "ok", "results": results,
                "branch": ",".join(x["branch"] for x in pushed),
                "worktree": pushed[0]["worktree"]}
    if not results:
        return _decline(tag, "no-candidate",
                        "SKIP — no worktree carries this run's work (registered "
                        "tree still on its start branch, none created since "
                        "dispatch)")
    # A push that ERRORED outranks a refusal: it is the one outcome that needs
    # a human, so it must be the reason reap() sees.
    failed = [x for x in results if x["reason"].startswith("push-failed")]
    lead = failed[0] if failed else results[0]
    return {"pushed": False, "branch": lead["branch"], "reason": lead["reason"],
            "worktree": lead["worktree"], "results": results}


def push_run_branch(r, other_active=()):
    """Push this run's OWN branch(es), or explain (loudly) why not.

    Returns {"pushed": bool, "branch": str|None, "reason": str, "results": [...]}.
    Every refusal is a named reason so the reap log reads as a decision, not a
    silence. Two places the work can be, and both are checked:
      A. the run's REGISTERED worktree, if the run moved it onto a new branch;
      B. a worktree created AFTER dispatch — the charter §4 default.
    """
    tag = f"push[{r.get('run_id')}]"
    if os.path.exists(NO_PUSH_FILE):
        return _decline(tag, "kill-switch",
                        "SKIP — NO_PUSH kill switch present "
                        "(rm tools/team_dispatcher/NO_PUSH to re-enable)")
    wt = r.get("worktree")
    if not wt or not os.path.isdir(wt):
        return _decline(tag, "no-worktree",
                        f"SKIP — no usable worktree on the run record ({wt!r}); "
                        "pre-#195 run → preflight (2) still covers it")
    if "branch0" not in r and "worktrees0" not in r:
        return _decline(tag, "no-start-branch",
                        "SKIP — the run recorded neither a start branch nor a "
                        "worktree snapshot, so 'did this run create it?' is "
                        "unanswerable")
    results = []
    if "branch0" in r:
        results.append(push_worktree_branch(wt, tag, branch0=r["branch0"]))
    for new_wt in fresh_worktrees(r, wt, tag, other_active):
        results.append(push_worktree_branch(new_wt, tag))
    return _aggregate(tag, results)


def try_push_run_branch(r, other_active=()):
    """FAIL-OPEN wrapper — the outer rail. push_run_branch() handles the
    failures it foresaw; this one exists for the rest (git missing, subprocess
    timeout, a worktree that vanished mid-reap). A reap must complete."""
    try:
        return push_run_branch(r, other_active)
    except Exception as e:  # noqa: BLE001 — a push can never fail a reap
        print(f"  push[{r.get('run_id')}]: ERROR — {e} (run reported normally; "
              f"preflight (2) still covers the branch)", flush=True)
        return {"pushed": False, "branch": None, "worktree": None,
                "reason": f"error: {e}", "results": []}


def rh_record_push(run_id, branch):
    """Record the pushed branch on run_history so #193's dashboard can show
    built → pushed → PR'd → merged without re-deriving it from git. Its OWN
    try/except: an un-migrated DB (no pushed_branch column → PostgREST 400)
    must not be able to break a reap that has already reported."""
    try:
        api("PATCH", f"run_history?run_id=eq.{run_id}", body={
            "pushed_branch": branch,
            "pushed_at": datetime.now(timezone.utc).isoformat()})
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  push[{run_id}]: pushed {branch} but run_history did not record it "
              f"({e}) — apply src/migrations/run_history_pushed_branch.sql", flush=True)
        return False


def report_push_failure(task_id, run_id, branch, reason):
    """LOUD half of fail-open: a push that ERRORED is the one outcome a human has
    to act on, so it lands on the thread next to the agent's report. Refusals
    (dirty, protected, not-ours) stay in the log — they are the design working."""
    try:
        api("POST", "dev_task_comments", body={
            "task_id": task_id, "author": "system",
            "body": f"⚠️ auto-push FAILED (run {run_id}, branch `{branch}`) — the "
                    f"work is COMMITTED BUT UNPUSHED on local disk. Reason:\n"
                    f"```\n{reason[:1000]}\n```"})
    except Exception as e:  # noqa: BLE001 — reporting the failure can't be fatal
        print(f"  push[{run_id}]: could not post the push-failure comment ({e})",
              flush=True)


def reap(st):
    """Collect finished live runs → post comments, move the task to Review.
    Runs in state were always live-dispatched, so reporting is unconditional.
    Collect triggers (board #131): process gone; lease expired (kill +
    Blocked); or the log already carries the terminal result JSON while the
    process lingers — those report normally and the leftover session group is
    killed. Iterates ALL active-flagged runs, NOT active_runs(): its
    within-lease filter excluded exactly the runs the lease branch exists for,
    making that branch unreachable. Board #136: Review STATUS replaced the
    needs-review tag — output done, awaiting sign-off (M always; Kevin closes
    iff kevin_final)."""
    for r in [x for x in st["runs"] if x.get("active")]:
        proc_alive = os.path.exists(f"/proc/{r['pid']}")
        expired = time.time() - r["t0"] >= RUN_TIMEOUT_S
        full = open(r["log"]).read() if os.path.exists(r["log"]) else "(no log)"
        last = parse_terminal(full)
        if proc_alive and not expired and last is None:
            continue  # still running, within lease
        r["active"] = False
        tail = full[-REAP_TAIL:]
        if proc_alive:
            kill_run_group(r["pid"])  # lease kill, or a completed run's leftovers
        if last is None and expired and proc_alive:
            api("POST", "dev_task_comments", body={
                "task_id": r["task_id"], "author": "system",
                "body": f"dispatch LEASE EXPIRED ({r['agent']}·auto run {r['run_id']}) — killed. Log tail:\n{tail[-LEASE_COMMENT_TAIL:]}"})
            api("PATCH", f"dev_tasks?id=eq.{r['task_id']}",
                body={"status": "Blocked", "notes": "dispatch failure — see thread"})
            rh_finish(r["run_id"], "lease-expired", tail)
        else:
            # Only what we POST/store is capped: comment RESULT_COMMENT_MAX,
            # run_history LOG_TAIL_DB_MAX. No terminal JSON + dead proc =
            # run died mid-stream → error.
            result = last.get("result", tail) if last is not None else tail
            outcome = ("error" if last.get("is_error") else "ok") if last is not None else "error"
            api("POST", "dev_task_comments", body={
                "task_id": r["task_id"], "author": f"{r['agent']}·auto",
                "body": str(result)[:RESULT_COMMENT_MAX]})
            api("PATCH", f"dev_tasks?id=eq.{r['task_id']}",
                body={"status": "Review"})
            api("POST", "dev_task_comments", body={
                "task_id": r["task_id"], "author": "system",
                "body": f"status: In Progress → Review (by dispatcher — "
                        f"run {r['run_id']} finished, output awaiting sign-off)"})
            rh_finish(r["run_id"], outcome, tail)
        # Board #195 — LAST, and after rh_finish: reporting is the job, the push
        # is the follow-through. Runs on the lease-expired leg too (a run that
        # committed and then hung is exactly the work most likely to be lost;
        # the dirty-tree rail declines the ones that were killed mid-edit).
        # The still-active runs go with it: a worktree they could also have
        # created is not this run's to push (see fresh_worktrees). `r` is no
        # longer active by here, so it cannot be its own rival.
        res = try_push_run_branch(
            r, [x for x in st["runs"] if x is not r and x.get("active")])
        if res["pushed"]:
            r["pushed_branch"] = res["branch"]
            rh_record_push(r["run_id"], res["branch"])
        elif res["reason"].startswith("push-failed"):
            report_push_failure(r["task_id"], r["run_id"], res["branch"],
                                res["reason"])
    save_state(st)


def one_pass(live, only_task=None):
    if os.path.exists(PAUSE_FILE):
        # flush=True on every print: --loop stdout is usually redirected to a
        # file, where block buffering hides hours of output (board #131).
        print("PAUSED (remove tools/team_dispatcher/PAUSE to resume)", flush=True)
        return
    agents, source = headless_agents()
    st = load_state()
    reap(st)
    slots = CONCURRENCY - len(active_runs(st))
    cap_left = DAILY_CAP - today_run_count(st)
    done_ids = {t["id"] for t in api("GET", "dev_tasks?status=eq.Done&select=id")}
    # Button presses first (priority-jump), then the organic Todo queue.
    requested, bad_requests = run_requested(agents, done_ids)
    req_ids = {t["id"] for t in requested}
    organic, skipped = triage_todo(agents, done_ids, st)
    cands = requested + [t for t in organic if t["id"] not in req_ids]
    # Board #171: LOUD, deduped reporting for the organic queue's skips — only in
    # the continuous --loop (a one-shot --task run must not comment on unrelated
    # tasks). Excludes ids the run-request path already reported this pass.
    if only_task is None:
        process_skips(st, skipped, live,
                      req_ids | {t["id"] for t, _ in bad_requests})
    if only_task is not None:
        hit = [t for t in cands if t["id"] == only_task]
        if not hit:
            why = next((r for t, r, _ in skipped if t["id"] == only_task), None)
            detail = (f" — {why}" if why else
                      " (not AI-eligible and not Todo / wrong lane / blocked / "
                      "this step already ran)")
            print(f"task #{only_task} is NOT eligible{detail} — nothing dispatched",
                  flush=True)
        cands = hit
    now = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"[{now}] agents={list(agents)} ({source}) · dispatchable={len(cands)} "
          f"(button-requested={len(requested)}, skipped={len(skipped)}) "
          f"· slots={slots} · cap_left={cap_left} · mode={'LIVE' if live else 'DRY-RUN'}",
          flush=True)
    for t, reason in bad_requests:
        if live:
            clear_run_request(t, reason)
            print(f"  CLEARED ineligible run-request #{t['id']}: {reason}", flush=True)
        else:
            print(f"  WOULD CLEAR ineligible run-request #{t['id']}: {reason}", flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # Per-lane serialization (board #132): ONE active run per agent letter.
    # All of a letter's runs share one worktree, so same-lane concurrency
    # collides on git/file state (a registry flip mid-loop launched 3 E·auto
    # runs into one tree, 07-25). Seeded from live runs and grown per
    # dispatch, so a lane can't double up within a pass either. Skipped tasks
    # aren't claimed (button tags survive) — they retry when the lane frees.
    busy = {r["agent"] for r in active_runs(st)}
    budget = max(0, min(slots, cap_left))
    for t in cands:
        if budget <= 0:
            break
        if t["assignee"] in busy:
            print(f"  SKIP #{t['id']} — {t['assignee']}·auto already has an active run "
                  f"(one per lane)", flush=True)
            continue
        busy.add(t["assignee"])
        budget -= 1
        agent = agents[t["assignee"]]
        # Board #143 — this lane's unseen @-mentions ride along in the prompt.
        # Nothing is marked seen here: the prompt has not reached a process yet.
        m_block, m_ids = mentions_for(t["assignee"])
        prompt = build_prompt(agent, t, mentions_block=m_block)
        run_id = f"r{int(time.time())}-{t['id']}"
        log_path = f"{LOG_DIR}/{run_id}.log"
        flag = " [run-requested]" if t["id"] in req_ids else ""
        if not live:
            open(f"{LOG_DIR}/{run_id}.DRY.prompt.txt", "w").write(prompt)
            print(f"  WOULD DISPATCH #{t['id']}{flag} '{t['title'][:60]}' → {t['assignee']}·auto "
                  f"(prompt saved: {run_id}.DRY.prompt.txt)", flush=True)
            continue
        # Claim = status flip + run-requested tag cleared in ONE patch.
        tags = [x for x in (t.get("tags") or []) if x != RUN_REQUESTED_TAG]
        api("PATCH", f"dev_tasks?id=eq.{t['id']}",
            body={"status": "In Progress", "tags": tags})
        api("POST", "dev_task_comments", body={
            "task_id": t["id"], "author": "system",
            "body": f"dispatched to {t['assignee']}·auto (run {run_id})"})
        rh_claim(t, run_id)
        p = spawn(agent, t, prompt, log_path)
        # Board #195 — the run records WHERE it ran, WHICH branch it started on,
        # and WHICH worktrees already existed. reap()'s push leg needs all three:
        # the worktree to act in, the start branch to answer "did this run move
        # off it?", and the snapshot to answer "which worktree did this run
        # create?" — charter §4 has agents cut a fresh one, so that is where the
        # work usually is. All captured at dispatch: by reap time none of it is
        # recoverable.
        # `step_sig` (board #198) is what stops an armed task re-running the step
        # it just ran: recorded at CLAIM, read by step_already_ran() once the run
        # is no longer active. Recorded for every run, both arms, so the guard
        # works the moment a task is armed later.
        wt = agent_worktree(agent)
        st["runs"].append({"run_id": run_id, "task_id": t["id"], "agent": t["assignee"],
                           "pid": p.pid, "t0": time.time(), "log": log_path,
                           "worktree": wt, "branch0": current_branch(wt),
                           "worktrees0": list_worktrees(wt),
                           "step_sig": step_sig(t),
                           "ts": datetime.now(timezone.utc).isoformat(), "active": True})
        save_state(st)
        # ONLY NOW are the mentions delivered — spawn() has the prompt and the
        # run is recorded. Marking earlier would lose a message on any failure
        # between here and there; marking later (or never, if this errors) only
        # costs a duplicate showing on the next dispatch.
        mentions_delivered(t["assignee"], m_ids)
        print(f"  DISPATCHED #{t['id']} → {t['assignee']}·auto pid={p.pid}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--task", type=int, default=None,
                    help="dispatch ONLY this task id (must still pass eligibility)")
    ap.add_argument("--poll", type=int, default=POLL_S,
                    help="seconds between --loop passes (lower it when the "
                         "loop is serving Run buttons — board #109)")
    args = ap.parse_args()
    while True:
        one_pass(args.live, only_task=args.task)
        if not args.loop:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
