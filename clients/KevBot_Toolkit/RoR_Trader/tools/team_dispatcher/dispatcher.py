#!/usr/bin/env python3
"""Team dispatcher (V4.20) — dispatches board tasks to headless Claude agents.

V4.20 (board #197): A FAILED RUN STOPS LYING ABOUT ITS STATUS. reap() used to
move EVERY finished run's task to `Review` ("output awaiting sign-off") — including
runs that died mid-stream on an API 500, a crash or a kill, where there is no
output at all. Three costs: the status lied; the truth (`run_history.outcome`)
was recorded but invisible on the board; and `Review` is not dispatch-eligible,
so a dead run silently PARKED its task until a human noticed — the same shape as
#171's 41-hour stall. Now the terminal status follows the outcome: on error the
task is returned to the status it held BEFORE dispatch (recorded on the run at
claim time, since the board no longer holds it by reap) so a transient failure
self-heals on the next poll, and after MAX_AUTO_RETRIES it escalates to `Blocked`
with the blocker NAMED — matching the lease-expired leg, which always got this
right. `ok` runs are unchanged.

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
FAIL_COMMENT_TAIL = 2000   # chars of tail quoted in the run-FAILED comment (#197);
                           # bigger than the lease one — a dead run's tail IS the
                           # diagnosis, and no agent result comment accompanies it

# Board #197 — automatic retries granted to a task whose run DIED, before the
# dispatcher stops re-arming it and escalates to Blocked. Kevin's ruling in
# step 1: cap at 2, so a transient API 500 self-heals but a genuinely broken
# task cannot loop against the daily cap.
MAX_AUTO_RETRIES = 2

# Board #109 (Registry Phase 2): the Run button is DECLARATIVE — the API adds
# this tag + a run_history row (outcome='requested'); this --loop is what
# EXECUTES button presses. Requested tasks jump the queue (still eligibility-
# gated); the tag is cleared on claim. Ineligible requests are cleared LOUDLY
# (comment + run_history 'ignored') instead of rotting on the task.
RUN_REQUESTED_TAG = "run-requested"

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


def triage_todo(agents, done_ids):
    """Classify the Todo queue → (dispatchable, skipped).

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
    tasks = api("GET", "dev_tasks?status=eq.Todo&select=*"
                       "&order=is_urgent.desc,priority_phase,priority_seq")
    out, skipped = [], []
    for t in tasks:
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
        out.append(t)
    return out, skipped


def dispatchable(agents, done_ids):
    """Back-compat shim: the dispatchable subset only. one_pass() calls
    triage_todo() directly because it also needs the skip diagnostics."""
    return triage_todo(agents, done_ids)[0]


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
ready. You cannot push (headless) — name any branch you created in your final report so
it is not lost; do NOT background a push.

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
            cwd=agent.get("worktree") or REPO, stdout=lf,
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


def retries_used(st, task_id):
    """Automatic retries already granted to this task after a FAILED run (board
    #197). Kept in dispatcher STATE rather than on the board: it is loop
    bookkeeping, not a task property, so it needs no schema change — and a lost
    state.json only ever costs a task two extra self-heal attempts."""
    return int((st.get("retries") or {}).get(str(task_id), 0))


def set_retries(st, task_id, n):
    """Set (n>0) or clear (n==0) a task's retry counter. Cleared on a successful
    run and on escalation to Blocked — a human unblocking the task gets a fresh
    budget rather than a task that can never be auto-retried again."""
    r = st.setdefault("retries", {})
    if n:
        r[str(task_id)] = n
    else:
        r.pop(str(task_id), None)


def report_failed_run(st, r, result, tail):
    """Board #197 — terminal reporting for a run that DIED (API error, crash,
    kill). Never `Review`: that status asserts reviewable output exists, and a
    Review task is not dispatch-eligible, so a dead run used to park its task
    until a human noticed.

    Under the cap the task goes back to the status it held BEFORE dispatch, so
    the next poll re-dispatches it with no human in the loop; once the cap is
    spent it goes `Blocked` with the blocker NAMED, matching the lease leg. One
    system comment carries the failure, the transition and the log tail — the
    agent is NOT credited with a result it never produced."""
    used = retries_used(st, r["task_id"])
    status0 = r.get("status0")
    # Runs claimed before V4.20 carry no status0, and the board no longer holds
    # the pre-dispatch value by reap time. Todo is the queue's home for a
    # dispatchable task and the only fallback that keeps self-heal working —
    # say so in the comment rather than defaulting silently.
    unrecorded = "" if status0 else (
        " — NOTE: this run recorded no pre-dispatch status (claimed by a "
        "pre-V4.20 dispatcher), so Todo was assumed")
    if used < MAX_AUTO_RETRIES:
        restore = status0 or "Todo"
        set_retries(st, r["task_id"], used + 1)
        api("PATCH", f"dev_tasks?id=eq.{r['task_id']}", body={"status": restore})
        # Only the Todo queue re-dispatches on its own — a button run can be
        # claimed from Backlog and would sit there. Promising a retry that will
        # never happen is the same species of lie #197 exists to kill.
        retry_note = (f"automatic retry {used + 1} of {MAX_AUTO_RETRIES}"
                      if restore == "Todo" else
                      f"NOT auto-retried from {restore} — press Run again or move it "
                      f"to Todo (that would be retry {used + 1} of {MAX_AUTO_RETRIES})")
        transition = (f"status: In Progress → {restore} (pre-dispatch status "
                      f"restored; {retry_note}){unrecorded}")
    else:
        set_retries(st, r["task_id"], 0)
        blocker = (f"{MAX_AUTO_RETRIES + 1} consecutive dispatched runs died — "
                   f"{MAX_AUTO_RETRIES} automatic retries spent. A human needs to read "
                   f"the log tails, fix the cause, and move this back to Todo.")
        api("PATCH", f"dev_tasks?id=eq.{r['task_id']}",
            body={"status": "Blocked", "notes": f"dispatch failure — {blocker}"})
        transition = f"status: In Progress → Blocked — blocker: {blocker}"
    api("POST", "dev_task_comments", body={
        "task_id": r["task_id"], "author": "system",
        "body": (f"dispatch run FAILED ({r['agent']}·auto run {r['run_id']}) — the run "
                 f"died before finishing; there is NO output to sign off.\n"
                 f"{transition}\nLog tail:\n{str(result)[:FAIL_COMMENT_TAIL]}")})
    print(f"  REAPED #{r['task_id']} FAILED (run {r['run_id']}) — {transition}",
          flush=True)


def reap(st):
    """Collect finished live runs → post comments, move the task to its
    outcome's terminal status (board #197: `ok` → Review; `error` → pre-dispatch
    status, or Blocked once the retry cap is spent; lease expired → Blocked).
    Runs in state were always live-dispatched, so reporting is unconditional.
    Collect triggers (board #131): process gone; lease expired (kill +
    Blocked); or the log already carries the terminal result JSON while the
    process lingers — those report normally and the leftover session group is
    killed. Iterates ALL active-flagged runs, NOT active_runs(): its
    within-lease filter excluded exactly the runs the lease branch exists for,
    making that branch unreachable. Board #136: Review STATUS replaced the
    needs-review tag — output done, awaiting sign-off (M always; Kevin closes
    iff kevin_final) — which is why #197 reserves it for `ok` runs only."""
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
            # Board #197: the terminal STATUS follows the outcome. Only an `ok`
            # run has output to sign off — an errored one is reported as the
            # failure it was and handed back to the queue (or to a human).
            if outcome == "error":
                report_failed_run(st, r, result, tail)
            else:
                api("POST", "dev_task_comments", body={
                    "task_id": r["task_id"], "author": f"{r['agent']}·auto",
                    "body": str(result)[:RESULT_COMMENT_MAX]})
                api("PATCH", f"dev_tasks?id=eq.{r['task_id']}",
                    body={"status": "Review"})
                api("POST", "dev_task_comments", body={
                    "task_id": r["task_id"], "author": "system",
                    "body": f"status: In Progress → Review (by dispatcher — "
                            f"run {r['run_id']} finished, output awaiting sign-off)"})
                set_retries(st, r["task_id"], 0)  # a good run clears the streak
            rh_finish(r["run_id"], outcome, tail)
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
    organic, skipped = triage_todo(agents, done_ids)
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
                      " (not Todo / wrong lane / blocked / awaiting review)")
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
        # Board #197: a FAILED run is now returned to the queue, so the same task
        # can be re-dispatched within the same SECOND (a run that dies instantly,
        # or a --task pass right after a reap) — and this id is second-resolution.
        # A duplicate would overwrite the previous run's log file and make
        # rh_finish() patch both run_history rows. Suffix instead.
        taken = {r["run_id"] for r in st["runs"]}
        if run_id in taken:
            n = 2
            while f"{run_id}.{n}" in taken:
                n += 1
            run_id = f"{run_id}.{n}"
        log_path = f"{LOG_DIR}/{run_id}.log"
        flag = " [run-requested]" if t["id"] in req_ids else ""
        if not live:
            open(f"{LOG_DIR}/{run_id}.DRY.prompt.txt", "w").write(prompt)
            print(f"  WOULD DISPATCH #{t['id']}{flag} '{t['title'][:60]}' → {t['assignee']}·auto "
                  f"(prompt saved: {run_id}.DRY.prompt.txt)", flush=True)
            continue
        # Claim = status flip + run-requested tag cleared in ONE patch.
        # Board #197: read the PRE-DISPATCH status FIRST — the claim overwrites
        # it on the board, and reap() has no way to recover it afterwards (the
        # same "unrecoverable at reap" lesson as #195's branch0). A failed run
        # restores this value instead of falsely claiming Review.
        status0 = t.get("status")
        tags = [x for x in (t.get("tags") or []) if x != RUN_REQUESTED_TAG]
        api("PATCH", f"dev_tasks?id=eq.{t['id']}",
            body={"status": "In Progress", "tags": tags})
        api("POST", "dev_task_comments", body={
            "task_id": t["id"], "author": "system",
            "body": f"dispatched to {t['assignee']}·auto (run {run_id})"})
        rh_claim(t, run_id)
        p = spawn(agent, t, prompt, log_path)
        st["runs"].append({"run_id": run_id, "task_id": t["id"], "agent": t["assignee"],
                           "pid": p.pid, "t0": time.time(), "log": log_path,
                           "ts": datetime.now(timezone.utc).isoformat(), "active": True,
                           "status0": status0})
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
