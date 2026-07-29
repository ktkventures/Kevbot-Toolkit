#!/usr/bin/env python3
"""Team dispatcher (V4.18) — dispatches board tasks to headless Claude agents.

V4.18 (board #182 Step 7, second half): THE TM GATE. Every step hand-off is
reviewed by TM — Task Manager — a fresh, tool-less, time-boxed agent that asks
exactly one question: *was what the step asked for actually delivered?* Three
outcomes and nothing else (PASS / BOUNCE / ESCALATE), a bounce limit of two,
and it FAILS OPEN — if TM cannot run, the hand-off proceeds flagged
"unreviewed". Silent on pass, loud on problem. It gates M's steps too; that is
the point (see the charter's PM-vs-TM section). Ships in `observe` mode: TM
posts real verdicts but takes NO board action until TM_GATE=enforce.

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
  python3 dispatcher.py --tm-review 182:4         # evaluate ONE hand-off, print verdict
  python3 dispatcher.py --tm-review 182:4:569     # ...against a named submission comment

Kill switches: touch tools/team_dispatcher/PAUSE (idles the loop);
touch tools/team_dispatcher/TM_OFF (or TM_GATE=off) to disable the TM gate;
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
TM_OFF_FILE = f"{HERE}/TM_OFF"
STATE_FILE = f"{HERE}/state.json"
LOG_DIR = f"{HERE}/logs"

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

# Board #109 (Registry Phase 2): the Run button is DECLARATIVE — the API adds
# this tag + a run_history row (outcome='requested'); this --loop is what
# EXECUTES button presses. Requested tasks jump the queue (still eligibility-
# gated); the tag is cleared on claim. Ineligible requests are cleared LOUDLY
# (comment + run_history 'ignored') instead of rotting on the task.
RUN_REQUESTED_TAG = "run-requested"

# ── TM gate (board #182 Step 7, second half) ─────────────────────────────────
# TM_GATE=off      no gate at all (equivalent to touching TM_OFF)
# TM_GATE=observe  DEFAULT — TM runs and posts its real verdict, but takes ZERO
#                  board action. This is the calibration mode: read a week of
#                  verdicts before letting it move anything.
# TM_GATE=enforce  verdicts act: BOUNCE reassigns to the submitter, ESCALATE
#                  inserts an origin='audible' PM step.
TM_MODE = (os.environ.get("TM_GATE") or "observe").strip().lower()
TM_TIMEOUT_S = 300         # hard time box — rail 1 ("narrow, cannot creep")
TM_CONCURRENCY = 2         # own pool; TM must never starve agent dispatch
TM_MAX_BOUNCES = 2         # rail 2 — third strike is an ESCALATE, not a loop
TM_SKIP_OWNERS = ("kevin",)  # Kevin's stamps are not operationally gated
TM_PER_PASS = 2            # tick-scan enqueue budget per pass
TM_SUB_MAX = 12000         # chars of submission handed to TM
TM_DESC_MAX = 14000        # chars of task description handed to TM
TM_THREAD_LOOKBACK = 8     # comments scanned to reconstruct a manual submission
TM_VERDICTS = ("PASS", "BOUNCE", "ESCALATE")

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


# ── TM — Task Manager: the hand-off gate (board #182 Step 7) ─────────────────
# TM asks ONE question — "was what the step asked for actually delivered?" — and
# answers with a route, never an investigation. PM asks "is this actually right?";
# TM asks "is this what was asked?". The separation exists because an agent that
# gates its own hand-offs is self-review wearing a badge (charter §1, PM vs TM).
#
# The reviewed-marker is a TM-AUTHORED COMMENT CARRYING `step_order` — the column
# F added in Step 3. That one choice means: no schema change, no F cycle, the
# verdict lives in the thread where the next owner will read it anyway, and the
# same row is the idempotency key ("has this hand-off been reviewed?"). Do not
# replace it with a state-file flag; state.json does not survive a fresh clone
# and a hand-off must not be re-reviewed just because the loop was restarted.

def tm_mode():
    """Effective mode. The TM_OFF file beats the env var so the gate can be
    killed without restarting the loop (same ergonomics as PAUSE)."""
    if os.path.exists(TM_OFF_FILE):
        return "off"
    return TM_MODE if TM_MODE in ("off", "observe", "enforce") else "observe"


def tm_verdicts(task_id):
    """Every TM verdict on a task → {step_order: [verdict, ...]} in thread order.
    Parsed from the comment's FIRST LINE, which the dispatcher writes (never the
    TM agent) precisely so the format cannot drift."""
    rows = api("GET", f"dev_task_comments?task_id=eq.{task_id}&author=eq.TM"
                      "&order=created_at&select=body,step_order") or []
    out = {}
    for r in rows:
        so = r.get("step_order")
        if so is None:
            continue
        head = (r.get("body") or "").strip().splitlines()
        first = head[0].strip() if head else ""
        v = first[3:].strip().rstrip(".") if first.upper().startswith("TM:") else ""
        out.setdefault(int(so), []).append(v.upper())
    return out


def tm_submission_from_thread(task_id):
    """Reconstruct the submission for a hand-off nobody dispatched (an M or F
    step ticked by hand). Everything the humans/agents said since the last TM
    verdict, system rows dropped. Deliberately NOT "the whole thread": TM judges
    a submission, and a submission is what was said after the previous gate."""
    rows = api("GET", f"dev_task_comments?task_id=eq.{task_id}"
                      f"&order=created_at.desc&limit={TM_THREAD_LOOKBACK}"
                      "&select=author,body") or []
    out = []
    for r in rows:                      # newest → oldest
        if r.get("author") == "TM":
            break                       # previous gate: stop, that is the window edge
        if r.get("author") == "system":
            continue
        out.append(f"[{r.get('author')}]\n{r.get('body') or ''}")
    return "\n\n".join(reversed(out))


def tm_build_prompt(task, idx, step, submission, prior):
    """TM's whole world. No tools are granted, so anything not in here does not
    exist for TM — which is the context-hygiene argument for a separate agent
    (charter §1): it carries the step and the submission and nothing else."""
    cl = task.get("checklist") or []
    sop = (step.get("body") or "").strip()
    if sop:
        sop_block = sop
        sop_note = ""
    else:
        # #182's own chain predates step bodies: its SOPs live in the description
        # under "# PROCESS CHAIN". Say so rather than letting TM guess at intent
        # — a guessed SOP is worse than no gate (the audible said exactly this).
        sop_block = "(this step object carries no SOP body)"
        sop_note = ("\nThe step object has NO SOP body. Find this step's SOP in the task "
                    "description below, under the 'PROCESS CHAIN' section, by matching the "
                    "step title. If you cannot find it, or what you find is too vague to "
                    "judge a submission against, answer ESCALATE with that as the reason — "
                    "do NOT invent the requirement.")
    hist = "\n".join(f"- earlier verdict on this step: {v}" for v in prior) or "- none (first review)"
    return f"""You are TM — Task Manager on the RoR Trader team board. You are a GATE, not a reviewer.

YOUR ONE QUESTION: was what this step ASKED FOR actually delivered by this submission?

You are NOT the Project Manager. You do NOT investigate, read code, run tests, check out
branches or verify claims. You have NO tools — everything you may consider is in this
prompt. You do not judge whether the work is a good idea, whether the approach is optimal,
or whether the code is correct. Someone else does that. You judge DELIVERY against the SOP,
then route. Escalating IS your answer to "this needs a deeper look."

=== THE STEP THAT WAS ASSIGNED — step {idx + 1} of {len(cl)} ===
Owner: {step_owner(step) or '—'}
Mode: {step.get('mode') or 'execute'}
Title: {step_title(step)}
SOP:
{sop_block}{sop_note}

=== TASK CONTEXT (background, and where an absent SOP lives) ===
Board #{task['id']}: {task['title']}
Description:
{(task.get('description') or '(none)')[:TM_DESC_MAX]}

=== THE SUBMISSION YOU ARE JUDGING ===
{submission[:TM_SUB_MAX] or '(the submitter left no report — that alone is a finding)'}

=== PRIOR TM VERDICTS ON THIS SAME STEP ===
{hist}

=== HOW TO DECIDE — exactly three outcomes ===
PASS — the submission addresses what the SOP asked for. Imperfection, extra polish,
  debatable style and unverified claims are all PASS: depth is the next owner's job, not
  yours. Default to PASS when the submission plausibly covers the SOP.
BOUNCE — something the SOP explicitly asked for is missing or was skipped, AND the
  submitter can fix it by resubmitting. You must name the specific missing thing. Do not
  bounce for things the SOP did not ask for.
ESCALATE — a resubmission will not fix it; a human/PM look is needed. Use it for: work
  that is good but is NOT what the step asked for (scope mismatch — extra steps done,
  a different approach taken, a step's boundaries crossed); an SOP constraint that appears
  violated; an SOP you cannot find or that is too vague to judge against; and any case
  where bouncing would clearly be worse than accepting the work as-is.

Bounce limits and all board actions are handled for you. Judge, do not act.

OUTPUT CONTRACT: reply with ONLY a JSON object — no prose before or after, no code fence:
{{"verdict": "PASS" | "BOUNCE" | "ESCALATE", "reason": "<1-2 specific sentences>", "missing": "<what must be fixed; BOUNCE only, else empty string>"}}"""


def tm_spawn(prompt, log_path):
    """A fresh, tool-less, single-shot judge. --disallowedTools is what keeps
    rail 1 honest: TM *cannot* wander into an investigation, prompt or no."""
    with open(log_path, "w") as lf:
        return subprocess.Popen(
            [CLAUDE_BIN, "-p", prompt, "--output-format", "json",
             "--disallowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep",
             "WebFetch", "WebSearch", "Task", "TodoWrite", "NotebookEdit"],
            cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)


def tm_parse(full):
    """(verdict, reason, missing) from a finished TM log, or None if the run has
    not produced a terminal result yet. A malformed verdict returns None-verdict
    so the caller FAILS OPEN — never guess what TM meant."""
    last = parse_terminal(full)
    if last is None:
        return None
    txt = str(last.get("result", "")).strip()
    if txt.startswith("```"):                       # tolerate a fenced answer
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    except Exception:
        return (None, f"TM returned unparseable output: {txt[:200]}", "")
    v = str(obj.get("verdict", "")).strip().upper()
    if v not in TM_VERDICTS:
        return (None, f"TM returned an unknown verdict {v!r}", "")
    return (v, str(obj.get("reason", "")).strip(), str(obj.get("missing", "")).strip())


def tm_comment(task_id, step_order, verdict, reason, missing, mode, submitter):
    """The verdict row — the reviewed-marker AND the human-readable outcome.
    Silent on pass / loud on problem (rail 4): PASS is one line, everything else
    gets real estate. The first line is machine-parsed by tm_verdicts()."""
    if verdict == "PASS":
        body = f"TM: PASS · step {step_order} — {reason}"
    elif verdict == "BOUNCE":
        body = (f"TM: BOUNCE · step {step_order}\n\n"
                f"**Not what the step asked for — back to `{submitter}` to resubmit.**\n\n"
                f"**Why:** {reason}\n\n**Fix and resubmit:** {missing or '(see above)'}\n\n"
                f"_TM gates delivery, not depth: this is *the step asked for X and X is not "
                f"here*, not a judgement on the quality of what is here. Bounce "
                f"{len(mode['prior']) + 1} of {TM_MAX_BOUNCES}; the next one escalates to M "
                f"instead of bouncing again._")
    elif verdict == "ESCALATE":
        body = (f"TM: ESCALATE · step {step_order}\n\n"
                f"**This needs a real look — routing to M.** A resubmission would not fix it.\n\n"
                f"**Why:** {reason}\n\n"
                f"_TM never investigates; escalating IS its answer to \"this needs more\"._")
    else:
        body = (f"TM: UNREVIEWED · step {step_order}\n\n"
                f"⚠️ The gate **failed open** — this hand-off proceeded WITHOUT review. "
                f"Reason: {reason}\n\n_Fail-open is deliberate (rail 3): losing the check on "
                f"one transition beats stalling every task mid-chain. Nothing is blocked; "
                f"the next owner just has no TM verdict to lean on._")
    if mode["mode"] == "observe" and verdict in ("BOUNCE", "ESCALATE"):
        body += ("\n\n> **OBSERVE MODE — no board action taken.** TM did not reassign or "
                 "insert anything; this is the verdict it *would* have acted on. "
                 "Set `TM_GATE=enforce` to arm it.")
    api("POST", "dev_task_comments", body={
        "task_id": task_id, "author": "TM", "step_order": step_order, "body": body})


def tm_apply(task_id, step_order, verdict, submitter, live):
    """The board writes — enforce mode only. BOUNCE puts the task back in the
    submitter's queue (Todo → the dispatcher re-runs it, with the bounce reason
    now in the thread). ESCALATE inserts an origin='audible' PM step right after
    the reviewed one and hands the task to M."""
    if not live or verdict not in ("BOUNCE", "ESCALATE"):
        return
    if verdict == "BOUNCE":
        api("PATCH", f"dev_tasks?id=eq.{task_id}",
            body={"status": "Todo", "assignee": submitter})
        return
    # ESCALATE — re-read the chain so a concurrent edit is not clobbered, and
    # insert AFTER the reviewed step. Completed steps are never touched (T4':
    # course-correct by inserting an audible, never by reopening).
    rows = api("GET", f"dev_tasks?id=eq.{task_id}&select=checklist") or []
    cl = list((rows[0].get("checklist") if rows else None) or [])
    at = min(step_order, len(cl))
    cl.insert(at, {"owner": "M", "role": "M", "origin": "audible", "done": False,
                   "title": f"PM look: TM escalated step {step_order}",
                   "text": f"PM look: TM escalated step {step_order}",
                   "mode": "execute",
                   "body": ("TM escalated the hand-off on this step — it judged that a "
                            "resubmission would not fix the problem. Read TM's verdict "
                            "comment above, then decide: accept as-is, re-scope the chain, "
                            "or insert the corrective step. TM routes; M decides.")})
    api("PATCH", f"dev_tasks?id=eq.{task_id}",
        body={"checklist": cl, "assignee": "M"})


def tm_active(st):
    now = time.time()
    return [r for r in st.get("tm_runs", []) if r.get("active")
            and now - r["t0"] < TM_TIMEOUT_S]


def tm_enqueue(st, task, idx, step, submission, live, why):
    """Spawn one TM evaluation. Non-blocking by construction: a synchronous gate
    in the poll loop would stall Run buttons and every other reap behind a
    judgement call — and rail 3 says never trade the pipeline for the check."""
    if tm_mode() == "off":
        return False
    if len(tm_active(st)) >= TM_CONCURRENCY:
        return False
    if step_owner(step) in TM_SKIP_OWNERS:
        return False
    tid, so = task["id"], idx + 1
    prior = tm_verdicts(tid).get(so, [])
    if prior:
        return False                    # already gated — the marker is the key
    prompt = tm_build_prompt(task, idx, step, submission, prior)
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = f"tm{int(time.time())}-{tid}s{so}"
    log_path = f"{LOG_DIR}/{run_id}.log"
    if not live:
        open(f"{LOG_DIR}/{run_id}.DRY.prompt.txt", "w").write(prompt)
        print(f"  WOULD TM-REVIEW #{tid} step {so} ({why}) → {run_id}.DRY.prompt.txt", flush=True)
        return True
    p = tm_spawn(prompt, log_path)
    st.setdefault("tm_runs", []).append({
        "run_id": run_id, "task_id": tid, "step_order": so, "pid": p.pid,
        "t0": time.time(), "log": log_path, "active": True,
        "submitter": step_owner(step) or "M", "prior": prior})
    save_state(st)
    print(f"  TM-REVIEW #{tid} step {so} ({why}) pid={p.pid} mode={tm_mode()}", flush=True)
    return True


def tm_reap(st, live):
    """Collect finished TM judgements. Every exit path posts a verdict row —
    including the failures — because that row is the idempotency key: without it
    a crashed gate would re-review the same hand-off on every single poll."""
    for r in [x for x in st.get("tm_runs", []) if x.get("active")]:
        alive = os.path.exists(f"/proc/{r['pid']}")
        full = open(r["log"]).read() if os.path.exists(r["log"]) else ""
        parsed = tm_parse(full) if full else None
        expired = time.time() - r["t0"] >= TM_TIMEOUT_S
        if alive and not expired and parsed is None:
            continue
        r["active"] = False
        if alive:
            kill_run_group(r["pid"])
        if parsed is None:
            verdict, reason, missing = None, (
                f"TM hit its {TM_TIMEOUT_S // 60}-minute time box without answering"
                if expired else "the TM run died without producing a verdict"), ""
        else:
            verdict, reason, missing = parsed
        # Rail 2 — the third strike is not a third bounce. Without this the gate
        # is a loop with no exit: TM bounces, the lane resubmits, TM bounces.
        if verdict == "BOUNCE" and len(r["prior"]) >= TM_MAX_BOUNCES:
            verdict = "ESCALATE"
            reason = (f"bounce limit reached ({TM_MAX_BOUNCES}) — TM's finding still "
                      f"stands and resubmission is not converging: {reason}")
        if live:
            tm_comment(r["task_id"], r["step_order"], verdict, reason, missing,
                       {"mode": tm_mode(), "prior": r["prior"]}, r["submitter"])
            if tm_mode() == "enforce":
                tm_apply(r["task_id"], r["step_order"], verdict, r["submitter"], live)
        print(f"  TM VERDICT #{r['task_id']} step {r['step_order']}: "
              f"{verdict or 'UNREVIEWED (failed open)'} — {reason[:120]}", flush=True)
    runs = st.get("tm_runs", [])
    st["tm_runs"] = [x for x in runs if x.get("active")] + \
                    [x for x in runs if not x.get("active")][-50:]
    save_state(st)


def tm_scan(st, live):
    """Trigger B — hand-offs nobody dispatched. A dispatched run is gated at reap
    (trigger A, on SUBMISSION, so TM's read reaches M *before* M reviews). This
    covers the rest: a step ticked by hand, which is how M's own steps complete —
    and gating M is explicitly the point.

    Only steps carrying `completed_at` are eligible. That is not a nicety: it is
    the line between a hand-off TM can date and a submission it would have to
    guess at. Historical steps ticked before the step-action API shipped have no
    timestamp, so TM leaves them alone instead of inventing a window. Review one
    on purpose with --tm-review."""
    if tm_mode() == "off":
        return
    budget = TM_PER_PASS
    rows = api("GET", "dev_tasks?select=id,title,description,checklist"
                      "&checklist=not.is.null&order=updated_at.desc&limit=60") or []
    for t in rows:
        if budget <= 0:
            break
        cl = t.get("checklist") or []
        if not is_process_chain(cl):
            continue
        reviewed = tm_verdicts(t["id"])
        for i, s in enumerate(cl):
            if budget <= 0:
                break
            if not (isinstance(s, dict) and s.get("done") and s.get("completed_at")):
                continue
            if (i + 1) in reviewed or step_owner(s) in TM_SKIP_OWNERS:
                continue
            if tm_enqueue(st, t, i, s, tm_submission_from_thread(t["id"]), live,
                          "ticked by hand"):
                budget -= 1


def tm_review_one(spec, live):
    """`--tm-review <task>:<step>[:<comment_id>]` — gate one hand-off on purpose.
    Two jobs: calibration against hand-offs that already happened, and the way M
    puts its OWN step through the gate today (M's ticks predate completed_at, so
    trigger B cannot see them). Synchronous — you asked for one answer."""
    parts = spec.split(":")
    tid, so = int(parts[0]), int(parts[1])
    cid = int(parts[2]) if len(parts) > 2 else None
    task = (api("GET", f"dev_tasks?id=eq.{tid}&select=*") or [None])[0]
    if task is None:
        print(f"task #{tid} not found", flush=True); return
    cl = task.get("checklist") or []
    if not 1 <= so <= len(cl):
        print(f"#{tid} has {len(cl)} steps — step {so} is out of range", flush=True); return
    idx, step = so - 1, cl[so - 1]
    if cid is not None:
        rows = api("GET", f"dev_task_comments?id=eq.{cid}&select=author,body") or []
        submission = (f"[{rows[0]['author']}]\n{rows[0]['body']}" if rows
                      else f"(comment {cid} not found)")
    else:
        submission = tm_submission_from_thread(tid)
    prior = tm_verdicts(tid).get(so, [])
    prompt = tm_build_prompt(task, idx, step, submission, prior)
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = f"tm{int(time.time())}-{tid}s{so}"
    log_path = f"{LOG_DIR}/{run_id}.log"
    print(f"TM reviewing #{tid} step {so} ({step_owner(step)}: {step_title(step)[:60]}) "
          f"· submission={'comment ' + str(cid) if cid else 'thread window'} "
          f"· {len(submission)} chars · mode={tm_mode()} · live={live}", flush=True)
    p = tm_spawn(prompt, log_path)
    try:
        p.wait(timeout=TM_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        kill_run_group(p.pid)
    parsed = tm_parse(open(log_path).read() if os.path.exists(log_path) else "")
    verdict, reason, missing = parsed if parsed else (
        None, "the TM run produced no verdict", "")
    print(f"\n  VERDICT: {verdict or 'UNREVIEWED (failed open)'}\n  REASON:  {reason}"
          f"{chr(10) + '  MISSING: ' + missing if missing else ''}\n", flush=True)
    if live:
        tm_comment(tid, so, verdict, reason, missing,
                   {"mode": tm_mode(), "prior": prior}, step_owner(step) or "M")
        if tm_mode() == "enforce":
            tm_apply(tid, so, verdict, step_owner(step) or "M", live)
        print("  posted to the thread", flush=True)


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


def build_prompt(agent, task):
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
        step_block = f"""
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


def tm_gate_submission(st, run, report):
    """Trigger A — a dispatched agent just submitted. This fires on SUBMISSION,
    not on the tick, so TM's read is already in the thread when M opens the task
    to review it. That ordering is the whole value: it is also what lets TM catch
    an M that would otherwise rubber-stamp its own lane's hand-off."""
    so = run.get("step_order")
    if so is None or tm_mode() == "off":
        return
    rows = api("GET", f"dev_tasks?id=eq.{run['task_id']}&select=*") or []
    if not rows:
        return
    task = rows[0]
    cl = task.get("checklist") or []
    if not (is_process_chain(cl) and 1 <= so <= len(cl)):
        return
    tm_enqueue(st, task, so - 1, cl[so - 1], f"[{run['agent']}·auto]\n{report}",
               True, "agent submission")


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
            if outcome == "ok":
                # #182 Step 7 — gate the hand-off. A failed run has nothing to
                # judge; that is a dispatch problem, not a delivery problem.
                tm_gate_submission(st, r, str(result)[:RESULT_COMMENT_MAX])
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
    if live:
        tm_reap(st, live)
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
    if only_task is None:
        tm_scan(st, live)      # trigger B — hand-offs nobody dispatched
    now = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"[{now}] agents={list(agents)} ({source}) · dispatchable={len(cands)} "
          f"(button-requested={len(requested)}, skipped={len(skipped)}) "
          f"· slots={slots} · cap_left={cap_left} · tm={tm_mode()}"
          f"({len(tm_active(st))} active) · mode={'LIVE' if live else 'DRY-RUN'}",
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
        prompt = build_prompt(agent, t)
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
        # Stamp WHICH step this run was dispatched for, at dispatch time. Reading
        # it back at reap would re-derive `current_step` from a chain that may
        # have been edited in the meantime — and TM must judge the step the agent
        # was actually handed, not the one that happens to be current later.
        d_idx, _ = current_step(t)
        st["runs"].append({"run_id": run_id, "task_id": t["id"], "agent": t["assignee"],
                           "pid": p.pid, "t0": time.time(), "log": log_path,
                           "step_order": (d_idx + 1) if d_idx is not None else None,
                           "ts": datetime.now(timezone.utc).isoformat(), "active": True})
        save_state(st)
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
    ap.add_argument("--tm-review", default=None, metavar="TASK:STEP[:COMMENT_ID]",
                    help="run the TM gate on ONE hand-off and print the verdict "
                         "(posts to the thread only with --live). Board #182.")
    args = ap.parse_args()
    if args.tm_review:
        tm_review_one(args.tm_review, args.live)
        return
    while True:
        one_pass(args.live, only_task=args.task)
        if not args.loop:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
