#!/usr/bin/env python3
"""Preflight invariant check (board #153) — detect coordination state drift.

Each invariant prints one line: `OK` or a loud one-line PROBLEM + the fix command.
Wired into the SessionStart hook (after the team-board queue, see the toolkit
`.claude/settings.local.json`) so EVERY session — M's and every dispatched agent's —
opens with the environment verified. Run `/preflight` (== `preflight.py --full`)
manually before any release for the deeper checks (Railway flag baseline).

FAIL LOUD, NEVER BLOCK: every check is isolated in try/except, degrades to a soft
note on error, and the script always exits 0 so it can never wedge session start.

Origin: most of 2026-07-27's coordination failures were silent state drift with no
detector — main checkout 95 behind origin/dev (local gates measured week-old code),
10 agent branches stuck local (headless agents cannot push), a branch cut from a
stale base carrying 10 unrelated commits, the charter on dev missing a week of rules.
These invariants would have caught 4 of the 5.

Invariants:
  1  main checkout on dev AND 0 behind origin/dev
  2  no worktree branch has unpushed commits (headless agents cannot push)
  3  no branch cut from a stale base (fork point far behind origin/dev)
  4  key M-lane docs (charter, Spec_*) exist on dev AND match local
  5  dispatcher loop alive AND its file matches dev
  6  armed RORT flags match the recorded baseline   (--full only; E-owned baseline)
  7  no task sits in Todo whose next-actor is kevin  (the #151 contradiction)
"""
import glob
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime

REPO = "/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader"
ENV = f"{REPO}/src/.env"
HERE = os.path.dirname(os.path.abspath(__file__))
FLAG_BASELINE = f"{HERE}/expected_flags.json"
DISPATCHER_REL = "tools/team_dispatcher/dispatcher.py"   # relative to REPO
# M-lane docs whose dev copy must match local. Charter + every Spec_* (globbed
# locally so a spec that exists locally but not on dev shows up as "missing").
DOC_RELS = ["docs/_active/Session_Charters.md"] + sorted(
    os.path.relpath(p, REPO) for p in glob.glob(f"{REPO}/docs/_active/Spec_*.md"))
# A branch whose fork point is > this many commits behind origin/dev was not cut
# fresh from latest dev (charter: branch from latest origin/dev, one PR per branch).
STALE_BASE_MAX = 12
LIST_CAP = 6            # worktree/task detail lines shown in compact mode

FULL = "--full" in sys.argv


def sh(args, timeout=12):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


# --- output helpers: exactly one line per invariant in compact mode ---------
def ok(n, msg):
    print(f"  OK  ({n}) {msg}")


def bad(n, msg, fix):
    print(f"  !!  ({n}) {msg}")
    print(f"         fix: {fix}")


def note(n, msg):
    print(f"  --  ({n}) {msg}")


def _fmt_list(items, cap=LIST_CAP):
    shown = ", ".join(items[:cap])
    if len(items) > cap:
        shown += f", (+{len(items) - cap} more — /preflight --full for all)" if not FULL \
            else ", " + ", ".join(items[cap:])
    return shown


# ---------------------------------------------------------------------------
def resolve_gitroot():
    try:
        top = sh(["git", "-C", REPO, "rev-parse", "--show-toplevel"]).stdout.strip()
        return top or REPO
    except Exception:
        return REPO


def git(gitroot, *args, timeout=12):
    return sh(["git", "-C", gitroot, *args], timeout=timeout)


def parse_worktrees(gitroot):
    out = git(gitroot, "worktree", "list", "--porcelain").stdout
    wts, cur = [], {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            cur = {"path": line[9:]}
        elif line.startswith("HEAD "):
            cur["head"] = line[5:]
        elif line.startswith("branch "):
            cur["branch"] = line[7:].replace("refs/heads/", "")
        elif line == "" and cur:
            wts.append(cur)
            cur = {}
    if cur:
        wts.append(cur)
    return wts


def fetch_board(query):
    url = key = None
    for line in open(ENV, encoding="utf-8"):
        line = line.strip()
        if line.startswith("SUPABASE_URL="):
            url = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
            key = line.split("=", 1)[1].strip().strip('"')
    r = urllib.request.Request(f"{url.rstrip('/')}/rest/v1/{query}",
                               headers={"apikey": key, "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(r, timeout=8) as resp:
        return json.loads(resp.read().decode())


# --- invariants -------------------------------------------------------------
def check_1_main_on_dev(gitroot):
    try:
        branch = git(gitroot, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        behind = int(git(gitroot, "rev-list", "--count", "HEAD..origin/dev").stdout.strip() or 0)
        if branch == "dev" and behind == 0:
            ok(1, "main checkout on dev, 0 behind origin/dev")
        else:
            bad(1, f"main checkout on '{branch}', {behind} behind origin/dev "
                   f"— local gates would measure STALE code",
                f"git -C {gitroot} checkout dev && git -C {gitroot} pull --ff-only origin dev")
    except Exception as e:
        note(1, f"could not verify (git error: {e})")


def _worktree_stats(gitroot):
    """Per non-main worktree: unpushed-commit count and fork-point staleness."""
    stats = []
    for wt in parse_worktrees(gitroot):
        if os.path.realpath(wt.get("path", "")) == os.path.realpath(gitroot):
            continue  # main checkout — covered by invariant 1
        head = wt.get("head")
        if not head:
            continue
        label = wt.get("branch") or f"detached@{head[:8]}"
        try:
            unpushed = int(git(gitroot, "rev-list", "--count", head, "--not", "--remotes").stdout.strip() or 0)
            own = int(git(gitroot, "rev-list", "--count", f"origin/dev..{head}").stdout.strip() or 0)
            mb = git(gitroot, "merge-base", "origin/dev", head).stdout.strip()
            fork_behind = int(git(gitroot, "rev-list", "--count", f"{mb}..origin/dev").stdout.strip() or 0) if mb else 0
        except Exception:
            continue
        stats.append({"label": label, "unpushed": unpushed, "own": own, "fork_behind": fork_behind})
    return stats


def check_2_unpushed(stats):
    hit = [f"{s['label']}(+{s['unpushed']})" for s in stats if s["unpushed"] > 0]
    if not hit:
        ok(2, "no worktree branch has unpushed commits")
    else:
        bad(2, f"{len(hit)} worktree branch(es) with unpushed commits "
               f"(headless agents cannot push): {_fmt_list(hit)}",
            "push each (or hand the branch to R): git push -u origin <branch>")


def check_3_stale_base(stats):
    hit = [f"{s['label']}(fork-behind {s['fork_behind']})"
           for s in stats if s["own"] > 0 and s["fork_behind"] > STALE_BASE_MAX]
    if not hit:
        ok(3, "no active branch cut from a stale base")
    else:
        bad(3, f"{len(hit)} branch(es) cut from a stale base "
               f"(will carry unrelated commits into their diff): {_fmt_list(hit)}",
            "rebase onto latest dev: git rebase origin/dev  (and cut future branches from origin/dev)")


def check_4_docs(gitroot, relprefix):
    missing, differ = [], []
    for rel in DOC_RELS:
        dev_path = f"{relprefix}/{rel}" if relprefix != "." else rel
        r = git(gitroot, "show", f"origin/dev:{dev_path}")
        if r.returncode != 0:
            missing.append(rel)
            continue
        try:
            local = open(f"{REPO}/{rel}", encoding="utf-8").read()
        except OSError:
            continue
        if r.stdout != local:
            differ.append(rel)
    if not missing and not differ:
        ok(4, f"{len(DOC_RELS)} M-lane docs present on dev and matching local")
    else:
        parts = []
        if missing:
            parts.append(f"MISSING on dev: {', '.join(os.path.basename(m) for m in missing)}")
        if differ:
            parts.append(f"DIFFER from dev: {', '.join(os.path.basename(d) for d in differ)}")
        bad(4, "M-lane docs drifted — a fresh agent on dev would read superseded rules: " + " · ".join(parts),
            "ship the local M-lane docs to dev (via R), or pull if dev is ahead")


def check_5_dispatcher(gitroot, relprefix):
    try:
        alive = sh(["pgrep", "-f", "team_dispatcher/dispatcher.py"]).returncode == 0
    except Exception:
        alive = None
    dev_path = f"{relprefix}/{DISPATCHER_REL}" if relprefix != "." else DISPATCHER_REL
    r = git(gitroot, "show", f"origin/dev:{dev_path}")
    matches = None
    if r.returncode == 0:
        try:
            matches = (r.stdout == open(f"{REPO}/{DISPATCHER_REL}", encoding="utf-8").read())
        except OSError:
            matches = None
    if alive and matches:
        ok(5, "dispatcher loop alive and file matches dev")
    elif not alive:
        bad(5, "dispatcher loop NOT running — board Run buttons and auto-dispatch are dead",
            f"nohup python3 {REPO}/{DISPATCHER_REL} --live --loop --poll 20 >/dev/null 2>&1 &")
    else:
        bad(5, "dispatcher is running but its file DIFFERS from dev (local edits not shipped/restarted)",
            "ship dispatcher.py to dev (via R), then restart the loop")


def check_6_flags():
    if not FULL:
        note(6, "armed-flag baseline check deferred — run /preflight --full before any release (E-lane, reads Railway)")
        return
    try:
        base = json.load(open(FLAG_BASELINE, encoding="utf-8"))
    except Exception as e:
        bad(6, f"flag baseline unreadable ({e})", f"E to create/fix {FLAG_BASELINE}")
        return
    if not base.get("_meta", {}).get("populated"):
        bad(6, "flag baseline NOT populated — cannot verify armed-flag drift (E owns flag state)",
            f"E: fill the `services` map in {FLAG_BASELINE} with the current armed RORT_* set, then set _meta.populated=true")
        return
    drift = []
    for svc, expected in base.get("services", {}).items():
        try:
            out = sh(["railway", "variables", "--service", svc, "--json"], timeout=15).stdout
            live = {k: str(v) for k, v in json.loads(out).items() if k.startswith("RORT_")}
        except Exception as e:
            drift.append(f"{svc}: could not read Railway ({e})")
            continue
        for k, v in expected.items():
            if live.get(k) != str(v):
                drift.append(f"{svc}:{k} expected {v} got {live.get(k, 'ABSENT')}")
        for k in live:
            if k not in expected:
                drift.append(f"{svc}:{k} armed but NOT in baseline ({live[k]})")
    if not drift:
        ok(6, "armed RORT flags match the recorded baseline")
    else:
        bad(6, f"{len(drift)} flag drift(s): {_fmt_list(drift)}",
            "reconcile: arm/disarm to match, or update the baseline if the change is intentional (E-lane)")


def check_7_todo_kevin():
    try:
        rows = fetch_board("dev_tasks?select=id,title,assignee&status=eq.Todo")
    except Exception as e:
        note(7, f"could not read the board ({e})")
        return
    hit = [f"#{r['id']}" for r in rows if (r.get("assignee") or "").strip().lower() == "kevin"]
    if not hit:
        ok(7, "no Todo task has next-actor kevin")
    else:
        bad(7, f"{len(hit)} Todo task(s) whose next-actor is kevin (dispatch-blocked, silently sitting): {_fmt_list(hit)}",
            "move each to Approval/Review — Todo means agent-dispatchable, never Kevin-actioned")


def main():
    print(f"[Preflight] Env invariants (task #153){' --full' if FULL else ''} — {datetime.now():%Y-%m-%d %H:%M}")
    gitroot = resolve_gitroot()
    relprefix = os.path.relpath(REPO, gitroot)
    try:
        git(gitroot, "fetch", "--quiet", "origin", "dev", timeout=8)
    except Exception:
        note(0, "git fetch failed — git checks compare against last-known origin/dev (may be stale)")
    for fn in (lambda: check_1_main_on_dev(gitroot),):
        try:
            fn()
        except Exception as e:
            note(1, f"check crashed: {e}")
    try:
        stats = _worktree_stats(gitroot)
    except Exception as e:
        stats = []
        note(2, f"worktree scan crashed: {e}")
    for label, fn in ((2, lambda: check_2_unpushed(stats)),
                      (3, lambda: check_3_stale_base(stats)),
                      (4, lambda: check_4_docs(gitroot, relprefix)),
                      (5, lambda: check_5_dispatcher(gitroot, relprefix)),
                      (6, check_6_flags),
                      (7, check_7_todo_kevin)):
        try:
            fn()
        except Exception as e:
            note(label, f"check crashed: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Preflight] soft-failed, session not blocked: {e}")
    sys.exit(0)
