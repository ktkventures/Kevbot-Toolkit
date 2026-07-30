---
name: preflight
description: Run the environment invariant check (board #153) — detect coordination state drift before it silently corrupts a session's work. Use before any release/merge, at the start of a coordination pass, or whenever git/board/doc/flag state feels off. The SessionStart hook already runs the compact check on every session; this skill runs the deeper `--full` (adds the Railway armed-flag baseline). M-lane skill.
---

# Preflight — verify the environment before trusting it

Root cause of most of 2026-07-27's coordination confusion: state drifted silently for
days with no detector (main checkout 95 behind origin/dev so local gates measured
week-old code and lied; 10 agent branches stuck local because headless agents cannot
push; a branch cut from a stale base carried 10 unrelated commits; the charter on dev
was missing a week of rules). Preflight asserts the invariants that would have caught
4 of those 5 — each prints `OK` or a loud one-line problem + the fix command.

## Run it

```bash
python3 /home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/tools/preflight/preflight.py --full
```

`--full` adds invariant (6), the Railway armed-flag baseline check, on top of the eight
the SessionStart hook already runs compact on every session. It fails loud and never
blocks — every check is isolated; the script always exits 0.

**Run `--full` before any release/merge.** A green board of invariants is the precondition
for trusting a gate result: if the checkout is behind or a branch is stale-based, the
gates you are about to run measured the wrong code.

## The invariants

1. main checkout on `dev` AND 0 behind `origin/dev` (else local gates measure stale code)
2. no worktree branch has unpushed commits (headless agents cannot push — branches rot local)
3. no branch cut from a stale base (fork point far behind `origin/dev` → carries unrelated commits)
4. key M-lane docs (charter, `Spec_*`) exist on dev AND match local (else a fresh agent reads superseded rules)
5. dispatcher loop alive AND its file matches dev
6. armed `RORT_*` flags match the recorded baseline — `tools/preflight/expected_flags.json` (**E owns/populates this**; `--full` only)
7. no task sits in `Todo` whose next-actor is `kevin` (dispatch-blocked, silently sitting — the #151 class)
8. the live SessionStart hook matches its tracked canonical — the hook runs by absolute path
   from untracked `/home/kevin/projects/Kevbot-Toolkit/.claude/hooks/team_board_context.py`,
   the canonical is `tools/team_dispatcher/hooks/team_board_context.py`, and arming is a
   manual `cp`. **Sibling of (5)**, added for exactly the same reason: (5) exists because the
   running `dispatcher.py` silently diverged from dev twice, and the hook has the identical
   shape (board #194)
9. SIM poller alive — the SIM Run button and nightly sweep are DECLARATIVE (they only write
   `replay_sim_requests` rows; this local poller claims them), so a dead poller is a silent
   no-op with no error anywhere. Its documented kill switch `tools/team_sim/PAUSE` reads as a
   `--` note, not an alarm. The fix line hands over `.venv/bin/python` — **bare `python3` dies
   instantly** on `ModuleNotFoundError: supabase`, and a handoff once shipped that command

Tests: `python3 tools/preflight/test_preflight_hook_sim_194.py` — standalone, no network,
proves (8) and (9) each go RED on drift/absence and GREEN once reconciled, and that a crash
in either degrades to a note without failing the run.

## Acting on the output

Each `!!` line ends with its own `fix:` command — run it, or hand the fix to the lane
that owns it (git/doc drift = ship via R; flag drift = E; a stale-based branch = rebase
onto `origin/dev`). Invariant (6) stays red until **E** populates the flag baseline and
sets `_meta.populated=true`; that is intentional — an empty baseline cannot certify
"no flag drift," so it must not read green.
