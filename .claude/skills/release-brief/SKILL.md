---
name: release-brief
description: Prepare the standardized paste-block that hands a finished branch to an ephemeral Release (R) session for validation, merge, deploy-watch, and Deploy_Log entry. Use when a working session (E/F/P) says its branch/PR is ready to ship, or Kevin says "prep the release", "release brief", or "hand this to R".
---

# Release Brief

## GENERATE IT FIRST — do not hand-author (board #211)

```
cd /home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader
python3 tools/release_brief/brief_gen.py                 # print the brief + verdict
python3 tools/release_brief/brief_gen.py --json          # the machine-readable plan
python3 tools/release_brief/brief_gen.py --create-task   # also create the R task
```

`Staged → shipped` used to be the last hand-typed hop, and it is where work died:
nothing polls `Staged`, so a task sat there until M noticed and typed ~40 lines of
largely templated prose. Everything a brief needs is already structured (the board)
or derivable (git), so the generator derives it:

| Part of the brief | Derived from |
|---|---|
| **cars** | `dev_tasks` where `status=Staged` and the CURRENT chain step is `R`-owned |
| **branch** | `run_history.pushed_branch` → the task thread → a local branch named `*-<id>` |
| **gates** | test files in the branch's diff; **the full dispatcher suite set** whenever `tools/team_dispatcher/dispatcher.py` is touched |
| **order** | docs cars first, code last — a code failure then leaves the docs landed |
| **conflicts** | overlapping paths across cars + `git merge-base` staleness per branch |
| **owed logs** | unmerged `docs/deploy-log-waveN-MMDD` branches (R leaves these behind on purpose) |
| **hands-off branches** | other worktrees' branches, filtered to unmerged + committed within ~3 weeks |
| **wave number** | highest `R · Release train Wave N` on the board, +1 |
| **hard limits / procedure** | constants — byte-identical in every brief M hand-wrote |

**Kevin's ruling (task #211, 07-30): generate + AUTO-DISPATCH**, with the judgement
enforced mechanically instead of by attention. Two distinct verdicts:

- a **refusal** drops ONE car and the rest still ship — an unreviewed chain (no
  ticked M-review step in front of the R step), an unresolvable branch, an empty
  diff, or a branch already merged into `dev`;
- a **hold** keeps the whole train and its brief but routes it to **M** — overlapping
  paths with no recorded resolution (two cars on `dispatcher.py`), or a stale base.

`--create-task` honours that: AUTO → `assignee=R`, `status=Todo`, `ai_eligible=true`
(the loop dispatches it). HELD → `assignee=M`, `status=Review`, unarmed — visible,
never dispatched.

The generator **never** merges, pushes, rebases, runs a gate, flips a flag, or touches
a worktree. It reads, prints, and at most writes one `dev_tasks` row.

Rails suite: `python3 tools/release_brief/test_brief_gen.py` (expect `ALL PASS`).

**Still M's job, every time:** read the generated brief, and resolve anything it held.
The shape it emits is what R is tuned to execute — when that shape needs to change,
edit the generator, not a one-off brief.

---

## Hand-authoring fallback

Use only when the generator cannot run (no board access) or the train is genuinely
unlike anything it models.

Produce ONE self-contained paste-block that an ephemeral `R` session executes
top-to-bottom. R has fresh eyes and zero context — the brief must contain everything.
R validates, merges, deploy-watches, logs, reports, and dies. R never starts feature
work and never fixes non-trivial failures (it reports back instead).

Charter §6 governs the protocol:
`/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/_active/Session_Charters.md`

## Before generating — verify in THIS session (do not delegate broken state to R)

1. `git fetch origin && git log origin/dev..HEAD --oneline` — only this branch's commits.
2. PR exists and is current (`gh pr view <n>`); body describes the change.
3. Any new flags are default-OFF, runtime-read. List every flag the change adds/reads.
4. Gates already run here (parity suite, role-specific checks) — record actual results;
   R re-runs them, but the brief must state what YOU got.

If any of these fail, fix them first — do not emit a brief for a broken branch.

## Output template (chat-only; fill every `<>` with real values, no placeholders left)

```
=== RELEASE BRIEF — paste into a fresh session named "R — release <branch> <MM-DD>" ===
You are R, an ephemeral release session. Execute this brief top-to-bottom. Abort loudly
on any gate failure and report back — do not improvise fixes, do not start feature work.
You are done (and retired) after the final report.

Charter: /home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/_active/Session_Charters.md (§6)
Repo: /home/kevin/projects/Kevbot-Toolkit  ·  App root: clients/KevBot_Toolkit/RoR_Trader

## What is shipping
- Branch: <branch>  ·  PR: #<n> (<url>)  ·  Author session: <role-name>
- Summary: <2-4 sentences: what changed and why>
- Commits: <git log origin/dev..HEAD --oneline output>
- Files touched: <list>
- Flags: <flag=default state, who flips, when> — or "none"
- Known risks / watch items: <list> — or "none"

## Gates (re-run ALL; author's results shown for comparison)
1. git fetch origin && git log origin/dev..HEAD --oneline → ONLY the commits above
2. cd clients/KevBot_Toolkit/RoR_Trader && <venv-python> src/fidelity_parity_suite.py
   → ALL PASS (author got: <n/n>)
3. <role-specific checks: frontend build/lint/screenshot, pack known-answer test, etc.>
   (author got: <results>)
4. /code-review on the PR — no CONFIRMED blockers.

## Merge + deploy
1. Backup: git branch backup/dev-pre-<slug> origin/dev && git push origin backup/dev-pre-<slug>
2. Merge PR #<n> into dev (merge via gh; never force-push dev).
3. Deploy watch: <which Railway services rebuild; boot lines / health checks to confirm>
   ⚠ shadow-worker never deploys from this path (E-lane railway-up only) — skip it.
4. Append Deploy_Log entry: clients/KevBot_Toolkit/RoR_Trader/docs/Deploy_Log.md
   (match existing entry format; include commit SHA + gates result).

## Rollback (if deploy watch fails)
<exact steps: revert PR via gh, or reset instructions using the backup branch — spelled
out by the author, not left to R's judgment>

## Final report, then die
Report: gates result, merge SHA, deploy status, Deploy_Log line added, any anomalies.
Remind Kevin to close this session; R sessions are never reused.
=== END BRIEF ===
```

## Hard rules

- Chat-only output; the brief is copy-paste material, not a file.
- Every command concrete and runnable — R must never have to guess a path, PR number,
  or expected result.
- Rollback section is mandatory and specific; "revert if needed" is not a rollback plan.
- If the change involves flags or engine files and the authoring session is not E,
  STOP — that change needs E's sign-off before a brief exists.
