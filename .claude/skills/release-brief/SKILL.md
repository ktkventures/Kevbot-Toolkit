---
name: release-brief
description: Prepare the standardized paste-block that hands a finished branch to an ephemeral Release (R) session for validation, merge, deploy-watch, and Deploy_Log entry. Use when a working session (E/F/P) says its branch/PR is ready to ship, or Kevin says "prep the release", "release brief", or "hand this to R".
---

# Release Brief

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
