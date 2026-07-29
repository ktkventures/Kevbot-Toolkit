# TM gate — verification review (board #182 Step 7, second half)

**Reviewer:** M — Project Manager (session c4764159) · **Date:** 2026-07-29
**Subject:** branch `feat/dispatcher-tm-gate-182` @ `47464f8b` — dispatcher V4.18, +451 lines
in `tools/team_dispatcher/dispatcher.py`, +330 lines `src/test_dispatcher_tm_gate_182.py`
**Provenance:** unknown. Preserved by M (c85c7f28), who did not write it. Treated as a
proposal from a stranger, per the handoff.

## VERDICT: DO NOT MERGE AS-IS. Do not arm, even in `observe`.

The design is sound and the code is genuinely well-written — the reviewed-marker choice
(a TM comment carrying `step_order`) is right, fail-open is implemented correctly, and the
prompt is a good piece of writing. **Two defects are blocking**, and one of them means
`observe` mode is not the safety net its banner claims.

---

## BLOCKER 1 — TM is not tool-less. `observe` mode does not contain this.

`tm_spawn()` relies on `--disallowedTools` to make TM "a fresh, tool-less, single-shot
judge", and the code comments call that flag "what keeps rail 1 honest: TM *cannot* wander
into an investigation, prompt or no."

**Empirically false.** Measured today against the real CLI, with the exact argv the code uses:

| Invocation | Tools the agent actually gets |
|---|---|
| TM's 11-name blocklist (as written) | `Artifact`, `ReportFindings`, `ScheduleWakeup`, **`Skill`**, **`ToolSearch`**, **`Workflow`** |
| Add those 6 to the blocklist | The entire deferred registry unmasks: `CronCreate`, `SendMessage`, `RemoteTrigger`, `PushNotification`, `EnterWorktree`, `Monitor`, `TaskStop`, … **plus every MCP tool** (ClickUp, Drive, Higgsfield) |
| `--allowedTools ""` | **No restriction at all** — `Bash`, `Edit`, `Read`, `Write`, `Agent`, `Skill`, `Workflow` |
| `--allowedTools NoSuchToolXyz --strict-mcp-config` | **Everything** except MCP — `Bash`, `Write`, `Agent`, `Skill`, `Workflow`, … |

Findings:

1. **`--allowedTools` is a permission/auto-approve list, NOT a capability filter.** It cannot
   be used to restrict. An empty or bogus allowlist grants *more*, not less.
2. **`--disallowedTools` is the only working lever, and it is a blocklist — allow-by-default.**
   Every name added reveals more of the registry. It cannot be made complete by inspection,
   and it silently re-opens whenever the harness adds a tool or an MCP server is connected.
3. The blocklist's `Task` entry *does* correctly deny this harness's `Agent` tool (verified).
   The gap is not `Task`; it is the ~20 names nobody knew to list.
4. **`--strict-mcp-config` does work** — it dropped all `mcp__*` tools. Keep it regardless.

**Why this outranks everything else:** `TM_GATE=observe` only suppresses `tm_apply()` — the
dispatcher's *board writes*. It does nothing about what the spawned TM agent can do to the
machine. In observe mode TM still runs, unattended, on every hand-off, with `Bash`-class
capability available via `Skill`/`Workflow`, `cwd=REPO`.

Concretely reachable from an unattended TM run today:
- **`Skill` → `/local-update`** — that skill *writes to the production database*.
- **`Workflow`** — spawns dozens of subagents; unbounded token burn.
- **`CronCreate`, `SendMessage`, `RemoteTrigger`, `PushNotification`** — outward-facing actions.

**Prompt-injection surface is real, not theoretical.** `tm_build_prompt()` embeds the task
`description` and the submission **verbatim** — both are text written by other agents. A
submission containing instructions aimed at TM is executed by a tool-capable agent. The one
thing standing between that and a prod-DB write is TM's own compliance with its prompt.

### Fix
Do not ship the "tool-less" claim as a flag-only guarantee. Required:
- Full blocklist (all ~34 current names) **plus `--strict-mcp-config`**, AND
- a **live canary check** — spawn one TM-shaped run that asks "list your tools" and assert
  it answers `NONE`; run it at dispatcher start and fail loud on drift. A hermetic test
  cannot catch this, which is exactly why it was missed.
- Reconsider the premise: if TM must be *provably* inert, the flag layer is the wrong place.

---

## BLOCKER 2 — rail 2 ("bounce limit of two") is unreachable. The verdict parser is broken.

`tm_verdicts()` extracts the verdict token from the comment's first line:

```python
v = first[3:].strip().rstrip(".") if first.upper().startswith("TM:") else ""
out.setdefault(int(so), []).append(v.upper())
```

The first line written by `tm_comment()` is:

```
TM: PASS · step 3 — <reason>
```

so `first[3:]` yields the **entire remainder of the line**, not the token:
`"PASS · STEP 3 — <REASON>"`. No stored verdict is ever equal to `"BOUNCE"`, `"PASS"`, or
`"ESCALATE"`.

Consequences:
- The rail-2 check `if verdict == "BOUNCE" and len(r["prior"]) >= TM_MAX_BOUNCES` counts
  *any* prior verdict, not bounces.
- It is **unreachable anyway**: `tm_enqueue()` does `if prior: return False`, so a step that
  already carries any verdict is never re-gated. A second bounce cannot occur.
- Therefore in `enforce` mode: TM bounces → task returns to Todo → the agent resubmits →
  **the resubmission is never reviewed again.** The gate fires exactly once per step, ever.

That is *safe* (no infinite bounce loop) but it is **not the specified rail**, and the test
suite claims to pin rail 2. Either the parser is fixed and re-gating allowed on resubmission,
or the design is honestly restated as "one review per step" and rail 2 deleted.

**Fix:** parse the token only — e.g. `first[3:].split("·")[0].strip().upper()` — and let
`tm_enqueue` re-gate when the newest prior verdict is a BOUNCE.

---

## NON-BLOCKING, but Kevin should rule

3. **"Silent on pass" isn't silent.** Every PASS posts a one-line comment on the task thread.
   The code justifies it — the comment *is* the idempotency key, and a state-file flag would
   not survive a fresh clone. That reasoning is correct. But it means every hand-off adds a
   row to the thread. Kevin's rail said "silent on pass". Needs a ruling, not a fix.

4. **First-arming burst — material at 79% weekly usage.** `tm_scan()` (trigger B) sweeps 60
   tasks/pass and enqueues up to `TM_PER_PASS=2` unreviewed completed steps, `TM_CONCURRENCY=2`
   at a time, each a **full `claude -p` session**. On first arming it will work through the
   entire historical backlog of every completed step on every process-chain task — plausibly
   tens to hundreds of Claude sessions, unattended. Needs a first-run cap or a date floor.

5. **`tm_scan()` polling cost.** It calls `tm_verdicts()` — one PostgREST GET — per
   process-chain task per pass. When everything is already reviewed, `budget` never
   decrements, so it scans all 60 every pass. At `--poll 20` that is a sustained ~3 req/s
   against the board DB, forever. Bound it, or filter reviewed tasks in the query.

---

## What is genuinely good (keep)

- Reviewed-marker as a TM-authored comment carrying `step_order` — no schema change, no F
  cycle, idempotency key and human-readable outcome in one row. Right call; do not replace.
- Fail-open is correctly implemented: **every** exit path posts a verdict row, including the
  crash and timeout paths, so a dead gate cannot cause re-review on every poll.
- `step_order` stamped at **dispatch** time, not re-derived at reap — correct, and the
  comment explains why (the chain may be edited in between).
- Non-blocking by construction, with its own concurrency pool so TM cannot starve dispatch.
- The `observe` rollout instinct is right, and better than originally specified — it is just
  **mis-scoped**: it gates board writes, not the agent's capabilities (see Blocker 1).

## Recommendation

Keep the branch, do not open a PR yet. Fix Blocker 2 (small, contained). Blocker 1 needs a
decision from Kevin on how hard the tool-less guarantee has to be, because the flag layer
cannot deliver it alone. `TM_OFF` stays armed until both are closed.

## Not yet done in this review
- The hermetic suite `src/test_dispatcher_tm_gate_182.py` has **not been run** — pending.
- Not yet checked against #182's audible spec comment on the board (three outcomes, bounce
  limit two, fails open, gates M's steps too) beyond what the handoff quoted.
