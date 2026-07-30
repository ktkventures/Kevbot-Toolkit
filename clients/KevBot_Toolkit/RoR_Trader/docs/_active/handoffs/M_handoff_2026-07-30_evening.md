# M — Project Manager · handoff 2026-07-30 EVENING (from session 1805c263)

You are **M — Project Manager**. State your name in your first reply so Kevin can set the
sidebar title.

**Supersedes `M_handoff_2026-07-30.md` (written this morning).** That file is still correct
about the traps; almost everything else in it has moved. Written as insurance, not because
this session is retiring.

## FIRST ACTS
1. **ARM A HEARTBEAT BEFORE ANYTHING ELSE.** M's turn ends when M stops talking; nothing
   wakes it. This is still the single highest-value act — see "Monitoring" below.
2. Charter: `docs/_active/Session_Charters.md` — §1 roster, §4 the ASSIGNEE rule, §7 process
   chains, §8 release protocol + the DEFAULT CADENCE (ship each reviewed branch immediately).
3. `python3 tools/preflight/preflight.py` from the REPO ROOT.
4. Read the board's `In Progress` / `Review` / `Staged` columns before believing anything here.

---

## WHAT CHANGED TODAY (07-30) — the parts that invalidate the morning handoff

### The lane model changed: session vs auto
**`M-A` (Manager Assistant) now exists in the `agents` registry** (board #222). The model, in
Kevin's words:

> *"The session-based agent is more of the big-picture thinker. Their auto version is trained
> on the same things, but automates the work that falls in line with the session version. The
> session version retains long-term memory and acts as a fallback when we have automation
> issues."*

- **`M` → `live-session`** (the session; not dispatchable). ⚠️ **The `UPDATE` doing this was
  authored and HELD** in `src/migrations/agents_registry_ma_split_222.sql` — check whether it
  has been applied. Until it is, `M·auto` still dispatches and the trap below still fires.
- **`M-A` → `headless`** — builds M-lane infrastructure, writes specs, runs sweeps, diagnoses,
  verifies trains.
- **`R`/`R-A` is APPROVED and scoped (#229)**, `blocked_by` #222 — it will not dispatch until
  #222 is genuinely Done (flip applied AND verified).
- **`E`/`E-A` and `F`/`F-A`: deliberately NOT done.** Kevin: hold until M and R prove out.
  `E2` is plan-specific and obsolete — **retire** it rather than repurpose; `E2` does not fit
  the `X`/`X-A` naming.

**The dividing line** (from #222's classification, which judged 15 tasks and 22 steps):

| goes to the AUTO lane | stays with the SESSION |
|---|---|
| building lane infrastructure | reviewing another agent's work |
| writing a spec or design doc | merge reconciles needing judgement |
| sweeps, audits, inventories | restarting the loop · applying a migration · prod |
| diagnosing a bounded defect | deciding priority, talking to Kevin |
| verifying a train's claims | authoring a chain · closing a task |

`M·auto` drew the last one itself and it is worth keeping verbatim: *"status changes sit
outside a dispatched agent's contract."*

### The daily cap is now RUNTIME-ADJUSTABLE (#228)
`DAILY_CAP` was a module constant read at import; the comment above it had claimed
*"read at runtime, no deploy"* since #219 and **that was never true**. Raising 40 → 50 cost a
branch, gates, a merge, a deploy and a restart.

**Now:** `effective_daily_cap()` reads `system_settings.dispatcher_daily_cap` **every poll**.
Change the row, the running loop honours it in ~20s. Verified live: `50 → 55 → 50` with the
loop untouched.

- The poll line prints `cap_left=N/CAP [source]`. **`[settings]` = healthy. `[constant (no
  row)]` = the feature is inert while showing an identical number** — treat that as an alarm.
- Out-of-range values are CLAMPED to `[1, 500]` and reported. No setting can disable the breaker.
- A `/admin/dispatch` control for it is built and Staged (#228 step 4, Kevin's call on placement).

### Dispatcher is V4.27
V4.26 = the GIT CONTRACT stopped lying (**agents CAN push**; four runs did on 07-29/30).
V4.27 = **worktree ownership is DECLARED, not inferred** — the run's id must appear in the
worktree directory name. Ownership by elimination was wrong in both directions and stranded
real commits three times on 07-30.

---

## THE TRAP THAT FIRED FOUR TIMES TODAY
**Moving an ARMED task's status or assignee is a DISPATCH DECISION.** Every time, `M·auto` was
dispatched into a step reserved for the session (#195, #219, #223, #227), and twice it created
a genuine race (#222, #226) where a concurrent run picked up work M was mid-way through.

**Rule, without exception: DISARM FIRST, then restatus.** M knew this for *review* steps and
kept missing it for status changes.

**#222's flip makes this structurally impossible for M** — which is the real reason it matters.
Until it is applied, disarm by hand.

---

## OPEN DEFECTS THAT WILL BITE YOU
1. **#225 — the stamp does not tick the Kevin step on process chains.** `stamp_approval`
   matches the legacy `role` field; chains use `owner`. **Every approval Kevin makes** sets
   `status → Todo` and arms the task but leaves the step unticked and the assignee on Kevin.
   It fired **four times** on 07-30. Until it ships, hand-tick after every stamp.
   ⚠️ **And hand-ticking has a hazard:** it moves which step Kevin's next "complete" click
   lands on. On #224 that marked an unbuilt step done — the build had never run. **Tell Kevin
   when you hand-tick.**
2. **The release-brief generator has 5 known gaps** (#224 — three fixed and shipped, gaps 4
   and 5 open): `review_step()` matches the word *review* in the step **TITLE** only (so
   **title every review step with "review"**), and **the wave number is derived wrongly** —
   it returned `22` when Wave 22 already existed. Take the wave number from
   `Deploy_Log.md`'s highest entry, not from the tool.
3. **`test_steps_actor_auth_217.py` is NOT hermetic** — `build_prompt` embeds the task's LIVE
   comment thread, so assertions scanning the whole prompt can be broken by a board comment.
   One was, by M's own review comment.
4. **#158 — a running agent never sees comments posted after dispatch.** Corrections cannot
   reach an in-flight run.

## TRAPS THAT COST REAL TIME (still true from the morning handoff)
1. **`git merge --ff-only` refuses SILENTLY** on untracked-vs-tracked collisions. Check HEAD moved.
2. **Stale-base diffs lie.** Use `git show --stat <sha>`, never a two-dot diff on a fork-behind branch.
3. **A merge can break a TEST rather than the code.** Happened again today (rail 5 on #217).
4. **`pgrep -f` matches your own argv.** Use the bracket trick AND check the process shape.
5. **Keep board comments SHORT** — long ones broke the task modal (#215).
6. **Per-branch green says nothing about a train.** Merge the cars into a scratch worktree and
   run the suites on the COMBINED tree. On 07-30 that caught #218's two branches conflicting
   with each other while each was clean against dev.

---

## MONITORING — the thing that keeps M alive
Arm a `Monitor` (~15 min) reporting: loop up/down + version · dev head · active runs ·
`cap_left` **and its source** · open PRs · counts on-M / on-kevin · **stranded branches**.

Refinements learned today, each from a false alarm:
- **Exclude branches belonging to ACTIVE runs** — agents push at run end, so an in-flight
  branch is not stranded.
- **Exclude R's per-train scratch** (`gate/*`, `*-preview`, `*-ship`).
- **Re-read constants from the FILE each beat** — a monitor that imports `dispatcher` once
  holds the old `DAILY_CAP` forever and misreports headroom.
- **"Commits ahead of origin" overstates risk after a rebase** — it counted 10 when the real
  new work was 1.

Also arm a **board watch (~60s)** for Kevin's comments and status changes. **`@M` mentions
(the `🤖 Ask AI` button) are TOP PRIORITY** — Kevin's words: *"usually when I ask the AI within
the task board, I'm looking for a response pretty quickly."* It posts a comment prefixed
`@M `; there is an unread-by-role queue and a Messages tab.

---

## HOW KEVIN WANTS TO BE WORKED WITH
- **Chat = TLDRs and pointers. Detail lives on the task.** *"I'm reading the same thing twice."*
- **Anything assigned to him must be SCOPED** — a chain with real SOP bodies. He will bounce
  unscoped approvals.
- **He reviews; M closes.** He does not mark things Done.
- **Releases go via R**, never M self-merging.
- **Migrations: author, never apply.** Kevin authorizes **by name**; a relayed OK is not
  authorization.
- **Don't ask him to do the mechanical half.** If something is blocked, ask a yes/no and then
  do it.

## WHAT IS IN FLIGHT (verify before trusting — this is a snapshot)
- **Wave 23 merged** (`f35d1fa0`): #218 stacked · #217 · #224 · #222. 12 suites green on the
  merged tip. R was on deploy-watch at handoff.
- **Owed immediately after:** restart the loop (three cars touched `dispatcher.py`) → apply the
  held `UPDATE M → live-session` → verify an armed `M` task does NOT dispatch (#222 step 5).
  That unblocks #229.
- **#220 M-Session Dashboard** — fully reviewed, 118 checks. **Blocked on Kevin authorizing
  `src/migrations/m_session_state.sql` BY NAME.** Needs a rebase after Wave 23 (overlaps #217
  on `dev_tasks.py`).
- **#228 cap control** — Staged, rides the next train.
- **On Kevin:** #220 migration · #232 Stand By status · #114 carry policy (a real A/B/C
  decision) · #225 · #193 · #184.

## WHERE M WAS WRONG TODAY (so you do not repeat it)
- Shipped #219 and **missed its frontend mirror** although the review step was titled *"both
  constants + the frontend boundary"*. dev sat RED for hours; `F·auto` found it.
- Claimed *"read at runtime, no deploy"* for a constant read at import. Kevin caught it.
- Hand-ticked a step and **did not tell Kevin**, so his next click marked an unbuilt step done.
- Restatused armed tasks four times, causing two races.
- Named a train "Wave 21" that R had already used — R self-corrected to 22.
- Told Kevin a run was "the trap firing" when that run had done good, in-bounds work.
