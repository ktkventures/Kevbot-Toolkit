# Design notes — M monitoring, and the M-session/M·auto split

**Status: IDEAS FOR A CONVERSATION. Not scoped, not a task.** Kevin, 2026-07-30:
*"don't scope it quite yet... certainly list out some ideas. I just want to have a
conversation about it, but I have to go to bed."*

Written by M (1805c263) at the end of the 07-29/30 session, while the evidence is fresh.

---

## 1. The diagnosis Kevin arrived at, and why it holds

> *"I think I may have just realized why things felt like they ran so smoothly on Saturday,
> but we've been running into more issues lately... if you don't have the monitor going,
> things kind of stall out."*

**This is correct, and the mechanism is structural rather than a matter of diligence:**
**M's turn ends when M stops talking.** Absent an external event, nothing wakes M. So M is
reactive by construction — it acts on a Kevin message or a monitor firing, then goes quiet
even with work in flight.

The dispatcher keeps *agents* moving. Nothing keeps *M* moving.

**Evidence from this session.** Kevin caught M stalling twice, explicitly (*"Did you pause
again? Come on man!"*). Both times M had just posted a status report and ended the turn,
because reporting felt like the deliverable. After a 15-minute heartbeat was armed, the same
session shipped 4 trains, closed 5 tasks, and caught the push mis-attribution — because
something asked "what is stuck?" every 15 minutes.

**And it explains Saturday.** Kevin was supplying the cadence himself: monitor this, kick
that off, check on this. When he stepped back expecting self-drive, the agents self-drove
and the coordinator did not.

**There is no documentation for this.** `Session_Charters.md` has zero mentions of
monitoring or heartbeats. `bug-hunt` and `replay-check` document monitoring for E's lane.
Board #133 is about the *dispatcher loop* running on cron — agent liveness, not M's.

---

## 2. What M should actually be watching

Every item below is drawn from a real failure in this session, not invented.

| Watch | Why — measured on 07-29/30 |
|---|---|
| **Stranded / unpushed agent branches** | **8 rescues.** Every one found by an ad-hoc check M reinvented, not a routine. |
| **`Staged` items and the AGE of the oldest** | The thing Kevin described as "gets staged and then sort of dies". PRs #129/#130 sat mergeable ~4h. |
| **Failed runs — did they self-heal or park?** | Two runs died on 07-29 and parked their tasks in a lying `Review`. That became #197. |
| **Running loop version vs dev** | M forgot the restart twice; preflight (5) caught it. Preflight cannot see the *in-memory* version — that gap is #206. |
| **Untracked-vs-tracked collisions** | **3 silent `git pull` refusals.** `git merge --ff-only` reports it in a way that is easy to miss when you read only the last line. |
| **Anything armed that should not be** | M moved #195 Staged→Review while armed and it dispatched into an M-reserved step. |
| **What is waiting on Kevin** | So it is surfaced, not discovered. 10 items sat on him for most of the session. |
| **Chain drift** (finished steps left unticked) | 6 hand-ticks. Became #202 — which then failed its own first live test. |

**Cadence:** 15 minutes was about right during heavy work. Quieter periods could stretch;
the number matters less than that *something* fires.

**Output:** the heartbeat should write a durable line per beat, not just notify. Kevin's
words: *"I also want to document things in a way that I'm able to look at."* Chat scrollback
is not a record.

**Two parts, and both are needed:** a **skill** says WHAT to check and what to do about each;
a **monitor** is what WAKES M to run it. A skill alone changes nothing, because a skill is
only instructions followed when invoked — and the failure mode is not being invoked.
So step one of the skill has to be "arm the heartbeat".

---

## 3. Kevin's question: should `M` split into `m_session` and `m_automate`?

> *"Are there situations where it's better just assigned to the auto, and then there are
> other situations where it's better to assign to the session because you are the M session?"*

**The distinction is real and it caused a live incident.** `M` currently means both, so a
task assigned `M` may be picked up by a headless `M·auto` run *or* worked by the M session —
and on 07-30 both happened to #195 at once: M moved it to `Review`, which un-refused it
while still armed, and the loop dispatched `M·auto` into a step whose SOP said *"M restarts
the loop"* — an action explicitly reserved for the session.

**Steps that genuinely want the SESSION, not a headless run:**
- reviewing another agent's work (the whole value is that the reviewer holds context the
  builder did not)
- merge reconciles requiring judgement (#197's 372-line reaper conflict)
- authoring a train brief (until #211 lands)
- anything that restarts the loop, applies a migration, or touches prod

**Steps that are fine headless:** build, test, scope, write a spec, mechanical reconciles.

### Two ways to express it

**(a) Two agent names — `m_session` / `m_automate`.**
Explicit and readable on the board. Cost: touches the agents registry, every prompt
template, ~16 existing board assignees, and the roster in the charter. Also slightly
misleading — `m_session` is not dispatchable at all, so it is not really an agent.

**(b) A step-level flag — e.g. `headless: false`.** *(M's preference.)*
Same shape as the `ai_eligible` toggle Kevin just shipped, one level down: the task says
*may an AI run this*, the step says *may a HEADLESS run take this step*. Owner stays `M`;
the dispatcher refuses the step to `M·auto` and reports it as **waiting, not stuck** —
exactly how `mode=discuss` already behaves. No registry change, no assignee migration, and
it composes with everything already built.

**Open question for the conversation:** is the readability of (a) worth its migration cost?
M leans (b), but Kevin reads the board far more than M does, so his instinct on legibility
should probably win.

---

## 4. Kevin's question: board dispatch vs spinning up agents in-session

> *"The thing I like about it is that you respond to it right away when it comes back, but
> the thing I don't like about it is that I can't really see too much of the details."*

He has named the exact trade-off.

| | in-session agents | board dispatch |
|---|---|---|
| latency | M reacts instantly | up to one poll (20s) + reap |
| Kevin's visibility | **poor** — lives in M's context; disappears | **good** — task thread, `run_history`, PR |
| audit trail | none | full, and diggable only when he wants |
| cost | cheaper, no re-briefing | pays context re-establishment per dispatch |

**M's view: the board is right for anything that produces an artifact** — code, a branch, a
spec, a decision. This session's entire audit trail exists *because* work went through the
board; none of it would be reconstructable from chat.

**In-session agents are right for research that only informs M's next move** and produces
nothing to keep — "find every caller of X", "does this doc still say Y".

Note: M has not been using in-session agents at all this session, because its operating
instructions say not to unless Kevin asks. Worth deciding explicitly rather than by default.

---

## 5. Kevin's own read, which is worth recording

> *"I think a lot of my obsession with trying to get the AI agents kicking off, executing,
> and working was a large part because of my frustration with not having monitoring active
> during this M session."*

Probably right, and it reframes the priority. The pipeline was never as broken as it felt —
it was **unattended**. Which means the next investment is arguably the heartbeat and the
documented watch-list, not more automation of execution.

**Related open items:** #211 (generate the release brief — the last manual hop in
`Staged → shipped`) · #206 (preflight cannot see the loop's in-memory version) · #133
(dispatcher loop on cron) · #193 (the dashboard, now live — the *visibility* half of this).

---

## 6. Kevin's follow-up: in-session subagents fed the dispatcher's own prompt

> *"would it be possible for you to use in-session agents that you feed with the same thing
> you would feed an agent that is on the board directly... so that they're functionally the
> same thing as an on-board agent except it actually makes you more responsive?"*

**Technically yes, trivially.** `build_prompt()` is a function. M could import it, generate the
identical prompt — task, current step SOP, thread, all four contracts — and hand it to an
in-session subagent. Subagents have bash, so they could write to the board themselves rather
than M transcribing.

**But it optimises the wrong variable.** The dispatcher's poll is 20 seconds. **M's stalls on
07-29/30 were 15 to 60 MINUTES**, because nothing woke M — the cause Kevin identified himself
in §1. In-session agents would buy back 20 seconds and cost four things:

1. **They die with M's session.** A headless run is `setsid`-detached and survives; a subagent
   is bound to the session. This box killed sessions **eight times on 07-29** (board #188).
   Everything that survived tonight survived *because* it was detached.
2. **The machinery stops applying.** `run_history`, the reaper, auto-retry (#197), the
   step-guard, the push leg (#195), `CONCURRENCY=3`, one-run-per-lane, the 24/day cap — all of
   it exists because runs go through the dispatcher. Subagents run ungoverned unless it is all
   reimplemented.
3. **Kevin's board controls stop working when M is not watching.** Flipping `ai_eligible` or
   pressing Run triggers the *dispatcher*, not M. Moving execution in-session makes the toggle
   he just shipped dependent on M's attention.
4. **M's context fills with their output.** ~20 runs tonight. Headless keeps the bulk on the
   board, where M reads only what it chooses.

**His "agent names as shorthand for prompts" instinct is right — and it is already the
design.** The registry's per-lane scope/boundaries/worktree *is* that shorthand, and
`build_prompt` composes it with the step SOP. The value was always in the prompt composition,
not in which process executes it.

**Where subagents genuinely win:** short research that informs M's next move and leaves no
artifact — "find every caller of X", "does this doc still say Y". No survivability needed,
nothing to audit. **M has not used them at all**, because its operating instructions say not
to unless Kevin asks. Worth deciding deliberately rather than by default.

## 7. Kevin's follow-up: a fourth agent status

> *"could we add a fourth option that's called in-session subagent or whatever? I think live
> session kind of implies it's like a chat thread... but that's what I thought headless was."*

**His confusion is the naming's fault.** `headless` does not mean in-session — it means *"the
dispatcher may spawn this lane as a detached `claude -p` process."* That is a poor name for
that, and renaming it is cheap and worth doing on its own.

**A status is only as real as the code that reads it.** Today the sole consumer is the registry
query `status=eq.headless`. A fourth value means nothing until something honours it — so it is
a dispatcher change, not a label.

**On subscription vs API:** the dispatcher shells out to `CLAUDE_BIN = "claude"` — the same CLI,
on the same machine, under the same auth as the M session. So there is no evident
subscription-vs-API difference between a detached run and an in-session one. **Flagged as
unverified**: neither Kevin nor M has confirmed how the CLI bills in each path, and it should
not be leaned on until someone does.
