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
