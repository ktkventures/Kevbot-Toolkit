# SOP — Nightly Bug-Hunt (unattended daily divergence routine)

**Status: ARMED-READY 2026-07-20.** The recurring routine that drives live↔backtest divergence
to zero, one night at a time. Kevin launches it as he leaves for the day; it babysits itself,
fixes what it can prove, arms the fixes behind kill-switch flags, and leaves a morning brief.
Engine = the `bug-hunt` skill, **Mode 3 (nightly)**. This doc is the operator's page.

## The daily loop (the routine we're standardizing)
> **update → fresh data → hunt bugs → fix → arm → document → repeat**

Every night: the nightly Update-All recompute settles the fleet on fresh data → the hunt runs
against that honest reference → confirmed bugs get fixed, proven, and armed behind flags → they
go live the next session → a morning brief records what happened. Keep it running day after day
until the fleet is clean, then stop.

## What Kevin runs (when leaving for the day)
Start a **fresh session** and run one of:
- `/bug-hunt nightly` — single night: waits for tonight's recompute, hunts, arms, briefs, stops.
- `/loop /bug-hunt nightly` — repeats day-after-day without relaunching.

Optional args you can pass:
- **expected nightly time** (default 00:20Z) — if you know it'll run at a different time.
- **time-box / budget** — e.g. "~3h" wall-clock, or a token budget ("+300k").
- **WIP-exclude list** — sids not to touch (mid-migration, known-broken, your parity canaries).

That's the whole ask of you. Everything below is what it does on its own.

## The two goals (the bar we're driving to)
- **Goal 1 — the tradable bar:** every monitored **gated** strategy **≥90% combined-% @10s** from
  its recent-activation window. This is the first milestone; hit it fleet-wide before chasing more.
- **Goal 2 — the ideal:** fills within **≤5s** of where the backtest says. Pursue only once Goal 1 holds.
- **Outliers always:** even at ≥90%, any *individual* trade firing >30s late gets surfaced — a few
  late fills in a passing strategy is a plumbing signal, not noise.

Non-gated strategies are deprioritized (high parity but economically useless); the loop spends its
budget on gated strategies with real profit factor.

## Phases (what it does unattended)
**Phase 0 — WAIT for the nightly, then verify it's clean.** It does NOT hunt against a
mid-recompute fleet. It monitors (via a background poll, not a fragile timer) for the nightly
recompute to *complete* — fleet lanes advanced past today's close, shadow heartbeat current,
batch-worker nightly-complete log — and confirms it didn't error. If the nightly errors or never
lands (max-wait ~3h), it drops to **diagnose-only, arms nothing**, and says so loudly in the brief.

**Phases 1–8 — the hunt (the existing bug-hunt loop).** SCAN the dashboard combined-% (5s/10s,
post-deploy window) → DOCUMENT candidates → DIAGNOSE with the replay harness + three-lane
arbitration + trade-by-trade drill → **CLASSIFY** each bug (see below) → BATCH FIX (kill-switched)
→ VALIDATE (byte-identical + 18/18 parity + prove-the-fix-resolves-or-STOP + replay-predict) →
**AUTO-ARM** proven fixes (6 rails) → MONITOR for regressions (revert on any) → RE-SCAN + LOOP
until the goal or the time-box.

**Bug classification (Kevin's taxonomy, from the triangulation):**
- **LOGIC** — engine reproduces the miss offline (`REPLAY<BACKTEST`) → a real code fix.
- **PLUMBING** — live underperforms the achievable ceiling (`LIVE<REPLAY`): stalls, dispatch lag,
  lost alerts, 30s-late fills → operational/routing fix.
- **INFRA/FLOOR** — decision-time vs settled-bar disagreement only (`algo≈live≠backtest`, WS≠REST):
  irreducible without infrastructure (canonical bar store) → NOTE it, don't "fix" it.

## Auto-arm — the 6 rails (why it's safe to let it flip flags ON)
A fix goes live next session ONLY if ALL hold:
1. **Default-OFF + byte-identical when OFF** — a wrong fix at OFF changes nothing. Non-negotiable.
2. **Proven, not plausible** — demonstrably raises the target's paired-% on settled/replay data,
   regresses no healthy strategy (incl. canary 267), passes 18/18. **If the harness can't prove it
   (coarse / sub-primary / short-fine gate classes — the known secondary-TF gap), it does NOT arm —
   it diagnoses and holds for you.**
3. **Per-fix kill switch** — each fix its own flag; the brief gives the exact one-line revert.
4. **Capped** — ≤3 flags/night, never >1 thing unproven-by-replay (ideally zero).
5. **Shadow-worker fixes** need the fingerprint SOP (`railway up`, not var-set); if it can't verify
   the fingerprint unattended, it holds the shadow-worker arm for you (auto-arms only api/batch/Worker).
6. **Irreversible = never auto-armed** (schema / data-delete → checks in).

**Why now is the safe window to trust this:** live trading is dormant, so armed fixes touch the
algo/alert lanes — **not real money**. The routine earns its track record during the pause, so by
the time live-trading resumes it's proven. (The rails stay strict regardless.)

## Staged rollout (honest about what it can prove today)
The loop is only as trustworthy as its measurement. The replay harness currently **cannot
reconstruct coarse / sub-primary / short-window-fine gate ribbons** (it produced 0 for 314 and 339
on 07-20). For those strategy classes it **diagnoses + proposes but does not auto-arm** until the
harness secondary-TF gap is fixed (the same blocker tracked in `Plan_Strategy_Health_V2.md`). Once
that lands, full-fleet auto-arm becomes trustworthy. Fixing the harness gap unlocks BOTH the V2
validation columns and confident nightly auto-arm — one prerequisite, two payoffs.

## The morning brief (what you read first)
Prepended, dated, to `docs/_active/Divergence_Hunt_Log.md`:
- **Fleet metric** — N/M gated ≥90%@10s (Goal 1) and @5s (Goal 2); delta vs last night.
- **Bugs found + classified** — sid, symptom, class (LOGIC/PLUMBING/INFRA), root cause.
- **Armed last night** — flag, sid(s), before→after (predicted + observed), service(s), revert cmd.
- **Held for Kevin** — irreversible / unprovable / ambiguous, with why.
- **Outliers** — individual >30s-late fills even where the aggregate passed.
- **Open / next.**

## What you do in the morning
1. Read the brief (top of `Divergence_Hunt_Log.md`).
2. Spot-check the **Armed last night** list on the Strategy Health page — the armed sids should show
   their predicted improvement in the fresh window.
3. If anything looks wrong, the brief gives the **exact revert command** per fix (one flag flip).
4. Review the **Held for Kevin** items — these need your call (irreversible, or harness-gap classes).

## Maintenance
- **Weekly:** fold flags that have held for a week into default-ON and retire the flag, so the flag
  list doesn't grow unbounded. (Ask Kevin before retiring — it's a small permanence step.)
- If the nightly recompute schedule or completion signal changes, update Phase 0 in the skill.
- Cross-refs: `bug-hunt` skill (the engine), `Plan_Strategy_Health_V2.md` (the harness-gap blocker +
  the reporting side), `Roadmap_Divergence_Hunting.md` (the strategy-level roadmap),
  `project_replay_harness` / `feedback_trading_target_90pct_gated_focus` (memory).
