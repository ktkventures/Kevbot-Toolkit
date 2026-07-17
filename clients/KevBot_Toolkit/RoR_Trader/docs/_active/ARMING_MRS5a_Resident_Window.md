# M-RS5a Resident Window — ARMING DECISION (for Kevin)

**Status: all offline work complete + validated; PAUSED for your arming call.** Nothing is armed;
`RORT_SHADOW_RESIDENT_FRAME` defaults OFF, so the shadow-worker is byte-identical to today until you flip it.
This is the SHADOW-WORKER only (separate Railway service; deploy via `railway up` from repo root — NOT git
auto-deploy). Branch `feat/mrs5a-resident-window` (commits 3704df7, 2dd09e2, bda0670, + Phase 2c step 2 pending commit).

## What it does (why we built it)
The shadow-worker re-preps the full warmup window from the DB EVERY poll just to feed the resident engine
1–5 new rows (~1000:1 waste; ~42h of DB exec time / 4 days; forced POLL_S=60 + a compute upgrade + spend-cap
disabled). M-RS5a keeps a RESIDENT prepared frame per slot and reads only the delta. Payoff: ~100–1000× fewer
DB reads → restores the 5s poll cadence, ~5× fleet headroom, much less DB pressure.

## Coverage (the whole monitored fleet is frame-eligible)
- **28 user-pack TRIGGER slots** (10Sec/15Sec/1Min/5Min) — OHLCV-only feed; the engine computes each pack's
  indicators+triggers internally (all fleet packs ship an incremental_class).
- **~39 SECONDARY-gated slots** — OHLCV + `__<tf>` columns from the cheap snapshot-extend pipeline.
- **2 excluded** (136, 329): carry a PRIMARY-`1M-` user-pack-interp gate leg (kept on full-prep — still correct,
  just not optimized). Every ineligible/erroring slot fails SAFE to today's full-prep path.

## Validation (offline, byte-identity)
- **frame == FULL-PREP manager, byte-identical** (the correct acceptance = reproduce the production path):
  proven by `_frame_userpack_probe.py` on SETTLED windows, bar-aligned AND partial-bar (live-cadence) edges,
  across 6 trigger pack archetypes (incl. rvol_v2) + secondary archetypes (2m/5m/10m/multi-TF/coarse).
- **frame == from-cold** for 1Min (269) exactly; the small 10Sec from-cold gap is PRE-EXISTING sub-minute
  MANAGER behavior (full-prep shows the identical divergence: 263 1/1/1, 290 3/3/119) — M-RS5a adds ZERO.
- REVISION RE-FEED intentionally NOT added (would make the frame better than full-prep = change outputs).
- Flag-OFF verified byte-identical; `compute_secondary_columns` refactor verified byte-identical (267/271 hash).

## Flags / kill switches (all default OFF)
- `RORT_SHADOW_RESIDENT_FRAME=1` — arm the resident frame (per-slot kill switch = set back to 0).
- `RORT_SHADOW_FRAME_COMPARE=1` — (Phase 3, if we build it) run both paths + log diffs, compare-only.
- Truth backstop: the nightly from-cold recompute stays authoritative regardless.

## Arming options (YOUR call — pick one)
1. **Compare-mode canary first (most conservative):** deploy, arm `RORT_SHADOW_FRAME_COMPARE=1` on a canary
   cohort (frame computed + diffed but NOT used for writes) for ~1 RTH week; zero diffs → then frame ON.
   (Needs the small comparator mechanism built — ~1 session; not built yet.)
2. **Direct kill-switched canary (faster):** deploy, arm `RORT_SHADOW_RESIDENT_FRAME=1` on a canary cohort
   (frame ON, writes), watch the nightly from-cold diff = zero for a week, per-slot kill switch ready.
   Archetype cohort: 263 (10Sec ut_bot_v4), 269 (1Min), 267 (2m sec), 271 (5m sec), 325 (coarse 30m/4h).
3. **Hold** — more offline soak / more archetypes first.

**Recommendation:** Option 2 with the archetype cohort — the offline byte-identity is thorough (frame ==
full-prep across every archetype), the per-slot kill switch + nightly backstop make it low-risk, and it
starts recovering the 5s cadence + DB headroom immediately. Option 1 if you want the live both-paths proof
before any frame write (I'll build the comparator on request).

## Deploy mechanism (when you say go)
- From repo root: `railway up` targeting the shadow-worker service (M-RS4 deploy-mechanism memory — NOT git
  auto-deploy; the main Worker deploys from `dev` and is untouched by this branch).
- Set the env flag in the shadow-worker's Railway service, arm the canary sids via `RORT_SHADOW_SIDS` if
  scoping to a cohort.
- Rollback: set `RORT_SHADOW_RESIDENT_FRAME=0` (or per-slot) — instant revert to full-prep.

## Open items / notes
- Cost caveat (v1): a secondary slot's per-poll `_secondary_snapshot_load_extend` recent-load grows over a
  long session (snapshot last_ts is from bootstrap). Still far cheaper than the full-primary re-read; a
  periodic drop-frame→re-bootstrap bounds it. v1.1 optimization = resident coarse snapshot advanced in-frame.
- Restores `SHADOW_WORKER_POLL_S=5` once armed + stable (currently 60 as an M-RS5 mitigation).
