# Roadmap — Divergence Hunting (Live ↔ Backtest Pair-Rate to 95%+)

**Last updated:** 2026-06-05 EOD
**Goal:** Drive fleet-wide live↔backtest pair rate to 95%+ across the canary cohort.
**Status (2026-06-05):** **96.7% achieved on sid 302.** Architectural work delivered.
Remaining issue is OPERATIONAL (BT-lane backfill), not engine divergence.

This doc is the single source of truth for what's done, what's open, and where the
related artifacts live. Update at each session's end.

---

## TL;DR — The Big Discovery (2026-06-05)

After a full Update All Data on sid 302 (UT Bot V4 canary), pair rate jumped
from ~32% to **96.7% (321 paired / 332 events)** across the entire trading day.

| hour | before recompute | after recompute |
|---|---|---|
| 13:30-14:30 | 34.7% | **98.0%** |
| 14:30-15:30 | 26.9% | **100.0%** |
| 15:30-16:30 | 39.5% | **100.0%** |
| 16:30-17:30 | 96.7% | **100.0%** |
| 17:30-18:30 | 32.6% | **100.0%** (gap filled!) |
| 18:30-19:30 | 58.3% | **100.0%** |
| 19:30-20:30 | 0.0% → 72.2% (still streaming) | **96.3%** |

**The "low pair rate" we measured all day was a MEASUREMENT ARTIFACT.** When the
backtest data is fresh, live and backtest agree on essentially every trade. The
architecture is trustworthy.

**Why the measurement was off:** BT-lane writes happen via Data Worker's streaming
pipeline. Each Data Worker restart (from a deploy) causes a 5-10 min warmup window
during which the engine processes bars but rejects entries because stateful stops
(swing/ATR) need ~30 bars to populate their high/low buffers. During warmup: bars
processed, no trades fire. Those bars get NO BT trade record. Today's 5 deploys
in 52 minutes (17:31-18:23 UTC) created cumulative BT-lane gaps that made the
pair-rate metric look ~50% when it was actually 100% — the trades just weren't
in the database to pair against.

---

## Today's deliverables (2026-06-05)

### Architectural fixes shipped (all in production)

| commit | what | impact |
|---|---|---|
| `c1cfb4b` | #57C engine write hooks for bar_diagnostics | Per-bar indicator state capture (live/BT/algo) |
| `8f745a5` | `_probe_bar_diagnostics.py` CLI tool | Side-by-side live↔BT divergence inspection |
| `5d634ae` | #57E+F per-bar REST verification (4-pass settle) | Layer 1 drift_uncorrected 0.7% → 0.1% (**7× reduction**) |
| `b5f40c7` + `3c5861c` | #57G `live_corrected` source rows | Post-splice state captured for analysis |
| `0901baf` + `c256900` | #57H phantom_pre_correction reclassification | Structural WS noise labeled out of needs_investigation |
| `3429dce` | #42 session filter on injected primary_df (services.py:264) | Sid 268-class bug fixed (RTH/Extended Hours respected on cache_locked path) |

### Migrations applied (you ran these via Supabase SQL editor)

- `bar_diagnostics_table.sql` — per-bar indicator state ✓
- `bar_diagnostics_live_corrected.sql` — `live_corrected` source value ✓
- `alerts_would_fire_post_correction.sql` — phantom reclass flag ✓

### Validation results

- **Layer 1 — Bar fidelity:** GREEN (drift_uncorrected 0.1%, verified 87%, corrected 11%)
- **Layer 2 — Pair rate on sid 302 with current BT data:** **96.7%** (321/332)
- **Per-bar verify firing:** confirmed via worker logs (multiple SPY/TSLA corrections per minute)
- **live_corrected rows landing:** 86+ rows accumulating since deploy
- **phantom_pre_correction reclass:** working (2 alerts flagged in first window post-migration)

---

## Open work (in priority order)

### P0 — Smart gap-detection cron (today/tomorrow's work)

**State:** Designed but not built. This is the operational missing piece to make
the dashboard pair-rate metric trustworthy without manual intervention.

**Why we need it:** BT-lane gaps from deploy churn or any transient Data Worker
restart create false "phantom" alerts in the dashboard. Manual Update All Data
backfills them but doesn't scale to tens of thousands of strategies.

**Design:**
- Detection: each strategy gets a 2h lookback; bucket alerts + BT trades into 5-min
  windows; flag bucket as gap if `alerts ≥ 2 AND bt_trades < alerts/3`
- Action: queue targeted `update_jobs` over `[gap_start - 30min warmup, gap_end + 5min buffer]`
- Schedule: hourly cron; rate-limit (skip strategy if recomputed in last 30 min)

**Cost at scale:**
- Today (46 strategies): ~25s detection + ~5min recomputes if gaps exist
- 10K strategies (your future): detection parallelizable to 5-10 min; recomputes
  via existing `update_jobs` queue with multiple workers

**Verification approach (Kevin's idea, 2026-06-05):**
Run detector on one strategy → backfill identified gaps → then click manual
Update All Data → compare results. If detector's targeted backfill ≈ full
recompute, validated.

### P1 — Algo-lane cache_* trades dead (task #60)

**State:** Last `cache_cache_locked` trade written 2026-06-04T21:38 UTC. 23h silent.
Independent of today's architectural work.

**Investigation needed:** Why algo-history cron writes nothing (despite running —
last_recompute_until_ts stamps are fresh). Possible regression from yesterday's
`3322298` algo-lane incremental fix.

**Impact:** Doesn't affect live↔BT pair rate (we filter to `backtest_%`), but it's
a latent bug. Algo lane was intended to mirror "what live actually did" — without
it we lose that secondary check.

### P2 — Operational: MANUAL streaming during active dev

**Pattern (your insight, 2026-06-05):**
- Active dev session → flip streaming to MANUAL via /admin/update-jobs
- Deploy freely without causing BT-lane gaps from warmups
- After deploys settle → flip to AUTO + run Update All Data on touched strategies
- Avoids today's gap pattern entirely

No code change needed — toggle exists from yesterday.

### P3 — Investigation cleanups

- #44 swing-stop cluster: REFRAMED — only sid 274 is genuinely silent across 5 days
  (other "silent" strategies fire normally per-week, just market-conditional gates).
  Investigate sid 274 specifically when there's time.
- #55 split BT/Algo UAD buttons — deferred until scale strain
- #56 cross_exec_type_mismatch root cause investigation — classifier ships; underlying
  state-drift mechanism understood (same as #53 in different indicator)

---

## Linked artifacts

| Doc / Tool | Purpose |
|---|---|
| `docs/SOP_Strategy_Health_Check.md` | 4-layer methodology + active classifier buckets |
| `docs/Known_Bugs.md` | Active bug log |
| `src/_remeasure_pair_rates_5s.py` | Macro pair-rate measurement |
| `src/_divergence_walkthrough.py` | Layer 1 + 2B walkthrough |
| `src/_fill_delta_analysis.py` | Layer 3 fill_ts delta |
| `src/_probe_bar_diagnostics.py` | Live vs BT vs algo state at any bar |
| `src/_verify_streaming_toggle.py` | Streaming-toggle observability |
| Backup branches | `dev-backup-pre-streaming-toggle-test`, `dev-backup-2026-06-03-eod` |

---

## Measurement methodology (post 2026-06-05 update)

**Critical:** Before measuring pair rate, ensure BT data is current. If you measure
against stale BT data, "phantoms" will be measurement artifacts, not engine divergence.

**Correct workflow:**
1. Click Update All Data on the strategies you're measuring (or run gap-detection cron once it exists)
2. Wait for completion (~3-5 min per strategy)
3. THEN run pair-rate analysis
4. Result is the true engine divergence

**Incorrect (what we did most of today):**
- Measure pair rate against live-streaming BT data
- See low numbers
- Mistake operational artifacts for engine bugs
- Spend hours on "fixes" that aren't actually needed

This methodology update should be added to the SOP.

---

## Run-order for next session

1. **Build smart gap-detection cron** — code the detector + action + cron wiring (~2 hrs)
2. **Verify on one strategy** — run detector → backfill → compare to manual Update All Data
3. **Deploy as hourly cron** — once verified accurate
4. **Re-measure pair rate fleet-wide** — should be 95%+ across all canaries with gap-detection running
5. **Investigate #60 dead algo lane** — separate latent bug, ~30 min
6. **Update SOP** with measurement methodology + new operational patterns
