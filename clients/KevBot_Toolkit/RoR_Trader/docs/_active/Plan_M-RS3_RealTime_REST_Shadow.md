# Plan — Real-Time REST Shadow (backtest-lane near-live refresh) — 2026-06-28

**Lineage:** evolves M-RS3 Phase 3 **Step 1 (append-as-cron)** into the larger vision Kevin
described 2026-06-28. Phase 3 Step 0 (DB-health: #7 bar leak + #6 autovacuum) is DONE; see
`Plan_M-RS3_Phase3.md`. This doc owns everything about keeping the **backtest (REST) lane** fresh,
from a 15-min cron up to a per-second stateful shadow.

---

## 1. Vision
Make the **backtest/REST lane track the live market as closely as the REST feed allows** — ideally
per-minute, aspirationally per-second — so we can **detect live↔backtest divergence in near-real-time**
and let a trader in a diverging position exit early. Mirror how the live model consumes the WebSocket,
except the shadow is fed by **Polygon REST** and is purely the backtest model.

### THE invariant (non-negotiable — Kevin 2026-06-28)
> The REST Bars (`bar_cache`) are the **backbone / source of truth**. Anything the shadow writes —
> bars or trades — must be **byte-identical to what a from-cold backtest recompute produces over the
> same settled bars.** The shadow must *track* the backtest, never *become a different backtest.*

If we break this, the shadow stops being a divergence **detector** and becomes a third divergence
**source**. Every phase below is gated on this invariant via the trade-snapshot tool.

---

## 2. The mode selector (single variable, default = today's behavior)
One runtime env var chooses how the backtest lane is refreshed. **Default OFF / status-quo.**

`RORT_BACKTEST_LANE_MODE` ∈:
- **`button`** *(default)* — today's behavior: the backtest lane only refreshes on an explicit
  Update-New / Update-All click (in-process). No automatic refresh. Zero new risk.
- **`cron`** — Phase A. A 15-min scheduled job enqueues fleet-wide `append_recent` to the batch-worker.
- **`shadow`** — Phase B. The stateful per-(symbol,tf) REST loop keeps warmed indicator state and
  applies new bars incrementally; the lane is refreshed continuously.

Rollback is always "set it back to `button`." `cron` and `shadow` are independently shippable; we run
`cron` in prod while `shadow` is still being validated offline.

### Service placement (decided 2026-06-28) — A reuses, B dedicates
The two phases are categorically different workloads, so they live in different services:
- **Phase A (`cron`) → the EXISTING batch-worker.** It's the same work the batch-worker already runs
  (`append_recent` jobs); a cron just enqueues to `compute_jobs`. No new infra. Append is light/bursty
  (every 15 min) so contention with Update-All/Mass-Backtester jobs is negligible. (The cron *trigger*
  may be a tiny Railway cron or a loop, but the *workers* are the batch-worker.)
- **Phase B (`shadow`) → a NEW dedicated, isolated service** (`Dockerfile.shadow-worker`, same mold as
  `Dockerfile.batch-worker`). Reasons it's not a close call: (1) **different workload shape** — the
  batch-worker is ephemeral (claim-job→run→exit); the shadow is persistent + stateful (resident warmed
  indicator state, ticks per-second), i.e. the live-Worker model not the queue-drain model; (2)
  **cadence protection** — a full_recompute pegs a core for minutes; sharing a box makes the per-second
  shadow stutter → false/missed divergence signals; (3) **memory + crash isolation** — RAM-hungry batch
  workers (~2.5 GB×8, occasional OOM) must not kill a resident real-time service or vice versa; (4)
  **different scaling axis** — batch-worker scales by job throughput (replicas drain a queue), the
  shadow scales by sharding symbols/strategies across instances. Avoid the middle option (reserved-core
  process *on* the batch-worker box) — no hard isolation on a shared box defeats the cadence guarantee.

---

## 3. Measured reality (the numbers that shape this) — 2026-06-28
In-process `append_new_backtest_trades_for_strategy`, market-open, on a representative sample:
- **Per-strategy latency: ~3–34 s, highly variable** (10Sec/30Sec/1Min), even when it appends **0 new
  trades** (lane already current).
- **Profile of the cost (sid 325):** ~**66 % indicator recompute** (`run_indicators_for_group`, much of
  it slow pandas `iterrows`/per-row `Series`), ~**19 % interpreters**, ~**9 % data load** (Polygon REST
  fetch, ~3 s), DB negligible.
- **Conclusion:** the wall is **recomputing all indicators over the full warmup window from cold every
  call.** Faster data/DB won't unlock per-second; **stateful incremental indicators** will. The
  primitive partially exists (`user_packs/sr_channels/indicator_incremental.py:update_bar`) but the
  append path doesn't use it statefully (called `_build_channels` 2095× in one run).

Fleet (68 strategies) implications at N=8 on the batch-worker:
- **15-min cron: comfortable** (drains in minutes even at the slow end).
- **Per-minute full fleet: risky** (variable 10–34 s/strategy bleeds past 60 s; needs shadow or replicas).
- **Per-second: only via the stateful shadow.**

---

## 4. Phase A — 15-min cron (ship now; QoL win, low risk)
Keep the backtest lane continuously current without manual clicks.

**A1. Remote-enqueue Update-New (de-risk + reuse).** Wire `update_strategy_lanes` (mode='new',
`api/routers/strategies.py` — NOT a protected file) to enqueue an `append_recent` `compute_job`
(drained by the batch-worker, which already runs both lanes: `recompute_jobs.py:382`) instead of the
3–5 min in-process blocking call. Endpoint returns a `job_id`; UI polls the existing `compute_jobs`
status contract. Gated by the mode flag.

**A2. The cron.** A scheduled enqueuer (Railway cron service OR a lightweight loop) that, when
`RORT_BACKTEST_LANE_MODE=cron`, enqueues per-strategy `append_recent` jobs fleet-wide every 15 min.
- **Per-strategy jobs** (not one fleet job) — a slow/failing strategy can't stall the rest; parallelizes.
- **Full fleet each cycle** (Kevin's choice) — simplest, guaranteed coverage; revisit "stale-only" if load matters.
- **Anti-pileup guard:** skip a cycle if the prior cycle's jobs are still draining (don't queue faster
  than we drain). This is what makes "tighten the interval later" safe.
- **Backtest lane is the target;** the algo lane stays on the live Worker's per-bar append (Kevin
  doesn't need the cron hammering algo). Decide: cron does backtest-only vs both (append_recent does
  both today — may want a backtest-only job variant).

**A3. Correctness backstop.** Frequent incremental append amplifies known append fragilities
(under-production ~50 % on the recent window — `project_append_underproduces_recent_window`; append-edge
fossilization — `project_append_fossilization`). Pair the cron with a **periodic full_recompute**
(e.g. nightly) to re-true the lane so drift can't accumulate. Step 2's diff-upsert makes that cheap.

**Phase A gate:** a cron-appended lane over a window must match a from-cold full_recompute of the same
window (trade-snapshot diff = identical on settled bars).

---

## 5. Phase B — Stateful REST shadow (north star; per-second, scalable)
A dedicated service that, per (symbol, timeframe), **keeps warmed indicator state resident and applies
each new REST bar incrementally** — the live WS model's architecture, fed by REST. Eliminates the ~66 %
cold-recompute, making per-second feasible and scaling toward tens of thousands of strategies.

**B1. Reuse, don't reinvent.** The live model already maintains stateful per-bar indicator updates;
the shadow should feed that same machinery from REST instead of WS. Audit incremental coverage first:
which packs have an `indicator_incremental.update_bar()` vs need one (gap sizes the effort).

**B2. Dedicated, isolated service.** Its own Railway service so the real-time shadow's load never
contends with full_recompute / Mass Backtester jobs on the batch-worker (Kevin: "so it doesn't share
stuff with other important services"). Horizontally scalable (shard by symbol/strategy cohort).

**B3. Writes the SAME `bar_cache` the backtester reads** (the invariant). The shadow refreshes REST
bars and marks them settled/in-flux (§6); backtests keep reading `bar_cache` unchanged. Shadow-produced
trades on **settled** bars must be byte-identical to full_recompute (trade-snapshot gate).

**Phase B gate (offline-validatable — works on a weekend):** replay a fixed historical day's REST bars
through the stateful loop one bar at a time; the resulting trades on settled bars must be **byte-
identical** to a from-cold full_recompute of that day. This proves correctness without live markets.

---

## 6. The settled / in-flux primitive
REST bars get revised (Polygon revises recent 1Min ~5 min; our revision horizon is 45 min). To append
the fresh edge for latency *and* stay correct, we mark provisional bars/trades.

- **Today:** "settled" is *derived* at read-time (`ts < now − revision_horizon`); `bar_cache` has no
  settled column (cols: symbol, timeframe, ts, ohlcv, cached_at).
- **Proposed:** add an explicit marker so it's queryable and can be confirmed by observation
  (a bar is "settled" once successive REST pulls stop changing it, not just by clock):
  - Option 1: `settled boolean` (+ maybe `revised_at`) column on `bar_cache`.
  - Option 2: mark **trades** derived from unsettled bars as `provisional`, re-trued on settle.
  - Likely **both**: bar-level for data truth, trade-level so the UI/divergence-detector knows which
    shadow trades are still soft.
- **Divergence detection only fires on settled bars** (or flags provisional-divergence distinctly) so
  we don't cry wolf on a bar that's about to be revised.

---

## 7. Risk & validation (Kevin's main concern: this can break live↔backtest agreement)
**Risk surface:** incremental indicator state drifting from cold-recompute; provisional-bar revisions
silently changing closed trades; the shadow writing bars that diverge from the backtester's pull;
append under-production/fossilization at higher frequency; load on shared services.

**Mitigations (all gated, all reversible):**
1. **Mode flag default `button`** — nothing changes until we opt a mode in.
2. **Trade-snapshot byte-identical gate** before/after every phase (tool already exists).
3. **Offline replay validation** (§5 B-gate) — provable on a weekend; markets-closed is fine for
   correctness, only live latency/divergence-UX needs RTH.
4. **Nightly full_recompute re-true** backstop (§4 A3).
5. **`fidelity_parity_suite.py` 18/18** before any PR ready.
6. **Dedicated isolated service** for the shadow so a bug can't starve api/Worker/data-worker.

**Weekend (Sun, markets closed):** can fully build + offline-validate Phase A (cron mechanism against
historical windows) and Phase B correctness (replay). Cannot validate live latency / real-time
divergence UX until RTH Monday.

---

## 7.5 FINDINGS — offline fidelity harness (2026-06-28)
Built `src/_shadow_replay_harness.py` (offline, no live market). It compares, over one
historical RTH window, trades from **PATH A** (one-shot continuous run) vs **PATH B** (same
window fed in K segments, each resuming the prior segment's serialized snapshot — what a
periodic/per-second shadow does naively).

**Map result (huge de-risk):** the incremental engine ALREADY EXISTS and is complete —
`IncrementalIndicatorEngine.update_bar` ([unified_engine.py:1314]); **all 15 active user
packs have `indicator_incremental.py`** (100% coverage, zero to build); `run_unified_backtest`
**already iterates bar-by-bar** via `process_bar`; snapshot serialize/resume exists. So Phase B
is "wrap the existing engine in a resident service," not greenfield.

**Harness result (the constraint):** PATH B (snapshot-chained) is **NOT byte-identical** to
PATH A. On sid 263, 0.5-day, 4 chunks (3 internal boundaries): **A=164, B=162 → 2 dropped + 5
changed (~4%)**, ALL at chunk boundaries. Trades open *across* a boundary drop; entries just
*after* one get a different risk/`r_multiple` (same prices, ~2× different `r_multiple` →
different stop/ATR). **Invariant to three negative controls** — `inherit_position` (True/False),
model (`rest_hifi`/`rest_only`), and warmup (300/3000, which the resume path ignores). So it's
**structural to snapshot-resume**: the snapshot stores indicator state only (not position —
Tier 3 §8.2 always-start-flat), and even the indicator restore isn't risk-byte-perfect at the
boundary.

**Design conclusion (firm):** a byte-identical shadow MUST be **ONE continuous resident engine**
that never re-windows/re-warms/snapshot-resumes in steady state (that IS PATH A's internal
per-bar loop → identical by construction). Snapshots are for **cold bootstrap / crash recovery
only**, and every bootstrap boundary needs healing (edge-band-replace) or a full re-warm — never
a steady-state per-tick resume.

**Cross-cutting (affects Phase A too):** the EXISTING append/cron path uses this same
`get_strategy_trades_for_window` snapshot-resume, so the 15-min cron inherits the ~4% boundary
divergence — which is presumably WHY production append has the `edge-band-replace` healing, and
may relate to the known append under-production (`project_append_underproduces_recent_window`).
Phase A fidelity therefore depends on edge-band healing; validate that explicitly.

**Next proof to build:** a continuous-resident-engine harness that drives `process_bar` one bar
at a time into a single warmed engine (no re-window) and shows byte-identity to PATH A — the
positive validation of the correct design.

### 7.6 Polygon + empirical findings (2026-06-28, cont.)
- **Polygon API shape:** REST is poll-based (not a stream); recent bars revise within ~15min+2s
  (FINRA), settled bars immutable; **no native "revised" signal — poll-and-diff only.** WS streams
  per-second `A.*` continuously (the live tip). We ALREADY poll REST to reconcile (rest_verifier,
  data_worker_ingest ~60s). `live_bars.first_*` columns already record first-seen-vs-corrected.
- **Settle threshold is a TUNABLE pad, not physics:** the 45min frozen horizon is conservative over
  the real ~15min+2s. For the shadow use a tighter `RORT_SHADOW_SETTLE_MIN` (~16–20) so the
  provisional tail is small / frozen truth is fresh; tune against live Monday.
- **Empirical WS-vs-REST (settled 1Min, 06-26, TSLA/SPY):** WS close == REST close **~99%** of bars
  (median AND p95 = **0.00 bps**); ~1% differ (gap/rest_insert bars, $1–3). WS self-revision on close
  = 0% in our data (volume revises). → **a pure-REST shadow tracks the live model very closely; the
  ~1% gap + intra-bar timing IS the divergence signal.** Caveats: only TSLA/SPY live-tracked; 1Min
  only (fleet is mostly 10Sec — check sub-minute); settled data (live cadence needs Monday).
- **USER-PACK PARITY IS BY CONSTRUCTION:** every pack's batch `indicator.py` is a thin wrapper that
  replays the df through its `XIncremental.update_bar` (verified sr_channels/ut_bot_v4/vwap_v2/
  supertrend). So batch == incremental for user packs — the feared seam doesn't exist; an incremental
  shadow's user-pack columns are byte-identical to the backtest by construction.
- **Remaining fidelity surface (much smaller than feared):** (1) BUILT-IN indicators (ATR/EMA/MACD —
  the earlier boundary `r_multiple`/risk divergence was ATR, a built-in); (2) secondary-TF
  resample/shift; (3) position-state at restart boundaries. The complex user-pack layer is safe.
- **Pure-REST shadow architecture:** poll REST continuously (reuse existing poll infra), feed a
  CONTINUOUS resident engine, freeze settled bars as permanent truth, recompute the small unsettled
  tail each poll with trades marked provisional. No WS blend, no steady-state snapshot-resume.
  Snapshots/`bar_cache` only for fast cold-bootstrap warmup. DB-safe via settled/in-flux discipline
  + upsert-on-tail (no churn) on top of the #7/#6 fixes.
- **Artifacts:** `src/_shadow_replay_harness.py` (trade-level boundary fidelity),
  `src/_ws_rest_empirical.py` (WS-vs-REST study), `src/_measure_append.py` / `src/_profile_append.py`.

## 7.7 REFRAME — the shadow is largely the existing DATA-WORKER (2026-06-28)
Investigation found the streaming REST shadow is **mostly already built** as the **data-worker**
(`data_worker.py`, `data_worker_engine.py`, `bar_store.py`, created 2026-05-22, its own Railway
service, LIVE):
- **Already fleet-capable** — dynamic symbol discovery shipped 2026-05-26; runs ~68 strategies /
  ~5 symbols. "TSLA→fleet" is essentially done (TSLA was the cautious initial rollout).
- **SymbolBarStore (1s) + CoarseBarStore (1min)** = the **continuous self-correcting REST stream**
  (1s ingest + 60s recon, last-write-wins) — exactly the "self-correcting REST" Kevin described. BUT
  it's **in-memory and feeds only the streaming engines.**
- Bar-close-gated tick at `[snapshot.last_bar_ts, now-15min]`; writes `backtest_<model>` trades;
  snapshot flush every 5min. Snapshot user-pack-drop bug FIXED 2026-06-01 (codec); residual ~7%
  under-fire = always-start-flat at boundaries (continuous-warm engine trades this away).

### THE SYSTEM SPRAWL (Kevin's "how many systems / are they unified?")
THREE REST/bar systems, NOT unified:
| System | Update | Fresh? |
|---|---|---|
| **bar_cache** (persistent) | on-demand/pull-on-read + cron (default OFF) | **DELAYED** |
| **SymbolBarStore+CoarseBarStore** (data-worker, in-memory) | **continuous** (1s+60s recon) | real-time |
| **live_bars** (WS, immutable) | WS event | real-time |

Two separate REST copies (in-memory stream + persistent bar_cache) updated by different mechanisms,
not feeding each other. **bar_cache is NOT auto-updated** — answers Kevin's freshness question.

### THE UNIFICATION (Kevin's intuition — correct)
**Make the data-worker's continuous stream WRITE THROUGH to `bar_cache`.** Then bar_cache is
continuously fresh + gap-reconciled, and EVERY consumer (backtests, charts, recompute, Mass Builder)
reads ONE live source. Collapses the in-memory store + on-demand cache into one continuously-fresh
layer. `live_bars` stays separate (WS forensic, immutable — different goal).

**Divergence payoff (why this matters):** append fossilization/under-production → killed by continuous
write-through; confluence-gate blowups → strongly reduced by aligned fresh bars; snapshot/append
"messiness" → sidestepped by the continuous-warm engine (+ the 06-01 codec fix); WS-vs-REST seam →
removed (leaves ~1% inherent floor). Irreducible floor: ~1% WS/REST timing + intra-bar jitter +
built-in-indicator boundaries (built-ins being retired anyway).

### PER-SECOND clarification
The COMMITTED lane is correctly bar-close + 15min-settle gated (faster ticks yield no new committable
trades). Per-second belongs to the PROVISIONAL/divergence-detection layer (evaluate the unsettled
<15min tail sub-minute, marked provisional) — not the committed lane.

### Built-in indicators
Catalog migration to user packs is COMPLETE — no strategy is *driven* by built-ins (verified sids
136/271/274/283/285/287 = all user-pack triggers; earlier matches were gate/record key substrings).
Built-ins are legacy/being retired; user packs are parity-by-construction (§7.5).

### Revised Phase B = (1) write-through unification + (2) divergence/fidelity polish, NOT a new build.

## 8. Sequencing
1. **Phase A1** — remote-enqueue Update-New (gated) + measure single/fleet on batch-worker. *(next)*
2. **Phase A2/A3** — 15-min cron + nightly re-true backstop. Ship to dev, validate offline, enable.
3. **Phase B audit** — incremental-indicator coverage gap across packs.
4. **Phase B design → build** — stateful REST shadow on a dedicated service; offline replay gate.
5. **Settled/in-flux schema** — land alongside B (needed for provisional-edge handling + divergence UX).
6. **Monday RTH** — live latency + real-time divergence validation.

## 9. Open decisions
- Cron mechanism: Railway cron service vs in-process loop in a small enqueuer.
- Cron scope: backtest-only job variant vs reuse both-lane `append_recent`.
- Settled marker: bar-level vs trade-level vs both; column vs derived.
- Shadow sharding model for 10k+ strategies (by symbol? cohort?).
- How divergence surfaces (reuse divergence-hunting tooling / Strategy Health).

## 10. Pointers
- `Plan_M-RS3_Phase3.md` (Step 0 done, Steps 2–5), `Design_M-RS3_Parallel_Recompute.md`
- `Roadmap_Divergence_Hunting.md` (the consumer of real-time divergence signals)
- Code: `bar_cache.py` (REST Bars), `recompute_jobs.py` / `compute_jobs_store.py` / `batch_worker.py`
  (queue+pool), `trade_snapshot.py` (fidelity gate), `api/routers/strategies.py` (Update-New endpoint),
  `user_packs/*/indicator_incremental.py` (the incremental primitive to reuse)
- Memory: `reference_db_health_settings`, `project_mrs3_parallel_recompute`,
  `project_append_underproduces_recent_window`, `project_append_fossilization`
