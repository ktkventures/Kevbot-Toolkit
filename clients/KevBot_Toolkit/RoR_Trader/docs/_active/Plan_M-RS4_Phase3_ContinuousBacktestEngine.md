# Plan — M-RS4 Phase 3: Continuous Resident Backtest Engine (the "never Append-New again" build) — 2026-06-30

**Parent:** `Plan_M-RS4_PipeThrough_SingleSource.md` (Phase 3) ← `Plan_M-RS3_RealTime_REST_Shadow.md`
(§5 Phase B design, §7.5 the byte-identity finding). This doc is the concrete build plan for the one big
remaining piece. **Default OFF, gated, fidelity-gated at every step.**

---

## 1. Goal — what this unlocks
Make the **backtest lane populate continuously, byte-identically, off the REST Bar Cache** — so:
- **You never click "Append-New" / "Update Backtest" again.** The lane is always current.
- The **stale-snapshot problem disappears** (default strategy view, KPIs, Health/Divergence all read a
  continuously-fresh `backtest_%` lane — **Phase 2 folds in here**).
- It scales toward tens of thousands of strategies (the per-bar cold-recompute wall is gone).

**The north star this serves:** backtest ↔ live trade-for-trade trust. When the backtest lane is
continuously, faithfully fresh, divergence vs the algo/alert lanes becomes a clean, trustworthy signal.

## 2. The core principle (non-negotiable — from M-RS3 §7.5)
A byte-identical continuous backtest **MUST be ONE continuous resident engine** per (strategy, or symbol
cohort) that keeps warmed indicator state and applies each new bar incrementally — and **NEVER
re-windows / snapshot-resumes in steady state.** That is exactly `run_unified_backtest`'s own internal
per-bar loop (PATH A) → identical *by construction*.

> **Why this is the whole game:** the offline harness proved snapshot-chained resume is **NOT**
> byte-identical (~4% of trades drop/change at chunk boundaries — indicator state restores but
> position/ATR/risk are not byte-perfect). The current append/cron/data-worker paths all snapshot-resume,
> which is the root of the "append breaks things" pain. The fix is architectural: resident, never resume.
> Snapshots are for **cold bootstrap / crash recovery only** — and every bootstrap boundary gets healed.

**The engine already exists** (zero to build on the indicator side): `IncrementalIndicatorEngine.update_bar`
(`unified_engine.py:1314`); all 15 user packs have `indicator_incremental.py` (parity-by-construction);
`run_unified_backtest` already iterates bar-by-bar via `process_bar`; snapshot serialize/resume exists.
Phase 3 is **"wrap the existing engine in a resident, sharded service,"** not greenfield.

## 3. Architecture
- **A dedicated, isolated `shadow-worker` Railway service** (`Dockerfile.shadow-worker`, batch-worker
  mold). Resident + stateful (the live-Worker model, not the queue-drain model). Crash/RAM/cadence
  isolation from the batch-worker (recompute) and data-worker (ingest). Resolved with Kevin.
- **Data in:** reads **`bar_cache`** (REST Bars) only — the continuously-fresh source. No Polygon, no WS.
  Settled bars are permanent truth; the <16-min unsettled tail is provisional.
- **Compute:** per strategy, a resident warmed engine; on each tick it applies only the NEW settled bars
  since its last processed `ts` (incremental `process_bar`), never re-warming. Bars are loaded once per
  symbol and fanned out to the strategies on that symbol (per-symbol feed, per-strategy engine state).
- **Data out:** writes `backtest_<model>` trades to the `trades` table (the lane the UI reads). Settled
  trades are committed; provisional-tail trades are marked and re-trued on settle.
- **Sharding:** by symbol (then cohort) across instances — scales horizontally.
### Current services → target (verified from the Dockerfiles + entry modules 2026-06-30)
| Service | Runs | Does today | Phase 3 target |
|---|---|---|---|
| **Worker** | `worker.py` | **The LIVE MODEL** — DB-backed RalphEngine per user; fires alerts off WS/live-model bars. Sub-second latency is a hard req. | **Unchanged. Do not touch.** |
| **Data Worker** | `data_worker.py` | **TWO jobs:** (1) REST ingest — `SymbolBarStore` 1s + write-through → `bar_cache`/`live_bars`; (2) **per-strategy streaming backtest engines** that tick at bar-close and write `backtest_<model>` trades **via snapshot-resume** (the ~4–7% boundary loss). | **SPLIT: keep job (1) ingest only.** Move job (2) to the shadow-worker (and fix it to continuous-resident). |
| **batch-worker** | `batch_worker.py` | **Update-All / Append-New** — claims `compute_jobs` (`full_recompute`/`append_recent`), fans across a process pool. Own service so CPU burn never contends with the live path. | **Unchanged** (keeps `full_recompute` = the nightly re-true backstop + on-demand). Append-New retired in Step G. |
| **api** / **frontend** / **Streamlit** / **flat-file-cron** | site + legacy + ingest cron | the site + flat-file ingest. | Unchanged. |
| **shadow-worker** *(NEW)* | `shadow_worker.py` | — | **The continuous-resident backtest engine** (this plan). Fully dedicated + isolated; never contends with the site, the live model, or the batch recompute. |

> **Key realization:** the Data Worker's job (2) is ALREADY "per-strategy engine state + per-symbol bar
> feed" — it's the **prototype** of Phase 3, just snapshot-resumed and sharing a box with ingest. So
> Phase 3 is **promote-and-isolate**: lift the streaming-engine code onto its own service and make it
> continuous-resident. Lower risk than greenfield. Net of the target: **4 isolated compute services —
> live model (Worker), data ingest (Data Worker), REST backtest engine (shadow-worker), heavy recompute
> (batch-worker)** — each fast and unbogged, which is exactly the isolation you want.

## 4. Build steps (ordered; each gated; offline-validatable where noted)

### Step A — PROVE the continuous-resident design is byte-identical (foundational, offline) — ✅ GREEN 2026-06-30
Build the positive-validation harness M-RS3 §7.5 called for: drive `process_bar` one bar at a time into a
SINGLE warmed engine over a historical RTH day (no re-window, no resume) and show the resulting trades are
**byte-identical** to a from-cold `full_recompute` of that day (PATH A). Extends `_shadow_replay_harness.py`.
**Gate: byte-identical on ≥3 canaries (incl. a sub-minute + a secondary-TF-gated strategy). Until this is
green, do NOT build the service.** Weekend-validatable (markets closed).

**DONE — `src/_resident_replay_harness.py`** (sibling of the negative-result `_shadow_replay_harness.py`).
PATH A = `run_unified_backtest(full_df)`; PATH C = one `UnifiedStrategy` warmed on the SAME df and fed
bar-by-bar via `process_bar` (a `ResidentEngine` prototype of the shadow-worker), replicating the per-bar
input construction verbatim. Both paths share an identical prepared df, isolating "one-shot loop vs resident
feed." **Result: 4/4 canaries byte-identical (added=removed=changed=0):** 263 (10Sec TSLA, no sec, 481t),
267 (10Sec TSLA + 2Min, 529t — sub-min AND secondary-gated), 194 (1Min TSLL + 2Min/10Min, 5t), 136 (1Min
SPY + 15Min, 44t). Confirms the resident design is faithful BY CONSTRUCTION across sub-minute primaries,
coarse + multi secondary-TF gates, and multiple symbols. **Step B unlocked.** Run:
`PYTHONPATH=. ../.venv/bin/python _resident_replay_harness.py 263,267,194,136 1`.

### Step B — Scaffold the `shadow-worker` service (inert) — ✅ SCAFFOLDED 2026-06-30
`Dockerfile.shadow-worker` + a resident loop gated by `RORT_BACKTEST_LANE_MODE` (∈ button/cron/**shadow**;
default `button` = today). Heartbeat/liveness (batch-worker mold). Claims a symbol shard. Does NOTHING
until the mode flag flips. Ship to dev inert.

**DONE — `src/shadow_worker.py` + `Dockerfile.shadow-worker`** (batch-worker mold). SIGTERM/SIGINT
graceful stop, `/tmp/shadow_worker_alive` health file on a daemon thread, `SHADOW_WORKER_DISABLED`
kill-switch. `RORT_BACKTEST_LANE_MODE` read at startup (default `button`; **fails loud** on an invalid
value — never silently picks a different lane source); `RORT_SHADOW_SHARD` for the symbol shard. Inert in
every mode (idle heartbeat only) — in `shadow` it logs "engine manager not implemented (Step C)" and idles.
Validated locally: button/shadow idle, invalid-mode fail-loud, disabled-exit, and the Polygon CI guard
(`_check_no_direct_polygon.py`) stays green (shadow_worker calls no Polygon). **Railway service itself is
created in the dashboard (points at `Dockerfile.shadow-worker`) — a deploy step, not in-repo.** NOT yet
flipped: the flag stays `button` everywhere; Kevin/backend agent sequence the flip.

### Step C — The resident engine manager (the core)
Per symbol shard: load + warm engines for that symbol's strategies; poll `bar_cache` for new settled bars
since each engine's last `ts`; apply incrementally; emit `backtest_<model>` trades (settled committed,
unsettled-tail provisional). Never re-window in steady state. Reuse the existing engine + snapshot codec.

### Step D — Cold bootstrap + crash recovery (the only place snapshots are used)
On startup / new strategy / crash: warm from a bounded from-cold window (or a snapshot) ONCE, then go
resident. Heal the bootstrap boundary (edge-band-replace) so the one cold-start boundary doesn't leak the
~4%. Handle **strategy edits** → re-warm that engine; **secondary-TF / cross-TF** resident state.

### Step E — Point consumers at the fresh lane (Phase 2 folds in here)
Default strategy view + KPIs + Health/Divergence/Parity read the continuously-fresh `trades` `backtest_%`
lane instead of the stale `stored_trades` JSONB. Done in this phase so the read-path fix and the
trustworthy data land together (no interim throwaway).

### Step F — Validate + roll out (gated, canary → fleet)
1. Step A replay gate green. 2. `fidelity_parity_suite.py` 18/18. 3. `trade_snapshot` byte-identical:
shadow lane vs a from-cold `full_recompute` over a settled window, per canary. 4. Flip
`RORT_BACKTEST_LANE_MODE=shadow` for ONE canary symbol; watch a full RTH session; compare to from-cold AND
to the algo/alert lanes. 5. Expand to a cohort, then fleet — each gated. Instant rollback = `button`.

### Step G — Retire manual append
Once shadow is trusted fleet-wide, "Append-New" / the cron append become redundant (the lane is always
current). Keep the **nightly `full_recompute` re-true** as the drift backstop (settled-bar truth).

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Boundary divergence (the ~4%) | Continuous-resident by construction (Step A proof); snapshots only at cold bootstrap, with edge-band healing; nightly re-true. |
| Provisional revisions silently changing CLOSED trades | Commit only on **settled** bars; provisional tail clearly marked; divergence fires on settled only. |
| Resident service load / a bug starving prod | Dedicated isolated service; sharded; gated default-OFF; heartbeat + crash isolation. |
| Strategy edited mid-stream | Re-warm that engine on config change (detect via `updated_at`). |
| Secondary-TF / cross-TF resident state | Covered in Step C/D; validate with a cross-TF-gated canary in Step A. |

## 6. Decisions (resolved — Kevin 2026-06-30)
- **Engine granularity → per-strategy engine state + per-symbol bar feed.** Each strategy holds its OWN
  resident engine (its indicators, triggers, position, ATR/risk); the BARS for a symbol are loaded once
  and fanned out to every strategy on that symbol. A "shared-per-symbol engine" does NOT fit: strategies
  on the same symbol use different packs/params/triggers, so their indicators+state can't be shared —
  only the raw bars can. (This is exactly what the Data Worker already does today.)
- **Sharding → by symbol now**, cohort/consistent-hash at 10k+ strategies. "Sharding" = how strategies are
  distributed across multiple shadow-worker INSTANCES; grouping by symbol means a symbol's bar feed loads
  once per instance. Not needed until we run multiple instances.
- **Data-worker role → SPLIT.** Data Worker = ingest only; the shadow-worker takes its streaming-engine
  job (continuous-resident). Migration, not a parallel third system (see the service table above).
- **Provisional trades → a `provisional` boolean flag on the trade.** Provisional-tail trades STILL appear
  in trade history (you want to see them), just **marked as not-finalized (e.g. an asterisk)**; the flag
  flips false when the bar settles. One lane, one flag — no separate provisional lane.

## 7. Pointers
- Code: `unified_engine.py:1314` (`IncrementalIndicatorEngine.update_bar`), `run_unified_backtest` /
  `process_bar`, `data_worker_engine.py` (today's snapshot-resume backtest append — the thing being
  superseded), `bar_cache.py` (REST Bars source), `trade_snapshot.py` + `fidelity_parity_suite.py`
  (gates), `Dockerfile.batch-worker` (the service mold), `recompute_jobs.py` (full_recompute backstop).
- Harness: `_shadow_replay_harness.py` (extend for Step A's positive proof).
- Memory: `project_realtime_rest_shadow`, `project_append_underproduces_recent_window`,
  `project_append_fossilization`, `feedback_two_bar_caches_ws_vs_rest`, `project_ws_rest_spliced_canary`.
