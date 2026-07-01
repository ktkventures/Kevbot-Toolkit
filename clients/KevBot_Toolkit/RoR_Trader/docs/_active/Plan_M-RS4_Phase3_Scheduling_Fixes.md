# Plan — M-RS4 Phase 3 scheduling fixes (before fleet rollout F / retire-append G) — 2026-06-30

**Parent:** `Plan_M-RS4_Phase3_ContinuousBacktestEngine.md`. The build (Steps A–E) is complete and live on
dev (PR #17, 2026-06-30 22:15 UTC). The first extended-hours live run validated **correctness** but surfaced
a **scheduling bottleneck** that must be fixed before Step F (fleet rollout) and is a hard blocker for Step G
(retire manual append). **Default OFF / gated / validated against the byte-identity + KPI gates at every step.**

---

## 0. STATUS / NEXT-SESSION PICKUP (2026-06-30 EOD)
**The two-phase loop (§3) was REVERTED (c0cc45d)** — it introduced a **per-second FIDELITY REGRESSION**:
deferring Hi-Fi (§1b) made shadow-written stop-loss exits land on **primary-TF bar edges** (10s boundaries)
instead of intra-candle, which also broke Strategy-Health pairing (everything TBD — bar-aligned exits don't
match the live lane's per-second fills). Verified in DB on sid 280: trades created ≤21:45 UTC were ~2-5%
bar-aligned (correct), ≥21:54 UTC were 100% bar-aligned. The two-phase change **never reached dev** (only
Phase 3 A-E + the 314 fix are on dev). The **anchor fix (b14657c) is KEPT** (good, orthogonal). Shadow is
**PARKED inert**. Reconciled the 20 polluted SPY lanes via parallel full_recompute (job b99ac3c6, batch-worker)
to restore per-second exits + pairing.

**NEXT SESSION — build Fix 1 (§2, incremental LOAD for KPI *and* Hi-Fi), the REAL fix, with an OFFLINE TEST
BEFORE any deploy:**
1. Make `run_hifi_pass2` + the KPI recompute load only the RECENT window (not load-all-then-filter) so Hi-Fi
   can run PROMPTLY every poll (cheap) → exits stay intra-candle in real time + no starvation.
2. **Gate/test:** on a canary (e.g. sid 280/288), confirm shadow-written stop-loss exits are INTRA-candle
   (NOT on 10s boundaries) AND trades byte-identical to from-cold, BEFORE arming live or pushing to dev.
3. Only then re-attempt a scheduling change (bounded/parallel), re-validate live during market hours (all
   strategies fresh + intra-candle exits + pairing), then F/G.
Do NOT re-ship a KPI/Hi-Fi decoupling that lets Hi-Fi lag — per-second exits are the fidelity contract.

## 0b. REFRAME (Kevin 2026-06-30) — Hi-Fi may lag *a little* if it's VISIBLE + Health-aware
Hi-Fi does NOT drive entries, so a SHORT lag is acceptable — the real problem was the lag being ~30min AND
INVISIBLE (silently mis-classified Health as bar-aligned exits didn't pair). So the target is **reasonably
prompt (minutes, via Fix 1 incremental-load) + VISIBLE per-trade status**, not "instant Hi-Fi". Two deliverables:

**A. Per-trade status emblem** (Strategy Details → Charting Trades / trade list). Two flags per trade:
`settled` = `provisional=false`; `hifi` = `data->>'hifi_resolved' = true`.
- 🟠 ORANGE = neither (provisional AND not-hifi — raw bar-aligned, "we have something").
- 🟡 YELLOW = exactly one met (settled-not-hifi, the common in-progress state).
- 🟢 GREEN  = both (settled AND hifi = final per-second price).
Backend: surface `provisional` + `hifi_resolved` on the trade DTO (both already stored). Frontend: emblem column.

**B. Health-page TBD gate:** a trade stays TBD until BOTH `settled` AND `hifi`. Only then classify
paired / phantom / missed. This fixes the mis-classification that un-Hi-Fi'd (bar-aligned) trades caused
(everything went TBD/unpaired during the shadow runs). Find the Health pairing/classification code and add the
`settled AND hifi_resolved` precondition before bucketing.

This makes the Hi-Fi lag honest + non-breaking, and makes Fix 1 a "prompt enough" goal rather than "instant".

## 1. The finding (what the live run showed)
Armed the shadow on the 34 extended-hours strategies (SPY 30, TSLA 2, KO 1, DIA 1) for ~2.5h. Result:
- ✅ **Correct + read-only.** SPY 422 ext-hours trades, sid 301 clean 5,076-trade backfill (no dups), 0
  bar_cache POSTs, 0 errors, 0 provisional. The shadow **fills the append under-production gaps** — the payoff.
- ⚠️ **22 of 34 strategies got attention (<60m); 12 starved** (TSLA 337/338, KO 307, DIA 306, SPY 276–279,
  SPY 302–305 — stale `Last BT`/`KPIs`/`data`). Confirmed on the Strategy Health page.

**Root cause:** the shadow runs a **single-threaded poll loop** (`shadow_worker._resident_loop`:
`for slot in mgr.slots.values(): mgr.poll(slot)`), and each `poll()` can do unbounded heavy work:
1. **KPI recompute reloads ALL of a strategy's trades** every time (`recompute_kpis_for_strategy` →
   `load_trades_kpi_fields_admin`, paginated — 14k rows for sid 296, 5k for 301) + equity curve + Hi-Fi.
2. **Bootstrap backfill** writes a strategy's whole under-production gap in one pass (464 trades for 301).

A busy strategy hogs the single thread, so the loop never cycles back to the tail → starvation. At fleet
scale (67+ now, 10k target) this is fatal: quieter strategies would go stale, so the lane can't be trusted to
replace append (G).

---

## 1b. CRITICAL refinement (2026-06-30) — Hi-Fi is fidelity-critical AND reloads-all
The two-phase fix (§3, shipped) decouples KPI **and Hi-Fi** (Hi-Fi runs inside
`recompute_kpis_for_strategy`) onto the bounded Phase-2 cadence. **Hi-Fi Pass 2 is what makes a `rest_hifi`
backtest match LIVE down to the SECOND** — it walks 1-sec bars to pin entry to the exact second and resolve
which exit level (stop/target) hit first (it can CHANGE the outcome on ambiguous bars). The resident engine
emits trades at PRIMARY-TF bar resolution; Hi-Fi is the per-second refinement. So **deferring Hi-Fi lags
per-second alignment**: recent trades stay bar-aligned (possibly wrong-exit) until Phase 2 reaches them
(~every N/KPI_PER_CYCLE cycles → minutes, growing with fleet size). Eventually-correct, but a real-time
per-second/divergence gap.

**You cannot just "run Hi-Fi promptly" — Hi-Fi ALSO reload-alls** (`run_hifi_pass2` → `load_trades_admin`
loads ALL trades then incremental-FILTERS; the filter is incremental, the LOAD is not). So both KPI and Hi-Fi
hit the same reload-all wall. **⇒ Fix 1 (incremental LOAD) is required for per-second FIDELITY, not just
throughput** — make both load only the recent window so Hi-Fi can run every poll cheaply and keep per-second
alignment current. This RE-ELEVATES Fix 1 (was deprioritized after the two-phase pivot).

## 2. Fix 1 — Incremental load for KPI **and Hi-Fi** (kill the reload-all)
**Today:** `maybe_recompute_kpis` (shadow_manager) → `data_worker_engine.recompute_kpis_for_strategy`, which
**reloads every backtest trade** for the strategy, recomputes KPIs + equity curve from scratch, runs Hi-Fi.
The DB reload (N paginated GETs) is the dominant cost and it repeats on every debounce fire.

**Design (recommended): load-once + append-in-memory, recompute DB-free.**
- Per `EngineSlot`, hold the strategy's KPI-field trade series in memory: `slot.kpi_series` (list of
  `{entry_fill_ts, exit_fill_ts, r_multiple, pnl, win}` — the fields `calc_kpis` + `_build_equity_curve` need).
- **Load it ONCE at bootstrap** (`load_trades_kpi_fields_admin`, the existing call) — not on every recompute.
- As the shadow **writes** new settled trades (`commit`), **append** them to `slot.kpi_series`.
- On the KPI debounce, recompute KPIs + equity from `slot.kpi_series` **in memory** (no DB read). Persist the
  `kpis` / `equity_curve_data` columns as today.
- Hi-Fi pass is already incremental (`run_hifi_pass2(incremental=True)` walks only trades since last pass) —
  keep it, but move it off the hot path (see Fix 2).

**Lighter alternative (if memory is a concern at 10k strats):** compute the simple KPIs via a single SQL
aggregate (`SELECT count(*), sum(r_multiple), avg(...), count(*) FILTER (WHERE win), sum(pnl) ...`) instead of
loading rows; keep a windowed/incremental equity curve. More queries, less memory. Prefer the in-memory series
first (simpler, one load); fall back to SQL-aggregate only if memory pressure shows up.

**Gotcha:** the series must match what the resident lane wrote. Provisional rows are excluded (settled only).
On a strategy edit (fingerprint change → engine re-bootstrap), drop + reload `kpi_series`.

**Validation:** KPIs computed from `kpi_series` must equal KPIs from a full `recompute_kpis_for_strategy`
(byte-equal `kpis` dict + `equity_curve_data`) for several canaries. Add to `_shadow_manager_validate.py`.

---

## 3. Fix 2 — Bound + parallelize the loop (kill starvation)
Three complementary changes; do (a)+(b) first (they remove starvation deterministically), (c) for throughput.

**(a) Fairness: round-robin by oldest-polled + bounded per-poll work.**
- Process slots in order of **oldest `last_tick_at` first** (so no slot can be skipped indefinitely).
- **Cap per-poll work**: e.g. a max-bars-per-feed and a max-trades-per-commit-batch. A heavily-under-produced
  strategy backfills in bounded chunks across several cycles instead of blocking the loop on one giant pass.
  (Bootstrap of a deep gap becomes incremental, not a single 464-trade stall.)

**(b) Decouple KPI recompute from the hot poll path.**
- Move `maybe_recompute_kpis` OUT of `poll()` into a **separate low-priority worker/thread** (mirrors
  `data_worker_engine`'s separate streaming + snapshot-flush + debounced-KPI loops). The poll loop then only
  does advance + write (fast, O(new bars)); KPI/Hi-Fi run on their own cadence and can lag without starving
  trade freshness. A bounded KPI queue (slots that traded) drained by the KPI worker.

**(c) Parallelism: thread-pool the polls (I/O-bound win).**
- Run polls across a `ThreadPoolExecutor` (size from cgroup like batch-worker). Most per-poll cost is **I/O**
  (bar_cache reads, trade writes, KPI loads) → threads parallelize it well despite the GIL; per-slot engine
  state is independent so it's thread-safe. **NOT** a process pool — resident engines are stateful in-process
  and can't be shared across processes (that's why sharding is by symbol across *instances*, §6 of the parent).
- Concurrency caveats: snapshot the slot list under a lock vs `discover()` mutation; ensure each thread uses
  its own DB connection (the direct-PG path opens per-call; PostgREST client is shared-safe).

**Sharding interplay:** §6 of the parent already shards by symbol across instances at 10k+. (a)+(b)+(c) make a
*single* instance keep its shard fresh; sharding scales across instances. Both are needed.

---

## 4. Sequencing & gates
1. **Fix 1 (incremental KPI)** — biggest single cost; validate KPIs byte-equal vs full recompute.
2. **Fix 2a+2b (fairness + decoupled KPI)** — removes starvation deterministically.
3. **Fix 2c (thread pool)** — throughput; optional if 2a+2b already keep the shard fresh.
4. **Re-validate live:** re-arm the 34 ext-hours shard (or a bigger one) and confirm **all** strategies stay
   fresh (every slot's `kpis_computed_at` < one loop interval) — i.e. 34/34 attention, 0 starved, over a full
   session. Trades still byte-identical to from-cold (`_shadow_manager_validate.py`).
5. Only then: Step F fleet rollout → Step G retire append.

**Invariants preserved:** read-only bar_cache (`no_backfill`), byte-identical settled trades, gated default-OFF
(`RORT_BACKTEST_LANE_MODE`, `RORT_SHADOW_DRY_RUN`). New flags suggested: `RORT_SHADOW_KPI_INMEM` (Fix 1),
`RORT_SHADOW_PARALLELISM` (Fix 2c), `RORT_SHADOW_MAX_BARS_PER_POLL` (Fix 2a) — all default to today's behavior.

## 5. Pointers
- Loop: `shadow_worker.py::_resident_loop` (the sequential `for slot … poll`). 
- Poll/commit/KPI: `shadow_manager.py::poll`, `commit`, `maybe_recompute_kpis`; `EngineSlot` (add `kpi_series`,
  `last_tick_at`).
- KPI compute: `data_worker_engine.recompute_kpis_for_strategy` (the reload-all to make incremental);
  `db.load_trades_kpi_fields_admin`, `services.calc_kpis`, `api.services.backtest_service._build_equity_curve`,
  `api.routers.strategies.run_hifi_pass2(incremental=True)`.
- Mold for the decoupled loops + thread pool: `data_worker.py` (separate streaming/flush/KPI loops),
  `batch_worker.py` (cgroup-sized pool).
- Validation: `_shadow_manager_validate.py` (extend with a KPI-equality check + a fairness/no-starvation check).
