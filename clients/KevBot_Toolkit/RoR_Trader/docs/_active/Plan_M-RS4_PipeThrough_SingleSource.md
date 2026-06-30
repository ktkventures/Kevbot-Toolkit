# Plan — M-RS4: Single-Source Pipe-Through (put the REST Bar Cache to practice) — 2026-06-29

**Lineage:** M-RS3 built the **shadow design** + shipped the **write-through** (the REST Bar Cache is now
continuously fresh — Design_BarCache_WriteThrough_Unification, Steps 1–4 DONE). M-RS3 §5 Phase B is the
*continuous resident engine* that appends the backtest lane byte-identically. **M-RS4 is the
put-it-to-practice layer: route EVERY consumer through the caches (no ad-hoc Polygon), feed the backtest
model continuously from the REST Bar Cache, and read the fresh lane everywhere — so Kevin can trust that
any historical/backtest number is built on the one cached source.** Gated; live model untouched.

---

## 0. CANONICAL GLOSSARY — three bars, three lanes (read this first; kills the confusion)

There are **THREE bar sources**, each maintained, each kept in alignment by the others. "Single source of
truth" does **NOT** mean collapsing to one bar — it means **consolidating all Polygon calls** so every
consumer reads one of these three caches instead of making its own ad-hoc API call.

| # | Bar source | Table / view | Feed | Purpose / who reads it |
|---|---|---|---|---|
| 1 | **REST Bars** | `bar_cache` (Postgres) | **Polygon REST** (1Sec + 1Min, write-through stream) | **Backtest lane** + ALL historical/backtest/chart reads. The ONLY thing allowed to call Polygon REST. |
| 2 | **WS Bars** (raw, historically "cache") | `live_bars`, sources `ws`/`ws_agg` | **Polygon WebSocket** | The raw live tip / WS reference. A *subset* of what the engine consumes (see #3). |
| 3 | **Engine-Consumed / Hybrid Bars** | `live_bars` via `ENGINE_CONSUMED_SOURCES` (`ws`+`ws_agg`+`rest_correction`+`rest_insert`+`warmup_seed`) | **WS-primary + REST corrections/fills/warmup** | "What the live engine actually saw." The **algo lane replays this**; the **live model decides on the same hybrid** (`ws_rest_spliced`). `rest_backfill` is excluded (cosmetic). The live decision path is **OUT OF SCOPE** this phase. |

**THREE trade lanes** (the `trades.data_source` prefix tells you the lane + its bar source — VERIFIED in code):

| Lane | `data_source` | Bar source | Written by |
|---|---|---|---|
| **Backtest** | `backtest_<model>` (e.g. `backtest_rest_hifi`) | **REST Bars (`bar_cache`)** — pure REST | data-worker (REST), Mass Builder, manual recompute (`data_worker_engine.py:280`) |
| **Algo** | `cache_<algo_model>` (e.g. `cache_cache_locked`) | **Engine-Consumed Hybrid** (`live_bars` ENGINE_CONSUMED_SOURCES) — **WS + REST corrections/fills, NOT pure WS** | the algo/replay path (`services.py:165`, `data_loader.py:972`) |
| **Alert** | `alerts` table | **Live-Model hybrid** (`ws_rest_spliced`) | the live Worker when a real alert fires |

> **Two corrections to the 2026-06-29 audit, both now verified in code:**
> 1. `cache_%` is the **algo lane**, NOT "the data-worker's REST append" (the data-worker writes
>    `backtest_%`, `data_worker_engine.py:280`).
> 2. The algo lane is **NOT pure WebSocket** — it reads the **engine-consumed HYBRID** (WS + REST
>    corrections/inserts/warmup, `ENGINE_CONSUMED_SOURCES`), i.e. a faithful replay of the live model's
>    bars. Only `rest_backfill` (cosmetic) is excluded. So the three "bars" are really: pure-REST
>    (`bar_cache`), raw-WS (`live_bars` ws/ws_agg), and the **hybrid the engine+algo actually consume**.
> This doc is the source of truth for the naming.

**Step 0 task:** verify + lock this glossary in code (confirm every `data_source` writer + which bar
store each lane reads), then keep this table current. No code changes — just confirm + correct if any cell
is wrong.

---

## 1. The invariant (non-negotiable)

> Everything historical/backtest in the app is built on the **REST Bars (`bar_cache`)**. `bar_cache` is the
> **only** caller of Polygon REST. Every `bar_cache` row must be **byte-identical to a direct Polygon REST
> pull** for the same settled bar (already gated — `fidelity_parity_suite --writethrough`). If we break
> byte-identity, the cache stops being a faithful mirror and becomes a divergence source.

Corollary: **no consumer calls Polygon REST directly** except `bar_cache`'s own ingest/backfill. Charts,
backtests, Mass Builder, KPIs, strategy builder, drill-downs — all read a cache.

---

## 2. Workstreams (what "pipe it through" means), with order

### Phase 1 — Consolidate ALL bar loading onto the caches (THE priority — Kevin's core ask)
Every consumer loads **bars** from a cache, NEVER direct Polygon. This is the heart of "single source":
not one bar, but **one place that calls Polygon** (`bar_cache` ingest), and everyone else reads a cache.
The audit found bar loading is ~95% on `bar_cache` already. Close the gaps so it's 100%:
- **`fetch_1s_bars_for_window`** (Hi-Fi 1Sec drill-down) defaults to per-day **Polygon-direct**; make it
  read `bar_cache` first (`prime_1s_cache_from_rest_bars`) — unifies it AND ~5.7× faster. *(data_loader.py:463)*
- **Strategy Builder + Mass Builder** must read `bar_cache` for bars — **never call Polygon**. (This is
  why Phase 4 is manual-add: the ticker must already be in the cache before you build on it — no
  builder-triggered Polygon fetch, no auto-enroll exception.)
- Sweep for any other direct `load_from_polygon` / Polygon-client call in a consumer path; each becomes a
  `bar_cache` read. The intentional callers stay: `bar_cache` ingest/backfill (the one Polygon caller),
  WS ingest (`live_bars`), crypto, gap-healer, diagnostics.
- **Gate:** a grep/CI check that no consumer module imports the Polygon client directly (only `bar_cache`
  + WS ingest do). Byte-identity already guaranteed by the write-through parity gate.

### Phase 2 — Trade-output read path → FOLDS INTO PHASE 3 (do-once, no rework)
The bars are fresh, but the UI shows stale **backtest *outputs*** layered on top — the `stored_trades`
JSONB blob (default strategy view + KPIs) "can lag the trades table by weeks"
*(StrategyDetailPage.tsx ~1504; strategies.py:1771 `use_stored`)*. **Per Kevin: do NOT fix this as an
interim quick win** — it's not urgent, and we want it done once. When **Phase 3** makes the `backtest_%`
lane continuously-fresh, flip the default view + KPIs + Health/Divergence/Parity to read **that lane** in
the same change — the read-path fix and the trustworthy data land together. (Reading the continuously-
written `backtest_%` lane IS the stable end state; the engine writes the lane, consumers read it.)

### Phase 3 — Backtest model continuously fresh from `bar_cache` (the "no more Append-New" goal)
This is the heart of it — **M-RS3 §5 Phase B**. The backtest lane should populate **going forward,
continuously**, as `bar_cache` streams — NOT via manual "Append New" (which causes the problems Kevin
hits) and NOT from a periodic snapshot.
- **Why append breaks things (root cause, from M-RS3 §7.5):** the append/snapshot-resume path is **NOT
  byte-identical** — chaining snapshots drops/changes ~4% of trades at chunk boundaries (indicator state
  restored but position/ATR not byte-perfect). That boundary divergence IS the "append messiness."
- **The fix:** **ONE continuous resident engine per (symbol, tf)** that keeps warmed indicator state and
  applies each new REST bar incrementally — exactly how the live model consumes WS, but fed by `bar_cache`.
  No re-windowing, no steady-state snapshot-resume → byte-identical by construction. The incremental
  engine **already exists** (`unified_engine.py:1314 update_bar`; all packs have `indicator_incremental.py`;
  `run_unified_backtest` already iterates bar-by-bar). The data-worker already writes `backtest_<model>`
  continuously — the work is making that append **byte-identical** (continuous-resident, not
  snapshot-chained) so the lane is trustworthy without manual re-runs.
- **"Snapshot" reframed (Kevin's idea):** bars load automatically from cache (fast). A "snapshot" = run
  indicators + trades on the already-loaded bars. With the continuous resident engine, that's incremental
  and cheap — no full cold recompute each time.
- **Settled / provisional:** commit trades on **settled** bars (the `settled` column we shipped); the
  unsettled <16-min tail is provisional (re-trued on settle). Per-second belongs to the
  provisional/divergence layer (M-RS3 §6, later), not the committed lane.
- **Service (resolved — Kevin):** a **dedicated, isolated `shadow-worker` service** (own Railway service,
  `Dockerfile.shadow-worker` in the batch-worker mold). Set up scalable infra now — don't conflate the
  resident real-time engine with the batch-worker (ephemeral, RAM-hungry) or the data-worker. Shards by
  symbol/cohort later. (M-RS3 §B2 made this case; Kevin confirmed dedicated over reuse.)

### Phase 4 — Bar Cache Supply as the single, AUTHORITATIVE control surface (manual-only)
- **MANUAL ADD ONLY (resolved — Kevin).** The Supply page is the **sole authority** on what's cached and
  how deep. **No auto-enroll** from the strategy builder (that would force a builder→Polygon exception —
  exactly what we're killing). Workflow: you **add the ticker here first** (enrolls **1Min + 1Sec**,
  default **1-year** backfill on both), THEN build strategies on it — the builder reads the cache, never
  Polygon. Keep the symbols already added. Depth is operator-controlled and **expandable** (you may push
  1Min back several years; 1yr is the floor). *(Shipped today: per-row backfill-to-date on the batch-
  worker. Add: an explicit "Add ticker" that seeds both layers at 1yr.)*
- **Tabs:** a **REST Bars** tab (today's coverage/freshness) + a **WS Bars** tab (`live_bars` freshness)
  so both streams are verifiable at a glance — the WS↔REST alignment view.
- **"Verify vs Polygon" — windowed, in a modal (resolved — Kevin: add it).** NOT a whole-cache scan
  (years of 1Sec = overkill). A self-serve modal: pick a **symbol + timeframe + start/end date-time**, it
  pulls *just that window* from Polygon and diffs vs `bar_cache`, showing byte-identical / mismatches.
  Use case: a trade fired weird → look at the previous few hundred candles and confirm they printed right.
  Reuses the `--writethrough` parity logic, scoped to the window.

---

## 3. Safety & gating (every phase)
- **One env var** gates the new behavior; default = today's behavior; rollback = unset. (Phase 3 reuses
  M-RS3's `RORT_BACKTEST_LANE_MODE`; the routing phases get their own flag if they change read behavior.)
- **Byte-identical gates:** `fidelity_parity_suite --writethrough` (bars) + `trade_snapshot` diff (trades,
  settled window) before/after each phase. The continuous engine must match a from-cold full_recompute.
- **Live model untouched** (bar source #3). We do not change how alerts fire this phase.
- **Nightly full_recompute backstop** re-trues the lane so any drift can't accumulate.

---

## 4. Sequencing (what order — reflects Kevin 2026-06-29)
1. **Step 0 — Canonical glossary** (verify the three-bars/three-lanes table in code; no code change). *(fast)*
2. **Phase 1 — bar-source consolidation (PRIORITY).** Every consumer loads bars from `bar_cache`; nothing
   but the cache calls Polygon. Mechanical, lockable with a CI grep. *(Kevin's true core ask.)*
3. **Phase 4 — Supply page (manual-add authority + REST/WS tabs + windowed Verify-vs-Polygon modal).**
   Independent, low-risk; needed so a ticker is in the cache before the builder relies on it.
4. **Phase 3 — dedicated shadow service + continuous resident backtest engine (the big one).** Byte-
   identical continuous append from `bar_cache`; offline replay gate; then enable. **Rolls in Phase 2** —
   in the same change, flip the default strategy view + KPIs + Health/Divergence/Parity to read the now-
   fresh `backtest_%` lane. Removes "Append New" entirely; this is the trade-for-trade trust win.
5. **Divergence/provisional layer** — per-second evaluation of the unsettled tail, marked provisional
   (= the old WriteThrough Step 6 / M-RS3 §6). Last, on top of the trustworthy continuous lane.

Phases 1 & 4 are independent, shippable, low-risk. Phase 3 is the deep one (M-RS3 Phase B) and unlocks the
real goal: **never click Append-New again; trust the backtest model trade-for-trade.** The default-view
freshness deliberately waits for Phase 3 so it's done once (no interim throwaway).

### Relationship to `Design_BarCache_WriteThrough_Unification` (sequencing clarity — Kevin asked)
That doc's **Steps 1–4 are DONE** (write-through live, enabled, consumers reading `bar_cache`). Its
**Steps 5–6 are ABSORBED into M-RS4 — do NOT go back to that doc separately:**
- **WriteThrough Step 5** (coverage / supply / auto-enroll) → **M-RS4 Phase 4**, REVISED to **manual-only**
  (Kevin reversed auto-enroll). The "stream all supply symbols" piece stays (data-worker discovery reads
  the supply registry).
- **WriteThrough Step 6** (provisional / divergence layer) → **M-RS4 sequencing #5** (above).
**M-RS4 is the single forward plan.** `Design_BarCache_WriteThrough_Unification` is now the historical
record of the shipped cache; this doc supersedes its open items.

---

## 5. Decisions

### Resolved (Kevin 2026-06-29)
- **Phase 2 read path** → read the continuously-written `backtest_%` lane; **fold into Phase 3** so it's
  done once (no interim `stored_trades`→lane swap that gets redone). Default view isn't urgent.
- **Phase 3 service** → **dedicated shadow service** (`Dockerfile.shadow-worker`). Set up scalable,
  isolated infra now rather than conflating with the batch-worker / data-worker and wrestling it later.
- **Supply page** → **manual-add only**, the authoritative decider of what's cached + how deep. No
  auto-enroll. Builder/Mass-Builder read the cache (never Polygon); ticker must be added here first.
- **Verify vs Polygon** → **yes, build it** — windowed (symbol + tf + start/end) in a modal, not a
  whole-cache scan.

### Still open
- Phase 3 **sharding** model for 10k+ strategies (by symbol? cohort?) — design when scaling there.
- **Trade-level provisional** marker for the divergence layer (bar-level `settled` already shipped).
- How divergence **surfaces** (reuse Strategy Health / divergence-hunting tooling vs a new view).

## 6. Pointers
- This plan's parent: `Plan_M-RS3_RealTime_REST_Shadow.md` (§5 Phase B = the continuous engine; §7.5 = why
  snapshot-resume isn't byte-identical), `Design_BarCache_WriteThrough_Unification.md` (the shipped cache).
- Code: `bar_cache.py` (REST Bars), `data_loader.py:463/714` (load + the fetch_1s gap), `data_worker_engine.py`
  (writes `backtest_%`), `unified_engine.py:1314` (incremental `update_bar`), `trade_snapshot.py` +
  `fidelity_parity_suite.py` (gates), `StrategyDetailPage.tsx` + `strategies.py` (the stale `stored_trades` read).
- Memory: `feedback_two_bar_caches_ws_vs_rest`, `project_ws_rest_spliced_canary`, `project_realtime_rest_shadow`.
