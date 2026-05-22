# Spec — Data-Worker Service & Streaming Backtest/Algo Models

**Status:** Phase 1 + Phase 2 + Phase 2.5 SHIPPED + LIVE on Railway,
2026-05-22 EOD. Phase 3 (algo lane) + Phase 4 (cold-start backfill +
mass-search migration) remain. See `project_session_2026-05-22.md`.

**Implementation notes (post-build, vs original §9 phasing):**

- Tick window contract: `[snapshot.last_bar_ts, now − 15min]` and
  snapshot advances ONLY to `now − 15min` (the `_ALGO_HISTORY_LAG_MINUTES`
  commit boundary). Snapshot boundary equals commit boundary → engine
  never advances past an un-settled trade → no lag-edge drops (strictly
  better than the prior cron's `until=now` + lag-filter design).
- A **Phase 2.5 was added** between 2 and 3 to fix a structural gap the
  90-min 1s Tier-1 store couldn't address: cross-TF confluence warmup.
  Solution: a per-symbol **Tier-2 1-Minute, 150-day rolling coarse-bar
  store** (`src/coarse_bar_store.py`) + a `BarStoreFacade`
  (`src/bar_store_facade.py`) that routes `get_timeframe(tf)` by
  `TIMEFRAME_SECONDS[tf] >= 120s`. Drop-in for `SymbolBarStore` — the
  engine path is unchanged. With Tier-2, every REST-model cross-TF
  strategy is eligible; the `≥1Hour secondary TF → ineligible` rule
  went away.
- **Weekend-gap recovery:** when the streaming tick hits `store_gap`
  AND the snapshot is more than 1 hour stale, the engine resets
  `catchup_done=False` so the next pass runs a fresh REST-fed
  `append_new_backtest_trades_for_strategy` (re-anchors the snapshot
  via REST — the only path that can bridge a multi-hour gap).
- Scope still TSLA-only; multi-symbol is a trivial extension (more
  `SymbolBarStore` + `CoarseBarStore` instances per symbol).
- Eligibility narrows only on non-REST `backtest_model` (`cache_locked`
  / `cache_corrected`) — those would need a 3rd tier sourced from
  `live_bars` (out of scope until Phase 3 algo lane).

Original v2 design follows.

## 1. Why

On 2026-05-22 a manual "Update New Data" run went pathological: 3+ hours
in, 22 of 70 strategies done; individual strategies running 26–53 min
full backtests; the PACKTEST canaries each running a full backtest then
*skipping* "no baseline"; and the whole thing hammering Supabase into
HTTP 522 connection-timeout failures — a self-inflicted DB outage.

Root causes: heavy jobs run as in-memory threads inside the `api`
process; no DB rate-limiting; full backtests inside the incremental
path; and a big catch-up batch instead of a steady stream.

**v1 of this spec proposed an alerts-gated cron. That was wrong** — see
§4. v2 is a streaming-engine model.

## 2. Service topology — one new service

The live / algo / backtest "models" are **data lanes, not processes**.
The split is by workload type:

| Service | Role | Status |
|---|---|---|
| `api` | FastAPI HTTP serving **only** | exists — heavy jobs move OUT |
| `worker` | Ralph live engine — real-time sub-second alerts | exists, unchanged (latency-critical, stays lean) |
| **`data-worker`** | **NEW** — shared bar layer + streaming backtest/algo engines + cold-start backfills + mass searches | to build |

One new Railway service. It must be separate from `worker` — periodic
heavy compute on the live-engine process would jeopardise sub-second
alert latency (`feedback_sub_second_latency`).

## 3. The model — two layers

### 3.1 Shared per-symbol bar layer

Today every strategy's engine loads its own bars — 10 TSLA strategies
do 10× the data work. Instead:

- A **steady, per-second, per-symbol ingest** pulls fine-grained data
  (Polygon) and extends a **canonical 1-second bar series per symbol**
  (decision §10.1 — 1s always, for consistency).
- Coarser timeframes are **resampled** from it (1s → 10s → 1Min → 5Min …
  — matches the existing "resample from 1-minute bars" convention in
  CLAUDE.md).
- Every strategy/engine on that symbol **subscribes** to the shared
  series — the bars exist **once per symbol**, not duplicated per
  strategy.

This is the live engine's `SymbolHub` pattern (per-symbol bars shared
across strategies + TFs) extended to the data-worker — a proven pattern
in this codebase, not a new invention.

**Memory:** with the shared layer, total footprint ≈ (symbols ×
bar-history) + (strategies × light engine state). The bars — the bulk —
are stored once per symbol. Each per-strategy engine holds only its
indicator + position state. This is what makes ~100+ engines feasible;
it resolves the v1 memory concern.

### 3.2 Per-strategy streaming engines

Each strategy runs a **persistent, in-memory engine instance** (the
unified engine — already O(1)-incremental):

- It holds its state across ticks (the `engine_snapshot_b64` mechanism
  already exists for this).
- When a bar **closes** for the strategy's timeframe, the engine
  processes **that one bar** — O(1) — reading from the shared bar layer.
- It writes any resulting trade to the `trades` table.
- Cadence is **bar-close-paced**: a 10Sec strategy ticks ~every 10s
  (+ a small grace lag), a 1Min strategy ~every 60s. Polling faster
  than the bar cadence does nothing — bar-close *is* the steady tick.

The steady per-second ingest (§3.1) *feeds* the bar-close engines —
one steady stream in, many engines ticking off it at their own cadence.

### 3.3 The three lanes

| Lane | Where it runs | Bar series it reads |
|---|---|---|
| **live model** | the `worker` service (real-time) | WebSocket / ws_agg |
| **backtest model** | data-worker streaming engine — **always on** | `rest_hifi` bar series |
| **algo model** | data-worker streaming engine — **opt-in per strategy** | `cache` bar series |

The algo model is the **same engine code** as the backtest model —
just subscribed to the *cache* bar series instead of the *rest_hifi*
series. So it costs almost nothing to keep: it's a second subscription,
not a third architecture. Decision (2026-05-22): **keep the algo
model.** It still localises a divergence (engine bug vs data gap) —
which matters most sub-minute, where backtest↔live alignment is not yet
perfect (1Min is near bit-perfect; 10Sec still has tails). Default:
backtest lane streams; algo lane is a per-strategy toggle for when a
strategy is being actively diagnosed.

## 4. Why streaming, not a poll / alerts-gated cron

v1 proposed gating updates on alert activity. That has a **circular
blind spot**: you cannot detect "the alert engine *failed* to fire a
trade" by only updating *when alerts fire*. A strategy that goes quiet
for days — you'd have no idea whether that's "no setup" or "the alert
engine is broken." For the actual goal — trustworthy **missed-trade**
detection — gating on alerts is exactly backwards.

A streaming engine runs **every bar, every strategy, regardless of
alerts** — so a missed trade (backtest fired, no alert) surfaces the
instant it would have happened. It also gives steady, predictable load
instead of the spikes an event-driven cron produces.

## 5. DB-safety

The streaming model is naturally steady (trickle of small per-bar
writes, not bursts). Still required:
1. **Circuit breaker** — on any Supabase 5xx/522, back off exponentially;
   resume when healthy. Today's self-DDoS must be impossible.
2. **Projected loads, no `select *`** on trades.
3. **Bounded ingest** — the per-symbol ingest paces itself; bounded
   connection concurrency from the data-worker.

## 6. Cold-start backfill (replaces "no-baseline full backtest")

A streaming engine needs a baseline before it can stream. A new
strategy (e.g. a freshly-saved Mass Builder strategy) gets a **one-time
cold-start backfill** — run the engine over history once to reach
current, then it joins the stream. This is:
- a **separate, throttled, off-peak** operation — never in the steady
  per-bar path;
- the cheap-check fix for today's bug (sid 228 wasted 26 min running a
  full backtest *then* skipping). A strategy is either *streaming* or
  *cold-starting* — never full-backtesting inside the steady loop.

## 7. What changes for `api`

Recompute jobs and mass searches move to the `data-worker`. `api`
becomes HTTP-only → memory stays flat, deploys stop killing jobs
(Task #39), the app stays responsive during heavy data work.

## 8. The "Update New Data" button

With streaming engines, every lane is always current — the bulk button
is obsolete. Repurpose as a per-strategy "refresh / cold-start now".
Retires the `force=True` alerts-gate-bypass path entirely.

## 9. Phasing

| Phase | Scope |
|---|---|
| **1** | Stand up the `data-worker` service. Build the shared per-symbol bar layer (steady ingest + canonical series + resample), reusing the `SymbolHub` pattern. |
| **2** | Per-strategy streaming **backtest-model** engines — bar-close tick, read shared bars, write trades. Circuit breaker + projected loads (§5). |
| **3** | **Algo-model** lane as an opt-in per-strategy subscription (§3.3). |
| **4** | Cold-start backfill lane (§6); move mass searches off `api` + serialise them; repurpose the "Update New Data" UI (§8). |

## 10. Resolved decisions (2026-05-22)

1. **Ingest granularity** — **1-second canonical series, always**, for
   every symbol. Consistency over per-symbol optimisation; revisit only
   if memory bites.
2. **Grace lag / accuracy** — two-stage:
   - **Provisional commit** ~7s after bar close — keeps the lane fresh.
   - **REST reconciliation pass** — a background sweep re-pulls the last
     ~15 min of bars from REST canonical; where a bar materially
     changed (late prints), the engine **re-evaluates from that bar**.
   So the backtest lane is fresh immediately and converges to
   bit-identical-with-REST within ~15 min — satisfying its
   source-of-truth role (must match a strategy-builder REST backtest).
3. **Mass searches** — a dedicated `mass-worker` service eventually
   (bursty profile, would starve the steady streaming engines of CPU),
   but it's **independent of the data-worker** — built as a follow-on,
   not entangled with Phases 1–3.
4. **Engine-state durability** — **persist snapshots periodically.**
   Each engine saves its `engine_snapshot_b64` to the DB every few min;
   on restart it resumes from there (replays only the gap). Snapshot
   resume is byte-identical (fingerprint-validated) so it does not
   compromise REST-match accuracy. Cold-restart is rejected — with
   ~100 engines it causes a re-warmup stampede.
5. **Rollout scope** — **per-symbol.** Stand it up for one busy ticker
   (TSLA) first, measure real memory/CPU/DB load, then add symbols.
   Note: only ~5 symbols are tracked total (≈70 strategies on them), so
   "per-symbol" rollout is short — TSLA first for a clean scaling read.
