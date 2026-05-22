# Spec — Data-Worker Service & Steady-Stream Updates

**Status:** DRAFT 2026-05-22. Triggered by the update-job incident below.

## 1. Why

On 2026-05-22 a manual "Update New Data" run went pathological:
- 3+ hours in, only 22 of 70 strategies done.
- Individual strategies ran *full backtests* of 26–53 min (sid 228:
  1,549s; sid 180: 3,166s).
- The PACKTEST canaries each ran a full backtest then **skipped**
  ("no baseline") — ~50+ min of pure waste.
- It hammered Supabase into **HTTP 522 connection-timeout** failures —
  the database stopped responding to *everything* (the job's own
  writes, the app, even ad-hoc queries). A self-inflicted DB outage.

Root causes:
1. Recompute jobs + mass searches run as **in-memory threads inside the
   `api` process** — they spike API memory, die on every deploy, and
   contend with HTTP serving.
2. The updater has **no DB rate-limiting** — it bursts `select *` trade
   loads and writes until Supabase falls over.
3. "Update New Data" runs **full backtests inside the incremental
   path** for no-baseline strategies — minutes of wasted compute.
4. It's a **big catch-up batch**, not a steady stream — the longer
   between runs, the worse the pile-up.

## 2. Service topology — ONE new service

The live / algo / backtest "models" are **data lanes, not processes** —
they do not each need a service. The right split is by *workload type*:

| Service | Role | Status |
|---|---|---|
| `api` | FastAPI HTTP serving **only** | exists — heavy jobs move OUT |
| `worker` | Ralph live engine — real-time, sub-second alerts | exists, unchanged (must stay lean — latency-critical) |
| **`data-worker`** | **NEW** — steady incremental cron (backtest + algo lanes), full-recompute backlog, mass searches | to build |

- **Live model** updates = the `worker` (already real-time; can't move).
- **Algo + backtest model** updates = identical workload (periodic
  engine replay → write trades) → **one** `data-worker`, not two.
- **PACKTEST canaries** are ordinary strategies — the same cron covers
  them.

So: **one new Railway service.** The `data-worker` must be separate
from `worker` — periodic heavy recompute on the live-engine process
would jeopardise sub-second alert latency (`feedback_sub_second_latency`).

## 3. The steady-stream cron (data-worker's core job)

Replaces big manual batches with small frequent increments — the data
is *always* near-current, so no run ever has hours to catch up.

- **Cadence:** every ~1–2 min.
- **Alerts-gated:** a strategy with no new exit alerts since the last
  cycle is skipped in **milliseconds** (cheap DB count) — only
  strategies with genuine new activity get an engine run. (The existing
  algo-history cron already works this way; this generalises it to the
  backtest lane too.)
- **Incremental only:** snapshot-resume (`engine_snapshot_b64`) +
  windowed load (`get_strategy_trades_for_window`). Small windows.
- **Both lanes** per active strategy: `backtest_model` and `algo_model`.
- **Sequential + paced:** one strategy at a time, small inter-strategy
  delay — never a burst.

At 70 strategies, full-coverage-every-minute is not sequential-feasible
(~5s each = ~6 min). But the alerts-gate means most strategies skip
cheaply each cycle; only the few with new activity cost real time. A
1–2 min gated cron is feasible and keeps everything fresh.

## 4. DB-safety requirements (the hard lesson from today)

The data-worker must be **structurally incapable** of repeating today's
self-DDoS:
1. **Bounded DB concurrency** — a hard cap on simultaneous Supabase
   connections from the data-worker.
2. **No `select *` on trades** — projected-column loads only (the
   recompute path still does `select *`; Task #30 only fixed the append
   KPI lane).
3. **Bounded page sizes + row caps** on every trade query.
4. **Inter-strategy pacing** — a deliberate small delay between
   strategies so load is a trickle, not a spike.
5. **Circuit breaker** — on any Supabase 5xx/522, back off exponentially
   and stop piling on; resume when the DB is healthy.

## 5. No-baseline strategies — never full-backtest in the steady loop

The incremental cron must **never run a full backtest**. A no-baseline
strategy (e.g. a freshly-saved Mass Builder strategy):
- is detected with a **cheap DB check** and skipped in milliseconds
  (today it ran a full backtest *then* skipped — sid 228 wasted 26 min);
- is enqueued into a **separate full-recompute backlog**.

The full-recompute backlog is drained slowly, **off-peak, one at a
time, throttled**. Full backtests are inherently expensive — they get
their own slow lane, isolated from the steady stream.

## 6. What changes for `api`

Recompute jobs and mass searches move to the `data-worker`. The `api`
process becomes **HTTP-only**. Consequences:
- API memory stays flat — no more spikes during data updates.
- API deploys no longer kill in-flight jobs (Task #39).
- The API stays responsive while heavy data work runs.

## 7. The "Update New Data" button

With a steady cron, data is ~always fresh, so the bulk button is mostly
obsolete. Repurpose it as a **per-strategy "refresh now"** — enqueues a
single high-priority cycle for one strategy. No more 70-strategy
force-bulk runs. (Also retires the `force=True` alerts-gate bypass that
made the manual path a full-replay-per-strategy.)

## 8. Phasing

| Phase | Scope |
|---|---|
| **1** | Stand up the `data-worker` Railway service. Move the incremental cron (algo + backtest lanes) into it. Add §4 DB-safety (concurrency cap, pacing, circuit breaker, projected loads). |
| **2** | Move mass searches off `api` onto `data-worker`; serialise them (Task #43). |
| **3** | No-baseline full-recompute backlog lane (§5). |
| **4** | Repurpose the "Update New Data" UI to per-strategy refresh (§7). |

## 9. Open decisions

1. **Cadence** — fixed 1 min, 2 min, or adaptive (faster during RTH,
   slower after hours)?
2. **Mass searches** — on the `data-worker`, or a 4th service later if
   they prove too bursty alongside the steady cron?
3. **Job state** — DB-backed job rows (survive restarts) vs. relying on
   the cron being naturally self-healing (next tick re-picks-up)?
4. **Backtest-lane cadence** — same frequency as the algo lane, or
   slower (the backtest lane is the canonical target; the algo lane is
   the live mirror — they may not need identical refresh rates)?
