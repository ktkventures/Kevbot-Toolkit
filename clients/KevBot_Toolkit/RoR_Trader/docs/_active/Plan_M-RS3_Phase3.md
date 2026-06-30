# M-RS3 Phase 3 — Scale-out plan (tenants + cron + churn-friendly storage) — 2026-06-28

## Where we are (Phase 1 + 2 shipped, cutover live)
- **Parallel pool** (`recompute_jobs`, `RORT_RECOMPUTE_PARALLELISM`) + **DB queue** (`compute_jobs`) +
  **dedicated `batch-worker` Railway service** (now tracking `dev`) + **supervised pool** (process-per-
  strategy + CPU/IO-progress heartbeat watchdog, `RORT_RECOMPUTE_SUPERVISOR=1`, no wall-clock cap).
- **Cutover done:** `RORT_COMPUTE_REMOTE=1` on api → Update-All/Append offload to the batch-worker.
- **Fidelity:** two full 67-strategy fleet runs (pool + supervisor) trade-for-trade identical (55/67
  bit-identical; rest = Hi-Fi exit-ts jitter + ~1e-9 float noise — no added/lost trades).
- **Tooling:** `trade_snapshot.py` CLI + Admin → Trade Snapshots (exact added/removed/changed diffs).

### Measured baselines (use these for Phase 3 targets)
| Thing | Number |
|---|---|
| Fleet, sequential (old) | 215 min |
| Fleet, N=8 on batch-worker (24 vCPU / 24 GB) | ~67 min pool / ~83 min supervisor, 0 failures |
| Effective parallelism at N=8 | ~6× (tail-bound, not throughput-bound) |
| Per-strategy cost | mean ~320 s; heavy 10Sec + Hi-Fi dominate the tail |
| Box ceiling | **RAM-bound: N≈8 on 24 GB** (~2.5 GB/worker); 32 GB → ~N=12 |
| **Bloat per full_recompute fleet run** | **~146k dead tuples** (DELETE+INSERT of the whole backtest lane) |

## The constraint that reshapes Phase 3
Phase 3 is, by definition, **high-frequency recompute** (constant Append, frequent Mass Backtester).
The current write pattern — `recompute_and_persist` does a **full DELETE+INSERT of every backtest
trade** per strategy per run — bloated `trades` to **1.4 GB / 680k dead tuples after only ~4 fleet
runs on 2026-06-28**, and queries started hitting Supabase's ~8 s statement timeout (My Strategies +
the snapshot tool broke). A 20 h idle-in-transaction `bar`-read leak ([[task #7]]) had also stalled
autovacuum, compounding it.

**So the Phase-3 bottleneck is NOT more parallelism — it's storage churn.** More replicas just
produce dead tuples faster. The real enablers are the DB-health + write-pattern fixes below.

---

## Sequenced plan (do in this order)

### Step 0 — DB-health prerequisites (unblock everything) — ✅ DONE 2026-06-28
- **#7 — bar leak — FIXED.** Re-analysis (terminal context from the incident was lost) pinned the
  real root cause: `bar_cache.read_bars` is the ONLY `psycopg.connect` site and is already correctly
  context-managed — NOT a client close/rollback bug. It's a pure SELECT that ran inside an *implicit
  transaction*; a slow read hitting `statement_timeout` (postgres role = **2 min**, not the 8 s REST
  role) left the **Supavisor-pooled** backend stuck `idle in transaction (aborted)` with
  `idle_in_transaction_session_timeout=0` → nothing reaped it (**~40 h session observed live**) → pins
  xmin → blocks autovacuum → `trades` bloat. **Fix:** `autocommit=True` on the `read_bars` connection
  (a transaction is never opened → leak signature structurally impossible). Verified byte-identical to
  baseline via `fidelity_parity_suite` (TSLA 12/12 + canary 267 pass; 3 SPY fails pre-exist on dev =
  cache-staleness, Kevin backfilling). Shipped **PR #5** (`fix/bar-leak-autocommit`).
  - DB backstop (out-of-band, defense-in-depth): `ALTER ROLE/DATABASE postgres SET
    idle_in_transaction_session_timeout='120s'`. NOTE: through Supavisor's warm-backend pooling this
    only applies as physical backends recycle — a belt, NOT the primary fix (autocommit is).
  - Stranded 40 h aborted session terminated; no idle-in-txn sessions remain.
- **#6 — autovacuum for `trades` — TUNED.** `ALTER TABLE trades SET (scale_factor=0.02, threshold=1000,
  cost_limit=2000, analyze_scale_factor=0.05, analyze_threshold=1000)`. Vacuum now triggers at ~6.5k
  dead (was ~55k) with 10× cost limit → continuous reclamation of the ~146k-dead-per-run churn.
- One-time: optional `VACUUM FULL trades` / `REINDEX` in a maintenance window to return the ~1.4 GB
  on disk (cosmetic — it's reusable free space now; locks the table). **STILL DEFERRED.**

### Step 1 — Append-as-cron → EXPANDED into a dedicated plan (2026-06-28)
Step 1 grew (per Kevin) from "a cron" into a full **backtest-lane real-time evolution**: a mode
selector (`button`/`cron`/`shadow`), a 15-min cron now, and an eventual per-second **stateful REST
shadow** for real-time live↔backtest divergence detection. Measurement+profiling 2026-06-28 showed the
per-call cost is ~66 % indicator recompute (not data/DB) → per-second needs stateful incremental
indicators, not just a tighter loop.
- **→ See `Plan_M-RS3_RealTime_REST_Shadow.md`** for the full sequencing, the source-of-truth
  invariant (shadow bars must stay byte-identical to the backtester's REST pull), the settled/in-flux
  primitive, and the fidelity/validation gates. Append IS still INSERT-only/low-churn, so Step 1
  doesn't depend on Step 2's write-pattern rework.

### Step 2 — Churn-friendly write pattern (the real Phase-3 enabler)
Replace full DELETE+INSERT in `recompute_and_persist` with **diff-and-upsert**: compute which trades
actually changed (we ALREADY do this — `trade_snapshot.diff_dicts` / the `trades_dedupe_idx` unique
key on `(strategy_id, entry_fill_ts, COALESCE(exit_fill_ts), COALESCE(data_source))`) and only
INSERT new + UPDATE changed + DELETE removed. On a deterministic recompute that's ~0–2% of rows (our
fleet diff: 55/67 strategies unchanged), vs 100% today → **~50–100× less churn**.
- Alternative/complement: **partition `trades`** by something (strategy cohort / month) so vacuum and
  rewrites are localized.
- **This is what makes Mass Backtester safe.** Touches `forward_test_service.recompute_and_persist`
  (backend-agent-owned file → coordinate / gate behind a flag, default OFF, byte-identical-gated).
- **Gate:** trade-level snapshot before/after must be IDENTICAL (the snapshot tool exists for exactly
  this).

### Step 3 — Mass Backtester onto `compute_jobs` + replicas (goal #1 kin)
Migrate MB's `_active_searches` enumeration to enqueue `compute_jobs` (a new `job_type`), drained by
the batch-worker. **Add Railway replicas** of batch-worker — the optimistic claim
(`UPDATE … WHERE status='queued'`) is replica-safe, so N replicas run N concurrent jobs. This is the
workload replicas are *for* (many jobs, not one big job).
- **Prerequisite: Step 2** (else MB reproduces the 2026-06-28 bloat incident at higher frequency).
- **Decisions:** replica count vs box size; per-job vs chunked enumeration; how MB progress/results
  surface (reuse the `compute_jobs` row contract).

### Step 4 — Faster single Update-All (only if needed)
A single Update-All is one job on one box's pool. To beat that, **chunk the fleet into K sub-jobs** →
claimed by K replicas → parallel across machines. Plus vertical bump (32 GB → N≈12). Low priority —
you said Update-All is rare, and ~67 min already beats the 215-min baseline.

### Step 5 — (Maybe) Hi-Fi decouple
Hi-Fi is the per-strategy tail. Decoupling (recompute fast; Hi-Fi as an incremental follow-up pass)
speeds the bulk — **but Hi-Fi is exactly where the run-to-run exit-timestamp refinement lives** (our
trade-level diff), so there's a fidelity tradeoff. Only pursue if Hi-Fi proves to be the critical
path after Steps 1–3 but be sure to check with Kevin first as fidelity is very important.

---

## Decision summary (to settle before building)
1. **Append cadence** + replace-vs-supplement the Worker's per-bar append.
2. **Diff-upsert vs partitioning** (or both) for the churn fix — and confirm it's byte-identical via
   the snapshot tool.
3. **Replica count vs vertical scale** for Mass Backtester.
4. **Hi-Fi decouple** — yes/no, and the fidelity bar.

## Open questions / risks
- Diff-upsert touches a backend-agent-owned file (`forward_test_service`) — coordinate; flag-gate
  default OFF; byte-identical gate via trade snapshots.
- Mass Backtester's existing checkpoint/enumeration machinery may not map cleanly to `compute_jobs`
  — scope its job model first.
- Statement-timeout ceiling (~8 s) on Supabase REST means any per-strategy query must stay fast even
  at scale — keyset pagination + a clean (low-bloat) table are both required.

## Pointers
- Design: `docs/_active/Design_M-RS3_Parallel_Recompute.md` · Build plan:
  `~/.claude/plans/optimized-crafting-clock.md`
- Code: `recompute_jobs.py` (pool + supervisor), `compute_jobs_store.py`, `batch_worker.py`,
  `trade_snapshot.py`, `api/routers/trade_snapshots.py`.
- Tasks: #6 (trades bloat/autovacuum), #7 (bar leak) — Step 0 prerequisites.
