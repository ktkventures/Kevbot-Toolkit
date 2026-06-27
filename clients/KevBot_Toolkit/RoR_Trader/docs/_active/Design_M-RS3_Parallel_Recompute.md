# M-RS3 — Parallel Recompute + Dedicated Railway Update Service — 2026-06-27

**Goal:** turn the fleet "Update All Data" from a **215-min sequential all-nighter** (67/67,
2026-06-27 baseline) into a routine **N×-faster** job, by running the *embarrassingly parallel*
per-strategy recompute unit across N processes on a **dedicated Railway service** that never
contends with live trading.

Supersedes the M-RS3 stub in `Recompute_Scalability_Findings.md`. Follows M-RS1 (warmup
right-size, shipped+enabled) and the coarse-secondary fix (b47bbbc). M-RS2 (shared 1s bar
store) is independent and can land before or after — they compose.

---

## Premise correction (vs. the original #46/#47 framing)

The handoff queued "#47 = the cheap `compute_parity=False` bulk lever, do first." **That lever
is already pulled and is a no-op:**

- The fleet loop already passes `compute_parity=False` (`recompute_jobs.py:320`).
- Parity is now fully **decoupled** from recompute — `_do_recompute` no longer triggers it at
  all (`forward_test_service.py:442–448`); it's a separate user-triggered action
  (`queue_parity_for_strategy`).

So there is no cheap latency to reclaim there. **Parallelism is the only real lever**, and the
per-strategy unit it parallelizes is already as cheap as M-RS1 made it (~193 s/strat avg).

---

## The unit being parallelized

`recompute_jobs._run_job_worker` iterates `strategy_ids` **strictly sequentially**
(`recompute_jobs.py:268`), per strategy calling **both lanes**:

- backtest: `recompute_and_persist_stored_trades(sid, user_id, compute_parity=False)`
- algo:     `recompute_and_persist_algo_trades(sid, user_id)`

Strategies are independent (no cross-strategy state) → embarrassingly parallel.

**Cost shape (Recompute_Scalability_Findings.md):** ~70 % CPU (engine `process_bar` +
user-pack indicators + interpreters) + ~30 % Polygon I/O. The CPU is **GIL-bound** Python →
**threads will not scale it**; we need **process-level** parallelism (or M-RS2's shared store
to kill the I/O, orthogonally).

---

## Why a *dedicated service* (the chosen venue)

The semaphore comment at `forward_test_service.py:25–33` documents an **18× slowdown** when
CPU-bound replays contend on one instance. Running a process pool on the **live API/Worker**
during RTH would starve live trading — and **sub-second latency is a hard requirement**.

A dedicated Railway service isolates the CPU burn entirely: it can run **any time**, scale
**vertically** (cores/RAM) and **horizontally** (replicas), and the live worker/API never feel
it. The project already runs multi-service from distinct Dockerfiles (`Dockerfile.api`,
`Dockerfile.worker`, `Dockerfile.data-worker`) — `data-worker` is the exact template.

---

## The central problem: cross-service job handoff

Today recompute jobs live **in-memory in the API process** (`recompute_jobs._active_jobs`, a
daemon thread; frontend polls `GET /api/jobs/{id}`). A separate service **cannot see** that
state. So M-RS3 needs a **DB-backed job queue** as the handoff:

```
Frontend ──POST /api/jobs/recompute──▶ API
                                        │  (RORT_RECOMPUTE_REMOTE=1)
                                        ▼
                              INSERT recompute_jobs row (status='queued')
                                        │
   Update Service  ──poll: claim oldest queued (FOR UPDATE SKIP LOCKED)──┘
        │
        ├─ ProcessPoolExecutor(N) over job.strategy_ids
        │     each worker: registry init → admin ctx → both lanes → result
        ├─ as futures complete: UPDATE row (progress, per_strategy_results, summary)
        └─ on done/error/cancel: UPDATE row status
                                        ▲
Frontend ──GET /api/jobs/{id}──▶ API ───┘  (reads the row; same response shape)
```

Backward-compatible: behind `RORT_RECOMPUTE_REMOTE` (**default 0**). At 0, `submit_recompute_job`
keeps today's exact in-process-thread behavior — **zero change** for single-strategy refresh
and the current admin button. At 1, the API enqueues a DB row and the update service runs it.

### Schema — `recompute_jobs` table

Mirror the in-memory job dict (already the API response contract), so `GET /api/jobs/{id}` is a
straight row read:

| col | type | note |
|---|---|---|
| `id` | uuid PK | job_id |
| `user_id` | text | scoping |
| `strategy_ids` | jsonb | int[] |
| `job_type` | text | `append_recent` \| `full_recompute` |
| `force` | bool | |
| `status` | text | `queued`/`running`/`completed`/`failed`/`cancelled` |
| `claimed_by` | text null | replica id (horizontal-scale safety) |
| `claimed_at` | timestamptz null | for stale-claim reclaim |
| `created_at`/`started_at`/`completed_at` | timestamptz | |
| `current_strategy_idx/_id/_name` | int/int/text | progress |
| `progress_label` | text | |
| `per_strategy_results` | jsonb | append-only list |
| `summary` | jsonb | totals |
| `error` | text null | |
| `cancelled` | bool | cooperative cancel flag the service polls |

Claim is `UPDATE ... SET status='running', claimed_by=:rid WHERE id=(SELECT id FROM
recompute_jobs WHERE status='queued' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING *` → safe for multiple replicas.

---

## The process pool (the throughput core)

Replace the sequential loop body with:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
N = int(os.getenv('RORT_RECOMPUTE_PARALLELISM', '1'))   # default 1 == today
with ProcessPoolExecutor(max_workers=N, initializer=_pool_init) as ex:
    futs = {ex.submit(_recompute_one, sid, user_id, job_type, force): sid
            for sid in strategy_ids}
    for fut in as_completed(futs):
        sid = futs[fut]
        res = fut.result()           # {bt, algo, elapsed} or {error}
        _write_progress_row(job_id, sid, res)   # DB update, parent side
        if _is_cancelled(job_id): break          # cooperative cancel
```

**Per-worker init is the landmine (`_pool_init`)** — each child process is a fresh interpreter:

1. **Pack registry** — call `scan_and_load_all()` or every recompute silently returns **0
   trades** (the packs=0 trap; `feedback_worker_pack_registry_init`,
   `feedback_local_script_pack_registry`). This is THE way a parallel run can be silently wrong.
2. **DB admin client** — each process opens its **own** Supabase client; transaction pooler
   (6543) needs `prepare_threshold=None` (the f46abbf fix). Do NOT share a client across procs.
3. **Admin user context** — `set_admin_user_context(user_id)` is thread/proc-local; set it in
   the worker, not inherited.

**Parallelism ceiling:** bounded by `min(cpu_count, Supabase pooler conn budget / clients-per-
proc, RAM/proc)`. Start conservative (N=4) on the dedicated service; raise with measurement.
The per-strategy lock (`_get_strategy_lock`) becomes a no-op across processes — but the DB
queue already guarantees each sid appears once per job, and the update service **owns** the
fleet queue so cron can be paused/exclusive during a fleet run (avoids cron×fleet double-write).

---

## Fidelity — byte-identical by construction

The per-strategy unit (`recompute_and_persist_stored_trades` / `_algo_trades`) is **unchanged**.
Parallelism only changes *scheduling*, not *computation*. So the only ways parallel output can
differ from sequential are **environmental**, and each is pinned:

- registry-not-loaded → 0 trades  → `_pool_init` asserts `packs > 0` (fail loud, no silent 0).
- DB pooler contention/prepared-stmt clash → `prepare_threshold=None` per-proc client.
- cron racing the fleet write → update service holds an advisory "fleet-running" flag; cron
  full-recompute defers (it already set-diffs, but we avoid the kpi flap).

**Gate (must pass before enable):**
1. `fidelity_parity_suite.py` 18/18 (unchanged unit).
2. **Sequential-vs-parallel byte-identity**: run a fixed subset (e.g. 263/267/273/325 — an
   ungated 15Sec, a sub-min+secondary, a coarse 1d/4h) once with N=1 and once with N=4; assert
   identical trades (entry/exit ts + r) and KPIs per strategy. A diff = an env landmine, not a
   logic change — fix the init, don't trust the run.
3. Kill-switch verified: `RORT_RECOMPUTE_REMOTE=0` → in-process path bit-for-bit as today.

---

## Phasing (de-risk the pool before the infra)

**Phase 1 — the pool, no new service.** Add `_recompute_one` + `_pool_init` +
`ProcessPoolExecutor` to `_run_job_worker`, gated `RORT_RECOMPUTE_PARALLELISM` (default 1).
Validate gate #2 (seq==parallel byte-identical) **locally / on the existing worker off-RTH**.
This proves the embarrassingly-parallel claim and the per-worker init **before** any Railway
work. No live risk at default 1.

**Phase 2 — DB queue + dedicated service.** `recompute_jobs` table + migration; `submit_recompute_job`
gains the `RORT_RECOMPUTE_REMOTE` enqueue branch; `GET /api/jobs/{id}` reads the row;
`src/update_worker.py` poll-claim-run loop; `Dockerfile.update-worker` (copy of data-worker);
new Railway service. Frontend untouched (same poll contract).

**Phase 3 — scale + (maybe) Hi-Fi decouple.** Horizontal replicas (claim is SKIP-LOCKED-safe);
if Hi-Fi (~8 min/strat) proves to be on the critical path, split it to an incremental follow-up
pass instead of inline.

**Order rationale:** Phase 1 is cheap, reversible, and *fully validates the hard part* (parallel
correctness). Only after seq==parallel is proven do we spend on the DB queue + new service.

---

## Baseline to beat

- **215 min** sequential, 67 strategies, 2026-06-27 (`project_uad_baseline_2026-06-27`).
- Target Phase 1 (N=4, one box): ≈ **60–75 min** wall (CPU-bound portion ~4×; I/O overlaps).
- Target Phase 3 (dedicated 8-core + a replica): **<20 min** — routine, not an all-nighter.

## What we will NOT do

- Run the pool on the live API/Worker during RTH (starves trading; that's *why* it's dedicated).
- Use threads for the CPU buckets (GIL-bound — no speedup).
- Trust a parallel run whose workers didn't assert `packs > 0` (the silent-0 trap).
- Change the per-strategy compute (byte-identity is the whole safety argument).
