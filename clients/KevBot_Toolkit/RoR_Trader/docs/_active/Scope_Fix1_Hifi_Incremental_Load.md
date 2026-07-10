# Scope — Fix 1 (Hi-Fi half): incremental LOAD for `run_hifi_pass2` — 2026-06-30 (night prep)

**Status:** SCOPE ONLY. Per `Plan_M-RS4_Phase3_Scheduling_Fixes.md` §0, this documents the exact
change; **do not ship the shared-fn edit tonight** (it touches a fidelity/KPI load path — implement
in the morning market-hours session against the ready offline gate).

**Why this is fidelity-critical (not just throughput):** §1b — Hi-Fi Pass 2 is what pins a
`rest_hifi` backtest to LIVE down to the second (walks 1-sec bars to resolve entry + which exit level
hit first; can change the outcome on ambiguous bars). The resident engine emits at PRIMARY-TF bar
resolution; Hi-Fi is the per-second refinement. `run_hifi_pass2` today **reload-alls** — so it can't
run promptly every poll, so recent exits stay bar-aligned (the regression §0 reverted). Windowing the
LOAD lets Hi-Fi run cheaply every poll → per-second exits stay current in real time.

---

## 1. Today's path (the reload-all)

`run_hifi_pass2` — `src/api/routers/strategies.py:722`.

1. **LOAD (unbounded):** `strategies.py:792-796`
   ```python
   from db import load_trades_admin
   trades_list = load_trades_admin(strategy_id, str(user_id),
                                   data_source_filter=data_source_filter) or []
   ```
   `load_trades_admin` (`db.py:1114`) paginates **ALL** of the strategy's trades (1000/page,
   `.order('entry_fill_ts')`, `.like('data_source', data_source_filter)`). **No lower-bound param
   exists today.** This is the dominant cost (14k rows for sid 296, 5k for 301).

2. **INCREMENTAL FILTER (in Python, after the full load):** `strategies.py:811-857`
   - `scope_key = data_source_filter or 'all'`
   - `last_pass_dt = strat.config.last_hifi_pass_at[scope_key]` (an ISO watermark; `strategies.py:815-825`)
   - Keep a trade iff **`data->>'hifi_resolved' is True`** (idempotency — the walker skips these anyway,
     `strategies.py:833-835`) **OR `created_at >= last_pass_dt`** (`strategies.py:847-848`).
   - Missing `created_at` → kept defensively (`strategies.py:838-840`).

3. **WALK:** `_hifi_resolve_trades` (`backtest_service.py:584`) — skips `hifi_resolved is True`
   (`backtest_service.py:664-666`); refines the rest via 1-sec bars; sets `hifi_resolved=True` +
   `behavior='HIFI'` on refined rows; persists `exit_fill_ts`/`exit_price`/`exit_reason`/`r_multiple`
   + merges the flags into `data` JSONB (`strategies.py:1003-1037`).

4. **BUMP WATERMARK:** `_bump_last_hifi_pass_at(sid, uid, scope_key)` at pass end
   (`strategies.py:861, 1056-1061`) → `config.last_hifi_pass_at[scope_key] = now()`.

**The filter is incremental; the LOAD is O(all trades).** That's the wall.

Callers today (all production incremental paths already pass `incremental=True` + scoped filter):
`data_worker_engine.py:802`, `forward_test_service.py:1517/2103/2465/2656` (`incremental=True`);
`forward_test_service.py:422` + bulk wrapper `strategies.py:1182` (non-incremental / full).

---

## 2. Recommended change — push the watermark into SQL (byte-identical to today's incremental)

**Window the LOAD by `created_at`, NOT by `entry_fill_ts`.** The existing incremental filter already
keys on `created_at >= last_hifi_pass_at`; moving that predicate into the DB query is **byte-identical
to today's behavior** and eliminates the reload-all.

### 2a. `db.load_trades_admin` — add an optional lower bound
`db.py:1114`
```python
def load_trades_admin(strategy_id, user_id=None, data_source_filter=None,
                      created_at_gte: str | None = None):   # NEW, default None = today
    ...
    q = c.table('trades').select('*').eq('strategy_id', strategy_id)
    if data_source_filter:
        q = q.like('data_source', data_source_filter)
    if created_at_gte:                       # NEW
        q = q.gte('created_at', created_at_gte)
    q = q.order('entry_fill_ts')
    ...  # pagination unchanged
```
Default `None` ⇒ no predicate ⇒ existing full-load behavior. (Confirm `created_at` is
selectable/indexed; `.gte` on it is a single server-side filter.)

### 2b. `run_hifi_pass2` — compute the bound from the watermark when incremental
`strategies.py:792`
```python
created_at_gte = None
if incremental:
    last_pass_iso = (strat.get('config') or {}).get('last_hifi_pass_at', {}).get(scope_key)
    created_at_gte = last_pass_iso or None        # None on first-ever pass → full load (today)
trades_list = load_trades_admin(strategy_id, str(user_id),
                                data_source_filter=data_source_filter,
                                created_at_gte=created_at_gte) or []
```
- Keep the Python filter block (§1.2) as a **belt-and-suspenders no-op** — with the SQL bound it drops
  0 additional rows, but it still handles the missing-`created_at` defensive case and the empty-result
  watermark bump (`strategies.py:858-869`). (Optional cleanup later; leave for now to minimize risk.)
- Non-incremental callers pass nothing new ⇒ `created_at_gte=None` ⇒ unchanged.

### 2c. Equivalence argument (why this is safe / byte-identical)
- Rows with `created_at >= last_pass` → loaded by both old and new paths. ✔
- Rows with `created_at < last_pass` **and** `hifi_resolved=True` → old path keeps-then-**skips** them
  in the walker (`backtest_service.py:664`); they produce **no writes**. New path doesn't load them.
  Net effect on output = identical (only the `skipped_idempotent` counter differs). ✔
- Rows with `created_at < last_pass` and **not** refined → old path already **drops** them
  (`strategies.py:849`). New path doesn't load them. Identical. ✔

⇒ Output trades are unchanged; only the number of rows transferred/iterated shrinks to "created since
last pass."

---

## 3. Gotcha — do NOT window by `entry_fill_ts`

The naive "load only recent trades" (`entry_fill_ts >= since`) is **wrong**: a **backfill** can write a
trade with an OLD `entry_fill_ts` but a RECENT `created_at` (filling a historical gap — the shadow's
bootstrap, nightly `full_recompute`, or a UAD all do this). An `entry_fill_ts` window would **drop a
freshly-created historical trade that still needs its first Hi-Fi pass**, silently leaving it
bar-aligned forever. `created_at` is the correct axis because it tracks "seen since last pass," which is
exactly what the watermark means. Flag this in the PR.

---

## 4. Scope boundaries (what this does NOT cover)

- **KPI/equity reload-all is a SEPARATE mechanism.** `recompute_kpis_for_strategy` →
  `load_trades_kpi_fields_admin` (`db.py:1161`) reloads all trades to rebuild KPIs + equity curve. That
  needs the **in-memory `slot.kpi_series`** design (parent plan §2 / Fix 1 main), not a `created_at`
  window (equity needs the FULL history, not a window). Both halves must be windowed for per-second
  promptness, but by **different means**. This doc is the Hi-Fi half only.
  - *Optional:* add the same `created_at_gte` kwarg to `load_trades_kpi_fields_admin` for a future
    SQL-aggregate fallback (parent §2 "lighter alternative"); not required for the in-memory path.
- **No flag needed** — the change is byte-identical by construction (default `None`), so it can ship
  unguarded once the gate is green. (If desired, gate behind `RORT_HIFI_INCREMENTAL_LOAD` defaulting
  to today's behavior for a staged rollout — cheap insurance.)

---

## 5. Validation (run in the morning, gate-first)

1. **Offline byte-identity** (already built tonight — `_shadow_manager_validate.py` Hi-Fi gate,
   `VALIDATE_HIFI=1`): manager trades refine to per-second **intra-candle** stop/target exits AND stay
   byte-identical to from-cold post-Hi-Fi. Green requires 1-sec data (market hours).
2. **Load-equivalence check:** on a canary with a real watermark, run `run_hifi_pass2(incremental=True)`
   twice — once on `main` (reload-all) and once with the `created_at_gte` change — and diff the resulting
   `trades` rows (`trade_snapshot.py`). Must be byte-identical; only latency/rows-loaded differ.
3. **Latency:** log rows-loaded before/after on a fat strategy (sid 296 ~14k) — expect the load to drop
   from O(all) to O(created-since-last-pass).
4. Only then wire into the shadow poll cadence (parent §2b) + re-arm live.

## 6. Pointers
- `run_hifi_pass2` — `src/api/routers/strategies.py:722` (load 792, filter 811-857, persist 1003-1037,
  bump 1056-1061).
- `load_trades_admin` — `src/db.py:1114`; `load_trades_kpi_fields_admin` — `src/db.py:1161`.
- `_hifi_resolve_trades` — `src/api/services/backtest_service.py:584` (idempotency skip 664; exit-ts
  persist 915-927).
- Offline gate — `src/_shadow_manager_validate.py` (Hi-Fi section; `VALIDATE_HIFI`).
