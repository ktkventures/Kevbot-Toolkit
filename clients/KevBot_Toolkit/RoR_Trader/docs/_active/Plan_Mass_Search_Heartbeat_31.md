# Plan — Mass search DB heartbeat during cold-cache first load (board #31)

**Author:** F·auto · **Date:** 2026-07-25 · **Re-verified against HEAD:** 2026-07-28 (F·auto)
**Status:** LANDED 2026-07-29 by E·auto on `fix/mass-search-heartbeat-31` — see §6.
Flag `RORT_MASS_SEARCH_HEARTBEAT` (default OFF). NOT pushed, NOT armed.
**Lane note:** the entire fix is in `src/mass_builder.py` (background worker). That is
in F's hard-boundary "must NOT touch" list (worker lanes). This doc is the ready patch;
**E should land it** (sibling #32 mass-search worker fix was E·auto), or Kevin can
explicitly waive F's boundary for this observability-only change.

> **2026-07-28 re-verify (F·auto, 2nd dispatch):** patch re-checked against current HEAD.
> Structure UNCHANGED and patch still applies cleanly; only line numbers drifted (code
> grew ~50 lines above the anchor). All primitives the patch relies on are present:
> `import threading` (`:1743`), `import time as _time` (`:15`/`:1744`), `_search_lock =
> threading.Lock()` (`:1749`), resume-path precedent `update_mass_search(id,{'status':
> 'running'})` (`routers/mass_builder.py:161-162`), `update_mass_search` (`db.py:2197`).
> **Current line map** (old → new): seed `last_db_flush` :1954→**:2004** · `_progress`
> def :1956→**:2006** · inline 60s flush block :1996-2019→**:2046-2069** · `raw =
> run_mass_search(...)` :2072→**:2122** · `finally`/`_cleanup` :2159→**:2209-2215**.
> Still landing-blocked on the routing decision in §5 — F did not cross the boundary.

---

## 1. Symptom (from the pressure test)

A mass-search row sat `status='queued'` for ~42 min while the worker thread was ALIVE
and healthy, loading 365-day 30Sec data (cold cache). The dashboard `/progress` endpoint
correctly showed "preparing indicators" (it reads the in-memory `_active_searches` map),
but the **DB row** and **any external monitor** that reads the row looked stuck/queued.

## 2. Root cause (confirmed, line-precise)

Two independent facts combine:

**(a) The fresh-start worker never writes `'running'` to the DB up front.**
- Router creates the row with `status: "queued"` and returns `{"status":"running"}` to
  the client — but that's a lie w.r.t. the DB row.
  `src/api/routers/mass_builder.py:28` (`"status": "queued"`) → `:33` `save_mass_search`
  → `:38` `start_mass_search_async`.
- `start_mass_search_async` sets `_active_searches[search_id]['status'] = 'running'`
  **in memory only** (`src/mass_builder.py:1932`). It does NOT write `'running'` to the DB.
- The **resume** path already does the right thing —
  `src/api/routers/mass_builder.py:160` `update_mass_search(search_id, {'status':'running'})`
  with the comment *"Flip status first so any concurrent polls see 'running'."* The
  fresh-start path is missing exactly that flip. Strong internal precedent for fix part 1.

**(b) The only DB `'running'` writes are gated inside `_progress` on a 60s wall-clock,
and `_progress` is never called during the blocking cold-cache load.**
- `_progress` (`src/mass_builder.py:1956`) flushes the DB only when
  `now - last_db_flush > 60` (`:2005`). `last_db_flush` is seeded at worker start (`:1954`).
- In the load loop, `progress_callback` fires ONCE with "Loading {symbol} {tf} data..."
  at `src/mass_builder.py:1011-1016` (t≈0, i.e. <60s after worker start → the 60s gate
  does NOT flush), then calls the long blocking `load_strategy_data`
  (`src/mass_builder.py:1055`). For 365-day 30Sec that blocks ~42 min and fires **zero**
  progress callbacks. The next callback is "data ready" at `:1113` — AFTER the load.
- Net: no callback fires for the whole load → the 60s DB flush never runs → the row is
  never flipped off `'queued'` until prep begins (~42 min later). `_checkpoint` (`:2025`)
  and `_partial` (`:2059`) also only fire after backtests start, so they don't help either.

The 60s flush is a real, deliberate throttle ("Each flush rewrites the full config_data
JSONB blob … expensive at 70+ strategy scale", `:1996-2002`) — the fix must NOT increase
its write budget.

## 3. Fix (two parts — minimal, observability-only, no engine/fidelity impact)

**Part 1 — Immediate status flip at worker start.** A `{'status':'running'}`-only update
is cheap: `update_mass_search` writes only the `status` column + `updated_at` when no
JSONB key (`results`/`progress`/`summary`/`checkpoint`) is present — no SELECT+merge of
`config_data` (`src/db.py:2174-2203`). This instantly cures the "stuck at queued"
symptom for external monitors. Mirrors the resume path.

**Part 2 — Wall-clock heartbeat independent of callbacks.** A daemon thread flushes the
current in-memory snapshot every ~60s even when `progress_callback` isn't firing,
coordinated with `_progress` via the shared `last_db_flush` timer so the two paths never
double-write inside the 60s window → **total write budget unchanged** vs today; the flush
is simply guaranteed to happen even during a blocking load.

### 3.1 Ready-to-apply patch (`src/mass_builder.py`, inside `start_mass_search_async._worker`)

**(i) Replace the seed + `_progress` inline flush.** Where `last_db_flush` is seeded
(`:1954`), add the shared heartbeat primitives and a single shared flush helper:

```python
            last_db_flush = _time.monotonic()
            _hb_stop = threading.Event()  # signals the heartbeat thread to exit

            def _flush_running_snapshot():
                """Flush the current in-memory progress to the DB as
                status='running'. Shared by the _progress callback and the
                wall-clock heartbeat so the row reflects live phase even during
                long blocking loads (a cold-cache data load fires NO progress
                callbacks — 365d 30Sec ~42 min). Coordinated via last_db_flush
                so the two paths never double-write inside the 60s window; the
                write budget is unchanged vs the old callback-only flush."""
                nonlocal last_db_flush
                last_db_flush = _time.monotonic()  # set first: races only cost 1 extra flush
                with _search_lock:
                    info = dict(_active_searches.get(search_id, {}))
                if info.get('status') != 'running':
                    return
                try:
                    update_mass_search(search_id, {
                        'status': 'running',
                        'progress': {
                            'current_step': info.get('current_step', 0),
                            'total_steps': info.get('total_steps', 0),
                            'current_label': info.get('current_label', ''),
                            'phase': info.get('phase'),
                            'phase_detail': info.get('phase_detail'),
                        },
                    })
                except Exception:
                    pass
```

Then in `_progress`, replace the inline 60s flush block (`:1996-2019`) with a call to the
shared helper:

```python
                # Flush to DB every 60 seconds (was 10s before 2026-04-22).
                # See _flush_running_snapshot for the write-budget rationale.
                nonlocal last_db_flush
                if _time.monotonic() - last_db_flush > 60:
                    _flush_running_snapshot()
```

**(ii) Immediate flip + start the heartbeat**, just before `raw = run_mass_search(...)`
(`:2072`):

```python
            # Immediately flip the DB row to 'running' (status-column-only write —
            # cheap, no config_data JSONB read/modify). The fresh-start path used to
            # leave the row at 'queued' until the first >60s progress callback, but a
            # cold-cache data load fires NO callbacks for its whole duration (365d
            # 30Sec ~42 min), so the DB row + external monitors looked stuck. Mirrors
            # the resume path (routers/mass_builder.py) which already flips up front.
            try:
                update_mass_search(search_id, {'status': 'running'})
            except Exception:
                pass

            # Wall-clock heartbeat: flush the in-memory snapshot every ~60s even when
            # progress_callback isn't firing (blocking data load). Independent of
            # callbacks; coordinated with _progress via last_db_flush so the two never
            # double-write in the same window. Polls at 15s so it stops promptly.
            def _heartbeat():
                while not _hb_stop.wait(15):
                    with _search_lock:
                        running = (_active_searches.get(search_id, {})
                                   .get('status') == 'running')
                    if not running:
                        break
                    if _time.monotonic() - last_db_flush > 60:
                        _flush_running_snapshot()
            threading.Thread(target=_heartbeat, daemon=True,
                             name=f"mass_hb_{search_id}").start()
```

**(iii) Stop the heartbeat in `finally`** (`:2159`):

```python
        finally:
            _hb_stop.set()  # stop the wall-clock heartbeat thread
            # Clean up after a delay so the UI can read final status
            def _cleanup():
                _time.sleep(60)
                ...
```

### 3.2 Notes / options for the landing engineer (E)
- **Write budget:** unchanged. Old code flushed `{status,progress}` at most once/60s via
  callbacks; new code flushes the same at most once/60s via callback-or-heartbeat
  (`last_db_flush` gates both). The only NET-new write is the single cheap status-only
  flip at start (Part 1).
- **Cheaper variant of Part 2 (optional):** during a pure blocking load the in-memory
  label doesn't change, so the heartbeat could do a status-only `{'status':'running'}`
  write (no JSONB read/modify) instead of the full snapshot, refreshing `updated_at` for
  external liveness monitors at near-zero cost. Snapshot flush is fine too — it just
  matches the existing budget. E's call.
- **No flag needed:** observability-only; no trading/fidelity/paired-% surface. (E may
  still prefer a flag per lane convention — not required by the change itself.)
- **Race on `last_db_flush`:** benign — worst case one redundant flush; every flush is
  already best-effort try/except. Setting the timestamp first in the helper minimizes it.

## 4. Test plan
- **Unit:** patch `update_mass_search` to record calls; drive `_worker` with a
  `run_mass_search` stub whose data-load phase sleeps >60s and fires no callbacks.
  Assert: (a) a `{'status':'running'}` write lands within the first heartbeat tick
  (≤~15s), and (b) at least one snapshot flush lands during the ≥60s no-callback stretch.
  Assert the heartbeat thread exits after `finally` (`_hb_stop.set()`).
- **Manual/prod-mirror:** launch a cold-cache 365-day 30Sec search; poll the DB row
  directly (not `/progress`) — it should read `'running'` within seconds and keep a fresh
  `updated_at` throughout the load, instead of sitting `'queued'` for ~42 min.
- **Parity:** N/A (no engine/fidelity path touched); `fidelity_parity_suite.py` unaffected.

## 5. Why F did not land this
`src/mass_builder.py` is a background worker (worker lane) — F's charter hard-boundary
"must NOT touch". This lane carries subtle DB-write-at-scale, JWT-expiry, and
flush-throttle constraints that E owns (see the dense comments at `:1919-1926`,
`:1996-2002`), and the sibling mass-search worker fix (#32, recover-queued) was executed
by E·auto. Handing E a ready patch (this doc) rather than crossing the lane.

## 6. Landing record (E·auto, 2026-07-29)

Landed on `fix/mass-search-heartbeat-31`, cut from `origin/dev` @`3397e070`. F's plan was
implemented as written, with three deviations, all tightening:

**(a) Gated behind `RORT_MASS_SEARCH_HEARTBEAT` (default OFF).** F noted no flag was
strictly required. Taken anyway, to match the sibling `RORT_MASS_RECOVER_QUEUED` (#32) in
the same file and keep the arm reversible by var-set. OFF is byte-identical legacy: no
up-front flip, no heartbeat thread.

**(b) `_progress` left completely untouched.** F's §3.1(i) refactored the inline 60s
flush to call a shared helper. Not done — the heartbeat instead shares `last_db_flush`
by `nonlocal` and does its own flush. Same coordination, same write budget, and OFF is
provably legacy because the callback path has zero diff.

**(c) NEW — write-ordering lock (not in F's plan).** F's heartbeat checked in-memory
status before flushing, which closes the common case but leaves a real race: a heartbeat
that passed the check and is inside its DB round-trip when the search finishes can land
its `'running'` write AFTER the terminal `'completed'` write. That would leave a finished
search reading `'running'` forever — and with #32 armed, the orphan sweeper would then
reap or auto-resume it, re-running a completed 42-minute search. Closed with a per-worker
`_hb_write_lock`: every terminal path (`completed`/`cancelled`/`failed`) does
`_hb_stop.set()` and takes the lock around its write, so an in-flight heartbeat either
sees the stop flag and skips, or completes first and is overwritten by the terminal
write. The primitives are declared **before** the `try:` so the `except` handlers can
always reference them even if the try body dies on its first statement.

`_HB_POLL_SECS = 15` / `_HB_FLUSH_SECS = 60` are module-level so tests can shrink them.
F's "cheaper status-only heartbeat" option (§3.2) was NOT taken — the snapshot flush
re-publishes the last known `phase`/`phase_detail`, so the row says *what* the worker is
stuck on ("Loading SPY 30Sec data...") rather than only that it is alive, and it costs
nothing extra because it replaces the flush `_progress` would have done.

### 6.1 Verification
- `src/test_mass_search_heartbeat_31.py` — **13/13 pass**. Covers OFF parity (no write at
  all during a callback-less load — the #31 symptom, asserted on purpose), the up-front
  flip landing *while* the load is still blocking, snapshot flush with no callbacks,
  phase carried from memory, the throttle window still bounding writes, thread exit on
  completed/cancelled/failed, and the resurrection race.
- The race test is **load-bearing**: with `_hb_write_lock` removed from the completed
  path it fails (`12/13`), and passes again when restored.
- Regression: `test_mass_recover_queued_32.py` 9/9, `test_mass_required_confluences.py`
  all pass, `test_mass_builder_fidelity_parity.py` PASS (6 cases).
- Not run: prod-mirror cold-cache launch (§4 manual step) — needs an attended session.

### 6.2 Arming
Single var-set on the **API** service (mass searches run in the API process, not the
workers): `RORT_MASS_SEARCH_HEARTBEAT=1`. Restart cost applies — arm post-close
(a var-set restart kills any in-flight mass search). Verify by launching a cold-cache
365-day 30Sec search and polling the `mass_searches` row directly: it should read
`'running'` within seconds with a refreshing `updated_at`, instead of `'queued'` for
~42 min.
