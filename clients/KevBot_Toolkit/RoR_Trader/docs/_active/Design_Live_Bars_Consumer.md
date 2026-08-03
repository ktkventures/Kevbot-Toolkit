# Design — the live `live_bars` consumer and its write isolation

**Board #165 Phase B, step 28.** Author: E·auto (run `r1785776207-165`), 2026-08-03.
**Status: DESIGN ONLY. Nothing was built. The migration is AUTHORED, NOT APPLIED.**

Reviewed against: `src/replay_harness.py`, `src/live_bars_writer.py`, `src/data_loader.py`,
`src/env_role.py`, `src/migrations/live_bars_table.sql`, `src/migrations/pilot_alerts_table.sql`
(branch `feat/pilot-alerts-migration-163`), `docs/_active/Plan_Measurement_Trust.md`,
`V51_Pilot_Session_Verdict_2026-07-31.md`, `V51_Pilot_Feed_Noise_Floor_2026-07-31.md`,
`docs/_active/Env_Write_Surface_Audit.md`.

Deliverable pair:
- this document
- `src/migrations/dev_alerts_table.sql` — **authored, not applied** (§7)

---

## 0 · What exists, and what this phase is actually for

**Proven already.** A second engine instance consumed **stored** `live_bars` through the real
`StrategyMonitor` / `SymbolHub` / shadow-engine stack via `src/replay_harness.py` — no
re-implementation — and matched live decisions **to the exact second on 66 of 72 decisions
(91.7%)** on the 1Min slice of 2026-07-31 RTH. It found something real on its first session: a
live-lane exit that was never recorded, independently corroborated by an entry/exit alternation
violation in the live record itself.

**Not proven, and the whole of this phase.** That pilot was an **offline batch replay run after
the close**. A consumer that polls for newly-written bars and drives the engine *as they land*
has never been built. Everything below is about that live consumer: how it learns a row landed,
what it writes, and what it does when the feed misbehaves.

**This design builds nothing.** Its output is a reviewable shape plus one authored `.sql`.

---

## 1 · THE SCOPE LIMIT — 1Min and coarser, stated before anything else

A DB-fed fleet is structurally behind a socket-fed one. Measured on 2026-07-31 RTH over the
1,317 live decisions whose deciding bar exists in `live_bars`:

> **median +3.2s, p90 +4.1s, and the deciding bar reached the database *after* live had
> already decided in 1,317 of 1,317 cases — 100.0%, zero overlap.**

It is a **fixed write-path cost, not a tail**. The distributions do not intersect, so no amount
of sampling makes it average out.

| timeframe | 3.2s as share of one bar | verdict |
|---|---|---|
| 1Min | **5%** | same bar — trustworthy |
| **10Sec / 15Sec** | **32%** | **different bar boundary — NOT trustworthy** |

**10Sec and 15Sec are explicitly OUT OF SCOPE for Phase B and remain so until Phase C.** At 32%
of a bar period the consumer is not "the same decision, timestamped later" — it is evaluating a
*different bar boundary* than live was, and bar-boundary state is exactly what entry/exit gating
keys on ([[feedback_pb_boundary_semantics_ruling]]). That converts the asymmetry from a caveat
you annotate into a confound you cannot separate.

Two further independent reasons the sub-minute slice is unreachable by this design, both
measured in `V51_Pilot_Feed_Noise_Floor_2026-07-31.md`:

- **`source='ws_agg'` carries no bar shorter than 60 seconds at all** (§3 of that doc). Sub-minute
  bars arrive under `source='ws'`.
- **The deciding bar is absent from `live_bars` entirely** for 3.6% of 10Sec and **13.9% of 15Sec**
  live decisions, versus 2.8% at 1Min. A consumer cannot reproduce a decision whose input never
  arrived, at any speed.

**Say the cost out loud: Kevin's live money is on 15Sec. This phase cannot validate those
strategies.** Phase C is what fixes that. Any reading of Phase B's verdict that generalises to
the money path is a misreading, and §6 states the rail that keeps it from happening quietly.

---

## 2 · Polling mechanism

### 2.1 The constraint that decides the design

`live_bars`' tracked DDL (`src/migrations/live_bars_table.sql`) gives:

```
PRIMARY KEY               (symbol, timeframe_seconds, bar_start)
live_bars_lookup_idx      (symbol, timeframe_seconds, bar_start)
live_bars_source_idx      (source, symbol, timeframe_seconds)
written_at TIMESTAMPTZ NOT NULL DEFAULT now()   -- NO INDEX
```

**There is no index on any arrival-time column.** So the obvious poll — "give me every row
written since my cursor" — is a **sequential scan of a ~42M-row table, once per poll**.

⚠️ **The tracked migrations do not describe this table's current production shape.** Live code
selects `first_open…first_volume` (`data_loader.py:1447`, `replay_harness.py:180`) and
`last_updated_at` (`api/routers/strategies.py:3493`), and `data_loader.py:1408` refers to a
`migrations/live_bars_first_values.sql` that **is not in the repo** — those columns were applied
out of band. Whether an index exists on `written_at` or `last_updated_at` therefore **cannot be
answered from the repo**, and this design does not assume one. See §8 item 2: a post-close
`\d live_bars` is a prerequisite for review, not for the build.

### 2.2 The poll — keyed on the index that provably exists

One round trip serves the whole dev fleet:

```
GET /rest/v1/live_bars
   ?select=symbol,timeframe_seconds,bar_start,open,high,low,close,volume,
           first_open,first_high,first_low,first_close,first_volume,source,written_at
   &timeframe_seconds=eq.60
   &symbol=in.(SPY,TSLA,NVDA,TSLL)
   &bar_start=gte.<cursor − GRACE>
   &source=in.(ws,ws_agg,rest_correction,rest_insert,warmup_seed)
   &order=bar_start.asc
```

- **`bar_start` is the cursor axis**, because it is the only one with an index. Arrival order is
  recovered by the `GRACE` overlap (§2.4), not by an arrival-time cursor.
- **`symbol=in.(…)`** resolves to a bitmap-OR of per-symbol index range scans, each bounded below
  by `bar_start`. Rows *scanned* ≈ `S × (GRACE/60 + 1)`, not table-sized.
- **The `source` filter is a list, not `eq.ws_agg`.** `apply_rest_correction` **upserts the row
  and overwrites `source`** (`data_loader.py:1399-1412`), so a bar that was written as `ws_agg`
  becomes `rest_correction`. Filtering to `ws_agg` alone would silently drop every corrected bar —
  the [[feedback_lane_filter_symmetry]] failure, on the input side. `rest_backfill` stays excluded:
  it is cosmetic cache patching the live engine never consumed.

### 2.3 Named cost — because a naive poll is a measurable bill

`read_bars` (the `bar_cache` history read, `src/bar_cache.py:672`) is **88.8% of all Supabase
egress** — a 42M-row table re-read ~22×/day. That is the shape of mistake this section exists to
avoid repeating on `live_bars`.

Model, at `S = 4` symbols, 1Min only, `GRACE = 180s`, poll interval `P = 2s`:

| quantity | value |
|---|---|
| rows returned per poll | `S × (180/60 + 1)` = **16** |
| bytes per row (15 cols, PostgREST JSON) | ~450 B |
| bytes per poll (+ HTTP/TLS) | **~8 kB** |
| polls per RTH day (23,400 s ÷ 2 s) | 11,700 |
| **egress per day** | **~94 MB** |
| as a share of the measured 49.58 GB/day project total | **~0.19%** |

Detection latency it buys: bar closes at `T`, lands at `T+3.44s` p50 / `T+4.56s` p90
(`ws_agg` @60s, calibrated against the #264 baseline of 3.41 / 4.72). A 2s poll adds ≤2s, ~1s
expected. **Dev decides at ≈ T+4.4s median, T+6.6s p90 — 7% / 11% of a 1Min bar. Same bar.**

The poll interval **adds to** the measured +3.2s/+4.1s floor; it does not hide inside it. Both
numbers belong in the verdict.

### 2.4 `GRACE` — why 180s

`GRACE` is how far *below* the cursor the poll reaches, and it is what lets a `bar_start`-keyed
cursor see a late arrival at all. It must cover the worst observed arrival lateness:

| arrival class | measured lateness after bar close |
|---|---|
| `ws_agg` @60s p99 | 27.8 s |
| `rest_insert` @10–15s | 34–65 s |
| `warmup_seed` @15s | 54–124 s |

180s covers all of them with headroom and costs 3 extra rows per symbol per poll. It is a
constant to tune with evidence, not a magic number: raise it and the poll gets linearly more
expensive; lower it and late bars are missed silently, which is the one failure this design
refuses to allow (§3.4).

### 2.5 What it does when it falls behind

"Behind" is defined, not felt: `lag = now − (last_consumed_bar_start + tf)`.

1. **It never skips ahead.** Bars are fed in `bar_start` order, always. Jumping to the newest bar
   would advance engine state past a bar it never saw — state corruption, not a latency artifact.
2. **Bounded catch-up.** It consumes as fast as it can. Every row it writes carries
   `consumer_lag_s`, and a decision produced while `lag > tf` is stamped `data.catchup = true`.
   The decision is real; its *timing* is meaningless, and the row says so rather than leaving a
   later analyst to work it out.
3. **A hard ceiling, recorded positively.** Above `RORT_DEV_CONSUMER_MAX_LAG_S` (default 300s)
   the consumer stops deciding, writes one `outcome='bar_skipped'` row **per skipped bar**, and
   re-seeds warmup at the current instant. A fleet ten minutes behind is deciding on a market
   that no longer exists; continuing would contaminate the comparison with rows that look like
   logic divergence. The gap is written down, never inferred from silence
   ([[feedback_no_silent_defaults]]).

### 2.6 Alternatives considered and rejected

| approach | why rejected |
|---|---|
| **`written_at > cursor` cursor** | No index (§2.1) ⇒ a **seq scan of ~42M rows, 11,700×/day, on the shared production database during RTH**. The cost is DB CPU/IO on the instance the live worker depends on, not bytes — the [[feedback_local_analysis_starves_live_worker]] / worker-minute-boundary-saturation hazard. Admissible **only** if a verified index on an arrival column is confirmed to exist (§8 item 2), and even then §2.2's key is cheaper because it is fleet-scoped. |
| **No `bar_start` lower bound** | Returns the whole 1Min history per symbol on every poll. This is precisely the `read_bars` failure mode — re-reading a large table on a loop — and it turns a 94 MB/day design into a GB/day one. |
| **Supabase Realtime / logical replication on `live_bars`** | Push instead of poll, no scan — but it means enabling replication on a hot, high-write table, fanning out **every** fleet write rather than the dev fleet's 4 symbols, and adopting a transport with new failure modes. It buys ~1s on a lane that is 3.4s behind by construction. Bad trade for Phase B. Revisit only if Phase C's transport work needs it anyway. |

---

## 3 · Failure modes

### 3.1 Gap — the bar never arrives

Measured at **2.8% of 1Min live decisions** (and 3.6% / 13.9% at 10Sec / 15Sec, which is one of
the reasons those are out of scope).

- The consumer **does not synthesize a bar**. It holds, exactly as live holds when the WS misses
  one; gap healing happens out of band in the live path and must not be re-implemented here.
- At `expected_close + GRACE` with no row, it writes `outcome='bar_absent'` — **one row per
  strategy per missing bar**, carrying no decision.

This closes the design gap the pilot verdict carried forward (§6.1 of
`V51_Pilot_Session_Verdict_2026-07-31.md`): `pilot_alerts` can record a decision *taken* but has
no way to record a decision that was *impossible to take*, so absence is byte-identical to
"evaluated, chose not to fire" and inflates apparent logic divergence by the blind-spot rate.
**`dev_alerts.outcome` exists for exactly this, and it is the one column that is not in the
`pilot_alerts` shape.**

### 3.2 Duplicate — the same bar seen twice

Happens routinely, by design and by accident: the `GRACE` overlap re-reads the same rows every
poll, and `apply_rest_correction` upserts an already-consumed row (0.1% of 60s bars; in every
sample **volume-only**, price unchanged).

- **In memory:** dedupe on `(symbol, timeframe_seconds, bar_start)` in a `seen` set.
- **In the DB:** a `UNIQUE` index on
  `(fleet_tag, strategy_id, symbol, timeframe_seconds, bar_start, outcome)`. A dedupe bug then
  fails loudly on INSERT instead of producing a second decision on the same bar.
- **A corrected bar is NEVER re-decided.** The decision-time value is `first_*`; the bar has
  already been consumed. Re-running the engine over it would **double-apply that bar to shadow
  state** — the exact corruption `replay_harness.py:408-414` guards against ("a refresh must
  never run BEFORE the incremental close of a bar its own canonical df already contains… cost
  hours on 2026-07-14"). The revision is recorded as `data.revision_seen` and nothing else.

### 3.3 Restart mid-session

- **The cursor lives in `dev_alerts`, not in memory.** On boot the consumer recovers, per
  `(symbol, timeframe_seconds)`, `max(bar_start)` for its own `fleet_tag`. It therefore cannot
  re-decide a bar it has already recorded, and it needs no separate cursor table.
- **Warmup re-seeds as of the boot instant** — history *ending at now*. (The no-look-ahead rule
  in `replay_harness.warmup_asof` exists because a replay's "now" is in the past; for a live
  consumer, now-anchored **is** as-of-safe.)
- **The hole is recorded, not inferred.** On boot it writes one `outcome='consumer_restart'` row
  per `(symbol, tf)` naming the recovered cursor and the wall-clock hole.
- **Stated limitation:** post-restart state is *not* the state live holds. Live has continuous
  state; a re-warmed consumer converges over the first minutes, which is the same session-open
  convergence already characterized in `Plan_Measurement_Trust.md` (328's 07-10 first entry 3.5
  min late). **Restarts contaminate the comparison and must be counted in the verdict**, which is
  why they get a row rather than a log line.

### 3.4 Out-of-order arrival

`rest_insert` and `warmup_seed` land **30–124s after close**, so a bar with an *earlier*
`bar_start` can arrive *after* a later one has already been consumed. The `GRACE` overlap finds
it — but by then the engine has moved past it.

**Rule: strictly forward-only. Never feed a bar whose `bar_start` is older than the
last-consumed `bar_start` for that `(symbol, tf)`.** Feeding it would re-order the series and
corrupt shadow state.

The late bar is written as `outcome='bar_late'` with its arrival delta, and any divergence it
causes is attributable rather than mysterious. **This is a recorded loss, not a recovered one,
and the design does not pretend otherwise** — live cannot rewind either, so a consumer that did
would be measuring something live can never do.

---

## 4 · Write surface

### 4.1 The decision, restated so it is not re-litigated

**The dev fleet writes to a separate table. It never writes `alerts`.**

`trades` has a `data_source` lane, so a new value there is invisible to existing readers.
**`alerts` has no lane axis** — `source` is uniformly `'ralph'` and names the producer, not a
lane — and **not one of its call sites is lane-scoped**. (Verified 2026-08-03: **33**
`table('alerts')` call sites in `src/`, excluding one-off scripts and tests; the brief's "~17"
is the read subset that would need editing.) A discriminator column there would put dev rows
into every existing query on day one, including `get_strategy_health` (`strategy_health.py:377`,
which would pair dev alerts against backtest edges and **inflate paired-%** — the headline metric
*and* the #264 pre-split baseline ruler this whole split is measured against) and
`pack_canary_service.py:243-250`, **which is the parity gate**.

The failure being engineered out is **silence, not error**. A dev row that throws is fine; a dev
row that quietly joins the money lane's statistics costs a week chasing a divergence that was
never real.

### 4.2 A new table, not `pilot_alerts` with a new tag

`pilot_alerts` is the right *pattern* and the wrong *home*:

- Its documented retirement is `DROP TABLE pilot_alerts;`, and it is described in its own header
  as a throwaway experiment. A **standing environment lane must not live inside a table someone
  is already authorized to drop.**
- Its `CHECK` constrains `fleet_tag` to `pilot_%`. The dev fleet would have to call itself a
  pilot to satisfy a constraint — a tag that lies to pass a rail whose entire purpose is that the
  tag cannot lie.

So: **`dev_alerts`**, mirroring `pilot_alerts` column-for-column, plus the two columns Phase B
needs (`outcome`, `consumer_lag_s`), with its own `CHECK (left(fleet_tag,4) = 'dev_')`.

> Naming note, in the migration too: `dev_alerts` is **the dev environment's alert lane**. It has
> nothing to do with `dev_tasks`, which is the team board.

### 4.3 Shape

Full DDL is in `src/migrations/dev_alerts_table.sql`. The load-bearing properties:

| property | value | why |
|---|---|---|
| `fleet_tag` | `NOT NULL`, `CHECK left(…,4)='dev_'`; Phase B value `dev_live_bars_1min` | the tag cannot claim a live or backtest lane |
| **`outcome`** | `NOT NULL`, one of `decision · bar_absent · bar_late · bar_skipped · consumer_restart` | **absence recorded positively** (§3.1) — the one addition to the pilot shape |
| `consumer_lag_s` | `bar_read_at − (bar_start + tf)` | the DB-feed cost, per decision, not a session average |
| `bar_written_at` / `bar_read_at` | timestamps | separates write-path lag from consumer lag |
| `flag_set` | `JSONB NOT NULL DEFAULT '{}'` | the `RORT_*` delta vs live. Phase B runs a **null delta** so any divergence is attributable to the feed path and live process state, not to flags |
| `code_fingerprint` | `sha256(engine srcs)[:12]` | "which code produced this row" answerable from the row (the 2026-07-07 lesson, `shadow_worker.py:52-70`) |
| foreign keys | **none** | an FK to `auth.users` installs triggers on an existing table; this migration touches nothing that exists |
| RLS | **enabled, zero policies** | fail-closed; service-role only. Every other client sees zero rows rather than dev decisions rendered as real |
| deliberately absent | `acknowledged`, `webhook_sent`, `webhook_deliveries`, `verification_*`, `would_fire_post_correction`, `orphaned_*` | PostgREST rejects an unknown column (PGRST204), so a consumer that ever tries to record a webhook delivery **fails loudly**. That is the rail firing — do not "fix" it by adding the column |
| unique index | `(fleet_tag, strategy_id, symbol, timeframe_seconds, bar_start, outcome)` | §3.2's dedupe rail, enforced in the DB |

It touches **no existing table**. Net effect if applied and never used: one empty table nobody
reads. Retirement is `DROP TABLE dev_alerts;` and nothing has to be unwound.

### 4.4 Isolation rails the code must carry

Three, and they are structural rather than conventional:

1. **The consumer refuses to boot in production.** It is a worker process, so #286's HTTP
   middleware (`install_env_role_guard`) never sees it. The inverse assertion is required at
   boot: **if `env_role.is_write_guarded()` is False, abort.** `RORT_ENV_ROLE` unset means
   production, so a consumer accidentally deployed there dies on startup instead of running.
2. **The module never reaches the live alert writer.** No import or call of `save_alert_db` /
   `save_alert_admin` / `table('alerts')`. Enforce it the way #286 enforces its route list — a
   source-parsing test (`test_env_role_write_guard_286.py`'s pattern), so a later edit that
   reintroduces it fails a test instead of shipping.
3. **It reads production strategy configs and writes only `dev_alerts`.** This is a deliberate
   departure from the environment split's usual isolation (dev logs in as its own account, #285).
   A fleet-vs-fleet comparison **requires the same `strategy_id`s** — two fleets that share no
   strategies cannot be diffed, which is the reason the split shares one Supabase at all. So the
   consumer takes the pilot's posture: admin context, read prod strategies by id, **write
   nowhere but its own table**. Say this in review rather than discovering it as a surprise.

### 4.5 Which strategies — explicit, never defaulted

The sid list is required configuration (`RORT_DEV_CONSUMER_SIDS`), and the consumer **fails to
boot if it is unset or contains a strategy whose primary timeframe is < 60s**. An empty default
would silently produce a zero-decision session that reads as agreement.

---

## 5 · Where it runs

A new Railway service in the **`dev`** environment, modeled on `shadow_worker.py`'s shape:
SIGTERM/SIGINT graceful stop, `/tmp` health file on a separate daemon thread, a `*_DISABLED`
kill-switch, and a boot-line code fingerprint. It consumes only `live_bars` and writes only
`dev_alerts`.

**Standing it up is not this step, and not headless-scope** (`railway up` / `variables` are a
hard line for a dispatched run). Steps 31–32 of the chain own the build and the live session.

---

## 6 · How the Phase B verdict must be read

Written here so it is agreed before there are numbers to argue about.

- **It answers "do the two fleets take the same trades?" at 1Min+.** Nothing else.
- **It says nothing about *when*.** Every timing difference between `dev_alerts` and `alerts` is
  write-path latency by construction (1,317/1,317, §1). A latency statistic computed from that
  comparison measures the DB write path and nothing else.
- **It says nothing about 95% of today's decisions**, including every 15Sec live-money strategy.
- **No logic conclusion may be drawn until #161 (environment parity) is asserted.** Feed
  mechanics are unaffected by #161; attribution of a divergence to *logic* is not.
- **Sequence structure beat row pairing last time.** The pilot's strongest finding came from
  entry/exit alternation across the whole session, which no nearest-timestamp pairing would have
  surfaced. Score alternation balance as a first-class check, not an afterthought.

---

## 7 · ⛔ Migration authorization — the file is authored, NOT applied

`src/migrations/dev_alerts_table.sql` exists and **has not been run against any database.**

```
author the file  →  hand to M  →  KEVIN AUTHORIZES BY NAME  →  M or Kevin applies  →  code may merge
```

A dispatch brief is not authorization. A relayed "OK" is not authorization.

**Ordering, and it is not negotiable:** code that reads or writes `dev_alerts` **must not merge
before the table exists in prod**, or it 500s. The build step (chain step 31) is blocked on the
apply, not on the review.

---

## 8 · Open items for review — what M and Kevin need to decide or check

1. **Kevin authorizes `dev_alerts_table.sql` by name** (§7). Everything downstream waits on it.
2. **Verify `live_bars`' real shape post-close** — `\d live_bars`. The repo's DDL is incomplete
   (§2.1): `first_*` and `last_updated_at` exist in production with no migration file, and
   whether an index exists on `written_at` / `last_updated_at` is unanswerable from the repo. If
   one does exist, §2.6's rejected arrival-cursor becomes worth re-costing. **This is a read, not
   a change** — but it is a prod read, so post-close.
3. **Name the dev fleet's sids** (§4.5). 1Min+ only. The pilot used 136 / 194 / 269 / 340 / 345;
   269 alone contributed 51 of 71 decisions, so a repeat with the same set gives a verdict
   dominated by one ungated canary. Worth widening.
4. **Confirm `GRACE = 180s` and `P = 2s`** (§2.3–2.4), or say what evidence would move them.
5. **Confirm the null-delta A/B** (§4.3, `flag_set`): Phase B runs dev on the identical armed flag
   stack, so divergence is attributable to feed path and live process state. A flag-delta A/B is a
   later experiment and is only interpretable once this baseline exists.
