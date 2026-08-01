# Supabase egress — where the bill actually comes from

**Board:** goal #281 · children #290–#294, #298–#301 · **Date:** 2026-08-01 · **Author:** G281 (goal session), measured by E·auto (#292, #294) and F·auto (#293), method gated by M (#291)

**Status:** ✅ **COMPLETE against `done_when`.** The top 3 are named, each with a measured byte volume and a filed fix task; everything else has a written leave-alone reason. **The denominator is now exact** (§2) — `read_bars` alone accounts for **88.8% of all Supabase egress**.

**One thing outranks every recommendation here and must be read first: §2a.** The workload fell 43.4% because **~50 strategies were deleted on 07-20 — a `DELETE`, not a fix.** It reverts when the fleet regrows. **The single most useful output of this research is therefore not a smaller bill; it is a unit price: ~2.1 GB/day of egress per forward-testing strategy, ~$5.80/month each, currently unmetered and uncapped.**

---

## 1 · The answer in one table

Every number is **measured at the socket** over a **stated window**, with a **named caller**. Nothing here is a type-width calculation.

> ⚠️ **DO NOT rank fixes against these absolutes, or against the 78.1 GB/day they imply.** The measurement window **straddles the 07-20 cull** — 17.78 days before it, 12.08 after — so it blends two regimes and **overstates today by ~2.2×** (#301).
>
> **Rank against the measured current rate: 49.58 GB/day** (dashboard, last 7 days — §2). #301's independent model of the measured lanes gives 36–45 GB/day which, plus the 9.5% residual, is 39–49 — **agreeing with the measurement.**
>
> The **ranking below is unaffected** — source #1 is two orders of magnitude clear — but every *saving* must be sized against 49.58, not 87.53 and not 78.1.

| # | Source | **Measured egress** | Window | Named caller | Verdict |
|---|---|---:|---|---|---|
| **1** | **`read_bars` — `bar_cache` history reads over direct Postgres** | **2,327 GB** (+5.6 GB handshake = **2,333 GB**) | 29.86 d | [`src/bar_cache.py:672`](../../src/bar_cache.py#L672), 8 call sites | **FIX — and it buys speed, not just money** (#298, #299) |
| **2** | **Dispatcher board-triage poll** | **26.8 GB** | 30 d | [`tools/team_dispatcher/dispatcher.py:449`](../../../../tools/team_dispatcher/dispatcher.py#L449) | **FIX — pure waste, zero speed cost** (#300) |
| **3** | **R watcher board poll** | **14.1 GB** | 30 d | `tools/team_dispatcher/r_watcher.py:22` | **FIX — pure waste, zero speed cost** (#300) |

**Source #1 is 98.1% of all measured egress.** Sources #2 and #3 together are 1.7%, worth ~$3.70/month. That ratio is the finding.

### Everything else measured, for completeness

| Source | Measured | Verdict |
|---|---:|---|
| M board watcher | 3.06 GB / 30 d | **Leave alone** — §6 |
| The abandoned March Worker (#284) — *the recorded lead* | **1.23 GB / mo**, ~5.5 GB over its whole 134-day life | **Already gone. Leave alone** — §5 |
| Failed-query error responses | 0.19–0.76 GB / mo (**$0.02–$0.07**) | **Leave alone** — §5 |

---

## 2a · ⭐ The workload already fell 43.4% — and the saving is REVERSIBLE

**This is the largest single effect in the investigation and nothing in this document caused it.**

| | GB | GB/day |
|---|---:|---:|
| Last 30 days | 2,625.939 | **87.53** |
| **Last 7 days** | 347.070 | **49.58** |
| | | **−43.4%** |

**Forward run-rate: ~1,487 GB/mo ⇒ $111.37/mo, against $175.27 this cycle. ~$63.90/mo is already banked**, by something that shipped before this research began.

### Cause: the fleet cull — pinned to **07-20 18:00–19:00 UTC** (#301, measured)

**Not a code fix. A workload change.** Two independent production time series step in the same UTC hour:

| | through 07-20 | 07-21 onward |
|---|---|---|
| Nightly `full_recompute` (`compute_jobs`) | **69–73 strategies**, 61–98 min | **22–23**, 22–39 min |
| Live lane (distinct `strategy_id` in `alerts`) | **54–62 sids**, 11.2k–15.8k alerts/day | **12–17 sids**, 1.6k–2.0k/day |

Hour resolution pins the deletion itself: on 07-20 the fleet holds 44 sids at 17:00Z, 37 at 18:00Z, then **15 at 19:00Z**. Control days hold 38–40 at 19:00Z, so this is not an end-of-day taper. Magnitudes corroborate: fleet 3.3× · recompute wall clock 4.5× · alert volume 5.8× · chart egress ~3×.

**Both flag candidates were checked and neither is the cause.** `RORT_CANONICAL_FINE_TF_STATE` (armed 07-16) rebuilds fine-TF state from the hub's **in-memory** 1Min history — it neither adds nor removes a `bar_cache` read, and the four nights after it are the heaviest in the series. `RORT_RESAMPLED_STORE_SERVE`/`_LIVE` (armed **07-13**) is real and structural — `_load_warmup_df` skips up to 355 d of 1Min per warmup — but it fits the mild mid-July softening, not the 3×.

> ### 🔴 This inverts the optimistic branch
> **The saving was not earned by a fix. It was earned by deleting ~50 strategies, and it reverts the moment the fleet regrows.** `Roadmap_Trading_At_Scale` is a plan to regrow it. **~$64/mo is on loan, not banked** — and the same cull is the recorded heal for the worker minute-boundary saturation ceiling, so egress and that latency ceiling will come back **together**.
>
> **This makes #298/#299/#300 more relevant, not less.** They were filed as hygiene against a 3× larger workload; they are now the only *durable* reductions on the table.

### ⚠️ A wrong call of G's, corrected

G argued in this document that **"M's candidate, the 07-21 fleet cull, appears wrong on timing — the collapse is 17–20 Jul, before it."** **That was wrong. M's candidate was right.** The error came from a stale date: the cull is recorded everywhere as *07-21*, which is when the **latency heal was observed**, one RTH session after the deletion. The `Deploy_Log.md` hole across 07-17→07-20 hid the rest. **The record was a day late, and the argument built on it inverted a correct answer.** Memory `project_worker_minute_boundary_saturation` has been corrected.

**One more thing #301 established that bears on §2's retracted Case A/B:** the 07-18/19 weekend still ran **69 sids / 100 min** nightly. **Weekend egress stays high, because recompute is market-independent** — so Case A (the 03–05 Jul head ran high) was the right read, and the market-closed reasoning behind Case B was unsound. Moot for the result, but the reasoning error is worth naming.

### Ruling (#301 step 2): **WORKLOAD — fully reversible. Nothing was fixed.**

**A strategy created today costs exactly what one cost on 07-19.** Three independent instruments say so:

1. **No arm sits in the window.** The `Deploy_Log.md` hole across 07-17→07-20 held three flag-OFF M-RS5a commits, one egress *increase* (`RORT_MTF_STATE_REFRESH_S` restored to 120 on 07-17), and two commits landing ~4 h *after* the drop.
2. **The deletion is not parked.** No `strategies_deleted` / archive table exists — the rows are gone, and equally, **nothing caps a rebuild.**
3. **Per-strategy cost is unchanged, and this is the load-bearing evidence.** Nightly `full_recompute` wall clock went **1.28 min/strategy pre-cull → 1.40 min/strategy post-cull.** Per-strategy work got *more* expensive per unit once fixed overhead stopped amortising across 69. **That is the signature of a workload change and is incompatible with a structural fix.**

*(Note on instrument 3's necessity: #301's egress split selects the fleet-independent share f ≈ 0 to reproduce the chart's ~3×, which then makes the pre/post per-strategy figures identically 1.55 GB/day **by construction**. That match is tautological and must not be cited as confirmation. The recompute wall clock is the independent check, and it is the one that carries the ruling.)*

**One genuinely durable slice:** `RORT_RESAMPLED_STORE_SERVE`/`_LIVE`, armed **07-13**, best explains the chart's mid-July softening (~137 → ~112, ≈18%). Call it **~15–20% structural and durable, ~80%+ workload and reversible** — and the 18% is eyeballed, so treat it as indicative.

> ### 💰 The unit price — the durable deliverable of this research
> **≈ 2.16 GB/day of total egress and ≈ 1.4 min of nightly recompute per forward-testing strategy ⇒ ~$5.82/month each.**
>
> ⚠️ **Use the ALL-IN figure. Two are in circulation and the other is 39% low:** #301's `read_bars`-lane-only figure is **1.55 GB/day ⇒ $4.18/mo**, which excludes the PostgREST lane and the 9.5% residual. Anything quoting ~$4.20 — including "69 strategies became a $271/mo line" — is the lane, not the bill. **All-in at fleet 69 is ~$379/mo.**
>
> ⚠️ **These are marginal-cost projections from ONE measured point, not a measured curve.** Linearity in fleet size held across the 07-20 cull (3.0× fleet ⇒ ~3× egress) but is untested above 73. **Label them as projections wherever displayed (#303).**
>
> **The always-on floor is fleet-scaled too.** The measured 12.4–12.7 GB/day idle floor comes from `refresh_mtf_states()` running per shadow every 120 s, 24/7 regardless of market window — so **even the necessary cost of divergence monitoring is per-strategy, not fixed.**

| fleet | egress/day | monthly | **egress cost** | nightly recompute |
|---:|---:|---:|---:|---:|
| **23 — today** | 49.6 GB | 1,487 GB | **$111** | 31 min |
| 69 — pre-cull | 149 GB | 4,466 GB | **$379** | 94 min |
| 200 | 431 GB | 12,945 GB | **$1,143** | 4 h 32 m |

**Fleet is flat at 23–24 for 11 nights, all forward-testing, 5 symbols — with no cap, no quota and no alarm. Adding a strategy is a UI click.** Regrowth restores the early-July regime within a day.

**Two walls, and the money one arrives first.** Nightly recompute does not threaten the 13:30 UTC open until ~600 strategies. **The bill is linear and unbounded from 23.** And #292's buffer-cache pressure (70.8% hit ratio, 1.93B real disk reads on a 42M-row table) is a **pre-cull** measurement — **that speed cost returns with the fleet, which is the axis Kevin said he cares about more than the bill.**

⚠️ **This is a growth tax on the roadmap.** `project_monetization` and the multi-user direction multiply the fleet by users, and the per-strategy read cost is currently unpriced.

---

## 2 · The denominator — exact, and the reconciliation passes

Supplied by Kevin from the dashboard (#290), **read from text, not from chart pixels**:

```
Billing period   06 Jul 2026 – 05 Aug 2026   (Pro Plan)
EGRESS           2,197.474 GB   $175.27
Pricing          250 GB included, then $0.09/GB
Last 30 days     2,625.939 GB
Last 7 days        347.070 GB
```

**Arithmetic verified:** `(2,197.474 − 250) × 0.09 = $175.27` — matches the dashboard exactly. **Allowance and unit price are confirmed, not inferred.**

### Reconciliation — like-for-like windows

| | GB | share of the 30-day total |
|---|---:|---:|
| **`read_bars` alone** (2,333 GB / 29.86 d) | 2,333 | **88.8%** |
| **All measured sources (D2)** | 2,377 | **90.5%** |
| **Unattributed residual** | **249** | **9.5%** |
| Method §6 tolerance | ±25% | **PASS** |

**One query accounts for ~89% of everything that leaves this database.**

> **⚠️ Use the "last 30 days" figure, not the billing-period figure, for this comparison — two readers have now tripped on it.** Against the billing period's 2,197.474 GB, `read_bars` computes to **106.2%**, and a part cannot exceed its whole. That is a **window error, not a data error**: the billing period starts 06 Jul (excluding the highest days, which the measurement window includes) and runs to 05 Aug (four days that had not happened at capture). **The like-for-like comparison is against `last 30 days` = 2,625.939 GB, and it gives 88.8%.**

**The 249 GB residual is named, not shrugged at.** Candidates, none yet measured: **the per-service split (Database vs API/Auth/Storage vs Realtime) was requested and has not yet been supplied** — it would localise this immediately · the ~232M rows (0.82%) across the other 2,439 `pg_stat_statements` entries, ~20 GB at a similar byte/row · browser-side board loads (469 KB per load) · the product frontend's server-side reads, which #293 did not cover — it scoped to agent tooling · Storage · Realtime (billing already $0.00) · Auth · WAL, backups/PITR.

### ⚠️ A retracted alarm, kept on the record because of what it shows

**M raised "the measurement exceeds the bill" — #292's 2,333 GB against a chart-summed 1,800–2,000 GB — and it was correct to raise it.** G's response analysed two cases for whether #292's 3-day window head ran high or low, and declined to pick one.

**Both were wrong, and neither in the way expected: the chart-reading was the faulty instrument.** The true total is **2,625.939 GB**, 31–46% above M's eyeball. There was never a contradiction to resolve.

**Two things are worth keeping from that:** the ±10% eyeball error on a transcribed chart was **stated as a bound and it still misled** — the lesson is to get numbers from text, which #290 then did. And **declining to guess between Case A and Case B was right**: guessing would have produced a confident wrong answer built on a faulty denominator, which is exactly this project's recurring failure. **The oversight rail caught M here, not an agent.**

### The PAT — declined, recorded as a decision rather than a gap

Supabase Personal Access Tokens are **account-wide and cannot be scoped read-only**; a read-only PAT does not exist. Since the database-side measurement localised ~89% of the cost without one, the Logs Explorer would only refine an API lane already measured at ~2%. **Recorded as a stated blind spot: "per-client API attribution unavailable — no PAT by decision."**

**Window rail, carried everywhere:** `pg_stat_statements` was reset **2026-07-02 23:42:28 UTC**; the window to measurement is **29.86 days**, aligning closely with the dashboard's "last 30 days". Every measured figure is a **rate over that window**, not a billing-period total.

### The pre-registered prediction — CONFIRMED

Recorded in the method on 2026-08-01 **before** any byte was measured, explicitly as a prediction to be falsified:

> 27.95B rows × an assumed ~80 B/row ≈ **~2.2 TB** … *If it lands there, one query explains the bill.*

**Measured: 88.68 B/row median → 2,327 GB.** The prediction was right, and right for the right reason. **One query explains the bill.**

---

## 3 · Source #1 — `read_bars`, in detail

**Measured at the socket** through a local counting TCP relay (psycopg's `libpq`/C implementation makes a Python socket shim blind, so the relay counts real wire bytes including TLS framing). Production call shape exactly: psycopg3 binary, transaction pooler `:6543`, `prepare_threshold=None`, `autocommit=True`, one connection per call. 32 calls — 5 baselines + 3 (symbol, timeframe) pairs × 3 window sizes × 3 repeats.

| | min | **median** | max |
|---|---:|---:|---:|
| `bytes_per_row` | 85.27 | **88.68** | 89.75 |
| `bytes_per_call_overhead` | 4,307 | **4,307** | 4,331 |

Run-to-run spread <0.01%. Applied to the 29.86-day counters (**1,400,553 calls / 28,178,313,609 rows**):

```
row payload         2,327 GB    99.76%
per-call overhead       5.6 GB   0.24%
TOTAL               2,333 GB  =  78.1 GB/day
```

**The 1.4M handshakes are a real line item and they are not the lever.** One connection per call ([`bar_cache.py:689`](../../src/bar_cache.py#L689)) costs **5.6 GB — 0.24%**. Pooling every connection saves under a third of a percent. Measured rather than assumed, and then set aside.

### Three things that matter more than the egress

- **99.18%** of every row the entire database returns comes from this one query — 28.18B of 28.41B rows across 2,440 distinct statements.
- **`bar_cache` holds only 42.06M rows** (1Sec 40.38M, 1Min 1.68M, 7 symbols). 28.18B rows read means **the whole table is re-read ~670 times in 30 days — about 22× per day.**
- **`total_exec_time` = 395,300 s = 4.58 days of backend time inside 29.86 days of wall clock — 15.3%.** Buffer traffic 6.61B blocks (~52.9 TB) at only a **70.8% hit ratio**; 1.93B are real disk reads. A 42M-row table should live in cache. These reads are evicting each other.

> ### ⚡ This inverts Kevin's cost calculus for source #1
> Kevin, 07-31: *"speed is the biggest cost to us rather than the egress cost."* **Source #1 is not a speed trade-off. It is a speed problem that also happens to be the egress bill.** 15.3% of backend time and a 70.8% buffer hit ratio are latency the fleet pays every day. Fixing it makes the database faster **and** cheaper. There is nothing to weigh.

### Why the 1.4M calls could not be split by caller — and what would

**`pg_stat_statements` cannot split them, structurally.** All eight call sites share one SQL string → one `queryid` (`-1070455249156103102`). A `/*caller=…*/` comment would **not** help: `queryid` is computed from the parse tree, not the query text.

Eight production call sites reach `read_bars`; the task's premise that it is reached via `cached_load_market_data` covers only one:

| Call site | Layer | Rows/call signature |
|---|---|---|
| `data_loader.py:797` / `:918` | 1Sec | prime-1s / HiFi load-once, day-chunked — ~23k–57k |
| `resampled_bar_store.py:785` | 1Sec | `maintain_submin_recent`, `recent_days`=2 — ~115k |
| `resampled_bar_store.py:213` / `:220` | 1Sec/1Min | base read per (symbol, tf, session) |
| `services.py:1329` | 1Min | warmup estimator, cap_start→now — ~29k at a 30d cap |
| `bar_cache.py:606` | 1Min | `cached_load_market_data` range select |
| `bar_cache.py:893` | 1Min | `materialize_derived` |
| `bar_cache.py:155` | coarse | small |

The 20,119 rows/call average is **consistent with at least three different callers** — one 1Sec RTH day = 23,400 rows; 1Min over a 30-day cap ≈ 28,800; 1Min over ~50 trading days ≈ 20,000 — which is precisely why arithmetic cannot attribute it.

**A live post-close sample bounds the shape:** 7.8 minutes, 498 calls / 695,326 rows = **1,396 rows/call at 63.8 calls/min**. Steady state is high-frequency and low-rows. **The 20,119 average is therefore driven by a minority of very large bursty reads — nightly recompute, backtest, seed — not by the steady loop.** That is where to look, and it is why #298 (attribution) precedes #299 (any fix).

### One concrete instance found, not hypothesised

`canonical_resampled()` re-reads the base layer whenever `one_min_df` is not passed (`resampled_bar_store.py:236/287/392`). Two loops call it **per (symbol, tf, session) with no shared base**:

- `shadow_compare_targets` (`:992`) — the §7 shadow gate, `days=5`
- `resampled_supply_coverage` (`:1037`) — **drives the Resampled Bar Store admin page**, `sample_days=5`

`default_coarse_targets()` ≈ 70–84 targets. At 5 days of 1Min base each (~4,800 rows), **one admin page load ≈ 80 `read_bars` calls ≈ 384k rows.**

**`maintain_submin_recent` (`:785`) already does it correctly** — one base read shared across 4 targets, with a comment explaining why. The two loops above do not. This is a confirmed instance of M's "per strategy, not per symbol" hypothesis.

---

## 4 · Sources #2 and #3 — the board tooling

Measured by F against prod, GETs only, post-close. **Frequency from named sources**, per method §4.

| rank | source | bytes/pass | passes/day | **/30 d** |
|---|---|---:|---:|---:|
| **2** | Dispatcher loop | 207,318 | 4,314 | **26.8 GB** |
| **3** | R watcher | 408,264 | 1,152 | **14.1 GB** |
| — | M board watcher | 70,911 | 1,440 | 3.06 GB |
| | **API lane total** | | | **~44.0 GB (~2.0%)** |

### The `select=*` hypothesis: falsified as stated, replaced by a sharper one

It is **not** column count — every non-text column of all 265 `dev_tasks` rows measures **17,924 B total**. It is **two unbounded TEXT columns**. Same 27 gate rows, only the projection changing:

| variant | bytes | vs shipped |
|---|---:|---:|
| `select=*`, identity — **as shipped** | **199,544** | — |
| `select=*` + gzip | 83,935 | −57.9% |
| minus `description` + `checklist` | 6,789 | −96.6% |
| gate-fields-only + gzip | **1,749** | **−99.1%** |

**And the dispatcher reads neither column in triage:** `description` only as a truthiness test (`dispatcher.py:744`), `checklist` only via `is_process_chain` (`dispatcher.py:782`) — its own docstring says *"Nothing reads the checklist for control flow any more"* (`:699-701`). ~193 KB per pass is fetched and discarded.

**Poll-vs-change:** 63 of 265 rows changed in 24 h, 17 inside the gate, against 4,314 polls/day → **≥99.6% of dispatcher polls return a byte-identical gate set.** Per M's explicit rail this is **not** an argument to slow the loop — it is the argument for the *same cadence on a smaller payload*: the identical gate decision can be made on 1,749 B instead of 199,544 B. **The polling buys latency; the 193 KB buys nothing.**

**Honest sizing: fixing all of #2 and #3 saves ~$3.70/month.** Real as engineering, immaterial as money. Filed as #299 on the strength of "zero speed cost", not on the strength of the saving.

---

## 5 · Two hypotheses killed — both were expected to be large, both are noise

### The ~91% Postgres error rate is a denominator artifact

**It is not a query error rate.** It is the `postgres_logs` line count, and Postgres writes a log line for errors but essentially never for successful statements (`log_statement='none'`). **A population that only admits errors will always read ~90%.**

| | value |
|---|---:|
| Reports "Postgres" requests | 6,400/day |
| **Actual SQL statements** | **3,070,415/day (35.5/s)** |
| **The Reports panel as a share of real traffic** | **0.208%** |

The honest transaction-level figure is `xact_rollback` = **6.31%** — and even that overstates errors, because transaction-mode poolers issue `ROLLBACK` on connection release. **`pg_stat_database.stats_reset` is NULL, so that 6.31% has no bounded window and is deliberately not published as a rate.**

**Do failed queries egress? Yes, and it is nothing.** Measured at the socket: 1,096 B warm / 4,348 B cold per error response, at 5,802 errors/day → **0.19–0.76 GB/month = $0.02–$0.07.** M's hypothesis — *"could be large AND free to fix"* — was right in kind and wrong in size. **Closed with a no.**

### The recorded lead — the abandoned March Worker (#284) — was wrong by ~1,900×

`goal_params.leads` named it *"a strong candidate cause"* and instructed: *measure it rather than assume it.* Measured:

| poll | calls / 29.85 d | bytes/call | GB |
|---|---:|---:|---:|
| `alert_config` — `db.py:1755` ← `worker.py:961`, `CONFIG_CHECK_INTERVAL=30` | 193,032 | 10,360 | 2.00 |
| `desired_state` — `db.py:1746` ← `worker.py:1814`, `POLL_INTERVAL=15` | 342,012 | 1,320 | 0.45 |
| **both workers combined** | | | **2.45 GB (0.11%)** |

**The March Worker's share, upper bound (one of two workers): 1.23 GB/month ≈ $0.11. Over its entire 134-day life: ~5.5 GB ≈ $0.50 total.**

A clean independent corroboration fell out of it: the measured rate for `load_all_desired_states` is **0.1326/s** against **0.1333/s** for exactly two independent 15-second pollers — a 99.5% match, confirming #284's "two workers" premise without touching Railway.

> **This is the result the goal's own instruction was written to get.** The lead was plausible, prominent, four months long, and it is **0.05% of the bill**. Had it been assumed rather than measured, the top of the fix list would have been a phantom — and one already eliminated by its deletion on 2026-08-01.

---

## 6 · Left alone, with reasons

| Source | Measured | Why it is left alone |
|---|---:|---|
| **The March Worker (#284)** | 1.23 GB/mo | **0.05% of the bill, and already deleted 2026-08-01.** Nothing to fix. |
| **Failed-query error responses** | 0.19–0.76 GB/mo | **$0.02–$0.07/month.** Pure waste, and pure waste of nothing. |
| **M board watcher** | 3.06 GB/30 d | 0.14%. Same fix shape as #299 but a seventh of the size; not worth a separate touch. Folded into #300 as optional. |
| **`read_bars` per-call handshakes** | 5.6 GB/30 d | 0.24%. Connection pooling would save <⅓ of 1%. **Measured, then deliberately set aside.** |
| **All poll cadences** — dispatcher 20 s, M 60 s, R 60–90 s, browser 15 s | — | **They buy latency, which Kevin has said is his more expensive resource.** Explicitly out of scope per M, 07-31. Not slowed. |
| **`done_ids`, `agents`, comment tails** | sub-GB | Below the noise floor. |

---

## 7 · Method (as approved in #291, with M's two mandatory amendments folded in)

The full method is in board #291. Its load-bearing rules:

1. **D1 vs D2.** The billed denominator and the measured sum are different quantities; a ranking is not publishable until reconciled. **Tolerance ±25%.** Achieved: +9.5%.
2. **The window rail.** `pg_stat_statements` is cumulative since reset, not since the billing period. **No byte number may be published without its window stated in days.**
3. **Bytes measured at the socket, never computed from column types** — `done_when` refuses estimates and a type-width calculation is one.
4. **Every source needs a named caller** at `file:line` or a named process.
5. **Every finding reports as a pair** — saving *and* speed cost — in four buckets: pure waste / cheap win / real trade-off / do not touch.

**AMENDMENT 1 (M) — `D1_billed` vs `D1_forward`.** The March Worker ran ~26 of the billing period's ~30 days and was deleted 2026-08-01, so no live measurement can capture it; its share had to be **bounded from surviving evidence**, not measured. Done in §5: **1.23 GB.** The fix list must rank against `D1_forward = D1_billed − 1.23 GB` — **a 0.05% correction. The amendment was correct in principle and turns out to be immaterial in magnitude, which is worth recording as much as if it had been large.**

**AMENDMENT 2 (M) — a source constant is only acceptable when checked against the running invocation.** §4(c) originally permitted citing a hard-coded interval by `file:line`. **F falsified that within the hour:** `dispatcher.py:267` declares `POLL_S = 900`; the running process is invoked `--poll 20`. Citing the constant would have been **45× wrong and would have passed review.** The rule is now: `ps`/argv or instrumented observation. **A constant a CLI flag can override is not a literal constant.**

**A refinement this work added.** Direct-Postgres connections egress **PG wire bytes**; PostgREST requests egress **HTTP response bytes**, and their internal PG hop is not billed egress. The two lanes must therefore be counted with different instruments and **must not be summed from `pg_stat_statements` alone** — doing so would double-count the API lane and miscount its size.

### Blind spots — named, not assumed

Browser-side egress while a board tab is open (bursty, `GET /api/dev-tasks` = 469 KB/load) · Realtime (`useLiveBar.ts:94` subscribes; billing already $0.00) · Auth token refresh · Storage · WAL/replication — **the #263 read replica was never created, confirmed absent rather than carried as unknown** · backups/PITR · transient agent-run queries · **the ~232M rows/0.82% across the other 2,439 `pg_stat_statements` entries, not individually costed** (~20 GB at a similar byte/row) · **the measurement itself egressed into the period it measures.**

---

## 8 · Filed, not actioned

Per #281's boundary — *"the deliverable is understanding, not action"* — and its oversight line, **Kevin sees this list before any of these is actioned.**

**⚠️ Read §2a first.** Per #281's boundary — *"the deliverable is understanding, not action"* — and its oversight line, **Kevin sees this list before any of these is actioned.**

**Re-ranked against the measured 49.58 GB/day (#301):**

| # | Lever | Size vs current run-rate | Speed cost | Verdict |
|---|---|---|---|---|
| **1** | **Fleet size itself — #303** | **±2.16 GB/day per strategy — the entire 3× lives here** | it *buys* product | **Not a code fix — a unit price to display before adding strategies. Visibility only; no cap was asked for and none is proposed.** |
| **2** | **#298** — instrument `read_bars` by caller | unlocks attribution of **100%** of the run-rate | none (in-process counter) | **Do first.** Nothing below can be sized or verified without it. |
| **3** | **#299** — `canonical_resampled()` per-target base re-read | **unsized** — known instance ≈34 MB/admin-page-load | none, pure sharing | Right shape, real waste, **gated on #298 for its number** |
| **4** | **#300** — board-tooling hygiene | **1.15 GB/day = 2.3–3.2%**, ~$3.10/mo | negative (gzip is faster) | Genuine pure waste. **Do it because it is free, not because it is big.** |
| **5** | **#294** — error-response egress | $0.02–0.07/mo | none | **Closed** — the ~91% was a `postgres_logs` artifact |
| — | **#301** — the forensics above | identified 71 GB/day | — | **Delivered** |

### The two sentences that matter

1. **The cull removed ~71 GB/day. The entire measured remainder of the fix list removes ~1.2 GB/day — 1.6% of it.** No proposal on this board is within an order of magnitude of the thing that already happened by accident.
2. **Because the cause is workload and not an arm, #298 and #299 are worth MORE, not less.** The optimistic reading — *"already fixed by accident"* — does not fire. **The win is not banked, the per-strategy cost is exactly where it was, and the roadmap multiplies it.** The right output of #281 is not a smaller bill today; it is **a known, reduced marginal cost per strategy before the fleet regrows.**

**#290 is CLOSED** — D1 delivered exact (§2). Still outstanding and non-blocking: the **per-service egress split**, which would localise the unattributed 249 GB.
