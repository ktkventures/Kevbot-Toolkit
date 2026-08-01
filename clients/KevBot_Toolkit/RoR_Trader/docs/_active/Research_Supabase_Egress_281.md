# Supabase egress — where the bill actually comes from

**Board:** goal #281 · children #290–#294, #298–#301 · **Date:** 2026-08-01 · **Author:** G281 (goal session), measured by E·auto (#292, #294) and F·auto (#293), method gated by M (#291)

**Status:** The **top 3 and their measured byte volumes are final** — `done_when`'s requirement is met. **Two things are NOT settled and neither may be skipped when reading the fix list:**

1. **§2a — daily egress already collapsed ~3× around 17–20 July, cause unidentified (#301).** That is a larger effect than any fix proposed here. **Rank everything against the current ~25–55 GB/day run-rate, not the 78.1 GB/day period average.**
2. **§2 — the denominator reconciles only to within the ±10% of an eyeballed chart.** Two free readings close it (#290).

---

## 1 · The answer in one table

Every number is **measured at the socket** over a **stated window**, with a **named caller**. Nothing here is a type-width calculation.

⚠️ **These are rates over the measurement window (Jul 2 – Aug 1), which spans the ~3× step-down in §2a.** The *ranking* is robust — source #1 is two orders of magnitude clear — but **the absolute figures describe a workload that is already about a third smaller than the window average.**

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

## 2a · ⭐ Egress already collapsed ~3× around 17–20 July, and nobody was looking

**This is the largest single effect in the whole investigation and it was not caused by anything in this document.** From Kevin's usage chart (transcribed by M — **±10% eyeball against labelled gridlines 46.6 / 93.1 / 171.4 GB; bounds, not measurements**):

| window | daily egress |
|---|---|
| early Jul (03–10) | **~100–175 GB/day** |
| mid Jul (13–17) | ~85–140 GB/day |
| **late Jul – 01 Aug** | **~25–55 GB/day** |

**Cause unidentified — #301 (E).** M's stated candidate, the 07-21 fleet cull, **appears wrong on timing: the collapse is 17–20 Jul and the cull is 07-21, after it.** `Deploy_Log.md` has a hole across exactly that window (07-16, then nothing until 07-21). The nearest preceding change is **07-16 M-RS5b — `RORT_CANONICAL_FINE_TF_STATE`, "fine RTH gate state rebuilt from the canonical resample"** — mechanically the right shape to cut `bar_cache` reads several-fold. `RORT_RESAMPLED_STORE_SERVE` (M-RS2 P2) is the same shape.

**Why this must be resolved before any fix is actioned:**
- **A fix ranked against the period average is sized against a workload that no longer exists.**
- **If the cause is an already-shipped arm, the #1 egress source was largely fixed once already, by accident, while chasing fidelity** — and what remains to recommend is much smaller.
- **If the cause is the fleet cull, it is a workload change that reverts the moment the fleet regrows** — a forward risk, not a past win. The two have opposite implications for whether #299 and #300 are worth doing at all.

---

## 2 · The denominator — reconciles, but only inside an eyeballed ±10%

| | value | window | status |
|---|---:|---|---|
| **D1_chart** — summed from the usage chart | **1,800–2,000 GB** | 06 Jul → 01 Aug (**26.85 d elapsed**) | ±10%, read off pixels |
| **D1_inferred** — from the invoice + public price | **≈ 2,172 GB** | full 30 d | $172.94 ÷ $0.09/GB + 250 GB allowance |
| **D2** — Σ of everything measured here | **≈ 2,379 GB** | 02 Jul → 01 Aug (**29.86 d**) | measured at the socket |

### The apparent contradiction, and why it is a span artifact

M flagged that **#292's 2,333 GB exceeds the entire billed total of 1,800–2,000 GB — one query cannot egress more than the project does.** That comparison sets a **29.86-day** measurement against a **26.85-day** elapsed bill. #292's window carries **3.01 extra days at the front — the highest days on the chart — and misses the final 4, which are the lowest.** It is guaranteed to overshoot.

**Aligned on a common span the two denominators agree:** 1,800–2,000 GB over 26.85 d ⇒ 67–74 GB/day ⇒ **2,010–2,235 GB over 30 days — a range that contains D1_inferred's 2,172 GB.** The invoice-derived number was sound; the *span* was mismatched.

### Where it is still not clean — stated, not smoothed

Removing #292's 3.01-day head depends on what those days ran at:

| assumption for 03–05 Jul | head | aligned Σ | vs D1_chart |
|---|---:|---:|---|
| **A** — ran at the early-Jul high (~137 GB/d) | 413 GB | **1,964 GB** | **fits** |
| **B** — ran low | 120 GB | **2,257 GB** | **overshoots 13–23%** |

**Case B is live: 2026-07-04 is a Saturday, so Friday 03 Jul is the observed market holiday and 03–05 Jul is a three-day market-closed stretch.** Against that, `bar_cache` reads are dominated by bursty recompute/backtest which run nightly regardless of the market. **Which one holds is not currently known and has not been guessed.**

If Case B holds, the residual needs **cached-vs-uncached** to explain it — Supabase bills uncached egress only, while #292 measured total wire bytes. For a workload that is 98% direct Postgres that would be surprising, since PG connections are not CDN-cacheable, **so Case B would itself be a finding rather than a patch.**

**Two free readings close this**, both visible on the screenshot already captured — filed as stamped steps on #290: **the bar heights for 03/04/05 July**, and **how large the green (cached) band is.**

**Window rail, carried everywhere:** `pg_stat_statements` was reset **2026-07-02 23:42:28 UTC**; the window to measurement is **29.86 days**. The billing period is **06 Jul – 05 Aug**. Every figure here is a **rate over 29.86 days**, not a billing total.

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

**⚠️ Read §2a first. If #301 finds the ~3× collapse was an already-shipped arm, #299 and #300 shrink to near-nothing and may not be worth doing at all.** They are filed so the findings are not lost, not because they are established as worthwhile.

| Task | Source | Bucket |
|---|---|---|
| **#298** — instrument `read_bars` to attribute the 1.4M calls by caller | #1 | Prerequisite — **you cannot fix what you cannot attribute** |
| **#299** — `canonical_resampled()` re-reads the base per target; share it | #1 | Pure waste, concrete instance |
| **#300** — board-tooling egress hygiene: gate projection, `Accept-Encoding`, R watcher finished-rows filter | #2, #3 | Pure waste + cheap win, zero/negative speed cost |

| **#301** — identify what cut daily egress ~3× around 17–20 July | §2a | **Forensics — outranks every fix above** |

**#290 (Kevin) remains open** — two readings off the chart already captured (§2), which turn the reconciliation from *fits inside an eyeballed ±10%* into *settled*.
