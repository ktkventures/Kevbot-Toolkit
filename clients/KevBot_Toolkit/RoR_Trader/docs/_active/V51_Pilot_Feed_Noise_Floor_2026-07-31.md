# V5.1 pilot — measured feed noise floor, and what it means for the verdict

**Board:** #163 (V5.1), step 6. **Date:** 2026-07-31, post-close. **Author:** E·auto (`r1785532942-163`).
**Status:** the pilot fleet was **NOT run** — see §6. This is the null-hypothesis noise floor
measured from data already in the DB, which is the evidence the verdict rests on.
**Read-only.** No writes, no flags, no deploy. `pilot_alerts` still has 0 rows.

---

## 0. The one-paragraph verdict

**Fleet-vs-fleet divergence on a `live_bars`-fed pilot is interpretable for 1Min+ strategies
and NOT interpretable for sub-minute ones — which is 95% of what this fleet actually trades.**
Three measured reasons, all from the 2026-07-31 RTH session: the stream named in the task
(`source=ws_agg`) carries **none** of the sub-minute bars and serves only **4.8%** of live
decisions; **6.9%** of live deciding bars (13.9% at 15Sec) are **absent from `live_bars`
entirely**, a divergence floor no amount of pilot correctness can reduce; and the feed lag,
while a near-constant ~3.2s, is **32% of a 10Sec bar period** versus 6% of a 1Min one. The
consequence for #164/#165: **V5.2 (the gateway) is not an optimization of V5.1, it is the
precondition** for the comparison to mean anything below 1Min.

---

## 1. THE ASYMMETRY — stated up front, not as a footnote

Per the task's own instruction, this is in the result and not in a footnote.

A pilot reading bars from the DB runs **strictly behind** a fleet reading the socket. Measured
on today's session, over the 1,317 live decisions whose deciding bar exists in `live_bars`:

> **The deciding bar landed in the DB AFTER live had already decided in 1,317 of 1,317 cases —
> 100.0%, zero exceptions.** Median lateness **+3.2s**, p90 **+4.1s**, p99 **+23–27s**.

This is not "the pilot is on average a bit behind, so noise cancels." The two distributions do
not overlap at all. **Every timing difference a pilot-vs-live comparison produces is write-path
latency. None of it is logic.** Any future reader who computes a timing statistic from
`pilot_alerts` against `alerts` is measuring the DB write path and nothing else.

The ~3.5s figure in the task description **replicates and is confirmed** — but only on the
stream it was measured on (see §3).

---

## 2. Method

Prod Supabase, service key, post-close (21:2x–21:4xZ; RTH closed 20:00Z). Window
**2026-07-31 13:30–20:00Z**, the full session.

- `alerts` = the live fleet's decision record. (Per step 1/2: the `trades` dual-write was
  removed 2026-05-06, so `alerts` *is* the live lane. Verified in thread by M.)
- `live_bars` = candidate pilot feed, all sources.
- Join on `(symbol, timeframe, bar_time)`; lateness = `live_bars.written_at − alerts.trigger_ts`;
  feed lag = `written_at − (bar_start + timeframe)`.

**Calibration check:** `ws_agg` @60s measured **p50 3.44s / p90 4.56s**, against #264's
pre-split baseline of **3.41 / 4.72**. The instrument agrees with the established ruler, so the
numbers below are comparable to it.

Session shape: **1,414 live alerts**, 13 strategies, 4 symbols — **891 × 10Sec (63%), 452 ×
15Sec (32%), 71 × 1Min (5%)**.

---

## 3. Finding 1 — the task specifies the wrong stream

The task scopes the pilot to `live_bars (source=ws_agg)`. Measured over the session:

| `source` | timeframes carried | live decisions served |
|---|---|---|
| `ws_agg` | **60s and coarser only** | 68 / 1414 = **4.8%** |
| `ws` | 10s, 15s, 30s | 1,195 / 1414 = **84.5%** |
| `rest_correction` | 10s, 15s | 53 / 1414 = 3.7% |

**`ws_agg` contains no bar shorter than 60 seconds.** A pilot built exactly as the task
describes would be structurally blind to **95%** of the fleet's decisions — including every
one of the 15Sec first-live-money strategies.

**This is correctable** — the pilot must read `source IN ('ws','ws_agg')` — but it is not a
detail. It changes which write path is under test, and the ~3.5s asymmetry quoted in the task
was measured on `ws_agg`, i.e. **on the stream backing 4.8% of decisions, not the 95%.**

---

## 4. Finding 2 — a ~7% blind spot that no correct pilot can close

For each live decision, was its deciding bar in `live_bars` at all?

| timeframe | alerts | **bar MISSING** | present via `ws`/`ws_agg` | via `rest_correction` |
|---|---|---|---|---|
| 10Sec | 891 | **32 (3.6%)** | 819 (91.9%) | 40 (4.5%) |
| 15Sec | 452 | **63 (13.9%)** | 376 (83.2%) | 13 (2.9%) |
| 1Min | 71 | **2 (2.8%)** | 68 (95.8%) | 1 backfill |
| **all** | **1414** | **97 (6.9%)** | 1263 (89.3%) | 54 (3.8%) |

A DB-fed pilot **cannot reproduce those 97 decisions at any speed** — the input never arrived.
They are an irreducible divergence floor, ~7% overall and **~14% at 15Sec**.

### 4.1 The design gap this exposes — worth acting on before build

`pilot_alerts` records **a decision taken**. It has no way to record **a decision that was
impossible to take.** A bar that never arrives produces *no pilot row at all* — which is
byte-identical, to any later analyst, to *"the pilot evaluated this bar and chose not to fire."*

So the 7% floor would present as **phantom logic divergence**, and the `bar_written_at` /
`bar_read_at` columns (which are the right instinct and do handle the *lag* case) do not cover
it, because there is no row to carry them.

**Recommended before any build:** the pilot emits a row for every bar-close it *evaluates*, with
an explicit outcome including `bar_absent`. Absence must be recorded positively, or the headline
number of this whole exercise is inflated by ~7% in a way that reads as logic.
This is the same principle as [[feedback_no_silent_defaults]] — fail loud, don't infer from silence.

---

## 5. Finding 3 — the constant lag, and why its significance is not constant

Feed lag = `written_at − bar_close`, RTH, by stream and timeframe:

| source | tf | n | p50 | p90 | p99 | **p50 as % of bar period** |
|---|---|---|---|---|---|---|
| `ws` | 10s | 4585 | 3.23 | 3.69 | 8.45 | **32%** |
| `ws` | 15s | 1487 | 3.25 | 4.13 | 13.06 | **22%** |
| `ws` | 30s | 770 | 3.44 | 4.36 | 18.17 | 11% |
| `ws_agg` | 60s | 1532 | 3.44 | 4.56 | 27.76 | **6%** |
| `ws_agg` | 300s | 152 | 1.26 | 2.18 | 8.25 | 0.4% |
| `rest_insert` | 10s | 29 | **34.17** | 64.98 | 83.80 | 342% |
| `warmup_seed` | 15s | 15 | **53.74** | 124.41 | 139.41 | 358% |

The lag is a near-constant **~3.2–3.4s write-path cost**, independent of timeframe. Its
*significance* is therefore inversely proportional to the bar period.

**The task's framing — "fatal for the money path but fine for a comparison fleet" — is correct
at 1Min and wrong at 10Sec.** At 32% of a bar period, the pilot is not "the same decision
timestamped later": it is evaluating a *different bar boundary* than live was. Bar-boundary
state is exactly what entry/exit gating keys on ([[feedback_pb_boundary_semantics_ruling]]), so
a third-of-a-bar offset is **logic-visible**, not merely a timestamp offset. That converts the
asymmetry from a caveat you annotate into a confound you cannot separate.

Note also the gap-fill rows: `rest_insert`/`warmup_seed` at 10–15s land **30–60s** after bar
close. A pilot reading the DB either sees nothing at decision time or sees the bar a minute late.

### 5.1 Minor: bar values are revised under the pilot's feet
`first_*` (original WS publish) vs current values differ on **0.6% of 10s, 3.3% of 15s, 0.1% of
60s** bars — all `rest_correction`, and in every sample **volume-only** (price unchanged). Small,
but it lands precisely on the known volume-gate trap for sub-minute primaries
([[project_fine_tf_construction_falsified]]): a pilot re-reading a corrected bar can take a
volume-gated decision live never had the inputs to take.

---

## 6. What was NOT done, and why

**The pilot fleet was not stood up and no session was compared by two fleets.** Three blockers,
all structural rather than a matter of effort:

1. **No pilot exists to run.** `grep` over `src/` for `RORT_PILOT_FLEET_TAG` / `pilot_alerts`
   returns nothing. Steps 1–5 delivered a design, a migration, and an applied table — no runner,
   no `live_bars` feed adapter. E flagged this in step 1 ("**Chain gap — there is no BUILD
   step**"); the chain still has none.
2. **Standing one up is outside headless scope.** It needs a new Railway service + `railway up`
   / `variables` — a hard line for a headless run, denied at the permission layer, and correctly
   so.
3. **The session was already over.** RTH closed 20:00Z; this run was dispatched 21:22Z. The next
   full session is Monday 2026-08-03.

**Step 6 is therefore NOT ticked.** What is delivered instead is the measurement that decides
whether step 6 is worth its cost — computed from the same session it would have observed.

---

## 7. The verdict

**Is fleet-vs-fleet divergence a useful signal, or too noisy to interpret?**

**Split by timeframe. The answer is genuinely different on each side, and today's fleet is 95%
on the noisy side.**

**USEFUL — 1Min and coarser, for "same trades?" only.** 2.8% blind spot, lag 6% of a bar,
`ws_agg` coverage 95.8%. A divergence here is worth investigating. It answers *whether* the two
fleets take the same trades and says nothing about *when*.

**TOO NOISY — sub-minute (95% of today's decisions).** Not one cause but three compounding:
a 3.6–13.9% feed blind spot that mimics logic divergence; a lag worth 22–32% of the bar period,
which moves decisions across bar boundaries rather than merely delaying them; and a strict,
100%-of-the-time ordering that removes any hope of the asymmetry averaging out. An observed
sub-minute divergence **cannot be attributed to logic** without per-decision bar provenance that
the current design does not capture (§4.1).

### 7.1 What this means for #164 (V5.2) and #165 (V5.3)

- **V5.2 is promoted from optimization to precondition.** The design doc already says symmetric
  delivery is "the load-bearing property"; this measurement puts a number on it — below 1Min,
  asymmetric delivery does not degrade the comparison, it **invalidates** it. If the A/B is
  wanted for the sub-minute strategies that carry the live money, V5.2 is not optional and V5.1
  cannot substitute.
- **V5.1 is still worth finishing, cheaply and with its scope cut.** It can honestly answer
  "same trades?" at 1Min+, and it is the natural place to prove the `pilot_alerts` write path,
  the stop lever, and the tag isolation before V5.2 depends on all three. It should **not** be
  asked the sub-minute question.
- **Fix the stream before building** (§3) and **record bar-absence positively** (§4.1). Both are
  cheap now and expensive after a session has been run and interpreted.

---

## 8. Honest limits of this document

- **One session** (2026-07-31), one fleet shape, 4 symbols. Bar-availability is a market-activity
  function; a quieter or gappier day moves the 6.9%. Re-run before treating it as a constant.
- **This is a bound, not a run.** It measures what a pilot's input stream *could* have supported.
  It does not replace observing two fleets, and it cannot see divergences arising inside the
  engine. It says the pilot's ceiling is low at sub-minute; it does not report its floor.
- **[[project_presplit_divergence_baseline_264]] §9/§12.9 holes still apply**, and V5.1's own
  gate — **#161 environment-parity assertion** — remains unasserted. Per the design doc, V5.1 may
  run in parallel with #161 *provided no conclusions are drawn* until parity holds. §7's verdict
  is about **feed mechanics**, which #161 does not affect; any conclusion about *logic*
  divergence from an actual pilot run still waits on #161.
- Reproduce: probe scripts are in this run's scratchpad and are pure reads of `alerts` +
  `live_bars` over the fixed window above; the join is 20 lines and is described in §2 rather
  than shipped, deliberately — no new script enters `src/` for a measurement this task may not
  repeat.
