# V5.1 pilot — one session compared, and the verdict

**Board #163 step 6.** Session **2026-07-31 RTH (13:30–20:00Z)**, **1Min slice only**.
Run post-close by E·auto (run `r1785537537-163`).

---

## 0 · THE ASYMMETRY — read this before any number below

Two separate facts, and conflating them is the mistake this section exists to prevent.

**(a) A DB-fed fleet is structurally behind a socket-fed one, and it is not an average.**
Measured from this same session's real data (board #163, prior E run, verified by M):
**median +3.2s, p90 +4.1s, and the deciding bar landed in the DB *after* live had already
decided in 1,317 of 1,317 cases — 100.0%, zero overlap.** It is a fixed write-path cost, not
a distribution with a favourable tail. At 1Min that is **5% of a bar**; at 10Sec the identical
3.2s is **32% of a bar**, which means a different bar boundary — logic-visible, not a
timestamp offset. That is the whole reason this pilot is scoped to 1Min.

**(b) THIS RUN MEASURED NO TIMING AT ALL, AND CANNOT.**
Per M's revised scope, the pilot ran as an **offline replay over stored `live_bars`**, hours
after the close. Every timestamp compared below is a *bar-derived decision instant* computed
from stored data. Wall-clock timing from this run is an artifact of when the script ran.

**So the agreement numbers below mean: "the two fleets decided on the SAME BAR."**
They are **not** evidence that a live DB-fed pilot would have decided at the same *moment*.
Do not compute a latency statistic from this run — the real one is (a), already measured.

---

## 1 · What ran

A second engine instance consuming **stored `live_bars`** — never the socket — driven through
the real `StrategyMonitor` / `SymbolHub` / shadow-engine stack via `src/replay_harness.py`
(no re-implementation). Its entire write surface was **`pilot_alerts`**, `fleet_tag =
'pilot_ws_agg_db'`.

**Flag set: identical to the live armed stack** (`replay_harness.ARMED_FLAGS`, recorded per
row in `pilot_alerts.flag_set`). This is a deliberate **null-delta A/B**: with the logic held
constant, any divergence is attributable to the feed path and to live process state, not to a
flag difference. A flag-delta A/B is a later experiment, and it is only interpretable once
this baseline is known.

| sid | strategy | symbol | live alerts |
|---|---|---|---:|
| 136 | SPY LONG — Mass #11 [mirror 50] | SPY | 2 |
| 194 | TSLL LONG 1Min Mass #30 | TSLL | 4 |
| 269 | SPY-CANARY-1m-Control (ungated) | SPY | 51 |
| 340 | TSLA LONG 1Min Mass #31 | TSLA | 8 |
| 345 | NVDA LONG 1Min Mass #134 | NVDA | 6 |
| | | **total** | **71** |

---

## 2 · The comparison

**Pilot: 72 decisions (36 entries + 36 exits). Live: 71.**

| pairing tolerance | matched | pilot-only | live-only |
|---|---:|---:|---:|
| **exact second (0s)** | **66** | 6 | 5 |
| ≤10s | 67 | 5 | 4 |
| ≤60s (one bar) | **69** | **3** | **2** |

- **Same-bar agreement: 66/72 = 91.7%** of pilot decisions matched a live decision to the
  **exact second**.
- **Decision-set agreement: 69 of 74 distinct decisions = 93.2%.** Five genuine
  set-level disagreements; three further matches were the same decision *displaced in bar
  index* (below).
- **Every deciding bar for every disagreement EXISTS in `live_bars`.** The 2.8% 1Min
  feed blind spot found in the prior step did not bite this set — so **no disagreement below
  is a missing-bar artifact**. All five are real decision differences.

### 2.1 · All five set-level disagreements, enumerated

**sid 269 — SPY-CANARY-1m-Control (ungated) — 3 pilot-only decisions**

| # | pilot decision | live counterpart | bar in `live_bars`? |
|---|---|---|---|
| 1 | entry 14:19:00 | none | yes (`ws_agg`, written 14:20:03) |
| 2 | exit 14:20:00 | none | yes (`ws_agg`, written 14:21:03) |
| 3 | exit 14:55:00 | none | yes (`ws_agg`, written 14:56:03) |

**#1+#2 are a complete one-bar round trip the live fleet never took** — live's next alert in
that window is an entry at 14:23:00.

**#3 is attributable, and it is the most useful thing this run produced.** Over the whole
session the live lane on 269 has **exactly one entry→exit alternation violation**: entry
14:45:00 followed by entry 14:57:00 with **no exit between**. The pilot's unmatched exit at
**14:55:00 falls precisely in that hole.** Live is **26 entries / 25 exits (unbalanced)**;
the pilot is **27 / 27 (balanced, zero alternation violations)**. The pilot did not invent an
exit — **it recovered one the live lane failed to record.** That is a live-lane operational
loss (`#117` dangling-open family), surfaced by fleet-vs-fleet comparison on day one.

**sid 340 — TSLA, gated (`3M-VWAP_V2->+1σ` + `5M-BOLLINGER_BANDS-SQUEEZE_MID`) — 2 live-only**

| # | live decision | pilot counterpart | bar in `live_bars`? |
|---|---|---|---|
| 4 | entry 19:26:00 | none | yes (`ws_agg`, written 19:27:03) |
| 5 | exit 19:31:00 | none | yes (`ws_agg`, written 19:32:03) |

The bars were present, so the pilot **saw the data and declined the trade**. With flags held
identical, the residual is **gate state at decision time** — the known **live-process-state
class** (267/271/310/194: replay ceilings 99–100 against live 55–83, no offline fix, cure =
canonical live serving / state re-true, M-RS5). **Not attributed further in this run**; it is
consistent with that class rather than proven to be it.

### 2.2 · Three displaced matches (same decision, different bar)

| sid | pilot | live | Δ | reading |
|---|---|---|---:|---|
| 340 exit | 18:35:05 | 18:35:01 | **4s** | intra-bar exit (stop/target on 1Sec ticks) — same bar, sub-bar placement differs |
| 340 exit | 18:58:00 | 18:59:00 | **60s** | pilot exits **one bar earlier** |
| 345 exit | 18:36:00 | 18:37:00 | **60s** | pilot exits **one bar earlier** |

**Both 60s displacements go the same way — the pilot exits one bar earlier than live, 2 of 2.**
A consistent direction in a 2-sample set is suggestive, not conclusive; it is the pattern to
re-check first on the next run. It is **not** a latency reading (§0b) — both lanes' timestamps
are bar-derived. A one-bar-early exit is a state question, not a clock question.

---

## 3 · Coverage — say it out loud

**This covers 71 of 1,414 decisions = 5.0% of the session.** The other 95%
(10Sec 63.0%, 15Sec 32.0%) is **not** covered and cannot be by this design: `source='ws_agg'`
carries nothing below 60s, and the sub-minute feed blind spot (deciding bar absent from
`live_bars` at any source) is **3.4% at 10Sec and 13.9% at 15Sec** — a hole that mimics logic
divergence, on top of a 3.2s lag worth a third of a 10Sec bar. That 95% is blocked on **#164**.

---

## 4 · VERDICT

> **Fleet-vs-fleet divergence is a USEFUL signal at 1Min+, and it is not noise.**
> **It is uninterpretable below 1Min, and that is a feed problem, not a comparison problem.**

The load-bearing evidence that it is signal rather than noise:

1. **The disagreement rate is small and the disagreements are individually inspectable** —
   5 in 74, each traced to a named bar that provably existed.
2. **The comparison is falsifiable and it caught something real on its first session** — a
   live-lane missing exit, independently corroborated by an alternation violation in the live
   record itself. A noise generator does not produce a finding that a second, unrelated
   invariant confirms.
3. **4 of 5 disagreements are attributed** (3 → live-lane operational loss; 2 → consistent
   with the live-process-state class). A metric where most divergences resolve to a named
   cause is a diagnostic; one where they don't is a coin flip.
4. **Zero disagreements were feed artifacts** at 1Min — the 2.8% blind spot didn't fire, so
   the signal wasn't diluted by unreproducible decisions.

**What it does NOT establish:**

- **Nothing about timing.** §0. The money path's sub-second requirement is untouched by this.
- **Nothing about 95% of the fleet.** §3.
- **Nothing operational.** This was a replay, not a live second fleet. No Railway service, no
  deploy, no flags. Whether the plumbing behaves *while data flows* is unproven and needs a
  live session.
- **The 269 sample dominates.** 51 of 71 live decisions are one ungated canary. The gated
  strategies contributed 20 decisions — enough to enumerate, not enough to rate.

### What this means for #164 and #165

**#164 (the gateway) — JUSTIFIED, and its role changes.** It was scoped as an optimisation of
V5.1. It is a **precondition**: below 1Min, asymmetric delivery does not degrade the
comparison, it invalidates it. The 95% of the fleet that matters most — including every
15Sec first-live-money strategy — is unreachable without it.

**#165 (promote dev → main) — the A/B argument now has evidence, but thin evidence.** The
comparison works and is diagnostic, over 5% of decisions and one session. The blast-radius
argument for the split stands on its own; the A/B argument is now **supported rather than
speculative**, and should be re-weighed after #164 makes the sub-minute slice comparable.

**Recommended before spending on #164:** one more 1Min session, to test whether the two
one-bar-early exits (§2.2) repeat. If a directional exit bias is real it is a finding in its
own right, and it is cheap to check.

---

## 5 · Rails — isolation held

| check | result |
|---|---|
| `pilot_alerts` rows | **72**, all `fleet_tag='pilot_ws_agg_db'` (only distinct value) |
| `alerts` in session window | **1,414 before → 1,414 after** — unchanged |
| `alerts` with a pilot-ish name | **0** |
| `trades.data_source` values today | `{backtest_rest_hifi}` only — **no `pilot_%` lane** |
| pilot writes | exactly **one** `insert` into `pilot_alerts`; `replay_harness` is read-only |
| stoppable | the pilot was a foreground process that has exited; rows removable with `DELETE FROM pilot_alerts WHERE fleet_tag='pilot_ws_agg_db'` |

**No tag leak.** No live table was written, altered, or read as mixed.

`bar_read_at` is deliberately **NULL** on all 72 rows, with the reason recorded positively in
each row's `data.timing_note` (`feedback_no_silent_defaults`): read time in a replay is an
artifact of when the replay ran. A future reader who wants a latency number is pointed at the
measured one rather than handed a fabricated one.

---

## 6 · Design gaps carried forward

1. **The prior step's gap stands, untested.** `pilot_alerts` records a decision *taken*, not a
   decision *impossible to take*; a missing bar produces no row and reads as "evaluated, chose
   not to fire." This run could not exercise it — every 1Min deciding bar was present. It
   **will** fire the moment the pilot reaches sub-minute, where the blind spot is 3.4–13.9%.
2. **Row-level pairing was not where the value was.** The strongest finding (§2.1 #3) came
   from *set-level structure* — entry/exit alternation across the whole session — which no
   per-row comparison would have surfaced. A future pilot comparison should score sequence
   balance as a first-class check, not just nearest-timestamp pairing.

---

## 7 · Reproduce

```
# pilot fleet (5 × 1Min sids, stored live_bars → real engine)
.venv/bin/python replay_harness.py --sids 136,194,269,340,345 \
    --since 2026-07-31T13:30:00Z --until 2026-07-31T20:00:00Z
```
Driver + comparison scripts for this run: `pilot_run.py` / `pilot_compare.py` (scratchpad,
run `r1785537537-163`). Replay wall time: 8–55s per sid, 158s total.
