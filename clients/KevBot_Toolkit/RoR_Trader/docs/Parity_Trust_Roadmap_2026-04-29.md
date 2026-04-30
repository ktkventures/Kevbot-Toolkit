# Parity Trust Roadmap — 2026-04-29

## North star

> Add a new user pack (or general pack), back-test it, and trust that
> the back-test result reflects what live will actually do. No more
> "wait days to see if alerts fire" loop.

To get there, every bug class that has historically caused
"backtest-vs-live drift" needs to be caught at pack-creation time,
not discovered in production.

---

## Two-layer mental model

We have two parity test layers today. They catch *different* things,
and the gap between them is where every bug we've found has lived.

### Layer 1 — Pack 4Q simulator (`parity_simulator.run_pack_parity_test_4q`)

Tests the pack in isolation:

| Quadrant | What it verifies |
|---|---|
| Q1 | Trigger primary: trigger fires same in batch vs live for the pack alone |
| Q2 | Interpreter primary: incremental engine emits same column contract as batch |
| Q3 | Cross-TF: secondary-TF shadow engine values match a batch-resampled reference |
| Q4 | Data fidelity (SKIP) |

Coverage status: **all 8 v2 packs PASS** on default config (1Min × 15Min × 7d).

### Layer 2 — Strategy parity (`parity_service.run_strategy_parity`)

Replays a strategy's stored backtest trades through the live engine
on the same data window and compares matched/missed/extra fires.

Coverage status: **5 PARTIAL, 4 FAIL_LIVE_BLOCKED, 1 NO_TRADES** across
Tier-A mirrors (sids 135–144). New Tier B/C mirrors (sids 145–154) not
yet swept.

---

## Why pack-level PASS doesn't imply strategy-level PASS

Layer 1 deliberately exercises indicator math + interpreter logic +
shadow engine in isolation. It does **NOT** exercise:

1. **TriggerEvaluator dispatch path** — the wrapper that calls the
   pack's interpreter from inside the unified engine, with whatever
   DataFrame shape it constructs.
2. **Live data ingestion fan-out** — Polygon ticks → BarBuilder →
   primary engine → shadow secondary builders. A break here looks
   like "live emits zero," not "indicator drift."
3. **Parity-service matching logic** — anchor selection, timestamp
   truncation, replay window sizing.
4. **Full strategy state machine** — confluence gating, position
   state, stop validity guard, time exits, opposite-signal exits.

Every backtest-vs-live drift bug we've fixed lived in one of those
four areas, **not** in pack indicator/interpreter logic — which is
exactly what Layer 1 covers.

---

## Bug ledger (caught vs missed)

| Bug | Where it lived | Layer 1 verdict | Layer 2 verdict | Caught by |
|---|---|---|---|---|
| `ba85845` user-pack triggers fired silently for 2d | worker.py missing `pack_registry.scan_and_load_all()` | N/A — not a math bug | live emits 0 | manual debug |
| `6e42acc` tentative-state polluted user-pack `_prev_close` | unified_engine snapshot logic | PASS | drift | manual debug |
| `88be377` user-pack interpreters not running live | TriggerEvaluator dispatch | PASS | live emits 0 | manual debug |
| `ca57cac` Polygon migration broke cross-TF live | Polygon ingestion fan-out | Q3 PASS | live emits 0 on cross-TF strats | manual debug |
| `f68b410` 1-row DataFrame broke `df.shift(1)` | TriggerEvaluator dispatch | Q3 PASS on default config | partial (only 2 of 4 MACD_HIST states emitted) | strategy parity |
| `3aae5bc` parity replay anchored on wrong timestamp | parity_service matching | N/A | matched_count=0 by construction | manual drill of a PARTIAL |
| `swing_123 + cross-TF FAIL_LIVE_BLOCKED` (open) | unknown — likely cross-TF buffer or trigger eval | swing_123_test PASS Q3 | live emits 0 on 4 strats | strategy parity |

**Pattern:** Layer 1 has a 0/7 catch rate on engine-integration bugs.
Layer 2 catches them after-the-fact. We've been living in Layer 2 for
weeks.

---

## The leverage point — synthetic-strategy probe at pack creation

When an AI builds a new pack, the install flow already runs Q1+Q2.
Add an end-to-end probe that creates a one-off synthetic test
strategy and runs it through the full Layer 2 flow.

**Probe design:**

1. Spawn a sandboxed strategy in DB (admin context, hidden from My
   Strategies via a `is_synthetic_probe=true` flag or a synthetic
   user_id).
2. Configure it as:
   - `entry_trigger_confluence_id` = pack's bull/long entry trigger
   - `exit_trigger_confluence_ids` = pack's opposite-side trigger (or
     a default stop)
   - `confluence` = one record from a known-good external pack (e.g.
     `15m-MACD_LINE_V2-M>S+`) — exercises cross-TF gating + dispatch
   - Symbol: SPY, TF: 1Min, days: 7
3. Run `recompute_and_persist_stored_trades(probe_sid)` to materialize
   backtest trades.
4. Run `run_strategy_parity(probe_sid)`.
5. Surface verdict in wizard response. PASS = green-light install.
   PARTIAL/FAIL = block install with the same drill-down we use today
   (most_common_failing_gate + replay_only count).
6. Tear down the probe row + its stored_trades after — it's
   ephemeral.

**Why this works:** every bug in the ledger above would have been
caught by this probe because it runs the actual production
integration path on a controlled pack/config combo.

---

## Workflow loop

```
┌─ 1. Sweep strategy parity on existing mirrors ─┐
│                                                │
│   Goal: enumerate concrete bug classes          │
│   Output: a row per failure with                │
│           "what failed / where it lived"        │
└─────────┬──────────────────────────────────────┘
          │
          ▼
┌─ 2. Encode each bug as a regression case ──────┐
│                                                │
│   - Q3 default config exposed                  │
│     `f68b410` only on stress configs →         │
│     bump default to 14d / longer windows       │
│   - swing_123 cross-TF →                       │
│     add a Q3 variant with secondary_tf=5m,     │
│     primary_tf=10s                             │
│   - parity_service anchor →                    │
│     unit test on `_replay_strategy`            │
└─────────┬──────────────────────────────────────┘
          │
          ▼
┌─ 3. Wire synthetic probe into pack install ────┐
│                                                │
│   New pack → Q1 + Q2 + Q3 + synthetic probe    │
│   → install only on full PASS                  │
└─────────┬──────────────────────────────────────┘
          │
          ▼
┌─ 4. Deprecate manual strategy parity ──────────┐
│                                                │
│   Strategy parity becomes a debug tool, not a  │
│   release gate. New packs ship trustable.      │
└────────────────────────────────────────────────┘
```

---

## Phased execution

### Phase A — Sweep Tier B/C mirrors (DONE 2026-04-29 ~19:35 UTC)

`_rerun_all_parities.py --sids 145,146,147,149,150,151,152,153,154`.
9 strategies, ~28 min wall-clock with semaphore=1.

| sid | mirror of | entry | exit | confluence | verdict | matched/stored | replay_only |
|---|---|---|---|---|---|---|---|
| 145 | 59 | utv4 bull_flip | utv4 bear_flip | `1d-MACD_LINE_V2-M>S-`, `5m-UT_BOT_V4-BEAR_TREND` | **FAIL_LIVE_BLOCKED** | 0/132 | 0 |
| 146 | 64 | utv4 bull_flip | macd_line_v2 cross_bear | `1d-MACD_LINE_V2-M>S-`, `1M-MACD_LINE_V2-M<S-` | **FAIL_LIVE_BLOCKED** | 0/104 | 0 |
| 147 | 66 | eppv4 cross_short_up | utv4 bear_flip | `1d-MACD_LINE_V2-M>S-`, `5m-UT_BOT_V4-BEAR_TREND` | **FAIL_LIVE_BLOCKED** | 0/144 | 0 |
| 149 | 88 | stochastic | eppv4 cross_short_down | (none) | **PASS** | 200/200 | 0 |
| 150 | 114 | utv4 bull_flip | swing_123 c3 | (none) | **FAIL_LIVE_BLOCKED** | 3/4 | 0 |
| 151 | 117 | utv4 bull_flip | (none) | (none) | **PASS** | 1/1 | 0 |
| 152 | 124 | eppv4 cross_short_up | eppv4 cross_mid_down | (none) | **PASS** | 200/200 | 0 |
| 153 | 131 | utv4 bull_flip | swing_123 c2 | (none) | **PARTIAL** | 15/24 | 20 |
| 154 | 134 | eppv4 cross_short_up | eppv4 cross_mid_down | `15m-UT_BOT_V4-BULL_TREND` | **PARTIAL** | 175/200 | 21 |

(sid 148 skipped — broken backtest from migration outage, scale issue.)

**Three failure patterns identified — see Phase B.**

### Phase B — Enumerate concrete bug classes (in progress 2026-04-29)

Three patterns visible in the Phase A matrix. Each is a distinct bug
class; they may share a single root cause but should be drilled
separately.

#### Pattern 1 — Cross-TF MACD_LINE_V2 daily gate kills all live entries

Affected: **sids 145, 146, 147** — all three AAPL strategies, all use
`1d-MACD_LINE_V2-M>S-` as a cross-TF gate. All three: `0/N matches,
replay_only=0`. Live emits zero on the entry side.

**Hypothesis:** the shadow secondary-TF engine for the 1Day timeframe
classifies MACD_LINE_V2 state differently than the batch path. When
backtest builds the gate column from the resampled 1Day frame
(`prepare_data_with_indicators`), it gets `M>S-` on the relevant bars.
When live runs, the shadow `_user_pack_macd_line_v2` engine doesn't
emit the same state, so the gate is permanently False.

**Diff vs the f68b410 / 88be377 fixes:** those addressed the *primary*
TF user-pack dispatch. Cross-TF shadow dispatch may have a separate
bug (or the same fix didn't propagate to shadow engines).

**How to verify:** drill sid 145 — print live shadow's emitted
`MACD_LINE_V2` column on the 1Day frame for AAPL; compare to
`prepare_data_with_indicators(...secondary_tfs=('1Day',))` output for
the same window. Difference = the bug.

#### Pattern 2 — `utv4_default_bull_flip` mismatches even without cross-TF

Affected: **sid 150** (FAIL 3/4, replay_only=0) and **sid 153**
(PARTIAL 15/24, replay_only=20). No cross-TF gate. Pure entry-trigger
drift.

- sid 150: live fires 3 of 4 stored entries (75% match) but misses
  one specific trade.
- sid 153: live fires 35 entries vs 24 stored, only 15 overlap. So
  live is firing on different bars than backtest.

**Hypothesis:** flip-state triggers (`bull_flip`, `bear_flip`)
classify a bar as a flip when prev != current. In the unified engine
batch path, prev comes from `df.shift(1)` over the full pre-computed
df. In the live path, prev comes from the per-bar interpreter
dispatch. The 1-row→2-row dispatch fix (`f68b410`) was supposed to
unify these — but flip-state triggers may need MORE than one bar of
prev (e.g., they need persistent "trend regime" state, not just the
last bar's classification).

**How to verify:** drill sid 153 — for each of the 35 live entries
and 24 stored entries, log the bar timestamp, the prev/current
interpretation, and the trigger evaluation. The 20 replay_only
entries should reveal whether live is incorrectly seeing a flip
where backtest doesn't.

#### Pattern 3 — Cross-TF UT_BOT_V4 gate causes ~10% drift on otherwise-clean eppv4 strategy

Affected: **sid 154** (PARTIAL 175/200, replay_only=21). The base
strategy (eppv4 entry + eppv4 exit, no confluence) is sid 152 — which
PASSes 200/200. Adding the `15m-UT_BOT_V4-BULL_TREND` confluence drops
match rate to 87% with non-zero replay_only.

**Same root-cause family as Pattern 1**, but on a different
secondary-TF (15m, not 1d). Smaller magnitude because most bars agree
on `BULL_TREND`; the disagreement is at flip moments.

**Verdict:** Patterns 1 + 3 are likely the same bug, just different
TFs/packs. Pattern 2 is plausibly distinct.

---

## Drill findings (2026-04-29 ~21:00 UTC)

Used `_drill_parity_full.py` to invoke `run_strategy_parity()` directly
and inspect the per-bar `stored_only` reason breakdown (the persisted
`parity_status` only stores summary counts).

### Sid 145 (AAPL/1Min, 0/132) — REVISED hypothesis

**Original guess: cross-TF MACD_LINE_V2 1Day gate.**
**Drilled reality:** ALL 132 stored entries have `reason=TRIGGER_NOT_FIRED`
with `triggers_fired_in_replay=[]`. Live replay walked 31,775 bars and
fired ZERO triggers across the whole window. The gate hypothesis was
wrong — the trigger itself never fires.

```
verdict: FAIL_LIVE_BLOCKED  score: 0.0
stored: 132  matched: 0  stored_only: 132  replay_only: 0
reason breakdown: TRIGGER_NOT_FIRED  ×132
```

Same pattern on sids 146, 147 (also AAPL/1Min). Whatever the bug is,
it affects ALL user-pack triggers (utv4_bull_flip on 145/146,
eppv4_cross_short_up on 147) on AAPL specifically.

### Sid 153 (SPY/10Sec, 15/24, replay_only=20) — flip-state drift

```
verdict: PARTIAL  score: 0.34
stored: 24  matched: 15  stored_only: 9  replay_only: 20
reason breakdown: TRIGGER_NOT_FIRED  ×9
```

Live DOES fire `utv4_bull_flip` on SPY (35 fires total = 15 matched +
20 replay_only). Live and backtest just disagree on WHICH bars are
flips. This is real flip-state drift, not "live emits zero."

### Sid 154 (SPY/10Sec, 175/200, replay_only=21) — cross-TF None state CONFIRMED

```
verdict: PARTIAL  score: 0.79
stored: 200  matched: 175  stored_only: 25  replay_only: 21
reason breakdown:
  TRIGGER_NOT_FIRED  ×19
  GATE_FAILED        ×6

failing gates:
  '15m-UT_BOT_V4-BULL_TREND'  →  replay_actual: None  ×6
```

**The 6 GATE_FAILED entries are the smoking gun for Pattern 1+3.** Live
shadow engine for the 15m secondary TF emits `replay_actual=None` —
not "wrong state," literally null. The shadow's user-pack interpreter
dispatch is silently dropping output on these bars.

The other 19 are TRIGGER_NOT_FIRED on the primary side — same flip-state
drift as sid 153.

### Sid 150 (SPY/10Sec, 3/4) — confirms small-sample of Bug 3

1 TRIGGER_NOT_FIRED on `utv4_bull_flip`, no GATE_FAILED, no replay_only.
Tiny sample of the same Pattern 2 drift.

---

## Revised bug ledger — 3 distinct bugs

| Bug | Symptom | Evidence | Likely root cause |
|---|---|---|---|
| **Bug A — AAPL-specific user-pack trigger black hole** | Live replay fires ZERO triggers across 31,775 bars on AAPL strategies, regardless of which user-pack trigger is required | Sids 145/146/147: 380/380 entries TRIGGER_NOT_FIRED, replay's triggers_fired=[] every bar | Unknown — could be data-loader returning wrong bars for AAPL, RTH filter dropping AAPL bars, or pack engine instantiation broken for AAPL config. Needs a probe that loads AAPL/1Min via `_load_primary_and_secondary_bars` and runs `ut_bot_v4.update_bar()` on the bars to confirm/rule out incremental class |
| **Bug B — Shadow engine emits None for user-pack interpreter on secondary TF** | Cross-TF gate `15m-UT_BOT_V4-BULL_TREND` evaluates to None (not "wrong state") on 6 specific bars | Sid 154: 6 GATE_FAILED entries with `replay_actual=None` | f68b410 fixed the 1-row→2-row DataFrame issue for primary TF interpreter dispatch. May not have propagated identically to shadow engine's `_ShadowIndicatorEngine.on_bar_close()` path, or there's a remaining warmup-gap edge case |
| **Bug C — Flip-state trigger fires on different bars in live vs backtest** | Same trigger (utv4_bull_flip on sid 153, eppv4_cross_short_up on sid 154) fires on overlapping but non-identical bars between live replay and stored backtest | Sid 153: 15 matched + 9 stored_only + 20 replay_only. Sid 154: 175 matched + 19 TRIGGER_NOT_FIRED stored_only + 21 replay_only | Two candidate causes: (1) numerical drift between batch's full-df pass and incremental's bar-by-bar accumulation of ATR/EMA state over thousands of bars; (2) tick-level data difference between Polygon REST snapshot used for backtest and Polygon REST snapshot used for replay (different at-rest timestamps) |

### Why none of these are caught by current tests

- 4Q Q1+Q2+Q3 don't load real Polygon data per symbol — they synthesize a fixture. Bug A (AAPL-specific) cannot surface because pack tests don't iterate over symbols.
- 4Q Q3 default secondary_tf=15m × 7 days produces few state transitions. Bug B's "None state on specific bars" is a low-frequency edge case that needs longer windows.
- Strategy parity catches Bug C but only after-the-fact. The 4Q simulator never tests "trigger fires same in live as in batch over 30 days" — it tests "indicator values match" and "interpreter classifies same state," which are upstream.

### Action items derived from findings

1. **Bug A is highest-leverage to drill next** — single fix probably unblocks sids 145, 146, 147 (and explains the Tier-A swing_123 cluster on TSLA: 138, 139, 141, 143). Concrete next step: write `_probe_aapl_userpack.py` that loads AAPL/1Min bars via the same path parity_service uses, instantiates `ut_bot_v4.update_bar()` directly on those bars, and counts how many bull_flips are produced. If 0 → bar data is wrong. If many → integration bug between data and engine.

2. **Bug B is a shadow-engine specific check.** Read `_ShadowIndicatorEngine.on_bar_close` and verify it constructs a 2-row [prev, current] DataFrame for user-pack interpreters the same way `TriggerEvaluator.evaluate_bar_close` was fixed in f68b410.

3. **Bug C is the hardest to fix and probably the right thing to defer.** It's small-magnitude (10–20% drift), only matters on flip-state triggers, and a fix would require investigating tick-level data integrity vs replay reproducibility. Encode as a monitored metric ("flip-state trigger drift %") rather than blocking.

4. **All three bugs justify the synthetic-probe (Phase C).** The probe should run a real backtest + replay on a real symbol, exercising all three bug classes by construction. Pack creation can't ship if the probe surfaces any of them.

#### Cross-tier consolidation

Combining Tier A (Phase 2 from this doc) + Tier B/C (Phase A above):

| Pack used as entry | live-fires-zero count | other failure count |
|---|---|---|
| swing_123 (Tier A: 138, 139, 141, 143) | 4 | 0 |
| ut_bot_v4 (Tier B/C: 145, 146, 150) | 3 | 1 PARTIAL (153) |
| ema_pp_v4 (Tier B/C: 147 etc) | 1 (only when paired with cross-TF gate) | 1 PARTIAL (154) |
| ema_pp_v4 standalone | 0 | 0 (all PASS) |
| stochastic_oscillator | 0 | 0 (all PASS) |

**The two packs that look fine at 4Q level but break in real
strategies are exactly the two with flip-state triggers: swing_123
(c2/c3 confirmation states) and ut_bot_v4 (bull_flip/bear_flip).**
This strongly suggests a flip-state-specific bug in the live
interpreter dispatch path that 4Q's default config doesn't exercise
adequately.

#### Bug ledger updates (proposed for Phase D regression encoding)

| New entry | Layer that should catch | Stress config needed |
|---|---|---|
| Cross-TF user-pack interpreter mismatch (Pattern 1+3) | 4Q Q3 — but with secondary_tf=1Day, not just 15Min | Add 1Day secondary to default Q3 sweep |
| Flip-state trigger drift in live (Pattern 2) | Synthetic probe (Phase C) — needs full strategy run, not just pack 4Q | Probe must use a flip-state trigger as entry |

### Phase C — Build the synthetic-probe scaffolding

- New module: `src/api/services/synthetic_probe.py`
- Endpoint: extend `POST /api/packs/builder/user-packs/{slug}/parity-test`
  to also run the probe (or a separate `/synthetic-probe` endpoint
  depending on perf)
- Wire into `ai_builder.py` install handler (line 549 area)
- Frontend: surface verdict on PackBuilderPage wizard

### Phase D — Encode regressions in 4Q simulator

For each bug from Phase B, identify the minimal Q1/Q2/Q3 config that
would have caught it. Add as a default config or a "stress" sweep.

### Phase E — Retrofit existing packs

Re-run upgraded 4Q + synthetic probe on all 8 v2 packs. Any newly
caught failure goes to Phase B → fix → re-run.

### Phase F — Deprecate Layer 2 as a release gate

Strategy parity stays as a debugging/audit tool but no longer blocks
pack adoption. New packs ship trusted.

---

## Open questions / decisions deferred

1. **Synthetic probe runtime budget.** A full Layer 2 run on a probe
   strategy = ~30s of replay. Acceptable for AI install latency? Or
   run async and surface in a follow-up notification?
2. **Cross-pack confluence in probe** — should we test the pack
   gating *and* being gated, in two probe runs? (Add ~30s × 2.)
3. **Synthetic probe persistence** — keep historical probe runs in DB
   for "did this pack regress?" diffs, or always tear down?
4. **General packs** (time-of-day, day-of-week, calendar gates) —
   different test surface; probe shape needs adaptation.

---

## Anchor commitment

This doc is the source of truth for *why* we're investing in test
infrastructure instead of patching strategies one at a time. If we
catch ourselves debugging a strategy-level parity failure that doesn't
fit a known bug class in the ledger, **that's a signal to upgrade the
ledger and the synthetic probe**, not just fix the strategy.
